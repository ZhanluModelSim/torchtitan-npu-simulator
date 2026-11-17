# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-authoritative Engram storage, sparse training and Torch lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

try:  # torch builds differ in the opaque-object custom-class API surface
    from torch._library.opaque_object import (  # pyrefly: ignore [missing-module-attribute]
        CustomClassBase,  # pyrefly: ignore [missing-module-attribute]
        register_opaque_type,  # pyrefly: ignore [missing-module-attribute]
    )

    _HAS_OPAQUE_OBJECT_API = True
except ImportError:  # torch build without the opaque-object custom-class API
    _HAS_OPAQUE_OBJECT_API = False
from torchtitan.tools.logging import logger

from .core import EngramTable
from .lookup import HostEngramLookup, _lookup_grad_rows, _lookup_rows

_JOINT_TABLE_MESHES: dict[tuple[dist.ProcessGroup, dist.ProcessGroup], DeviceMesh] = {}


def _joint_table_mesh(parallel_dims) -> DeviceMesh:
    mesh = parallel_dims.get_mesh(["efsdp", "ep"])
    key = (mesh.get_group("efsdp"), mesh.get_group("ep"))
    if key not in _JOINT_TABLE_MESHES:
        ranks = mesh.mesh.flatten().tolist()
        group = dist.new_group(ranks=ranks, use_local_synchronization=True)
        assert isinstance(group, dist.ProcessGroup)
        _JOINT_TABLE_MESHES[key] = DeviceMesh.from_group(
            group, mesh.device_type, mesh=ranks, mesh_dim_names=("engram_shard",)
        )
    return _JOINT_TABLE_MESHES[key]


class HostEngramTable(EngramTable):
    """CPU table sharded over EP or EFSDP x EP, with Torch lookup and SparseAdam."""

    uses_host_offload = True

    @dataclass(kw_only=True, slots=True)
    class Config(EngramTable.Config):
        pin_memory: bool = False
        shard_over_efsdp: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.pin_memory = config.pin_memory
        self.shard_over_efsdp = config.shard_over_efsdp
        self._table_mesh: DeviceMesh | None = None
        self._efsdp_size = 1
        self._ep_rank = 0
        self._ep_size = 1
        self._pending_sparse_grad: torch.Tensor | None = None
        self._grad_keepalive = torch.zeros((), requires_grad=True)
        self._replica_group: dist.ProcessGroup | None = None
        self._replica_size = 1
        # Opaque custom-op handles may not be created inside a compiled
        # region, so materialize one per table up front and reuse it. On torch
        # builds without the opaque-object API the legacy lookup runs instead.
        self._host_lookup_handle = HostEngramTableHandle(self) if _HAS_OPAQUE_OBJECT_API else None
        self._mark_host_weight()

    def forward(
        self,
        input_ids_BL: torch.Tensor,
        positions_BL: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # EngramTable flattens the token/head axes with ``view``. Nested FSDP
        # warns that returning a view can lose its pre-backward hook if a
        # caller later performs an in-place op, so end the view chain at this
        # module boundary.
        return super().forward(input_ids_BL, positions_BL, **kwargs).clone()

    def _mark_host_weight(self) -> None:
        # Optimizer construction follows parallelize/to_empty, so preserve the
        # Host marker and checkpoint shard suffix on replacement parameters.
        self.weight._engram_host_offload = True  # type: ignore[attr-defined]
        self.weight._engram_checkpoint_suffix = self._checkpoint_suffix()  # type: ignore[attr-defined]

    @property
    def lookup_mesh(self):
        return self._table_mesh if self._table_mesh is not None else self.ep_mesh

    def _checkpoint_suffix(self) -> str:
        if self._table_mesh is not None:
            return f".efsdp_ep_shard_{self._table_mesh.get_local_rank():05d}_of_{self._table_mesh.size():05d}"
        if self._ep_size <= 1:
            return ""
        return f".ep_shard_{self._ep_rank:05d}_of_{self._ep_size:05d}"

    def parallelize(self, parallel_dims) -> None:
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        ep_size = ep_mesh.size() if ep_mesh is not None else 1
        self._table_mesh = None
        self._efsdp_size = 1
        if self.shard_over_efsdp:
            if ep_size <= 1:
                raise ValueError("Engram EFSDP sharding requires EP > 1.")
            efsdp_mesh = parallel_dims.get_optional_mesh("efsdp")
            self._efsdp_size = efsdp_mesh.size() if efsdp_mesh is not None else 1
            if self._efsdp_size > 1:
                self._table_mesh = _joint_table_mesh(parallel_dims)
        shard_size = ep_size * self._efsdp_size
        if self.num_embeddings % shard_size != 0:
            raise ValueError(
                f"Engram table has {self.num_embeddings} physical rows, not divisible by its shard degree {shard_size}."
            )

        # Keep weight out of generic SpmdLayout distribution. Hash metadata is
        # still parallelized by the inherited sharding config.
        global_weight = self._parameters.pop("weight")
        assert isinstance(global_weight, torch.nn.Parameter)
        try:
            super().parallelize(parallel_dims)
        finally:
            local_rows = self.num_embeddings // shard_size
            local_weight = torch.nn.Parameter(
                torch.empty(
                    local_rows,
                    self.embedding_dim,
                    dtype=global_weight.dtype,
                    device=global_weight.device,
                ),
                requires_grad=global_weight.requires_grad,
            )
            self.register_parameter("weight", local_weight)

        self._ep_rank = ep_mesh.get_local_rank() if ep_mesh is not None else 0
        self._ep_size = ep_size
        self._mark_host_weight()
        logger.info(
            "Engram layer %d: %s, CPU shard %s, EP%d x EFSDP%d storage shards, SparseAdam",
            self.layer_id,
            type(self).__name__,
            tuple(self.weight.shape),
            ep_size,
            self._efsdp_size,
        )

    def wire_sparse_grad_replicas(self, *, edp_mesh, edp_mesh_dims=None) -> None:
        """Record the ranks that hold a copy of this EP shard.

        By default EFSDP and DP-replicate carry replicas of each EP shard.
        With joint table sharding, only DP-replicate carries replicas. FSDP performs
        the equivalent reduction for the parameters it manages; this table is
        deliberately unmanaged, so it has to do it itself.

        Under the SPMD backends ``edp_mesh`` is the whole sparse storage mesh,
        EP axis included, and ``edp_mesh_dims`` is what names its data-parallel
        axes. Reducing over the EP axis as well would sum shards that own
        disjoint row ranges, so the axes are selected rather than flattened
        wholesale.
        """
        self._replica_group = None
        self._replica_size = 1
        if edp_mesh is None:
            return
        if edp_mesh_dims is not None:
            replica_axes = (edp_mesh_dims.replicate,)
            if self._efsdp_size == 1:
                replica_axes += (edp_mesh_dims.shard,)
            # Dense FSDP may fold (dp_shard, cp) into a tuple of shard axes.
            axes = tuple(name for axis in replica_axes if axis for name in ((axis,) if isinstance(axis, str) else axis))
            if not axes:
                return
            replica_mesh = edp_mesh[axes]
        elif self._efsdp_size > 1:
            if not edp_mesh.mesh_dim_names or "dp_replicate" not in edp_mesh.mesh_dim_names:
                return
            replica_mesh = edp_mesh["dp_replicate"]
        else:
            replica_mesh = edp_mesh
        if replica_mesh.size() <= 1:
            return
        # Replica groups for different EP shards are disjoint. Only members
        # participate in creating their group; no private mesh flattening API.
        self._replica_group = cast(
            "dist.ProcessGroup",
            (
                replica_mesh.get_group()
                if replica_mesh.ndim == 1
                else dist.new_group(ranks=replica_mesh.mesh.flatten().tolist(), use_local_synchronization=True)
            ),
        )
        self._replica_size = replica_mesh.size()
        logger.info(
            "Engram layer %d reduces its sparse gradient across %d replicas of its EP shard",
            self.layer_id,
            self._replica_size,
        )

    @torch.no_grad()
    def reduce_sparse_gradient_across_replicas(self) -> None:
        """Sum this step's sparse gradient over the replicas of this shard.

        Called from the gradient clipper, which TorchTitan runs unconditionally
        before every optimizer step, so the norm is taken on the reduced
        gradient. Each rank pads its owner-local rows to the group maximum,
        all-gathers them, and coalesces: every replica ends up with the same
        summed gradient, which keeps their weights and SparseAdam state in step
        without ever materializing a dense shard gradient.
        """
        group = self._replica_group
        if group is None:
            return

        device = self.token_id_map.device
        pending = self._pending_sparse_grad
        if pending is None:
            local_ids = torch.empty(0, dtype=torch.int64)
            local_values = torch.empty(0, self.embedding_dim, dtype=torch.float32)
        else:
            local_ids = pending.indices()[0]
            local_values = pending.values()

        # Row counts differ per replica, so the gather has to be padded to the
        # largest one; collecting the counts first also says where to cut.
        counts = torch.zeros(self._replica_size, dtype=torch.int64, device=device)
        counts[dist.get_rank(group)] = local_ids.numel()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        max_rows = int(counts.max())
        if max_rows == 0:
            return

        padded_ids = torch.zeros(max_rows, dtype=torch.int64, device=device)
        padded_ids[: local_ids.numel()] = local_ids.to(device=device)
        padded_values = torch.zeros(max_rows, self.embedding_dim, dtype=torch.float32, device=device)
        padded_values[: local_values.shape[0]] = local_values.to(device=device)

        gathered_ids = [torch.empty_like(padded_ids) for _ in range(self._replica_size)]
        gathered_values = [torch.empty_like(padded_values) for _ in range(self._replica_size)]
        dist.all_gather(gathered_ids, padded_ids, group=group)
        dist.all_gather(gathered_values, padded_values, group=group)

        ids_parts = []
        values_parts = []
        for replica in range(self._replica_size):
            rows = int(counts[replica])
            if rows:
                ids_parts.append(gathered_ids[replica][:rows].to(device="cpu", copy=True))
                values_parts.append(gathered_values[replica][:rows].to(device="cpu", copy=True))

        self._pending_sparse_grad = torch.sparse_coo_tensor(
            torch.cat(ids_parts).unsqueeze(0),
            torch.cat(values_parts),
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()

    def _apply(self, fn, recurse=True):
        """Apply device transforms to metadata while keeping weight on CPU."""
        weight = self._parameters.pop("weight")
        assert isinstance(weight, torch.nn.Parameter)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            if weight.is_meta:
                materialized = torch.empty(
                    weight.shape,
                    dtype=weight.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                weight = torch.nn.Parameter(materialized, requires_grad=weight.requires_grad)
            self.register_parameter("weight", weight)
            self._mark_host_weight()
        return result

    def storage_dtype(self, *, param_dtype: torch.dtype) -> torch.dtype:
        # The CPU shard is passed to FSDP as an ignored parameter, so the
        # mixed-precision policy never casts it.
        return self.weight.dtype

    def _init_self_parameters(self) -> None:
        if self.weight.device.type != "cpu":
            raise RuntimeError(
                f"Host-offload Engram weight must be on CPU during initialization, got {self.weight.device}."
            )
        # Rank-dependent streams avoid repeating the same local shard on every
        # EP owner. Exact cross-EP-degree initialization parity is not promised;
        # checkpoints preserve the initialized values thereafter.
        shard_rank = self._table_mesh.get_local_rank() if self._table_mesh is not None else self._ep_rank
        seed = (torch.initial_seed() + 1000003 * self.layer_id + 9176 * shard_rank) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._init_param("weight", self.weight)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # The table is CPU-resident, while hash metadata follows input_ids on
        # the accelerator. After to_empty(), token_id_map records that device.
        device = buffer_device if buffer_device is not None else self.token_id_map.device
        EngramTable._init_self_buffers(self, buffer_device=device)
        # The keepalive gradient arrives from the accelerator, so the anchor has
        # to live there too.
        self._grad_keepalive = torch.zeros((), device=device, requires_grad=True)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        suffix = self._checkpoint_suffix()
        if suffix:
            destination[prefix + "weight" + suffix] = destination.pop(prefix + "weight")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        suffix = self._checkpoint_suffix()
        shard_key = prefix + "weight" + suffix
        weight_key = prefix + "weight"
        if suffix and shard_key in state_dict:
            state_dict[weight_key] = state_dict.pop(shard_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def accumulate_sparse_gradient(
        self,
        local_row_ids: torch.Tensor,
        grad_rows: torch.Tensor,
    ) -> None:
        """Accumulate one microbatch's owner-local rows in FP32 on CPU."""
        local_row_ids = local_row_ids.reshape(-1)
        grad_rows = grad_rows.reshape(local_row_ids.numel(), self.embedding_dim)
        ids = local_row_ids.to(device="cpu", dtype=torch.int64)
        values = grad_rows.to(device="cpu", dtype=torch.float32)
        if ids.numel():
            min_id = int(ids.min())
            max_id = int(ids.max())
            if min_id < 0 or max_id >= self.weight.shape[0]:
                raise IndexError(
                    "Engram lookup returned local row outside "
                    f"[0, {self.weight.shape[0]}): got range [{min_id}, {max_id}] "
                    f"across {ids.numel()} sparse entries."
                )
        sparse_grad = torch.sparse_coo_tensor(
            ids.unsqueeze(0),
            values,
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()
        if self._pending_sparse_grad is None:
            self._pending_sparse_grad = sparse_grad
        else:
            self._pending_sparse_grad = (self._pending_sparse_grad + sparse_grad).coalesce()

    def prepare_sparse_optimizer_step(self) -> None:
        if self._pending_sparse_grad is None:
            self.weight.grad = None
            return
        self.weight.grad = self._pending_sparse_grad.to(dtype=self.weight.dtype)

    def clear_sparse_gradient(self) -> None:
        self._pending_sparse_grad = None
        self.weight.grad = None

    def mark_sparse_step_complete(self) -> None:
        self._pending_sparse_grad = None

    def refresh_lookup_storage(self, row_ids: torch.Tensor) -> None:
        """Refresh derived lookup rows after an optimizer update, if any.

        FP32 lookup uses the master weight directly and needs no refresh.
        """

    def pending_sparse_grad(self) -> torch.Tensor | None:
        """The accumulated CPU sparse gradient, before it reaches SparseAdam."""
        return self._pending_sparse_grad

    def scale_pending_sparse_grad(self, scale: float) -> None:
        """Apply the global gradient-clipping coefficient to the sparse gradient."""
        if self._pending_sparse_grad is None:
            return
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            self._pending_sparse_grad.indices(),
            self._pending_sparse_grad.values() * scale,
            self._pending_sparse_grad.shape,
        ).coalesce()

    def _distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        if not _HAS_OPAQUE_OBJECT_API:
            # Torch builds without the opaque-object API keep the legacy
            # autograd lookup: eager training only.
            if torch.compiler.is_compiling():
                raise RuntimeError("Torch Host Engram lookup currently supports eager training only.")
            return HostEngramLookup.apply(self.weight, row_ids_N, self, self._grad_keepalive)
        # The host lookup runs behind a custom-op boundary so torch.compile
        # fullgraph treats it as an opaque node instead of tracing the
        # data-dependent all_to_all / CPU row access. The same op serves the
        # eager path, so numerics are identical in both modes.
        return _host_engram_lookup(
            self.weight,
            row_ids_N,
            self._grad_keepalive,
            self._ep_size > 1,
            self._host_lookup_handle,
        )[0]


class _OpCtx:
    """Adapter that lets the module-level lookup helpers save state on a
    plain namespace when they run inside the custom-op implementations."""

    def __init__(self) -> None:
        self.saved_tensors: tuple = ()
        self.send_splits: list | None = None
        self.recv_splits: list | None = None
        self.group = None

    def save_for_backward(self, *tensors) -> None:
        self.saved_tensors = tensors


# The opaque-handle custom-op machinery below is only definable on torch
# builds that expose torch._library.opaque_object's custom-class API;
# otherwise HostEngramTable falls back to the legacy eager lookup above.
if _HAS_OPAQUE_OBJECT_API:

    class HostEngramTableHandle(CustomClassBase):
        """Opaque reference to a ``HostEngramTable`` across custom-op boundaries."""

        def __init__(self, value: HostEngramTable):
            self.value: HostEngramTable = value

        def __eq__(self, other):
            return isinstance(other, HostEngramTableHandle) and self.value is other.value

        def __hash__(self):
            return id(self.value)

    register_opaque_type(HostEngramTableHandle, typ="reference")  # pyrefly: ignore [unbound-name]

    @torch.library.custom_op("torchtitan_npu::host_engram_lookup", mutates_args=())
    def _host_engram_lookup(
        weight: torch.Tensor,
        row_ids: torch.Tensor,
        keepalive: torch.Tensor,
        distributed: bool,
        table: HostEngramTableHandle,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Opaque host-table lookup so fullgraph torch.compile never traces the
        data-dependent all_to_all / CPU row access. Returns ``rows`` plus the
        tensors the backward op needs at runtime."""
        if distributed:
            ctx = _OpCtx()
            group = table.value.lookup_mesh.get_group()  # pyrefly: ignore [missing-attribute]
            rows = _lookup_rows(ctx, weight, row_ids, group)
            inverse, received_ids = ctx.saved_tensors
            return (
                rows,
                inverse,
                received_ids,
                torch.tensor(ctx.send_splits, dtype=torch.int64),
                torch.tensor(ctx.recv_splits, dtype=torch.int64),
            )
        local_ids = row_ids.reshape(-1).to(device="cpu", copy=True)
        rows = weight.index_select(0, local_ids).to(device=row_ids.device).view(*row_ids.shape, weight.shape[1])
        # Custom-op outputs may not alias each other, so hand out distinct
        # placeholder tensors for the unused backward slots.
        e1 = row_ids.new_empty((0,), dtype=torch.int64)
        e2 = row_ids.new_empty((0,), dtype=torch.int64)
        e3 = row_ids.new_empty((0,), dtype=torch.int64)
        return rows, e1, local_ids, e2, e3

    @_host_engram_lookup.register_fake
    def _host_engram_lookup_fake(weight, row_ids, keepalive, distributed, table):
        # The auxiliary outputs are consumed only by the equally opaque backward
        # op, so placeholder shapes suffice; ``rows`` mirrors the real lookup.
        e1 = row_ids.new_empty((0,), dtype=torch.int64)
        e2 = row_ids.new_empty((0,), dtype=torch.int64)
        e3 = row_ids.new_empty((0,), dtype=torch.int64)
        e4 = row_ids.new_empty((0,), dtype=torch.int64)
        rows = row_ids.new_empty((*row_ids.shape, weight.shape[1]), dtype=weight.dtype)
        return rows, e1, e2, e3, e4

    def _engram_lookup_setup_context(ctx, inputs, output) -> None:
        _weight, row_ids, _keepalive, distributed, table = inputs
        _rows, inverse, received_ids, send_counts, recv_counts = output
        ctx.table = table
        ctx.distributed = distributed
        ctx.anchor_device = inputs[2].device
        ctx.anchor_dtype = inputs[2].dtype
        if distributed:
            ctx.save_for_backward(inverse, received_ids, send_counts, recv_counts)
        else:
            ctx.save_for_backward(row_ids)

    def _engram_lookup_backward(ctx, grad_rows, _g_inverse, _g_received, _g_send, _g_recv):
        if ctx.distributed:
            inverse, received_ids, send_counts, recv_counts = ctx.saved_tensors
            g_keepalive = _host_engram_lookup_backward(
                grad_rows, inverse, received_ids, send_counts, recv_counts, grad_rows, True, ctx.table
            )
        else:
            (row_ids,) = ctx.saved_tensors
            g_keepalive = _host_engram_lookup_backward(
                grad_rows, grad_rows, grad_rows, grad_rows, grad_rows, row_ids, False, ctx.table
            )
        # Sparse grads reach the host table via ``accumulate_sparse_gradient``
        # inside the backward op. The op's scalar output becomes the keepalive
        # gradient, which forces the compiled backward graph to keep (and run)
        # the op; a constant zeros grad would let it be DCEd away.
        return None, None, g_keepalive, None, None

    torch.library.register_autograd(
        "torchtitan_npu::host_engram_lookup",
        _engram_lookup_backward,
        setup_context=_engram_lookup_setup_context,
    )

    @torch.library.custom_op("torchtitan_npu::host_engram_lookup_backward", mutates_args=())
    def _host_engram_lookup_backward(
        grad_rows: torch.Tensor,
        inverse: torch.Tensor,
        received_ids: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        row_ids: torch.Tensor,
        distributed: bool,
        table: HostEngramTableHandle,
    ) -> torch.Tensor:
        """Runtime side effect: reduce grad rows and accumulate host sparse grads."""
        if distributed:
            ctx = _OpCtx()
            ctx.saved_tensors = (inverse, received_ids)
            ctx.send_splits = send_counts.tolist()
            ctx.recv_splits = recv_counts.tolist()
            ctx.group = table.value.lookup_mesh.get_group()  # pyrefly: ignore [missing-attribute]
            grad_local_rows, local_ids = _lookup_grad_rows(ctx, grad_rows)
        else:
            local_ids = row_ids.reshape(-1).to(device="cpu", copy=True)
            grad_local_rows = grad_rows.reshape(-1, grad_rows.shape[-1])
        table.value.accumulate_sparse_gradient(local_ids, grad_local_rows)
        # Returned as the keepalive gradient so the compiled backward graph
        # must execute this op (and its accumulate side effect) instead of DCEing
        # it as an unused node.
        return grad_rows.new_zeros((), dtype=torch.float32)

    @_host_engram_lookup_backward.register_fake
    def _host_engram_lookup_backward_fake(
        grad_rows, inverse, received_ids, send_counts, recv_counts, row_ids, distributed, table
    ):
        # Returned as the keepalive gradient so the compiled backward graph
        # must execute this op (and its accumulate side effect) instead of DCEing
        # it as an unused node.
        return grad_rows.new_zeros((), dtype=torch.float32)


__all__ = ["HostEngramTable"]
