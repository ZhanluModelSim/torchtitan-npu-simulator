# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture support for DeepSeek-V4.1's host-offload Engram table.

The production Engram table is CPU-resident (``uses_host_offload``) and its
initialization asserts a CPU weight. Meta capture builds the model on the meta
device, where the assertion is wrong but nothing else changes: initialization
performs no allocation and the table's real-device residency stays CPU, which
is a memory-budget fact rather than a graph fact.

The host sparse-gradient side effects are bridged explicitly for Meta (PR #19
review): the pending sparse gradient accumulates on meta tensors instead of
being dropped by the fake backward, and the host sparse gradient clipping keeps
its op structure without the value-dependent host scalars, so the captured
F/B/O includes the sparse work the dense gradient contract exempts.
"""

import torch
from torch.profiler import record_function

from torchtitan_simulator.capture.dispatch_capture import (
    annotate_active_capture_events,
    host_only_work,
)
from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)


@torch.library.custom_op("dsv41_sim::sparse_adam_update", mutates_args=("parameter", "exp_avg", "exp_avg_sq"))
def sparse_adam_update(
    parameter: torch.Tensor,
    grad_indices: torch.Tensor,
    grad_values: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """Virtual boundary of one native SparseAdam table update.

    The native functional coalesces the sparse gradient and masks the moments
    through SparseMeta kernels that do not exist, so the simulator records the
    update at this declared boundary instead: the same parameter selection,
    state materialization and per-table work as production, executed only in
    Meta capture. The sparse gradient enters as its dense index/value rows.
    """
    raise RuntimeError("DeepSeek-V4.1 virtual SparseAdam update is available only for Meta simulation")


@sparse_adam_update.register_fake
def _sparse_adam_update_meta(parameter, grad_indices, grad_values, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps):
    return None


register_tensor_semantics(
    "dsv41_sim.sparse_adam_update.default",
    OperatorTensorSemantics(
        inputs=(
            ("parameter", TensorMeaning("host_table_weight", ("table_row", "embedding_dim"))),
            ("grad_indices", TensorMeaning("host_sparse_gradient_row_ids", ("sparse_dim", "sparse_entry"))),
            ("grad_values", TensorMeaning("host_sparse_gradient_values", ("sparse_entry", "embedding_dim"))),
            ("exp_avg", TensorMeaning("first_moment", ("table_row", "embedding_dim"))),
            ("exp_avg_sq", TensorMeaning("second_moment", ("table_row", "embedding_dim"))),
        ),
        outputs=(
            TensorMeaning("host_table_weight", ("table_row", "embedding_dim")),
            TensorMeaning("first_moment", ("table_row", "embedding_dim")),
            TensorMeaning("second_moment", ("table_row", "embedding_dim")),
        ),
    ),
)


def _meta_sparse_adam_step(self, closure=None):
    """Meta twin of the native SparseAdam step (PR #19 review).

    Mirrors the native parameter selection, dense-gradient rejection and
    lazy state materialization exactly; the per-table update runs through
    the declared virtual boundary because the native functional needs
    SparseMeta kernels that do not exist.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    with torch.no_grad():
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if not grad.is_sparse:
                    raise RuntimeError("SparseAdam does not support dense gradients, please consider Adam instead")
                with host_only_work():
                    # State materialization and the sparse row extraction are
                    # CPU work in production; only the declared update boundary
                    # below belongs to the captured graph.
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["step"] += 1
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                sparse_adam_update(
                    parameter,
                    grad_indices,
                    grad_values,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                )
    return loss


def _meta_accumulate_sparse_gradient(self, local_row_ids: torch.Tensor, grad_rows: torch.Tensor) -> None:
    """Meta twin of the CPU accumulation: no device moves, no value reads.

    Production moves rows to CPU, range-checks ids by value and coalesces the
    sparse gradient; SparseMeta has no coalesce kernel and meta has no data
    to move, read or deduplicate, so this twin keeps the FP32 sparse
    accumulation as index/value concatenation and marks the result coalesced
    by declaration -- every downstream sparse consumer (the clipping rescale,
    ``prepare_sparse_optimizer_step`` and the SparseAdam update) requires the
    coalesced flag, and on meta the entries carry shapes only. The coalesce
    itself is host-only work in production and stays outside the device graph.
    """
    with host_only_work():
        local_row_ids = local_row_ids.reshape(-1).to(dtype=torch.int64)
        grad_rows = grad_rows.reshape(local_row_ids.numel(), self.embedding_dim).to(dtype=torch.float32)
        if self._pending_sparse_grad is None:
            self._pending_sparse_grad = torch.sparse_coo_tensor(
                local_row_ids.unsqueeze(0),
                grad_rows,
                size=self.weight.shape,
                dtype=torch.float32,
                check_invariants=False,
            )
        else:
            pending = self._pending_sparse_grad
            self._pending_sparse_grad = torch.sparse_coo_tensor(
                torch.cat((pending.indices(), local_row_ids.unsqueeze(0)), dim=1),
                torch.cat((pending.values(), grad_rows), dim=0),
                size=self.weight.shape,
                dtype=torch.float32,
                check_invariants=False,
            )
        self._pending_sparse_grad._coalesced_(True)


def _meta_scale_pending_sparse_grad(self, scale) -> None:
    """Meta twin of the clipping rescale, keeping the coalesced declaration."""
    if self._pending_sparse_grad is None:
        return
    with host_only_work():
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            self._pending_sparse_grad.indices(),
            self._pending_sparse_grad.values() * scale,
            self._pending_sparse_grad.shape,
            check_invariants=False,
        )
        self._pending_sparse_grad._coalesced_(True)


def _meta_prepare_sparse_optimizer_step(self) -> None:
    """Meta twin of the pending-to-weight.grad attach.

    Production converts the pending gradient to the weight dtype; on meta
    the conversion is FP32 identity but dispatches ``_to_copy`` with a
    sparse operand, which is host work and must not enter the graph.
    """
    if self._pending_sparse_grad is None:
        self.weight.grad = None
        return
    with host_only_work():
        self.weight.grad = self._pending_sparse_grad.to(dtype=self.weight.dtype)


def _meta_reduce_sparse_gradient_across_replicas(self) -> None:
    """Meta twin of the replica reduction (single-process capture view).

    Production pads the owner-local rows to the group maximum and
    all-gathers them across the dp replicas of this shard; meta cannot read
    row counts by value, and the single-process capture carries one logical
    rank, so the peer replicas' rows do not exist in-process and the
    reduction collapses to the local pending gradient. The missing host
    collective is a documented single-process gap, not a claim that
    production skips it.
    """
    group = self._replica_group
    if group is None:
        return


def _meta_clip_grad_norm_with_host_sparse_tables(
    parameters,
    max_norm,
    norm_type=2.0,
    *args,
    tables,
    clipper,
    scale_dense_gradients=None,
    **kwargs,
):
    """Meta twin of ``clip_grad_norm_with_host_sparse_tables``.

    The production wrapper reads the norms by value (``float``/``isfinite``
    guards) to decide whether to rescale; on meta the norms are symbolic, so
    this twin keeps the exact op structure and applies the clamped coefficients
    unconditionally -- the same convention the native dense clipper already
    follows on meta (``_foreach_mul_`` with a clamped coefficient tensor).
    """
    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    dense_norm = clipper(parameters, max_norm, norm_type, *args, **kwargs)
    # WeakSet order depends on local object addresses. Replica collectives
    # must visit the same layer on every rank, and norm summation must be stable.
    registered_tables = sorted(tables, key=lambda table: table.layer_id)
    for table in registered_tables:
        table.reduce_sparse_gradient_across_replicas()
    pending_tables = [table for table in registered_tables if table.pending_sparse_grad() is not None]
    if not pending_tables:
        return dense_norm
    if norm_type != 2.0:
        raise NotImplementedError(f"Host Engram gradient clipping supports the 2-norm only, got norm_type={norm_type}.")
    group_squares: dict = {}
    for table in pending_tables:
        lookup_mesh = table.lookup_mesh
        if lookup_mesh is not None and lookup_mesh.size() > 1:
            # Single-process capture owns one logical rank: summing row
            # squares across an EP group needs the multi-rank collective,
            # which is outside the single-process capture contract.
            raise RuntimeError("Host Engram row-squared reduction is outside the single-process capture contract")
        # Rows are owner-local at EP degree 1: the production all_reduce over
        # the trivial row-ownership group is the identity.
        group = lookup_mesh.get_group() if lookup_mesh is not None else None
        if group not in group_squares:
            group_squares[group] = torch.zeros((), dtype=torch.float32)
        pending = table.pending_sparse_grad()
        assert pending is not None
        # The sparse square is host-computed in production; only its scalar
        # joins the device coefficient math below.
        with host_only_work():
            group_squares[group] = group_squares[group] + pending.values().detach().float().pow(2).sum()
    sparse_sq = torch.zeros((), dtype=torch.float32, device=dense_norm.device)
    for group, local_sq in group_squares.items():
        # The row-ownership group was verified trivial above; the production
        # all_reduce is the identity, and the host scalar joins the device
        # coefficient math like production's transfer to the dense device.
        sparse_sq = sparse_sq + local_sq.to(dense_norm.device)
    dense_norm_f32 = dense_norm.detach().float()
    total_norm = torch.sqrt(dense_norm_f32.pow(2) + sparse_sq)
    dense_coef = torch.clamp(max_norm / (dense_norm_f32 + 1e-6), max=1.0)
    total_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    correction = total_coef / dense_coef
    # Production reads the correction scalar and rescales only when it is
    # not 1.0; meta cannot read values, so the rescale records
    # unconditionally under a distinct scope, keeping the dense gradients
    # at the combined coefficient like production's taken branch.
    with annotate_active_capture_events("gradient_clip_sparse_correction"):
        if scale_dense_gradients is not None:
            scale_dense_gradients(parameters, correction)
        else:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
    for table in pending_tables:
        table.scale_pending_sparse_grad(total_coef)
    return total_norm.to(dense_norm.dtype)


def _meta_host_sparse_container_step(self, closure=None):
    """Meta twin of ``HostSparseOptimizersContainer.step`` (PR #19 review).

    The native SparseAdam functional needs SparseMeta kernels that do not
    exist; the container twin keeps the production body -- prepare,
    per-optimizer steps, refresh, mark -- and routes the sparse table
    optimizers through the meta step, which records the per-table update
    through the declared virtual boundary.

    TorchTitan's ``OptimizersContainer._post_init`` runs
    ``Optimizer.__init__`` on the container instance, whose
    ``_patch_step_function`` rebinds the class ``step`` with
    ``profile_hook_step`` (its own escape hatch is the ``hooked`` marker on
    an already-wrapped step). The marker below keeps this binding intact
    through the first container construction; the profile wrapper only adds
    a ``record_function`` envelope that the twin already carries, and the
    step-boundary tracker still saves and restores the twin exactly.
    """
    assert closure is None, "OptimizersContainer does not support closures"
    with record_function("engram::sparse_grad_attach"):
        for table in self._iter_host_tables():
            table.prepare_sparse_optimizer_step()
    for optimizer in self.optimizers:
        if isinstance(optimizer, torch.optim.SparseAdam):
            _meta_sparse_adam_step(optimizer)
        else:
            optimizer.step()
    for table in self._iter_host_tables():
        pending = table.pending_sparse_grad()
        if pending is not None:
            with record_function("engram::refresh_lookup_storage"):
                table.refresh_lookup_storage(pending.indices()[0])
        table.mark_sparse_step_complete()
    return None


# Torch's Optimizer._patch_step_function skips its profile wrapper for steps
# already carrying the 'hooked' marker; without it the first container build
# would wrap this binding over and the patch ownership check would fire.
_meta_host_sparse_container_step.hooked = True  # pyrefly: ignore [missing-attribute]


def install_deepseek_v4_1_meta_support(scope):
    """Allow the host-offload Engram table to initialize and train on meta."""
    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
        HostSparseOptimizersContainer,
    )
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

    def _meta_init_self_parameters(self):
        # Mirror the production seed derivation exactly; only the device
        # assertion is relaxed for meta, where init allocates nothing.
        if self.weight.device.type not in {"cpu", "meta"}:
            raise RuntimeError(
                f"Host-offload Engram weight must initialize on CPU or meta, got {self.weight.device}."
            )
        shard_rank = self._table_mesh.get_local_rank() if self._table_mesh is not None else self._ep_rank
        seed = (torch.initial_seed() + 1000003 * self.layer_id + 9176 * shard_rank) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._init_param("weight", self.weight)

    def _meta_container_clip_grad_norm(self, parameters, max_norm, **kwargs):
        from torchtitan.distributed.utils import clip_grad_norm_

        return _meta_clip_grad_norm_with_host_sparse_tables(
            parameters,
            max_norm,
            tables=list(self._iter_host_tables()),
            clipper=clip_grad_norm_,
            scale_dense_gradients=self._scale_dense_gradients,
            **kwargs,
        )

    scope.replace(HostEngramTable, "_init_self_parameters", _meta_init_self_parameters)
    # The production accumulator moves rows to CPU and reads ids by value,
    # neither of which exists on meta; the meta twin keeps the FP32 sparse
    # accumulation structure (PR #19 review).
    scope.replace(HostEngramTable, "accumulate_sparse_gradient", _meta_accumulate_sparse_gradient)
    # The production clipping rescale coalesces; SparseMeta cannot.
    scope.replace(HostEngramTable, "scale_pending_sparse_grad", _meta_scale_pending_sparse_grad)
    # The production attach converts the pending gradient to the weight
    # dtype, which dispatches a sparse _to_copy; the meta twin keeps it as
    # host-only work (FP32 identity on the capture path).
    scope.replace(HostEngramTable, "prepare_sparse_optimizer_step", _meta_prepare_sparse_optimizer_step)
    # The native SparseAdam functional needs SparseMeta kernels that do not
    # exist; the container twin routes the sparse table optimizers through
    # the meta step, which records the per-table update through the declared
    # virtual boundary (see _meta_host_sparse_container_step).
    scope.replace(HostSparseOptimizersContainer, "step", _meta_host_sparse_container_step)
    # The production replica reduction pads and all-gathers on CPU with
    # value-read row counts; SparseMeta cannot run it and the single-process
    # capture has no peer rows to gather (see the twin docstring).
    scope.replace(
        HostEngramTable, "reduce_sparse_gradient_across_replicas", _meta_reduce_sparse_gradient_across_replicas
    )
    # The fake backward of the opaque host lookup never calls the accumulator,
    # so Meta capture bridges the side effect explicitly: a gradient hook on
    # the lookup rows accumulates the same owner-local rows the production
    # backward op would, in warmup and capture alike.
    _production_lookup = HostEngramTable._distributed_lookup

    def _meta_distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        rows = _production_lookup(self, row_ids_N)

        def _accumulate_meta_sparse_grad(grad_rows: torch.Tensor) -> torch.Tensor:
            self.accumulate_sparse_gradient(row_ids_N, grad_rows)
            return grad_rows

        rows.register_hook(_accumulate_meta_sparse_grad)
        return rows

    scope.replace(HostEngramTable, "_distributed_lookup", _meta_distributed_lookup)
    # The production wrapper reads norms by value; the meta twin keeps the
    # op structure with symbolic scalars instead.
    scope.replace(HostSparseOptimizersContainer, "clip_grad_norm", _meta_container_clip_grad_norm)


def _build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
    """The runner's transitional seam for the V4.1 attention-metadata map.

    V4.1 has no model-level ``build_attention_masks``; the pinned decoder
    patch constructs the ``AttentionMetadataMap`` itself through
    ``_get_attention_metadata``. The adapter reuses exactly that builder so
    the capture path matches the production prepare-input branch, and keeps
    CP out of scope as for the other families.
    """
    if cp_mesh is not None:
        if load_balancer_type is not None:
            raise NotImplementedError("DeepSeek-V4.1 simulator capture does not use a CP load balancer")
        raise NotImplementedError("DeepSeek-V4.1 simulator capture does not support context parallel")
    from torchtitan_npu.patches.torchtitan.models.common.decoder import _get_attention_metadata

    metadata = _get_attention_metadata(self, extra_inputs.get("positions"))
    extra = dict(extra_inputs)
    if metadata:
        extra["attention_metadata"] = metadata
    return tokens, labels, extra


def install_deepseek_v4_1_input_adapter(scope):
    """Bind the runner's transitional input contract to the metadata map."""
    from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model

    scope.replace(DeepSeekV41Model, "build_attention_masks", _build_attention_masks)


__all__ = ["install_deepseek_v4_1_input_adapter", "install_deepseek_v4_1_meta_support"]
