# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Intercepts torch.distributed collective calls so they can run on
meta-device tensors under a FakeProcessGroup without touching the real
c10d dispatcher (see design doc §2 finding #4 and §5.2). Generalizes the
`is_fake_process_group` short-circuit pattern already used by
`torchtitan_npu.converters.kernels.moe_dispatch.NpuExpertParallel` to every
collective entry point (FSDP2 all-gather/reduce-scatter, TP all-reduce, DP
grad all-reduce, broadcast)."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from torchtitan_npu.distributed.process_group import is_fake_process_group
from torchtitan_npu.simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes

_event_counter = itertools.count()


def _should_intercept(group: object) -> bool:
    """Check if communication should be intercepted (no-op + record).

    Returns True when:
    - ``_is_meta_simulation`` is True (set by SimulationTrainer for both
      fake_backend and multi_proc_meta modes), OR
    - the group is a FakeProcessGroup (legacy single-process mode)
    """
    from torchtitan_npu.simulator.meta_env import _is_meta_simulation
    if _is_meta_simulation:
        return True
    return is_fake_process_group(group)


@dataclass
class CommEvent:
    event_id: str
    comm_primitive: str
    group_name: str
    world_size: int
    tensor_shape: tuple[int, ...]
    dtype: str
    volume_bytes: int
    op_id: int = 0  # L0 OpNode ID if this event was also recorded as a synthetic op
    comm_dim: str = ""  # parallel dimension name (e.g. "tp", "ep", "fsdp")
    comm_ranks: list[list[int]] = field(default_factory=list)  # groups of ranks in this comm domain
    # P2P-specific fields (set by patched_isend/irecv for pipeline parallelism):
    p2p_peer_rank: int = -1      # peer rank (isend's dst / irecv's src)
    p2p_direction: str = ""      # "forward_send" / "forward_recv" / "backward_send" / "backward_recv"
    p2p_mb_idx: int = -1         # microbatch index
    p2p_stage: int = -1          # pipeline stage index


class _NoOpWork:
    """Minimal stand-in for `torch.distributed.Work`, so callers that use
    the `async_op=True` idiom (call `.wait()` on the return value) do not
    crash when we skip the real collective."""

    def wait(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def is_completed(self) -> bool:
        return True


class CommEventRecorder:
    def __init__(self) -> None:
        self.events: list[CommEvent] = []

    def record(
        self,
        comm_primitive: str,
        group: object,
        tensor: torch.Tensor,
        *,
        comm_dim: str = "",
        comm_ranks: list[list[int]] | None = None,
    ) -> CommEvent:
        """Record a communication event. Returns the CommEvent so the caller
        can set ``op_id`` after optionally registering a synthetic L0 op."""
        dtype_str = dtype_to_str(tensor.dtype)
        world_size = _resolve_world_size(group)
        event = CommEvent(
            event_id=f"comm_{next(_event_counter)}",
            comm_primitive=comm_primitive,
            group_name=_group_name(group),
            world_size=world_size,
            tensor_shape=tuple(int(d) for d in tensor.shape),
            dtype=dtype_str,
            volume_bytes=tensor_volume_bytes(tuple(tensor.shape), dtype_str),
            comm_dim=comm_dim,
            comm_ranks=comm_ranks or [],
        )
        self.events.append(event)
        return event


def _resolve_world_size(group: object) -> int:
    """Best-effort world-size extraction from any group type the functional
    collectives API accepts: ProcessGroup, DeviceMesh, list of ranks, or
    group-name string. Returns 1 if unresolvable (e.g. None or
    dist not initialized)."""
    if group is None:
        return dist.get_world_size() if dist.is_initialized() else 1
    # ProcessGroup
    if hasattr(group, "size"):
        try:
            return int(group.size())
        except (TypeError, RuntimeError):
            pass
    # DeviceMesh
    if hasattr(group, "ndm"):
        try:
            return int(group.size())
        except (TypeError, RuntimeError):
            pass
    # List of ranks
    if isinstance(group, (list, tuple)):
        return len(group)
    # Group-name string
    if isinstance(group, str) and dist.is_initialized():
        try:
            pg = dist.distributed_c10d._resolve_group_name(group, "")  # type: ignore[attr-defined]
            return int(pg.size())
        except Exception:
            pass
    return 1


def _group_name(group: object) -> str:
    """Extract a group-name string from any group type for RankTable
    dimension attribution. Falls back to "default" when unresolvable."""
    if group is None:
        return "default"
    # ProcessGroup
    name = getattr(group, "group_name", None)
    if name is not None:
        return str(name)
    # DeviceMesh -- try to get the underlying ProcessGroup's name
    if hasattr(group, "get_group"):
        try:
            pg = group.get_group()
            name = getattr(pg, "group_name", None)
            if name is not None:
                return str(name)
        except Exception:
            pass
    # Group-name string itself
    if isinstance(group, str):
        return group
    return "default"


def _resolve_comm_ranks(group: object) -> list[list[int]]:
    """Best-effort extraction of the rank lists that belong to this
    communication domain.  Returns a list of groups, where each group is a
    list of global rank IDs (e.g. ``[[0,1,2,3],[4,5,6,7]]`` for two TP
    groups of size 4).  Returns ``[]`` when unresolvable."""
    if group is None:
        ws = dist.get_world_size() if dist.is_initialized() else 1
        return [list(range(ws))] if ws > 1 else []
    # DeviceMesh: extract ranks from the mesh tensor
    if hasattr(group, "mesh") and hasattr(group, "ndm"):
        try:
            mesh = group.mesh
            if hasattr(mesh, "flatten"):
                ranks = [int(r) for r in mesh.flatten().tolist()]
                return [ranks] if len(ranks) > 1 else []
        except Exception:
            pass
    # ProcessGroup: only know our own group's size, not all groups
    if hasattr(group, "size"):
        try:
            ws = int(group.size())
            rank = int(group.rank()) if hasattr(group, "rank") else 0
            return [list(range(rank, rank + ws))]
        except Exception:
            pass
    return []


def _record_comm_with_l0(
    recorder: CommEventRecorder,
    comm_primitive: str,
    group: object,
    tensor: torch.Tensor,
    output_tensor: torch.Tensor | None = None,
) -> CommEvent:
    """Record a CommEvent AND register a synthetic L0 OpNode for the
    communication op, so it appears in the L0 compute graph and CSV.

    Returns the CommEvent (so P2P callers can set p2p_peer_rank etc.)."""
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    comm_dim = _group_name(group)
    comm_ranks = _resolve_comm_ranks(group)
    event = recorder.record(
        comm_primitive, group, tensor,
        comm_dim=comm_dim, comm_ranks=comm_ranks,
    )

    out = output_tensor if output_tensor is not None else tensor
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type=f"comm.{comm_primitive}",
            inputs=[tensor],
            outputs=[out],
        )
        # Link the L0 op to this CommEvent and store comm metadata in annotations
        if capture._events:
            event.op_id = capture._events[-1].op_id
            # Store comm_dim and comm_ranks in the L0 node's annotations
            # so they appear in CSV and JSON exports
            capture._events[-1].comm_dim = comm_dim
            capture._events[-1].comm_ranks_str = ";".join(
                ",".join(str(r) for r in g) for g in comm_ranks
            )
    return event


@contextmanager
def capture_fake_collectives() -> Iterator[CommEventRecorder]:
    """Monkeypatch the legacy (`torch.distributed.*`) and functional
    (`torch.distributed._functional_collectives.*`) collective APIs for the
    duration of the context.

    Legacy APIs receive a real `ProcessGroup` (or `None`) as their `group`
    argument, so they defensively check `is_fake_process_group(group)` and
    fall back to the real implementation when it is not fake (keeps this
    module safe to import even outside a simulation run).

    Functional-collective APIs (used internally by DTensor/FSDP2) accept a
    `ProcessGroup`, `DeviceMesh`, list of ranks, or group-name string as
    `group` -- resolving all of those reliably is fragile (see design doc
    §2 finding #4 discussion). Because this context manager is only ever
    active for the full duration of one simulated training step, and the
    simulator always runs entirely under a fake backend (never a mix of
    real and fake groups), the functional-collective patches always treat
    calls made while the context is active as fake, unconditionally.
    """
    recorder = CommEventRecorder()

    orig_all_reduce = dist.all_reduce
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_all_to_all_single = dist.all_to_all_single
    orig_broadcast = dist.broadcast
    orig_barrier = dist.barrier

    orig_funcol_all_reduce = funcol.all_reduce
    orig_funcol_all_gather_tensor = funcol.all_gather_tensor
    orig_funcol_reduce_scatter_tensor = funcol.reduce_scatter_tensor
    orig_funcol_all_to_all_single = funcol.all_to_all_single
    orig_funcol_all_gather_tensor_autograd = funcol.all_gather_tensor_autograd
    orig_funcol_reduce_scatter_tensor_autograd = funcol.reduce_scatter_tensor_autograd
    orig_funcol_all_to_all_single_autograd = funcol.all_to_all_single_autograd

    def patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
        _record_comm_with_l0(recorder, "allreduce", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)
        _record_comm_with_l0(recorder, "allgather", group, input_tensor, output_tensor)
        return _NoOpWork() if async_op else None

    def patched_reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)
        _record_comm_with_l0(recorder, "reduce_scatter", group, input, output)
        return _NoOpWork() if async_op else None

    def patched_all_to_all_single(  # noqa: ANN001
        output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
    ):
        if not _should_intercept(group):
            return orig_all_to_all_single(
                output, input, output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes, group=group, async_op=async_op,
            )
        _record_comm_with_l0(recorder, "all_to_all", group, input, output)
        return _NoOpWork() if async_op else None

    def patched_broadcast(tensor, src=0, group=None, async_op=False, group_src=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_broadcast(tensor, src=src, group=group, async_op=async_op, group_src=group_src)
        _record_comm_with_l0(recorder, "broadcast", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_barrier(group=None, async_op=False, device_ids=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)
        return _NoOpWork() if async_op else None

    def patched_funcol_all_reduce(self_tensor, reduceOp, group, tag=""):  # noqa: ANN001, N803
        out = self_tensor.clone()
        _record_comm_with_l0(recorder, "allreduce", group, self_tensor, out); return out

    def patched_funcol_all_gather_tensor(self_tensor, gather_dim, group, tag=""):  # noqa: ANN001
        world_size = _resolve_world_size(group)
        out_shape = list(self_tensor.shape)
        out_shape[gather_dim] *= world_size
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "allgather", group, self_tensor, out); return out

    def patched_funcol_reduce_scatter_tensor(self_tensor, reduceOp, scatter_dim, group, tag=""):  # noqa: ANN001, N803
        world_size = _resolve_world_size(group)
        out_shape = list(self_tensor.shape)
        dim_size = out_shape[scatter_dim]
        if world_size > 1 and dim_size % world_size != 0:
            raise ValueError(
                f"reduce_scatter: scatter_dim size {dim_size} is not divisible by "
                f"world_size {world_size}; check parallelism config / tensor shapes"
            )
        out_shape[scatter_dim] = dim_size // world_size
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "reduce_scatter", group, self_tensor, out); return out

    def patched_funcol_all_to_all_single(self_tensor, output_split_sizes, input_split_sizes, group, tag=""):  # noqa: ANN001
        out_shape = list(self_tensor.shape)
        if output_split_sizes:
            out_shape[0] = int(sum(output_split_sizes))
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "all_to_all", group, self_tensor, out); return out

    def patched_funcol_all_gather_tensor_autograd(self_tensor, gather_dim, group, tag=""):  # noqa: ANN001
        world_size = _resolve_world_size(group)
        out_shape = list(self_tensor.shape)
        if gather_dim < len(out_shape):
            out_shape[gather_dim] *= world_size
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "allgather", group, self_tensor, out); return out

    def patched_funcol_reduce_scatter_tensor_autograd(self_tensor, reduceOp, scatter_dim, group, tag=""):  # noqa: ANN001, N803
        world_size = _resolve_world_size(group)
        out_shape = list(self_tensor.shape)
        if scatter_dim < len(out_shape):
            dim_size = out_shape[scatter_dim]
            if world_size > 1 and dim_size % world_size != 0:
                raise ValueError(
                    f"reduce_scatter: scatter_dim size {dim_size} is not divisible by "
                    f"world_size {world_size}; check parallelism config / tensor shapes"
                )
            out_shape[scatter_dim] = dim_size // world_size
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "reduce_scatter", group, self_tensor, out); return out

    def patched_funcol_all_to_all_single_autograd(self_tensor, output_split_sizes, input_split_sizes, group, tag=""):  # noqa: ANN001
        out_shape = list(self_tensor.shape)
        if output_split_sizes:
            out_shape[0] = int(sum(output_split_sizes))
        out = torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)
        _record_comm_with_l0(recorder, "all_to_all", group, self_tensor, out); return out

    dist.all_reduce = patched_all_reduce
    dist.all_gather_into_tensor = patched_all_gather_into_tensor
    dist.reduce_scatter_tensor = patched_reduce_scatter_tensor
    dist.all_to_all_single = patched_all_to_all_single
    dist.broadcast = patched_broadcast
    dist.barrier = patched_barrier

    # P2P operations used by pipeline parallelism for actual tensor passing
    # (not just metadata).  Under fake PG these are no-ops: the tensor is
    # already in the same process, so "sending" it does nothing and
    # "receiving" it means using the pre-allocated buffer as-is.
    orig_isend = dist.isend
    orig_irecv = dist.irecv
    orig_send = dist.send
    orig_recv = dist.recv

    def patched_isend(tensor, dst=None, group=None, tag=0, group_dst=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_isend(tensor, dst=dst, group=group, tag=tag, group_dst=group_dst)
        # Record P2P send with PP context
        from torchtitan_npu.simulator.meta_env import _pp_context
        event = _record_comm_with_l0(recorder, "p2p_send", group, tensor)
        event.p2p_peer_rank = dst if dst is not None else (group_dst if group_dst is not None else -1)
        event.p2p_direction = f"{_pp_context['phase']}_send"
        event.p2p_mb_idx = int(_pp_context["mb_idx"])
        event.p2p_stage = int(_pp_context["stage"])
        # No actual data transfer — meta tensors have no data.
        # Shape inference is handled by DYNAMIC mode via _send_meta/_recv_meta.
        return _NoOpWork()

    def patched_irecv(tensor, src=None, group=None, tag=0, group_src=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_irecv(tensor, src=src, group=group, tag=tag, group_src=group_src)
        # Record P2P recv with PP context
        from torchtitan_npu.simulator.meta_env import _pp_context
        event = _record_comm_with_l0(recorder, "p2p_recv", group, tensor)
        event.p2p_peer_rank = src if src is not None else (group_src if group_src is not None else -1)
        event.p2p_direction = f"{_pp_context['phase']}_recv"
        event.p2p_mb_idx = int(_pp_context["mb_idx"])
        event.p2p_stage = int(_pp_context["stage"])
        # No actual data transfer — meta tensors have no data.
        # The recv buffer shape was already set by DYNAMIC mode's
        # _setup_forward_recv_info using metadata from _recv_meta.
        return _NoOpWork()

    def patched_send(tensor, dst=None, group=None, tag=0, group_dst=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_send(tensor, dst=dst, group=group, tag=tag, group_dst=group_dst)
        from torchtitan_npu.simulator.meta_env import _pp_context
        event = _record_comm_with_l0(recorder, "p2p_send", group, tensor)
        event.p2p_peer_rank = dst if dst is not None else (group_dst if group_dst is not None else -1)
        event.p2p_direction = f"{_pp_context['phase']}_send"
        event.p2p_mb_idx = int(_pp_context["mb_idx"])
        event.p2p_stage = int(_pp_context["stage"])
        return None

    def patched_recv(tensor, src=None, group=None, tag=0, group_src=None):  # noqa: ANN001
        if not _should_intercept(group):
            return orig_recv(tensor, src=src, group=group, tag=tag, group_src=group_src)
        from torchtitan_npu.simulator.meta_env import _pp_context
        event = _record_comm_with_l0(recorder, "p2p_recv", group, tensor)
        event.p2p_peer_rank = src if src is not None else (group_src if group_src is not None else -1)
        event.p2p_direction = f"{_pp_context['phase']}_recv"
        event.p2p_mb_idx = int(_pp_context["mb_idx"])
        event.p2p_stage = int(_pp_context["stage"])
        return 0

    dist.isend = patched_isend
    dist.irecv = patched_irecv
    dist.send = patched_send
    dist.recv = patched_recv

    # P2POp.__new__ checks `op in [isend, irecv]` by identity, but we replaced
    # dist.isend/irecv with wrappers.  Patch _check_op to accept our wrappers.
    from torch.distributed.distributed_c10d import _check_op as _orig_check_op

    def _patched_check_op(op):
        if op in (patched_isend, patched_irecv, orig_isend, orig_irecv):
            return
        return _orig_check_op(op)

    import torch.distributed.distributed_c10d as _c10d_mod
    _c10d_mod._check_op = _patched_check_op
    funcol.all_reduce = patched_funcol_all_reduce
    funcol.all_gather_tensor = patched_funcol_all_gather_tensor
    funcol.reduce_scatter_tensor = patched_funcol_reduce_scatter_tensor
    funcol.all_to_all_single = patched_funcol_all_to_all_single
    funcol.all_gather_tensor_autograd = patched_funcol_all_gather_tensor_autograd
    funcol.reduce_scatter_tensor_autograd = patched_funcol_reduce_scatter_tensor_autograd
    funcol.all_to_all_single_autograd = patched_funcol_all_to_all_single_autograd

    try:
        yield recorder
    finally:
        dist.all_reduce = orig_all_reduce
        dist.all_gather_into_tensor = orig_all_gather_into_tensor
        dist.reduce_scatter_tensor = orig_reduce_scatter_tensor
        dist.all_to_all_single = orig_all_to_all_single
        dist.broadcast = orig_broadcast
        dist.barrier = orig_barrier
        dist.isend = orig_isend
        dist.irecv = orig_irecv
        dist.send = orig_send
        dist.recv = orig_recv
        funcol.all_reduce = orig_funcol_all_reduce
        funcol.all_gather_tensor = orig_funcol_all_gather_tensor
        funcol.reduce_scatter_tensor = orig_funcol_reduce_scatter_tensor
        funcol.all_to_all_single = orig_funcol_all_to_all_single
        funcol.all_gather_tensor_autograd = orig_funcol_all_gather_tensor_autograd
        funcol.reduce_scatter_tensor_autograd = orig_funcol_reduce_scatter_tensor_autograd
        funcol.all_to_all_single_autograd = orig_funcol_all_to_all_single_autograd
