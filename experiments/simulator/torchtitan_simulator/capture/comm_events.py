# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Communication semantics attached to actual native transport dispatches."""

from __future__ import annotations

import contextvars
import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist

from torchtitan_simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes
from torchtitan_simulator.memory.records import FSDPResidencyEvent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torchtitan_simulator.ir.tensor_meta import TensorMeta

_event_counter = itertools.count()
_collective_dependencies = contextvars.ContextVar("simulator_collective_dependencies", default=())


@contextmanager
def collective_dependencies(*tensors):
    token = _collective_dependencies.set((*_collective_dependencies.get(), *tensors))
    try:
        yield
    finally:
        _collective_dependencies.reset(token)


_default_collective_context: contextvars.ContextVar[tuple[str, object] | None] = contextvars.ContextVar(
    "simulator_default_collective_context",
    default=None,
)


@contextmanager
def default_collective_context(
    dimension: str,
    group: object,
) -> Iterator[None]:
    """Retain the logical group when fake DTensor lowers it to WORLD/None."""
    token = _default_collective_context.set((dimension, group))
    try:
        yield
    finally:
        _default_collective_context.reset(token)


@dataclass
class CommEvent:
    event_id: str
    comm_primitive: str
    group_name: str
    world_size: int
    tensor_shape: tuple[int, ...]
    dtype: str
    volume_bytes: int
    raw_group_name: str = ""  # framework-generated ProcessGroup identifier
    op_id: int = 0  # L0 OpNode ID if this event was also recorded as a synthetic op
    comm_dim: str = ""  # parallel dimension name (e.g. "tp", "ep", "fsdp")
    comm_ranks: list[list[int]] = field(default_factory=list)  # groups of ranks in this comm domain
    # P2P-specific fields (set by patched_isend/irecv for pipeline parallelism):
    p2p_peer_rank: int = -1  # peer rank (isend's dst / irecv's src)
    p2p_direction: str = ""  # "forward_send" / "forward_recv" / "backward_send" / "backward_recv"
    p2p_mb_idx: int = -1  # microbatch index
    p2p_stage: int = -1  # pipeline stage index
    seq_idx: int = 0  # global execution sequence number (shared with L0 ops)
    comm_layer: str = ""  # "L1" (model compute) or "L2" (framework scheduling)
    src_exit_op: int = 0  # L1 op that produced the data being communicated (0 = unknown)
    dst_entry_op: int = 0  # L1 op that consumes the communicated data (0 = unknown)
    # Compute-graph class (mirrors _pp_context["comp_type"]) active when this
    # comm op ran. Lets the L2 schedule attribute P2P/FSDP comm to the correct
    # (microbatch, stage, comp_type) StepInstance instead of just the stage.
    comp_type: str = ""
    # FSDP sharding state active when this comm op ran.
    fsdp_state: str = "NA"
    transfer_id: str = ""  # stable cross-rank PP rendezvous key
    capture_process_rank: int = -1
    tensor_ordinal: int = 0
    action_order: int = -1
    fsdp_transition_id: str = ""
    fsdp_group_id: str = ""
    gradient_reduction: bool | None = None
    fsdp_module_fqn: str = ""
    fsdp_prefetch_source_fqn: str = ""
    fsdp_prefetch_type: str = ""
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()


class CommEventRecorder:
    def __init__(
        self,
        *,
        memory_tracking_enabled: bool = True,
        capture_process_rank: int = 0,
    ) -> None:
        self.memory_tracking_enabled = memory_tracking_enabled
        self.capture_process_rank = capture_process_rank
        self.events: list[CommEvent] = []
        self.timeline_events: list[dict] = []  # L2 scheduling events (MB 1+ pass-through)
        self.fsdp_residency_events: list[FSDPResidencyEvent] = []
        self.fsdp_schedule_events: list[FSDPResidencyEvent] = []
        self._action_counter = itertools.count()

    def next_action_order(self) -> int:
        return next(self._action_counter)

    def record_pipeline_action(
        self,
        *,
        action_type: str,
        stage: int,
        mb_idx: int = -1,
        sub_actions: list[dict[str, int | str]] | None = None,
    ) -> None:
        from torchtitan_simulator.capture.dispatch_capture import _seq_counter

        event: dict[str, object] = {
            "event_kind": "schedule_action",
            "action_type": action_type,
            "pp_stage": stage,
            "pp_mb_idx": mb_idx,
            "seq_idx": next(_seq_counter),
            "action_order": self.next_action_order(),
        }
        if sub_actions:
            event["sub_actions"] = [dict(sub_action) for sub_action in sub_actions]
        self.timeline_events.append(event)

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
        raw_group_name = _group_name(group)
        event = CommEvent(
            event_id=f"comm_{next(_event_counter)}",
            comm_primitive=comm_primitive,
            group_name=raw_group_name,
            world_size=world_size,
            tensor_shape=tuple(int(d) for d in tensor.shape),
            dtype=dtype_str,
            volume_bytes=tensor_volume_bytes(tuple(tensor.shape), dtype_str),
            raw_group_name=raw_group_name,
            comm_dim=comm_dim,
            comm_ranks=comm_ranks or [],
            capture_process_rank=self.capture_process_rank,
            action_order=self.next_action_order(),
        )
        self.events.append(event)
        return event

    def record_timeline_event(
        self,
        *,
        action: str,
        pp_stage: int,
        pp_mb_idx: int,
        phase: str,
        comp_type: str = "",
        start_seq_idx: int | None = None,
        start_action_order: int | None = None,
    ) -> None:
        """Record an L2 scheduling-level timeline event (e.g. forward_one_chunk,
        backward_one_chunk, backward_weight_one_chunk).  Used for every
        microbatch's compute chunk — including pass-through ones whose L0 graph
        was deduped against an already-captured (stage, comp_type) template —
        so the L2 schedule can instantiate the right template for each
        microbatch. ``comp_type`` carries the F/B/I/W class."""
        from torchtitan_simulator.capture.dispatch_capture import _seq_counter

        end_seq_idx = next(_seq_counter)
        end_action_order = self.next_action_order()
        instance_id = f"s{pp_stage}_{comp_type or phase}_mb{pp_mb_idx}"
        self.timeline_events.append(
            {
                "seq_idx": end_seq_idx,
                "start_seq_idx": end_seq_idx if start_seq_idx is None else start_seq_idx,
                "end_seq_idx": end_seq_idx,
                "action_order": (end_action_order if start_action_order is None else start_action_order),
                "end_action_order": end_action_order,
                "instance_id": instance_id,
                "action": action,
                "pp_stage": pp_stage,
                "pp_mb_idx": pp_mb_idx,
                "phase": phase,
                "comp_type": comp_type,
            }
        )

    def record_fsdp_residency(
        self,
        *,
        group_id: str,
        action: str,
        phase: str,
        num_bytes: int,
        tensor_ids: tuple[int, ...],
        storage_keys: tuple[str, ...] = (),
        allocation_storage_bytes: tuple[tuple[str, int], ...] = (),
        shard_world_size: int = -1,
        transition_id: str = "",
        include_schedule: bool = True,
        include_memory: bool = True,
        schedule_source: str = "state",
        module_fqn: str = "",
        prefetch_source_fqn: str = "",
        prefetch_type: str = "",
    ) -> None:
        from torchtitan_simulator.capture.dispatch_capture import _seq_counter
        from torchtitan_simulator.meta_env import _pp_context

        stage = int(_pp_context.get("stage", -1))
        mb_idx = int(_pp_context.get("mb_idx", -1))
        comp_type = str(_pp_context.get("comp_type", ""))
        event = FSDPResidencyEvent(
            group_id=group_id,
            action=action,
            seq_idx=next(_seq_counter),
            phase=phase,
            num_bytes=num_bytes,
            tensor_ids=tensor_ids,
            storage_keys=storage_keys,
            allocation_storage_bytes=allocation_storage_bytes,
            capture_process_rank=self.capture_process_rank,
            pp_stage=stage,
            pp_mb_idx=mb_idx,
            comp_type=comp_type,
            parent_compute_instance_id=(
                f"s{stage}_{comp_type}_mb{mb_idx}" if stage >= 0 and mb_idx >= 0 and comp_type else ""
            ),
            shard_world_size=shard_world_size,
            action_order=self.next_action_order(),
            transition_id=transition_id,
            schedule_source=schedule_source,
            module_fqn=module_fqn,
            prefetch_source_fqn=prefetch_source_fqn,
            prefetch_type=prefetch_type,
        )
        if include_schedule:
            self.fsdp_schedule_events.append(event)
        if include_memory and self.memory_tracking_enabled:
            self.fsdp_residency_events.append(event)


def _canonical_collective_group(group):
    if isinstance(group, torch.ScriptObject):
        group = dist.ProcessGroup.unbox(group)
    # Functional collectives accept (DeviceMesh, mesh_axis) in addition to
    # process groups and rank lists. Its tuple length is not a world size.
    if (
        isinstance(group, tuple)
        and len(group) == 2
        and hasattr(group[0], "get_group")
        and isinstance(group[1], (int, str))
    ):
        return group[0].get_group(group[1])
    return group


def _resolve_world_size(group: Any) -> int:
    """Best-effort world-size extraction from any group type the functional
    collectives API accepts: ProcessGroup, DeviceMesh, list of ranks, or
    group-name string. Returns 1 if unresolvable (e.g. None or
    dist not initialized)."""
    group = _canonical_collective_group(group)
    active_context = _default_collective_context.get()
    group_name = getattr(group, "group_name", None)
    if active_context is not None and (
        group is None or group is dist.group.WORLD or str(group) == "default" or str(group_name) == "default"
    ):
        return _resolve_world_size(active_context[1])
    if group is None:
        return dist.get_world_size() if dist.is_initialized() else 1
    # ProcessGroup
    if hasattr(group, "size"):
        try:
            return int(group.size())
        except (TypeError, RuntimeError):
            pass
    # DeviceMesh
    if hasattr(group, "ndim"):
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
            pg = cast("Any", dist.distributed_c10d._resolve_process_group)(group)
            return int(pg.size())
        except Exception:
            pass
    return 1


def _group_name(group: object) -> str:
    """Extract a group-name string from any group type for RankTable
    dimension attribution. Falls back to "default" when unresolvable."""
    group = _canonical_collective_group(group)
    active_context = _default_collective_context.get()
    raw_group_name = getattr(group, "group_name", None)
    if active_context is not None and (
        group is None or group is dist.group.WORLD or str(group) == "default" or str(raw_group_name) == "default"
    ):
        return active_context[0]
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


def _resolve_comm_ranks(group: Any) -> list[list[int]]:
    """Best-effort extraction of the rank lists that belong to this
    communication domain.  Returns a list of groups, where each group is a
    list of global rank IDs (e.g. ``[[0,1,2,3],[4,5,6,7]]`` for two TP
    groups of size 4).  Returns ``[]`` when unresolvable."""
    group = _canonical_collective_group(group)
    active_context = _default_collective_context.get()
    group_name = getattr(group, "group_name", None)
    if active_context is not None and (
        group is None or group is dist.group.WORLD or str(group) == "default" or str(group_name) == "default"
    ):
        return _resolve_comm_ranks(active_context[1])
    if group is None:
        ws = dist.get_world_size() if dist.is_initialized() else 1
        return [list(range(ws))] if ws > 0 else []
    if isinstance(group, (list, tuple)) and all(isinstance(rank, int) for rank in group):
        return [list(group)] if group else []
    if isinstance(group, str) and dist.is_initialized():
        return _resolve_comm_ranks(cast("Any", dist.distributed_c10d._resolve_process_group)(group))
    # DeviceMesh: extract ranks from the mesh tensor
    if hasattr(group, "mesh") and hasattr(group, "ndim"):
        try:
            mesh = group.mesh
            if hasattr(mesh, "flatten"):
                ranks = [int(r) for r in mesh.flatten().tolist()]
                return [ranks] if ranks else []
        except Exception:
            pass
    # ProcessGroup: c10d keeps the exact global-rank map for registered groups.
    if hasattr(group, "size"):
        try:
            get_process_group_ranks = getattr(dist, "get_process_group_ranks", None)
            if get_process_group_ranks is not None:
                ranks = [int(rank) for rank in get_process_group_ranks(group)]
                return [ranks] if ranks else []
            from torch.distributed import distributed_c10d as c10d

            rank_map = c10d._world.pg_group_ranks.get(group)
            if rank_map:
                ranks = [
                    int(global_rank) for global_rank, _local_rank in sorted(rank_map.items(), key=lambda item: item[1])
                ]
                return [ranks] if ranks else []
        except Exception:
            pass
    return []


def _record_comm_with_l0(
    recorder: CommEventRecorder,
    comm_primitive: str,
    group: object,
    tensor: torch.Tensor,
    output_tensor: torch.Tensor | None = None,
    *,
    transport_tensor: torch.Tensor | None = None,
    operation_input_metas: list[TensorMeta] | None = None,
    operation_output_metas: list[TensorMeta] | None = None,
    dependency_inputs: list[torch.Tensor] | None = None,
    memory_inputs: list[torch.Tensor] | None = None,
    memory_outputs: list[torch.Tensor] | None = None,
    captured_op_id: int | None = None,
    source_op_id: int | None = None,
    native_inputs: tuple[torch.Tensor, ...] = (),
    native_outputs: tuple[torch.Tensor, ...] = (),
) -> CommEvent:
    """Record a CommEvent AND register a synthetic L0 OpNode for the
    communication op, so it appears in the L0 compute graph and CSV.

    Returns the CommEvent (so P2P callers can set p2p_peer_rank etc.)."""
    from torchtitan_simulator.capture.dispatch_capture import get_active_capture

    dependency_inputs = [*(dependency_inputs or ()), *_collective_dependencies.get()]
    comm_dim = _group_name(group)
    comm_ranks = _resolve_comm_ranks(group)
    event = recorder.record(
        comm_primitive,
        group,
        transport_tensor if transport_tensor is not None else tensor,
        comm_dim=comm_dim,
        comm_ranks=comm_ranks,
    )
    if native_inputs:
        event.volume_bytes = sum(tensor.numel() * tensor.element_size() for tensor in native_inputs)
        event.input_shapes = tuple(tuple(tensor.shape) for tensor in native_inputs)
        event.output_shapes = tuple(tuple(tensor.shape) for tensor in native_outputs)
    # Assign a global seq_idx so comm events can be ordered in the timeline
    from torchtitan_simulator.capture.dispatch_capture import _seq_counter

    event.seq_idx = next(_seq_counter)

    # Record comm_layer from context (set by call-site patches)
    from torchtitan_simulator.meta_env import _comm_layer, _pp_context

    event.comm_layer = _comm_layer
    # Stamp the active compute-graph class + FSDP state so the L2 schedule can
    # attribute this comm op to the correct (microbatch, stage, comp_type)
    # StepInstance. Falls back to "" when no chunk is active (e.g. inter-chunk
    # UNSHARD/RESHARD actions in runtime zero-bubble schedules).
    event.comp_type = str(_pp_context.get("comp_type", ""))
    event.fsdp_state = str(_pp_context.get("fsdp_state", "NA"))
    try:
        from torchtitan_simulator.capture.fsdp_residency import (
            get_active_fsdp_communication_context,
            get_active_fsdp_group_dimension,
            get_active_fsdp_reduction_dependencies,
            get_active_fsdp_transition,
            is_active_fsdp_reduction_group,
        )

        transition_id, group_id = get_active_fsdp_transition()
        event.fsdp_transition_id = transition_id
        event.fsdp_group_id = group_id
        (
            event.fsdp_module_fqn,
            event.fsdp_prefetch_source_fqn,
            event.fsdp_prefetch_type,
        ) = get_active_fsdp_communication_context()
        # FSDP can flatten dp_shard/cp into a group absent from ParallelDims,
        # or reuse a loss/CP group. Bind the role to this call and primitive.
        if dimension := get_active_fsdp_group_dimension(group, comm_primitive):
            event.group_name = event.comm_dim = comm_dim = dimension
        # Replicate-only groups skip RS and all-reduce a packed gradient
        # buffer whose framework copy operations are hidden from L0.
        if comm_primitive in {"reduce_scatter", "allreduce"} and event.fsdp_group_id:
            if not is_active_fsdp_reduction_group(group):
                event.fsdp_group_id = ""
                event.fsdp_transition_id = ""
                event.fsdp_module_fqn = ""
            fsdp_dependencies = get_active_fsdp_reduction_dependencies(group)
            if fsdp_dependencies:
                dependency_inputs = [
                    *(dependency_inputs or ()),
                    *fsdp_dependencies,
                ]
    except ImportError:
        pass
    event.gradient_reduction = bool(event.fsdp_group_id) and comm_primitive in {"reduce_scatter", "allreduce"}
    # For FSDP collective comm (allgather/reduce_scatter), record the active
    # stage so build_schedule_plan can match the plan's UNSHARD/RESHARD action
    # (by stage) to this CommEvent + its L0 op_id. (P2P events already set
    # p2p_stage in patched_isend/irecv.) Requires _patch_pipeline_action_context
    # to have stamped the real stage before the action ran.
    if not event.p2p_direction:
        try:
            event.p2p_stage = int(_pp_context.get("stage", -1))
            event.p2p_mb_idx = int(_pp_context.get("mb_idx", -1))
        except (TypeError, ValueError):
            pass

    out = output_tensor if output_tensor is not None else tensor
    capture = get_active_capture()
    if capture is not None:
        # FSDP collectives are attributed to the residency transition's
        # module. All other collectives, including MoE all-to-all, belong to
        # the module that is active when the collective is issued.
        module_path = event.fsdp_module_fqn
        if not module_path and capture.module_path_tracker is not None:
            module_path = capture.module_path_tracker.current_path()

        # Resolve src_exit_op: find the L1 op that produced the input tensor
        producer_op = capture.producer_op(tensor)
        if producer_op is not None:
            event.src_exit_op = producer_op

        event_count_before = len(capture._events)
        if captured_op_id is not None:
            recorded_op_id = captured_op_id
            event_count_before -= 1
            native = capture._events[-1]
            event.src_exit_op = source_op_id or 0
            event.seq_idx = native.seq_idx
            native.extra_annotations = {
                **(native.extra_annotations or {}),
                "fsdp_group_id": event.fsdp_group_id,
                "gradient_reduction": event.gradient_reduction,
                "fsdp_module_fqn": event.fsdp_module_fqn,
                "fsdp_prefetch_source_fqn": event.fsdp_prefetch_source_fqn,
                "fsdp_prefetch_type": event.fsdp_prefetch_type,
            }
        else:
            recorded_op_id = capture.record_synthetic_op(
                raw_op_type=f"comm.{comm_primitive}",
                inputs=[tensor],
                outputs=[out],
                module_path=module_path,
                operation_input_metas=operation_input_metas,
                operation_output_metas=operation_output_metas,
                dependency_inputs=dependency_inputs,
                memory_inputs=memory_inputs,
                memory_outputs=memory_outputs,
                extra_annotations={
                    "fsdp_group_id": event.fsdp_group_id,
                    "gradient_reduction": event.gradient_reduction,
                    "fsdp_module_fqn": event.fsdp_module_fqn,
                    "fsdp_prefetch_source_fqn": (event.fsdp_prefetch_source_fqn),
                    "fsdp_prefetch_type": event.fsdp_prefetch_type,
                },
            )
        if recorded_op_id is not None:
            event.op_id = recorded_op_id
        # Link the L0 op to this CommEvent and store comm metadata in annotations
        if len(capture._events) > event_count_before:
            # The capture phase is authoritative for optimizer/clip and for
            # nested chunked-loss backward inside an outer forward block.
            event.comp_type = capture._events[-1].comp_type
            # Store comm_dim and comm_ranks in the L0 node's annotations
            # so they appear in CSV and JSON exports
            capture._events[-1].group_name = event.group_name
            capture._events[-1].raw_group_name = event.raw_group_name
            capture._events[-1].comm_dim = comm_dim
            capture._events[-1].comm_ranks_str = ";".join(",".join(str(r) for r in g) for g in comm_ranks)

        # Register pending link for dst_entry_op: if this comm produces
        # an output tensor, the next L1 op that consumes it will be the
        # dst_entry_op.  We register the tensor id → event mapping and
        # resolve it lazily in _record_event when the consumer is captured.
        tensor_id = capture.tensor_id(tensor)
        output_id = capture.tensor_id(out)
        if captured_op_id is not None or output_id != tensor_id:
            for output in native_outputs or (out,):
                capture._pending_comm_links[capture._storage_identity(output)[0]] = event

    return event


_active_recorder: CommEventRecorder | None = None


def get_active_recorder() -> CommEventRecorder | None:
    """Return the currently active CommEventRecorder, if any."""
    return _active_recorder


@contextmanager
def observe_collectives(
    *,
    memory_tracking_enabled: bool = True,
    capture_process_rank: int = 0,
) -> Iterator[CommEventRecorder]:
    """Observe native collective dispatches; never replace transport APIs."""
    global _active_recorder
    if _active_recorder is not None:
        raise RuntimeError("Nested collective observation is unsupported")
    recorder = CommEventRecorder(
        memory_tracking_enabled=memory_tracking_enabled,
        capture_process_rank=capture_process_rank,
    )
    _active_recorder = recorder
    try:
        yield recorder
    finally:
        _active_recorder = None
