# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expand deduplicated PP memory templates over a captured schedule."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from torchtitan_npu.simulator.memory.records import (
    AutogradSavedTensorEvent,
    CheckpointBoundaryEvent,
    FSDPResidencyEvent,
    MemoryActionSpan,
    MemoryPlan,
    RawMemoryEvent,
    TensorRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch.nn as nn

    from torchtitan_npu.simulator.ir.schedule_plan import ScheduleAction, SchedulePlan


_BACKWARD_COMP_TYPES = {"B", "I", "W", "F_RECOMPUTE"}


@dataclass(slots=True)
class ReplayedMemoryCapture:
    events: list[RawMemoryEvent]
    checkpoint_boundary_events: list[CheckpointBoundaryEvent]
    fsdp_residency_events: list[FSDPResidencyEvent]
    action_spans: list[MemoryActionSpan]
    dropped_duplicate_events: int = 0
    autograd_saved_tensor_events: list[AutogradSavedTensorEvent] = field(default_factory=list)


def _flatten_actions(actions: Iterable[ScheduleAction]) -> list[ScheduleAction]:
    flattened: list[ScheduleAction] = []
    for action in actions:
        if action.action_type == "OVERLAP_F_B" and action.sub_actions:
            flattened.extend(_flatten_actions(action.sub_actions))
        else:
            flattened.append(action)
    return flattened


def _action_phase(action: ScheduleAction) -> str:
    if action.action_type == "OPTIMIZER" or action.comp_type == "OPTIMIZER":
        return "optimizer"
    if action.comp_type in _BACKWARD_COMP_TYPES or action.action_type in {
        "SEND_B",
        "RECV_B",
        "REDUCE_GRAD",
    }:
        return "backward"
    if action.comp_type == "F" or action.action_type in {"SEND_F", "RECV_F"}:
        return "forward"
    return "comm"


def _template_key(event: RawMemoryEvent) -> tuple[int, str] | None:
    if event.pp_stage < 0 or event.pp_mb_idx < 0 or not event.comp_type:
        return None
    return event.pp_stage, event.comp_type


def _select_templates(
    events: list[RawMemoryEvent],
    compute_keys: set[tuple[int, str]],
    non_replayable_op_ids: set[int],
) -> tuple[
    dict[tuple[int, str], list[RawMemoryEvent]],
    dict[tuple[int, str], int],
    set[int],
    int,
]:
    candidates: dict[tuple[int, str], dict[int, list[RawMemoryEvent]]] = {}
    for event in events:
        key = _template_key(event)
        if key is None or key not in compute_keys or event.op_id in non_replayable_op_ids:
            continue
        candidates.setdefault(key, {}).setdefault(event.pp_mb_idx, []).append(event)

    templates: dict[tuple[int, str], list[RawMemoryEvent]] = {}
    source_microbatches: dict[tuple[int, str], int] = {}
    selected_event_ids: set[int] = set()
    duplicate_count = 0
    stages = {stage for stage, _ in candidates}
    for stage in stages:
        keys = [key for key in candidates if key[0] == stage]
        if any(key[1] in _BACKWARD_COMP_TYPES for key in keys) and (stage, "F") not in candidates:
            raise ValueError(f"PP memory replay requires a paired forward microbatch on stage {stage}")
        common = set.intersection(*(set(candidates[key]) for key in keys))
        if not common:
            raise ValueError(f"PP memory replay requires paired templates from one source microbatch on stage {stage}")
        source_mb = min(common, key=lambda mb: (-sum(len(candidates[key][mb]) for key in keys), mb))
        for key in keys:
            by_microbatch = candidates[key]
            template = sorted(by_microbatch[source_mb], key=lambda event: event.seq_idx)
            templates[key] = template
            source_microbatches[key] = source_mb
            selected_event_ids.update(event.event_id for event in template)
            duplicate_count += sum(len(group) for group in by_microbatch.values()) - len(template)
    return templates, source_microbatches, selected_event_ids, duplicate_count


def replay_pp_memory_capture(
    raw_events: Iterable[RawMemoryEvent],
    *,
    schedule_plan: SchedulePlan,
    comm_events: Iterable[Any] | None = None,
    fsdp_residency_events: Iterable[FSDPResidencyEvent] | None = None,
    checkpoint_boundary_events: Iterable[CheckpointBoundaryEvent] | None = None,
    persistent_tensor_ids: set[int] | None = None,
    persistent_storage_keys: set[str] | None = None,
    autograd_saved_tensor_events: Iterable[AutogradSavedTensorEvent] | None = None,
) -> ReplayedMemoryCapture:
    """Replay one captured template for every PP compute action.

    This transforms only the memory event stream. L0/L1 graph templates remain
    folded, and explicit framework-level communication/FSDP events remain
    single-source records instead of being cloned with compute templates.
    """
    events = sorted(raw_events, key=lambda event: event.seq_idx)
    autograd_saved_tensor_events = list(autograd_saved_tensor_events or [])
    actions = _flatten_actions(schedule_plan.actions)
    compute_actions = [
        action
        for action in actions
        if action.action_type == "COMPUTE" and action.stage >= 0 and action.mb_idx >= 0 and action.comp_type
    ]
    microbatches_by_stage: dict[int, set[int]] = {}
    for action in compute_actions:
        microbatches_by_stage.setdefault(action.stage, set()).add(action.mb_idx)
    if any(len(microbatches) > 1 for microbatches in microbatches_by_stage.values()) and any(
        event.phase == "optimizer" for event in events
    ):
        # TODO: capture first/subsequent backward variants and bind accumulated
        # gradients to the single optimizer step before replaying training.
        raise ValueError(
            "PP memory replay with multiple microbatches and an optimizer requires "
            "gradient accumulation bindings that are not yet supported. "
            "Disable memory tracking to inspect graph templates only; "
            "this does not provide a complete accumulated-gradient graph."
        )
    compute_keys = {(action.stage, action.comp_type) for action in compute_actions}
    comm_events = list(comm_events or [])
    raw_comm_op_ids = {
        event.op_id for event in events if event.raw_op_type.startswith("comm.")
    }
    non_replayable_op_ids = {
        int(getattr(event, "op_id", 0) or 0)
        for event in comm_events
        if getattr(event, "comm_layer", "") == "L2"
    } & raw_comm_op_ids
    p2p_op_ids = {
        int(getattr(event, "op_id", 0) or 0)
        for event in comm_events
        if getattr(event, "comm_layer", "") == "L2"
        and bool(getattr(event, "p2p_direction", ""))
    } & raw_comm_op_ids
    templates, source_microbatches, selected_event_ids, dropped_duplicates = _select_templates(
        events,
        compute_keys,
        non_replayable_op_ids,
    )
    boundary_templates: dict[tuple[int, str], list[CheckpointBoundaryEvent]] = {}
    for boundary in checkpoint_boundary_events or []:
        key = (boundary.pp_stage, boundary.comp_type)
        if boundary.pp_mb_idx == source_microbatches.get(key):
            boundary_templates.setdefault(key, []).append(boundary)
    missing_templates = sorted(compute_keys - templates.keys())
    if missing_templates:
        formatted = ", ".join(
            f"stage={stage}/comp_type={comp_type}"
            for stage, comp_type in missing_templates
        )
        raise RuntimeError(
            "PP memory replay cannot expand schedule actions without captured templates: "
            + formatted
        )

    persistent_tensor_ids = persistent_tensor_ids or set()
    min_tensor_id = min(
        (ref.tensor_id for event in events for ref in (*event.inputs, *event.outputs)),
        default=0,
    )
    next_tensor_id = min(-1, min_tensor_id - 1)
    min_op_id = min((event.op_id for event in events), default=0)
    next_op_id = min(-1, min_op_id - 1)
    tensor_ids: dict[tuple[int, int, int], int] = {}
    sequence_map: dict[tuple[int, int, int], int] = {}
    persistent_storage_keys = persistent_storage_keys or set()
    op_ids: dict[tuple[str, int], int] = {}
    canonical_mb = min((action.mb_idx for action in compute_actions), default=0)

    def tensor_id_for(stage: int, microbatch: int, original: int) -> int:
        nonlocal next_tensor_id
        if original in persistent_tensor_ids or microbatch == canonical_mb:
            return original
        key = (stage, microbatch, original)
        if key not in tensor_ids:
            tensor_ids[key] = next_tensor_id
            next_tensor_id -= 1
        return tensor_ids[key]

    def op_id_for(action_id: str, microbatch: int, original: int) -> int:
        nonlocal next_op_id
        if microbatch == canonical_mb:
            return original
        key = (action_id, original)
        if key not in op_ids:
            op_ids[key] = next_op_id
            next_op_id -= 1
        return op_ids[key]

    def storage_key_for(stage: int, microbatch: int, key: str) -> str:
        if not key or key.partition(":")[0] in persistent_storage_keys:
            return key
        return f"pp:s{stage}:mb{microbatch}:{key}"

    def clone_ref(ref: TensorRef, action: ScheduleAction) -> TensorRef:
        is_persistent = ref.storage_key.partition(":")[0] in persistent_storage_keys
        return replace(
            ref,
            tensor_id=ref.tensor_id if is_persistent else tensor_id_for(action.stage, action.mb_idx, ref.tensor_id),
            alias_of=(tensor_id_for(action.stage, action.mb_idx, ref.alias_of) if ref.alias_of is not None else None),
            storage_key=storage_key_for(action.stage, action.mb_idx, ref.storage_key),
        )

    replayed: list[RawMemoryEvent] = []
    replayed_boundaries: list[CheckpointBoundaryEvent] = []
    action_spans: list[MemoryActionSpan] = []
    consumed_event_ids: set[int] = set()
    logical_seq = 0
    next_event_id = 0

    def append_event(event: RawMemoryEvent, *, action: ScheduleAction | None = None) -> None:
        nonlocal logical_seq, next_event_id
        if action is None:
            cloned = replace(event, event_id=next_event_id, seq_idx=logical_seq)
        else:
            cloned = replace(
                event,
                event_id=next_event_id,
                op_id=op_id_for(action.action_id, action.mb_idx, event.op_id),
                seq_idx=logical_seq,
                phase=_action_phase(action),
                pp_stage=action.stage,
                pp_mb_idx=action.mb_idx,
                comp_type=action.comp_type,
                inputs=tuple(
                    clone_ref(ref, action)
                    for ref in event.inputs
                ),
                outputs=tuple(
                    clone_ref(ref, action)
                    for ref in event.outputs
                ),
            )
        if action is not None:
            sequence_map[(action.stage, action.mb_idx, event.seq_idx)] = logical_seq
        replayed.append(cloned)
        next_event_id += 1
        logical_seq += 1

    # Framework setup is not a microbatch template and remains single-instance.
    optimizer_events = [event for event in events if event.phase == "optimizer"]
    prelude_events = [
        event
        for event in events
        if event.phase != "optimizer"
        and _template_key(event) is None
        and event.op_id not in non_replayable_op_ids
    ]
    for event in prelude_events:
        append_event(event)
        consumed_event_ids.add(event.event_id)

    events_by_op: dict[int, list[RawMemoryEvent]] = {}
    for event in events:
        events_by_op.setdefault(event.op_id, []).append(event)

    def source_seq_for(action: ScheduleAction) -> int:
        if action.action_type == "COMPUTE" or action.action_type == "OPTIMIZER":
            return action.seq_idx
        if action.comm_op_id:
            source_events = events_by_op.get(action.comm_op_id, [])
            if source_events:
                return min(event.seq_idx for event in source_events)
        return -1

    for action in actions:
        start_seq = logical_seq
        if action.action_type == "COMPUTE":
            for event in templates.get((action.stage, action.comp_type), []):
                append_event(event, action=action)
                consumed_event_ids.add(event.event_id)
            for boundary in boundary_templates.get((action.stage, action.comp_type), []):
                replayed_boundaries.append(
                    replace(
                        boundary,
                        seq_idx=max(start_seq, logical_seq - 1),
                        inputs=tuple(
                            clone_ref(ref, action)
                            for ref in boundary.inputs
                        ),
                        outputs=tuple(
                            clone_ref(ref, action)
                            for ref in boundary.outputs
                        ),
                        pp_stage=action.stage,
                        pp_mb_idx=action.mb_idx,
                        comp_type=action.comp_type,
                    )
                )
        elif action.action_type == "OPTIMIZER":
            for event in optimizer_events:
                if event.event_id not in consumed_event_ids:
                    append_event(
                        replace(
                            event,
                            phase="optimizer",
                            pp_stage=action.stage,
                            pp_mb_idx=-1,
                            comp_type="OPTIMIZER",
                        )
                    )
                    consumed_event_ids.add(event.event_id)
        elif action.comm_op_id and action.comm_op_id not in p2p_op_ids:
            for event in events_by_op.get(action.comm_op_id, []):
                if event.event_id not in consumed_event_ids:
                    append_event(event)
                    consumed_event_ids.add(event.event_id)
                    break

        if logical_seq == start_seq:
            logical_seq += 1
        action_spans.append(
            MemoryActionSpan(
                action_id=action.action_id,
                action_type=action.action_type,
                stage=action.stage,
                microbatch=action.mb_idx,
                comp_type=action.comp_type,
                phase=_action_phase(action),
                start_seq=start_seq,
                end_seq=logical_seq - 1,
                source_seq_idx=source_seq_for(action),
            )
        )

    # Keep single-instance events that were not part of a selected compute
    # template. Events from duplicate pass-through chunks are intentionally
    # omitted; the corresponding template has already been replayed above.
    for event in events:
        if event.event_id in consumed_event_ids or event.event_id in selected_event_ids:
            continue
        if event.op_id in p2p_op_ids:
            continue
        key = _template_key(event)
        if key in templates and event.op_id not in non_replayable_op_ids:
            continue
        append_event(event)

    replayed_slots: list[AutogradSavedTensorEvent] = []
    for stage, microbatch in sorted({(action.stage, action.mb_idx) for action in compute_actions}):
        spans = [span for span in action_spans if span.stage == stage and span.microbatch == microbatch and span.action_type == "COMPUTE"]
        for saved in autograd_saved_tensor_events or ():
            key = (saved.pp_stage, saved.comp_type)
            if saved.pp_stage != stage or saved.pp_mb_idx != source_microbatches.get(key):
                continue
            pack_span = next((span for span in spans if span.comp_type == saved.comp_type), None)
            if pack_span is None:
                raise ValueError(f"Cannot replay saved slot {saved.slot_id} without its pack action")
            pack_seq = sequence_map.get((stage, microbatch, saved.pack_seq), pack_span.start_seq)
            unpack_seq = -1
            if saved.unpack_seq >= 0:
                mapped = sequence_map.get((stage, microbatch, saved.unpack_seq))
                if mapped is None:
                    raise ValueError(f"Cannot replay saved slot {saved.slot_id}: unpack anchor is outside paired microbatch templates")
                unpack_seq = mapped
            replayed_slots.append(replace(
                saved, slot_id=len(replayed_slots),
                tensor_id=(saved.tensor_id if saved.storage_key.partition(":")[0] in persistent_storage_keys else tensor_id_for(stage, microbatch, saved.tensor_id)),
                storage_key=storage_key_for(stage, microbatch, saved.storage_key),
                pack_seq=pack_seq, unpack_seq=unpack_seq, pp_mb_idx=microbatch,
            ))

    remapped_fsdp = _remap_fsdp_residency_events(
        list(fsdp_residency_events or []),
        action_spans,
    )
    return ReplayedMemoryCapture(
        events=replayed,
        checkpoint_boundary_events=replayed_boundaries,
        fsdp_residency_events=remapped_fsdp,
        action_spans=action_spans,
        dropped_duplicate_events=dropped_duplicates,
        autograd_saved_tensor_events=replayed_slots,
    )


def _remap_fsdp_residency_events(
    events: list[FSDPResidencyEvent],
    action_spans: list[MemoryActionSpan],
) -> list[FSDPResidencyEvent]:
    if not events or not action_spans:
        return events
    anchors = sorted(
        (
            (span.source_seq_idx, span)
            for span in action_spans
            if span.source_seq_idx >= 0
            and span.action_type in {"COMPUTE", "UNSHARD", "RESHARD"}
        ),
        key=lambda item: item[0],
    )
    if not anchors:
        return events
    source_seqs = [item[0] for item in anchors]
    remapped: list[FSDPResidencyEvent] = []
    for event in sorted(events, key=lambda item: item.seq_idx):
        if event.action == "alloc":
            idx = min(bisect_left(source_seqs, event.seq_idx), len(anchors) - 1)
        else:
            idx = max(bisect_right(source_seqs, event.seq_idx) - 1, 0)
        span = anchors[idx][1]
        seq_idx = span.start_seq if event.action == "alloc" else span.end_seq
        remapped.append(replace(event, seq_idx=seq_idx))
    return remapped


def estimate_schedule_memory(
    raw_events: Iterable[RawMemoryEvent],
    *,
    schedule_plan: SchedulePlan | None,
    model_parts: Iterable[nn.Module] | None = None,
    comm_events: Iterable[Any] | None = None,
    fsdp_residency_events: Iterable[FSDPResidencyEvent] | None = None,
    checkpoint_boundary_events: Iterable[CheckpointBoundaryEvent] | None = None,
    autograd_saved_tensor_events: Iterable[AutogradSavedTensorEvent] | None = None,
    parameter_storage_dtype: str | None = None,
    offload_ac_saved_tensors: bool = False,
    fsdp_allgather_transport_dtype: str = "",
) -> MemoryPlan:
    """Estimate memory, replaying templates only for PP schedules."""
    from torchtitan_npu.simulator.memory.estimator import estimate_static_memory

    if schedule_plan is None or schedule_plan.pp_degree <= 1:
        return estimate_static_memory(
            raw_events,
            model_parts=model_parts,
            comm_events=comm_events,
            fsdp_residency_events=fsdp_residency_events,
            checkpoint_boundary_events=checkpoint_boundary_events,
            autograd_saved_tensor_events=autograd_saved_tensor_events,
            parameter_storage_dtype=parameter_storage_dtype,
            offload_ac_saved_tensors=offload_ac_saved_tensors,
            fsdp_allgather_transport_dtype=fsdp_allgather_transport_dtype,
        )

    replayed = replay_pp_memory_capture(
        raw_events,
        schedule_plan=schedule_plan,
        comm_events=comm_events,
        fsdp_residency_events=fsdp_residency_events,
        checkpoint_boundary_events=checkpoint_boundary_events,
        persistent_tensor_ids=_persistent_tensor_ids(model_parts or []),
        persistent_storage_keys=_persistent_storage_keys(model_parts or []),
        autograd_saved_tensor_events=autograd_saved_tensor_events,
    )
    plan = estimate_static_memory(
        replayed.events,
        model_parts=model_parts,
        comm_events=comm_events,
        fsdp_residency_events=replayed.fsdp_residency_events,
        checkpoint_boundary_events=replayed.checkpoint_boundary_events,
        autograd_saved_tensor_events=replayed.autograd_saved_tensor_events,
        parameter_storage_dtype=parameter_storage_dtype,
        offload_ac_saved_tensors=offload_ac_saved_tensors,
        fsdp_allgather_transport_dtype=fsdp_allgather_transport_dtype,
    )
    plan.action_spans = replayed.action_spans
    plan.notes.append(
        "PP memory replay instantiated deduplicated compute templates over "
        f"{len(replayed.action_spans)} schedule actions; "
        f"{replayed.dropped_duplicate_events} pass-through raw events were omitted."
    )
    return plan


def _persistent_tensor_ids(model_parts: Iterable[nn.Module]) -> set[int]:
    persistent: set[int] = set()
    for model in model_parts:
        values = [
            *(parameter for _, parameter in model.named_parameters(recurse=True)),
            *(buffer for _, buffer in model.named_buffers(recurse=True)),
        ]
        for value in values:
            persistent.add(id(value))
            try:
                from torch.distributed.tensor import DTensor

                if isinstance(value, DTensor):
                    local = getattr(value, "_local_tensor", None)
                    if local is None:
                        local = value.to_local()
                    persistent.add(id(local))
            except Exception:
                pass
    return persistent


def _persistent_storage_keys(model_parts: Iterable[nn.Module]) -> set[str]:
    from torchtitan_npu.simulator.memory.estimator import _to_local_tensor

    return {
        str(local.untyped_storage()._cdata)
        for model in model_parts
        for value in (*model.parameters(), *model.buffers())
        if (local := _to_local_tensor(value)) is not None
    }
