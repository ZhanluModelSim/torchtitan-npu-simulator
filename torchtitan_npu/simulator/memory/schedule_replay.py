# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expand deduplicated PP memory templates over a captured schedule."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

from torchtitan_npu.simulator.memory.records import (
    FSDPResidencyEvent,
    MemoryActionSpan,
    MemoryPlan,
    RawMemoryEvent,
    TensorRef,
)

if TYPE_CHECKING:
    import torch.nn as nn

    from torchtitan_npu.simulator.ir.schedule_plan import ScheduleAction, SchedulePlan


_BACKWARD_COMP_TYPES = {"B", "I", "W", "F_RECOMPUTE"}


@dataclass(slots=True)
class ReplayedMemoryCapture:
    events: list[RawMemoryEvent]
    fsdp_residency_events: list[FSDPResidencyEvent]
    action_spans: list[MemoryActionSpan]
    dropped_duplicate_events: int = 0


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
) -> tuple[dict[tuple[int, str], list[RawMemoryEvent]], set[int], int]:
    candidates: dict[tuple[int, str], dict[int, list[RawMemoryEvent]]] = {}
    for event in events:
        key = _template_key(event)
        if key not in compute_keys or event.op_id in non_replayable_op_ids:
            continue
        candidates.setdefault(key, {}).setdefault(event.pp_mb_idx, []).append(event)

    templates: dict[tuple[int, str], list[RawMemoryEvent]] = {}
    selected_event_ids: set[int] = set()
    duplicate_count = 0
    for key, by_microbatch in candidates.items():
        # Full capture has far more events than a later pass-through chunk. The
        # earliest microbatch is the deterministic tie-breaker.
        source_mb, template = min(
            by_microbatch.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        del source_mb
        template = sorted(template, key=lambda event: event.seq_idx)
        templates[key] = template
        selected_event_ids.update(event.event_id for event in template)
        duplicate_count += sum(len(group) for group in by_microbatch.values()) - len(template)
    return templates, selected_event_ids, duplicate_count


def _clone_ref(ref: TensorRef, tensor_id: int) -> TensorRef:
    return replace(ref, tensor_id=tensor_id)


def replay_pp_memory_capture(
    raw_events: Iterable[RawMemoryEvent],
    *,
    schedule_plan: SchedulePlan,
    comm_events: Iterable[Any] | None = None,
    fsdp_residency_events: Iterable[FSDPResidencyEvent] | None = None,
    persistent_tensor_ids: set[int] | None = None,
) -> ReplayedMemoryCapture:
    """Replay one captured template for every PP compute action.

    This transforms only the memory event stream. L0/L1 graph templates remain
    folded, and explicit framework-level communication/FSDP events remain
    single-source records instead of being cloned with compute templates.
    """
    events = sorted(raw_events, key=lambda event: event.seq_idx)
    actions = _flatten_actions(schedule_plan.actions)
    compute_actions = [
        action
        for action in actions
        if action.action_type == "COMPUTE" and action.stage >= 0 and action.mb_idx >= 0 and action.comp_type
    ]
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
    templates, selected_event_ids, dropped_duplicates = _select_templates(
        events,
        compute_keys,
        non_replayable_op_ids,
    )
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
    tensor_ids: dict[tuple[int, int], int] = {}
    op_ids: dict[tuple[str, int], int] = {}
    canonical_mb = min((action.mb_idx for action in compute_actions), default=0)

    def tensor_id_for(microbatch: int, original: int) -> int:
        nonlocal next_tensor_id
        if original in persistent_tensor_ids or microbatch == canonical_mb:
            return original
        key = (microbatch, original)
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

    replayed: list[RawMemoryEvent] = []
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
                    _clone_ref(ref, tensor_id_for(action.mb_idx, ref.tensor_id))
                    for ref in event.inputs
                ),
                outputs=tuple(
                    _clone_ref(ref, tensor_id_for(action.mb_idx, ref.tensor_id))
                    for ref in event.outputs
                ),
            )
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

    remapped_fsdp = _remap_fsdp_residency_events(
        list(fsdp_residency_events or []),
        action_spans,
    )
    return ReplayedMemoryCapture(
        events=replayed,
        fsdp_residency_events=remapped_fsdp,
        action_spans=action_spans,
        dropped_duplicate_events=dropped_duplicates,
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
) -> MemoryPlan:
    """Estimate memory, replaying templates only for PP schedules."""
    from torchtitan_npu.simulator.memory.estimator import estimate_static_memory

    if schedule_plan is None or schedule_plan.pp_degree <= 1:
        return estimate_static_memory(
            raw_events,
            model_parts=model_parts,
            comm_events=comm_events,
            fsdp_residency_events=fsdp_residency_events,
        )

    replayed = replay_pp_memory_capture(
        raw_events,
        schedule_plan=schedule_plan,
        comm_events=comm_events,
        fsdp_residency_events=fsdp_residency_events,
        persistent_tensor_ids=_persistent_tensor_ids(model_parts or []),
    )
    plan = estimate_static_memory(
        replayed.events,
        model_parts=model_parts,
        comm_events=comm_events,
        fsdp_residency_events=replayed.fsdp_residency_events,
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
