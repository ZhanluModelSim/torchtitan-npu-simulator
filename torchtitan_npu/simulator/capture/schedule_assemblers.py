# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Action-skeleton assemblers for schedule families without a lowered plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.ir.schedule_plan import CommDetail
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent


_PP_DIRECTIONS: dict[str, tuple[str, str]] = {
    "forward_send": ("SEND_F", "send"),
    "forward_recv": ("RECV_F", "recv"),
    "backward_send": ("SEND_B", "send"),
    "backward_recv": ("RECV_B", "recv"),
}
_COMPUTE_ACTION_TYPES = {"F", "B", "I", "W"}
_P2P_ACTION_TYPES = {"SEND_F", "RECV_F", "SEND_B", "RECV_B"}
_RESIDENCY_ACTION_TYPES = {"UNSHARD", "RESHARD"}


@dataclass(slots=True)
class ActionSpec:
    action_type: str
    stage: int
    mb_idx: int
    seq_idx: int
    order_key: tuple[int, int, int]
    comp_type: str = ""
    template_ref: str = ""
    comm_op_id: int = 0
    comm: CommDetail | None = None
    is_noop: bool = False
    sub_actions: list[ActionSpec] | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


class CapturedTraceAssembler:
    """Build a rank-local action skeleton from observed execution events.

    Every pipeline schedule eventually executes the same PipelineStage and
    communication APIs. Building from those semantic events avoids coupling
    L2 capture to a particular TorchTitan lowered-plan representation.
    """

    def __init__(
        self,
        *,
        timeline_events: Iterable[dict[str, Any]],
        comm_events: Iterable[CommEvent],
        fsdp_residency_events: Iterable[FSDPResidencyEvent],
        rank: int,
    ) -> None:
        self.timeline_events = list(timeline_events)
        self.comm_events = list(comm_events)
        self.fsdp_residency_events = list(fsdp_residency_events)
        self.rank = rank

    def _schedule_intents(self) -> list[dict[str, Any]]:
        return sorted(
            (
                event
                for event in self.timeline_events
                if event.get("event_kind") == "schedule_action"
            ),
            key=lambda event: int(
                event.get("action_order", event.get("seq_idx", 0))
            ),
        )

    @staticmethod
    def _intent_key(intent: dict[str, Any]) -> tuple[int, int, str]:
        return (
            int(intent.get("pp_stage", -1)),
            int(intent.get("pp_mb_idx", -1)),
            str(intent.get("action_type", "")),
        )

    @classmethod
    def _bind_compute_intents(
        cls,
        compute_specs: list[ActionSpec],
        schedule_intents: list[dict[str, Any]],
    ) -> list[ActionSpec]:
        compute_intents = [
            intent
            for intent in schedule_intents
            if intent.get("action_type") in _COMPUTE_ACTION_TYPES
            or intent.get("action_type") == "OVERLAP_F_B"
        ]
        if not compute_intents:
            return compute_specs

        by_key: dict[tuple[int, int, str], list[ActionSpec]] = {}
        for spec in compute_specs:
            by_key.setdefault(
                (spec.stage, spec.mb_idx, spec.comp_type), []
            ).append(spec)

        def claim(
            descriptor: dict[str, Any],
            parent_intent: dict[str, Any],
        ) -> ActionSpec:
            key = cls._intent_key(descriptor)
            candidates = by_key.get(key, [])
            if not candidates:
                raise RuntimeError(
                    "pipeline schedule action was observed without matching "
                    f"compute execution event: {key}"
                )
            spec = candidates.pop(0)
            intent_order = int(
                parent_intent.get(
                    "action_order", parent_intent.get("seq_idx", 0)
                )
            )
            spec.order_key = (intent_order, 0, 0)
            spec.annotations["capture_schedule_intent"] = True
            spec.annotations["schedule_intent_order"] = intent_order
            return spec

        bound: list[ActionSpec] = []
        for intent in compute_intents:
            if intent.get("action_type") != "OVERLAP_F_B":
                bound.append(claim(intent, intent))
                continue

            child_descriptors = intent.get("sub_actions")
            if not isinstance(child_descriptors, list) or not child_descriptors:
                raise RuntimeError(
                    "OVERLAP_F_B schedule intent has no sub-action descriptors"
                )
            children = [
                claim(child, intent)
                for child in child_descriptors
                if isinstance(child, dict)
            ]
            if len(children) != len(child_descriptors):
                raise RuntimeError(
                    "OVERLAP_F_B schedule intent contains an invalid sub-action"
                )
            intent_order = int(
                intent.get("action_order", intent.get("seq_idx", 0))
            )
            bound.append(ActionSpec(
                action_type="OVERLAP_F_B",
                stage=-1,
                mb_idx=-1,
                seq_idx=int(intent.get("seq_idx", 0)),
                order_key=(intent_order, 0, 0),
                sub_actions=children,
                annotations={
                    "capture_schedule_intent": True,
                    "capture_action_order": intent_order,
                },
            ))

        unmatched = {
            key: len(candidates)
            for key, candidates in by_key.items()
            if candidates
        }
        if unmatched:
            raise RuntimeError(
                "pipeline compute executions were observed without matching "
                f"schedule actions: {unmatched}"
            )
        return bound

    @classmethod
    def _bind_p2p_intents(
        cls,
        p2p_specs: list[ActionSpec],
        schedule_intents: list[dict[str, Any]],
    ) -> list[ActionSpec]:
        p2p_intents = [
            intent
            for intent in schedule_intents
            if intent.get("action_type") in _P2P_ACTION_TYPES
        ]
        if not p2p_intents:
            return p2p_specs

        unmatched = list(p2p_specs)
        for intent in p2p_intents:
            key = cls._intent_key(intent)
            matches = [
                spec
                for spec in unmatched
                if (spec.stage, spec.mb_idx, spec.action_type) == key
            ]
            if not matches:
                raise RuntimeError(
                    "pipeline P2P action was observed without matching "
                    f"communication event: {key}"
                )
            intent_order = int(
                intent.get("action_order", intent.get("seq_idx", 0))
            )
            for tensor_ordinal, spec in enumerate(matches):
                spec.order_key = (intent_order, 0, tensor_ordinal)
                spec.annotations["capture_schedule_intent"] = True
                spec.annotations["schedule_intent_order"] = intent_order
                unmatched.remove(spec)
        if unmatched:
            identities = [
                (spec.stage, spec.mb_idx, spec.action_type)
                for spec in unmatched
            ]
            raise RuntimeError(
                "pipeline P2P communication events were observed without "
                f"matching schedule actions: {identities}"
            )
        return p2p_specs

    @staticmethod
    def _noop_residency_spec(intent: dict[str, Any]) -> ActionSpec:
        stage = int(intent.get("pp_stage", -1))
        mb_idx = int(intent.get("pp_mb_idx", -1))
        action_order = int(
            intent.get("action_order", intent.get("seq_idx", 0))
        )
        return ActionSpec(
            action_type=str(intent.get("action_type", "")),
            stage=stage,
            mb_idx=mb_idx,
            seq_idx=int(intent.get("seq_idx", 0)),
            order_key=(action_order, 0, 0),
            comm=CommDetail(
                src_stage=stage,
                dst_stage=stage,
                mb_idx=mb_idx,
                is_noop=True,
            ),
            is_noop=True,
            annotations={
                "capture_schedule_intent": True,
                "capture_action_order": action_order,
                "fsdp_intent_noop": True,
            },
        )

    @classmethod
    def _bind_residency_intents(
        cls,
        residency_specs: list[ActionSpec],
        schedule_intents: list[dict[str, Any]],
    ) -> list[ActionSpec]:
        residency_intents = [
            intent
            for intent in schedule_intents
            if intent.get("action_type") in _RESIDENCY_ACTION_TYPES
        ]
        if not residency_intents:
            return residency_specs

        matched_intents: set[int] = set()
        for spec_ordinal, spec in enumerate(residency_specs):
            # Only the transition emitted by the runtime's explicit
            # UNSHARD/RESHARD action belongs at the lowered schedule intent.
            # FSDP may reshard a module after a compute chunk and allocate it
            # again for a later microbatch. Those state transitions must keep
            # their observed action order instead of collapsing onto the
            # first stage-level prefetch intent.
            if spec.annotations.get("fsdp_schedule_source") != "intent":
                continue
            event_order = int(
                spec.annotations.get(
                    "capture_action_order", spec.order_key[0]
                )
            )
            candidates = [
                (intent_ordinal, intent)
                for intent_ordinal, intent in enumerate(residency_intents)
                if intent.get("action_type") == spec.action_type
                and int(intent.get("pp_stage", -1)) == spec.stage
                and int(
                    intent.get("action_order", intent.get("seq_idx", 0))
                ) <= event_order
            ]
            if not candidates:
                continue
            intent_ordinal, intent = max(
                candidates,
                key=lambda item: int(
                    item[1].get(
                        "action_order", item[1].get("seq_idx", 0)
                    )
                ),
            )
            matched_intents.add(intent_ordinal)
            intent_order = int(
                intent.get("action_order", intent.get("seq_idx", 0))
            )
            if spec.action_type == "UNSHARD":
                spec.order_key = (intent_order, 0, spec_ordinal)
            spec.annotations["capture_schedule_intent"] = True
            spec.annotations["schedule_intent_order"] = intent_order

        residency_specs.extend(
            cls._noop_residency_spec(intent)
            for intent_ordinal, intent in enumerate(residency_intents)
            if intent_ordinal not in matched_intents
        )
        return residency_specs

    @staticmethod
    def _compute_spec(event: dict[str, Any]) -> ActionSpec:
        comp_type = str(event.get("comp_type", "") or "")
        if not comp_type:
            action = str(event.get("action", ""))
            comp_type = "F" if "forward" in action else ("W" if "weight" in action else "B")
        stage = int(event.get("pp_stage", -1))
        mb_idx = int(event.get("pp_mb_idx", -1))
        start_seq = int(event.get("start_seq_idx", event.get("seq_idx", 0)))
        end_seq = int(event.get("end_seq_idx", event.get("seq_idx", start_seq)))
        action_order = int(event.get("action_order", start_seq))
        instance_id = str(event.get("instance_id", "") or f"s{stage}_{comp_type}_mb{mb_idx}")
        return ActionSpec(
            action_type="COMPUTE",
            stage=stage,
            mb_idx=mb_idx,
            comp_type=comp_type,
            template_ref=f"s{stage}_{comp_type}",
            seq_idx=start_seq,
            order_key=(action_order, 0, 0),
            annotations={
                "compute_instance_id": instance_id,
                "capture_start_seq": start_seq,
                "capture_end_seq": end_seq,
                "capture_action_order": action_order,
            },
        )

    @staticmethod
    def _p2p_spec(event: CommEvent) -> ActionSpec | None:
        mapping = _PP_DIRECTIONS.get(event.p2p_direction)
        if mapping is None:
            return None
        action_type, role = mapping
        stage = int(event.p2p_stage)
        mb_idx = int(event.p2p_mb_idx)
        direction = "forward" if event.p2p_direction.startswith("forward") else "backward"
        if event.transfer_id:
            transfer_id = event.transfer_id
        elif role == "send":
            dst_stage = stage + (1 if direction == "forward" else -1)
            transfer_id = (
                f"pp:{direction}:s{stage}->s{dst_stage}:mb{mb_idx}:"
                f"t{event.tensor_ordinal}"
            )
        else:
            src_stage = stage + (-1 if direction == "forward" else 1)
            transfer_id = (
                f"pp:{direction}:s{src_stage}->s{stage}:mb{mb_idx}:"
                f"t{event.tensor_ordinal}"
            )
        return ActionSpec(
            action_type=action_type,
            stage=stage,
            mb_idx=mb_idx,
            seq_idx=int(event.seq_idx),
            order_key=(
                event.action_order if event.action_order >= 0 else int(event.seq_idx),
                0,
                0,
            ),
            comm_op_id=event.op_id,
            comm=CommDetail(
                primitive="p2p_send",
                role=role,
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_stage=(stage if role == "send" else stage + (-1 if direction == "forward" else 1)),
                dst_stage=(stage + (1 if direction == "forward" else -1) if role == "send" else stage),
                mb_idx=mb_idx,
                peer_rank=event.p2p_peer_rank,
                comm_group_ranks=event.comm_ranks,
                src_exit_op=event.src_exit_op,
                dst_entry_op=event.dst_entry_op,
                comm_op_id=event.op_id,
                transfer_id=transfer_id,
            ),
            annotations={"transfer_id": transfer_id},
        )

    @staticmethod
    def _residency_spec(
        event: FSDPResidencyEvent,
        compute_by_instance: dict[str, ActionSpec],
        ordinal: int,
    ) -> ActionSpec:
        parent = compute_by_instance.get(event.parent_compute_instance_id)
        if parent is None:
            event_order = event.action_order
            candidates = [
                spec
                for spec in compute_by_instance.values()
                if spec.stage == event.pp_stage
                and (
                    event.pp_mb_idx < 0
                    or spec.mb_idx == event.pp_mb_idx
                )
            ]
            if event.action == "alloc":
                candidates = [
                    spec
                    for spec in candidates
                    if event_order < 0 or spec.order_key[0] >= event_order
                ]
                parent = min(candidates, key=lambda spec: spec.order_key, default=None)
            else:
                candidates = [
                    spec
                    for spec in candidates
                    if event_order < 0 or spec.order_key[0] <= event_order
                ]
                parent = max(candidates, key=lambda spec: spec.order_key, default=None)
            if parent is None:
                raise RuntimeError(
                    f"FSDP residency event for stage={event.pp_stage}, "
                    f"mb={event.pp_mb_idx}, group={event.group_id} has no "
                    "adjacent compute action"
                )
        event_order = (
            event.action_order if event.action_order >= 0 else ordinal
        )
        if (
            event.comp_type in {"UNSHARD", "RESHARD"}
            and event.action_order >= 0
        ):
            # Runtime schedules execute explicit residency actions outside a
            # compute chunk. Preserve their observed global action position.
            order_key = (event_order, 0, ordinal)
        else:
            # Ordinary FSDP invokes unshard/reshard inside a compute call.
            # Place those transitions immediately around their parent chunk.
            side = -2 if event.action == "alloc" else 2
            order_key = (parent.order_key[0], side, event_order)
        parent_instance_id = str(
            parent.annotations["compute_instance_id"]
        )
        return ActionSpec(
            action_type="UNSHARD" if event.action == "alloc" else "RESHARD",
            stage=int(event.pp_stage),
            mb_idx=int(event.pp_mb_idx),
            seq_idx=int(event.seq_idx),
            order_key=order_key,
            annotations={
                "fsdp_group_id": event.group_id,
                "parent_compute_instance_id": parent_instance_id,
                "residency_comp_type": parent.comp_type,
                "residency_bytes": event.num_bytes,
                "shard_world_size": event.shard_world_size,
                "fsdp_transition_id": event.transition_id,
                "fsdp_intent_noop": (
                    event.schedule_source == "intent"
                    and not event.transition_id
                ),
                "fsdp_schedule_source": event.schedule_source,
                "capture_action_order": event.action_order,
                "fsdp_module_fqn": event.module_fqn,
                "fsdp_prefetch_source_fqn": event.prefetch_source_fqn,
                "fsdp_prefetch_type": event.prefetch_type,
            },
        )

    @staticmethod
    def _reduce_grad_spec(event: CommEvent, ordinal: int) -> ActionSpec:
        stage = int(event.p2p_stage)
        return ActionSpec(
            action_type="REDUCE_GRAD",
            stage=stage,
            mb_idx=int(event.p2p_mb_idx),
            seq_idx=int(event.seq_idx),
            order_key=(
                event.action_order if event.action_order >= 0 else int(event.seq_idx),
                1,
                ordinal,
            ),
            comm_op_id=event.op_id,
            comm=CommDetail(
                primitive=event.comm_primitive,
                role="collective",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_stage=stage,
                dst_stage=stage,
                mb_idx=int(event.p2p_mb_idx),
                comm_group_ranks=event.comm_ranks,
                src_exit_op=event.src_exit_op,
                dst_entry_op=event.dst_entry_op,
                comm_op_id=event.op_id,
            ),
        )

    @staticmethod
    def _reduce_grad_intent_spec(
        intent: dict,
        event: CommEvent | None,
        ordinal: int,
    ) -> ActionSpec:
        stage = int(intent.get("pp_stage", -1))
        mb_idx = int(intent.get("pp_mb_idx", -1))
        seq_idx = int(intent.get("seq_idx", 0))
        action_order = int(intent.get("action_order", seq_idx))
        if event is None:
            comm = CommDetail(
                src_stage=stage,
                dst_stage=stage,
                mb_idx=mb_idx,
                is_noop=True,
            )
        else:
            comm = CommDetail(
                primitive=event.comm_primitive,
                role="collective",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_stage=stage,
                dst_stage=stage,
                mb_idx=mb_idx,
                comm_group_ranks=event.comm_ranks,
                src_exit_op=event.src_exit_op,
                dst_entry_op=event.dst_entry_op,
                comm_op_id=event.op_id,
            )
        return ActionSpec(
            action_type="REDUCE_GRAD",
            stage=stage,
            mb_idx=mb_idx,
            seq_idx=seq_idx,
            order_key=(action_order, 0, ordinal),
            comm_op_id=event.op_id if event is not None else 0,
            comm=comm,
            is_noop=event is None,
            annotations={
                "capture_schedule_intent": True,
                "capture_action_order": action_order,
            },
        )

    def build(self) -> list[ActionSpec]:
        schedule_intents = self._schedule_intents()
        compute_events = [
            event
            for event in self.timeline_events
            if event.get("event_kind") != "schedule_action"
        ]
        observed_compute_specs = [
            self._compute_spec(event) for event in compute_events
        ]
        compute_by_instance = {
            str(spec.annotations["compute_instance_id"]): spec
            for spec in observed_compute_specs
        }
        specs = self._bind_compute_intents(
            observed_compute_specs,
            schedule_intents,
        )

        p2p_specs = [
            spec
            for event in self.comm_events
            if (spec := self._p2p_spec(event)) is not None
        ]
        specs.extend(self._bind_p2p_intents(p2p_specs, schedule_intents))
        schedule_residency_events = [
            event
            for event in self.fsdp_residency_events
            if event.action in {"alloc", "free"}
            and event.pp_stage >= 0
        ]
        merged_residency_events: dict[
            tuple[str, str] | tuple[str, int], FSDPResidencyEvent
        ] = {}
        for ordinal, event in enumerate(schedule_residency_events):
            key: tuple[str, str] | tuple[str, int]
            if event.transition_id:
                key = (event.transition_id, event.action)
            else:
                key = ("legacy", ordinal)
            previous = merged_residency_events.get(key)
            if previous is None:
                merged_residency_events[key] = event
                continue
            if event.action == "alloc":
                # The explicit UNSHARD call is the schedule position; the
                # state event may occur later when async all-gather is waited.
                if (
                    event.schedule_source == "intent"
                    and previous.schedule_source != "intent"
                ):
                    merged_residency_events[key] = event
            elif (
                event.schedule_source == "state"
                and previous.schedule_source != "state"
            ):
                # Actual state loss is the memory release position. The later
                # explicit RESHARD call may already be a no-op.
                merged_residency_events[key] = event
        captured_alloc_transitions = {
            event.transition_id
            for event in merged_residency_events.values()
            if event.action == "alloc" and event.transition_id
        }
        merged_residency_events = {
            key: event
            for key, event in merged_residency_events.items()
            if not (
                event.action == "free"
                and event.transition_id
                and event.transition_id not in captured_alloc_transitions
            )
        }
        residency_specs = [
            self._residency_spec(event, compute_by_instance, ordinal)
            for ordinal, event in enumerate(merged_residency_events.values())
        ]
        specs.extend(
            self._bind_residency_intents(
                residency_specs,
                schedule_intents,
            )
        )
        reduce_intents = [
            event
            for event in self.timeline_events
            if event.get("event_kind") == "schedule_action"
            and event.get("action_type") == "REDUCE_GRAD"
        ]
        reduce_comm_events = [
            event
            for event in self.comm_events
            if event.comm_primitive in {"reduce_scatter", "allreduce"}
            and (
                event.comm_layer == "L2"
                or event.comp_type == "REDUCE_GRAD"
            )
        ]
        matched_reduce_event_ids: set[str] = set()
        for ordinal, intent in enumerate(reduce_intents):
            intent_stage = int(intent.get("pp_stage", -1))
            intent_order = int(intent.get("action_order", -1))
            candidates = [
                event
                for event in reduce_comm_events
                if event.event_id not in matched_reduce_event_ids
                and event.p2p_stage == intent_stage
                and (
                    intent_order < 0
                    or event.action_order < 0
                    or event.action_order >= intent_order
                )
            ]
            matched = min(
                candidates,
                key=lambda event: (
                    event.action_order
                    if event.action_order >= 0
                    else event.seq_idx
                ),
                default=None,
            )
            if matched is not None:
                matched_reduce_event_ids.add(matched.event_id)
            specs.append(
                self._reduce_grad_intent_spec(intent, matched, ordinal)
            )
        specs.extend(
            self._reduce_grad_spec(event, ordinal)
            for ordinal, event in enumerate(reduce_comm_events)
            if event.event_id not in matched_reduce_event_ids
        )
        def annotate(spec: ActionSpec) -> None:
            spec.annotations.setdefault("capture_process_rank", self.rank)
            for child in spec.sub_actions or []:
                annotate(child)

        for spec in specs:
            annotate(spec)
        return sorted(specs, key=lambda spec: spec.order_key)


def pp_transfer_id(event: CommEvent) -> str:
    """Return the stable rendezvous id generated by the trace assembler."""
    spec = CapturedTraceAssembler._p2p_spec(event)
    return "" if spec is None or spec.comm is None else spec.comm.transfer_id


# Compatibility for direct imports while callers migrate to the schedule-
# agnostic name.
SingleStageTraceAssembler = CapturedTraceAssembler
