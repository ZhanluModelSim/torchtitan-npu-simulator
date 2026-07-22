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
    annotations: dict[str, Any] = field(default_factory=dict)


class SingleStageTraceAssembler:
    """Build a 1F1B/GPipe action skeleton from captured execution spans."""

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
        instance_id = str(event.get("instance_id", "") or f"s{stage}_{comp_type}_mb{mb_idx}")
        return ActionSpec(
            action_type="COMPUTE",
            stage=stage,
            mb_idx=mb_idx,
            comp_type=comp_type,
            template_ref=f"s{stage}_{comp_type}",
            seq_idx=start_seq,
            order_key=(start_seq, 0, 0),
            annotations={
                "compute_instance_id": instance_id,
                "capture_start_seq": start_seq,
                "capture_end_seq": end_seq,
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
            transfer_id = f"pp:{direction}:s{stage}->s{dst_stage}:mb{mb_idx}:t0"
        else:
            src_stage = stage + (-1 if direction == "forward" else 1)
            transfer_id = f"pp:{direction}:s{src_stage}->s{stage}:mb{mb_idx}:t0"
        return ActionSpec(
            action_type=action_type,
            stage=stage,
            mb_idx=mb_idx,
            seq_idx=int(event.seq_idx),
            order_key=(int(event.seq_idx), 0, 0),
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
            raise RuntimeError(
                f"FSDP residency event for stage={event.pp_stage}, mb={event.pp_mb_idx}, "
                f"group={event.group_id} has no parent compute "
                f"{event.parent_compute_instance_id!r}"
            )
        side = -2 if event.action == "alloc" else 2
        order_key = (parent.order_key[0], side, ordinal)
        return ActionSpec(
            action_type="UNSHARD" if event.action == "alloc" else "RESHARD",
            stage=int(event.pp_stage),
            mb_idx=int(event.pp_mb_idx),
            seq_idx=int(event.seq_idx),
            order_key=order_key,
            annotations={
                "fsdp_group_id": event.group_id,
                "parent_compute_instance_id": event.parent_compute_instance_id,
                "residency_comp_type": event.comp_type,
                "residency_bytes": event.num_bytes,
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
            order_key=(int(event.seq_idx), 1, ordinal),
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

    def build(self) -> list[ActionSpec]:
        specs = [self._compute_spec(event) for event in self.timeline_events]
        compute_by_instance = {
            str(spec.annotations["compute_instance_id"]): spec
            for spec in specs
        }

        specs.extend(
            spec
            for event in self.comm_events
            if (spec := self._p2p_spec(event)) is not None
        )
        schedule_residency_events = [
            event
            for event in self.fsdp_residency_events
            if event.action in {"alloc", "free"}
            and event.pp_stage >= 0
            and event.pp_mb_idx >= 0
        ]
        specs.extend(
            self._residency_spec(event, compute_by_instance, ordinal)
            for ordinal, event in enumerate(schedule_residency_events)
        )
        specs.extend(
            self._reduce_grad_spec(event, ordinal)
            for ordinal, event in enumerate(self.comm_events)
            if event.comm_layer == "L2" and event.comm_primitive in {"reduce_scatter", "allreduce"}
        )
        for spec in specs:
            spec.annotations.setdefault("capture_process_rank", self.rank)
        return sorted(specs, key=lambda spec: spec.order_key)


def pp_transfer_id(event: CommEvent) -> str:
    """Return the stable rendezvous id generated by the trace assembler."""
    spec = SingleStageTraceAssembler._p2p_spec(event)
    return "" if spec is None or spec.comm is None else spec.comm.transfer_id
