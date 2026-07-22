# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_plan
from torchtitan_npu.simulator.capture.schedule_validation import (
    replay_1f1b_readiness,
    validate_1f1b_transfer_pairs,
)
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent


class _RankTable:
    def __init__(self, *, pp: int = 2, dp_shard: int = 1) -> None:
        self.dim_degrees = {
            "pp": pp,
            "dp_replicate": 1,
            "dp_shard": dp_shard,
            "tp": 1,
        }

    def to_dict(self) -> dict:
        return {"dim_degrees": dict(self.dim_degrees)}


def _timeline(stage: int, comp_type: str, mb_idx: int, start: int, end: int) -> dict:
    return {
        "pp_stage": stage,
        "pp_mb_idx": mb_idx,
        "comp_type": comp_type,
        "start_seq_idx": start,
        "end_seq_idx": end,
        "seq_idx": end,
        "instance_id": f"s{stage}_{comp_type}_mb{mb_idx}",
    }


def _p2p(
    event_id: str,
    direction: str,
    *,
    stage: int,
    mb_idx: int,
    seq_idx: int,
    peer_rank: int,
) -> CommEvent:
    primitive = "p2p_send" if direction.endswith("send") else "p2p_recv"
    return CommEvent(
        event_id=event_id,
        comm_primitive=primitive,
        group_name="pp",
        world_size=2,
        tensor_shape=(2, 8),
        dtype="bfloat16",
        volume_bytes=32,
        p2p_peer_rank=peer_rank,
        p2p_direction=direction,
        p2p_mb_idx=mb_idx,
        p2p_stage=stage,
        seq_idx=seq_idx,
        comm_layer="L2",
    )


def _templates(stage: int) -> dict[str, StepGraph]:
    return {
        f"s{stage}_F": StepGraph(f"s{stage}_F", "F", {}),
        f"s{stage}_B": StepGraph(f"s{stage}_B", "B", {}),
    }


def test_two_stage_1f1b_trace_has_complete_cross_rank_dependencies() -> None:
    rank0 = build_schedule_plan(
        step_templates=_templates(0),
        rank_table=_RankTable(),
        comm_events=[
            _p2p("f-send-0", "forward_send", stage=0, mb_idx=0, seq_idx=20, peer_rank=1),
            _p2p("f-send-1", "forward_send", stage=0, mb_idx=1, seq_idx=40, peer_rank=1),
            _p2p("b-recv-0", "backward_recv", stage=0, mb_idx=0, seq_idx=50, peer_rank=1),
            _p2p("b-recv-1", "backward_recv", stage=0, mb_idx=1, seq_idx=80, peer_rank=1),
            # CP uses similarly named metadata but is not a PP stage transfer.
            _p2p("cp-send", "cp_forward_send", stage=0, mb_idx=0, seq_idx=25, peer_rank=1),
        ],
        timeline_events=[
            _timeline(0, "F", 0, 10, 11),
            _timeline(0, "F", 1, 30, 31),
            _timeline(0, "B", 0, 60, 61),
            _timeline(0, "B", 1, 90, 91),
        ],
        pipeline_schedule="1F1B",
        num_micro_batches=2,
        rank=0,
    )
    rank1 = build_schedule_plan(
        step_templates=_templates(1),
        rank_table=_RankTable(),
        comm_events=[
            _p2p("f-recv-0", "forward_recv", stage=1, mb_idx=0, seq_idx=5, peer_rank=0),
            _p2p("b-send-0", "backward_send", stage=1, mb_idx=0, seq_idx=35, peer_rank=0),
            _p2p("f-recv-1", "forward_recv", stage=1, mb_idx=1, seq_idx=45, peer_rank=0),
            _p2p("b-send-1", "backward_send", stage=1, mb_idx=1, seq_idx=75, peer_rank=0),
        ],
        timeline_events=[
            _timeline(1, "F", 0, 15, 16),
            _timeline(1, "B", 0, 25, 26),
            _timeline(1, "F", 1, 55, 56),
            _timeline(1, "B", 1, 65, 66),
        ],
        pipeline_schedule="1F1B",
        num_micro_batches=2,
        rank=1,
    )

    assert rank0.annotations["assembler"] == "single_stage_trace"
    assert [action.action_type for action in rank0.actions] == [
        "COMPUTE",
        "SEND_F",
        "COMPUTE",
        "SEND_F",
        "RECV_B",
        "COMPUTE",
        "RECV_B",
        "COMPUTE",
    ]
    assert all("cp" not in action.action_id for action in rank0.actions)

    send = next(action for action in rank0.actions if action.action_type == "SEND_F")
    send_slot = rank0.data_slots[send.consumes[0]]
    assert send_slot.kind == "activation_local"
    assert send_slot.producer_action_id

    recv = next(action for action in rank1.actions if action.action_type == "RECV_F")
    recv_slot = rank1.data_slots[recv.produces[0]]
    assert recv_slot.kind == "activation_recv"
    assert recv_slot.consumer_action_ids

    validate_1f1b_transfer_pairs([rank0, rank1])
    replay_1f1b_readiness([rank0, rank1])


def test_fsdp_residency_and_gradient_reduction_are_distinct_dependencies() -> None:
    allgather = OpNode(
        op_id=900,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.allgather"},
    )
    reduce_scatter = OpNode(
        op_id=901,
        op_type="reduce_scatter",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.reduce_scatter"},
    )
    templates = {
        "s0_F": StepGraph("s0_F", "F", {900: allgather}),
        "s0_B": StepGraph("s0_B", "B", {901: reduce_scatter}),
    }
    comm_events = [
        CommEvent(
            event_id="allgather",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=900,
            comm_layer="L2",
            p2p_stage=0,
            p2p_mb_idx=0,
            seq_idx=12,
            comp_type="F",
        ),
        CommEvent(
            event_id="reduce",
            comm_primitive="reduce_scatter",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=901,
            comm_layer="L2",
            p2p_stage=0,
            p2p_mb_idx=0,
            seq_idx=80,
        ),
    ]
    residency = [
        FSDPResidencyEvent(
            group_id="block0",
            action="alloc",
            seq_idx=12,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
        ),
        FSDPResidencyEvent(
            group_id="block0",
            action="free",
            seq_idx=48,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
        ),
    ]

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[
            _timeline(0, "F", 0, 10, 50),
            _timeline(0, "B", 0, 60, 100),
        ],
        pipeline_schedule="1F1B",
        rank=0,
    )

    assert [action.action_type for action in plan.actions] == [
        "UNSHARD",
        "COMPUTE",
        "RESHARD",
        "COMPUTE",
        "REDUCE_GRAD",
    ]
    reshard = next(action for action in plan.actions if action.action_type == "RESHARD")
    reshard_slot = plan.data_slots[reshard.consumes[0]]
    assert reshard.comm is None
    assert reshard_slot.kind == "control"
    assert plan.action_map[reshard_slot.producer_action_id].comp_type == "F"

    reduce = next(action for action in plan.actions if action.action_type == "REDUCE_GRAD")
    reduce_slot = next(plan.data_slots[slot_id] for slot_id in reduce.consumes)
    assert reduce.comm is not None
    assert reduce.comm.primitive == "reduce_scatter"
    assert reduce_slot.kind == "grad_local"
    assert plan.action_map[reduce_slot.producer_action_id].comp_type == "B"


def test_repeated_fsdp_residency_reuses_folded_allgather_template() -> None:
    allgather = OpNode(
        op_id=910,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.allgather"},
    )
    templates = {
        "s0_F": StepGraph("s0_F", "F", {910: allgather}),
        "s0_B": StepGraph("s0_B", "B", {}),
    }
    comm_events = [
        CommEvent(
            event_id="first-mb-allgather",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=910,
            comm_layer="L2",
            p2p_stage=0,
            p2p_mb_idx=0,
            seq_idx=12,
            comp_type="F",
        ),
    ]
    residency = [
        FSDPResidencyEvent(
            group_id="metadata-inference",
            action="alloc",
            seq_idx=1,
            phase="forward",
            num_bytes=256,
        ),
        FSDPResidencyEvent(
            group_id="metadata-inference",
            action="free",
            seq_idx=2,
            phase="forward",
            num_bytes=256,
        ),
    ]
    for mb_idx, seq_idx in enumerate((12, 32)):
        residency.extend([
            FSDPResidencyEvent(
                group_id="block0",
                action="alloc",
                seq_idx=seq_idx,
                phase="forward",
                num_bytes=256,
                pp_stage=0,
                pp_mb_idx=mb_idx,
                comp_type="F",
                parent_compute_instance_id=f"s0_F_mb{mb_idx}",
            ),
            FSDPResidencyEvent(
                group_id="block0",
                action="free",
                seq_idx=seq_idx + 8,
                phase="forward",
                num_bytes=256,
                pp_stage=0,
                pp_mb_idx=mb_idx,
                comp_type="F",
                parent_compute_instance_id=f"s0_F_mb{mb_idx}",
            ),
        ])

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[
            _timeline(0, "F", 0, 10, 20),
            _timeline(0, "F", 1, 30, 40),
            _timeline(0, "B", 0, 50, 60),
            _timeline(0, "B", 1, 70, 80),
        ],
        pipeline_schedule="1F1B",
        num_micro_batches=2,
        rank=0,
    )

    unshards = [action for action in plan.actions if action.action_type == "UNSHARD"]
    assert len(unshards) == 2
    assert [action.comm_op_id for action in unshards] == [910, 910]
    assert all(action.comm is not None and action.comm.primitive == "allgather" for action in unshards)
