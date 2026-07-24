# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

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


def test_action_order_is_independent_from_l0_sequence_ids() -> None:
    send = _p2p(
        "f-send",
        "forward_send",
        stage=0,
        mb_idx=0,
        seq_idx=1,
        peer_rank=1,
    )
    send.action_order = 1
    timeline = _timeline(0, "F", 0, 100, 200)
    timeline["action_order"] = 0

    plan = build_schedule_plan(
        step_templates={"s0_F": StepGraph("s0_F", "F", {})},
        rank_table=_RankTable(),
        comm_events=[send],
        timeline_events=[timeline],
        pipeline_schedule="custom",
        rank=0,
    )

    assert [action.action_type for action in plan.actions[:2]] == [
        "COMPUTE",
        "SEND_F",
    ]
    assert [action.schedule_order for action in plan.actions[:2]] == [0, 1]
    assert [action.seq_idx for action in plan.actions[:2]] == [100, 1]


def test_schedule_compute_intent_without_execution_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="without matching compute execution"):
        build_schedule_plan(
            step_templates={},
            rank_table=_RankTable(),
            comm_events=[],
            timeline_events=[
                {
                    "event_kind": "schedule_action",
                    "action_type": "F",
                    "pp_stage": 1,
                    "pp_mb_idx": 0,
                    "seq_idx": 10,
                    "action_order": 0,
                }
            ],
            pipeline_schedule="custom",
            rank=1,
            captured_trace_primary=True,
        )


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

    assert rank0.annotations["assembler"] == "captured_trace"
    assert rank0.annotations["capture_schema_version"] == 2
    assert rank0.annotations["capture_process_rank"] == 0
    assert all(
        action.annotations["capture_schema_version"] == 2
        for action in rank0.actions
    )
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


def test_split_backward_trace_builds_semantic_dependencies() -> None:
    optimizer_node = OpNode(
        op_id=999,
        op_type="optimizer",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=60,
    )
    templates = {
        "s0_F": StepGraph("s0_F", "F", {}),
        "s0_I": StepGraph("s0_I", "I", {}),
        "s0_W": StepGraph("s0_W", "W", {}),
        "s0_OPTIMIZER": StepGraph(
            "s0_OPTIMIZER", "OPTIMIZER", {999: optimizer_node}
        ),
    }
    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(),
        comm_events=[
            _p2p(
                "f-send",
                "forward_send",
                stage=0,
                mb_idx=0,
                seq_idx=20,
                peer_rank=1,
            ),
            _p2p(
                "b-recv",
                "backward_recv",
                stage=0,
                mb_idx=0,
                seq_idx=30,
                peer_rank=1,
            ),
        ],
        timeline_events=[
            _timeline(0, "F", 0, 10, 11),
            _timeline(0, "I", 0, 40, 41),
            _timeline(0, "W", 0, 50, 51),
        ],
        pipeline_schedule="zero-bubble",
        rank=0,
    )

    forward = next(action for action in plan.actions if action.comp_type == "F")
    backward_input = next(action for action in plan.actions if action.comp_type == "I")
    backward_weight = next(action for action in plan.actions if action.comp_type == "W")
    backward_recv = next(
        action for action in plan.actions if action.action_type == "RECV_B"
    )
    optimizer = next(
        action for action in plan.actions if action.action_type == "OPTIMIZER"
    )

    recv_slots = [
        plan.data_slots[slot_id] for slot_id in backward_recv.produces
    ]
    assert any(
        backward_input.action_id in slot.consumer_action_ids
        for slot in recv_slots
    )

    forward_state = next(
        plan.data_slots[slot_id]
        for slot_id in forward.produces
        if plan.data_slots[slot_id].kind == "forward_state"
    )
    assert set(forward_state.consumer_action_ids) == {
        backward_input.action_id,
        backward_weight.action_id,
    }
    assert any(
        plan.data_slots[slot_id].kind == "dataloader_input"
        for slot_id in forward.consumes
    )
    assert any(
        plan.data_slots[slot_id].producer_action_id == backward_weight.action_id
        for slot_id in optimizer.consumes
    )


def test_virtual_stages_build_local_split_backward_dependency() -> None:
    schedule = type(
        "_VirtualSchedule",
        (),
        {"stage_index_to_group_rank": {0: 0, 1: 1, 2: 1, 3: 0}},
    )()
    plan = build_schedule_plan(
        step_templates={
            "s1_F": StepGraph("s1_F", "F", {}),
            "s1_I": StepGraph("s1_I", "I", {}),
            "s2_F": StepGraph("s2_F", "F", {}),
            "s2_I": StepGraph("s2_I", "I", {}),
        },
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=[
            _timeline(1, "F", 0, 10, 11),
            _timeline(2, "F", 0, 20, 21),
            _timeline(2, "I", 0, 30, 31),
            _timeline(1, "I", 0, 40, 41),
        ],
        pp_schedule_obj=schedule,
        pipeline_schedule="virtual",
        rank=1,
        captured_trace_primary=True,
    )

    stage2_backward = next(
        action
        for action in plan.actions
        if action.stage == 2 and action.comp_type == "I"
    )
    stage1_backward = next(
        action
        for action in plan.actions
        if action.stage == 1 and action.comp_type == "I"
    )
    local_grad = next(
        slot
        for slot in plan.data_slots.values()
        if slot.kind == "grad_input" and slot.is_local_transfer
    )

    assert local_grad.producer_action_id == stage2_backward.action_id
    assert local_grad.consumer_action_ids == [stage1_backward.action_id]
    assert not any(
        slot.kind == "loss_grad" for slot in plan.data_slots.values()
    )


def test_virtual_last_stage_receives_external_loss_gradient() -> None:
    schedule = type(
        "_VirtualSchedule",
        (),
        {"stage_index_to_group_rank": {0: 0, 1: 1, 2: 1, 3: 0}},
    )()
    plan = build_schedule_plan(
        step_templates={
            "s3_F": StepGraph("s3_F", "F", {}),
            "s3_I": StepGraph("s3_I", "I", {}),
        },
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=[
            _timeline(3, "F", 0, 10, 11),
            _timeline(3, "I", 0, 20, 21),
        ],
        pp_schedule_obj=schedule,
        pipeline_schedule="virtual",
        rank=0,
        captured_trace_primary=True,
    )

    loss_grad = next(
        slot
        for slot in plan.data_slots.values()
        if slot.kind == "loss_grad"
    )
    assert loss_grad.external
    assert loss_grad.dst_stage == 3


def test_reduce_grad_schedule_intent_without_collective_is_nonblocking() -> None:
    optimizer_node = OpNode(
        op_id=999,
        op_type="optimizer",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=60,
    )
    templates = {
        "s0_F": StepGraph("s0_F", "F", {}),
        "s0_I": StepGraph("s0_I", "I", {}),
        "s0_W": StepGraph("s0_W", "W", {}),
        "s0_OPTIMIZER": StepGraph(
            "s0_OPTIMIZER", "OPTIMIZER", {999: optimizer_node}
        ),
    }
    forward = _timeline(0, "F", 0, 10, 11)
    forward["action_order"] = 0
    backward_input = _timeline(0, "I", 0, 20, 21)
    backward_input["action_order"] = 1
    backward_weight = _timeline(0, "W", 0, 30, 31)
    backward_weight["action_order"] = 2

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=[],
        timeline_events=[
            forward,
            backward_input,
            backward_weight,
            {
                "event_kind": "schedule_action",
                "action_type": "REDUCE_GRAD",
                "pp_stage": 0,
                "pp_mb_idx": -1,
                "seq_idx": 40,
                "action_order": 3,
            },
        ],
        pipeline_schedule="ZBVZeroBubble",
        rank=0,
    )

    reduce_grad = next(
        action for action in plan.actions if action.action_type == "REDUCE_GRAD"
    )
    optimizer = next(
        action for action in plan.actions if action.action_type == "OPTIMIZER"
    )
    weight = next(action for action in plan.actions if action.comp_type == "W")

    assert reduce_grad.is_noop
    assert reduce_grad.comm is not None and reduce_grad.comm.is_noop
    assert not reduce_grad.consumes
    assert not reduce_grad.produces
    assert any(
        plan.data_slots[slot_id].producer_action_id == weight.action_id
        for slot_id in optimizer.consumes
    )


def test_explicit_fsdp_action_resolves_adjacent_compute_by_observed_order() -> None:
    timeline = _timeline(0, "F", 0, 20, 30)
    timeline["action_order"] = 5
    residency_alloc = FSDPResidencyEvent(
        group_id="block0",
        action="alloc",
        seq_idx=10,
        phase="forward",
        num_bytes=256,
        pp_stage=0,
        pp_mb_idx=0,
        comp_type="UNSHARD",
        parent_compute_instance_id="s0_UNSHARD_mb0",
        shard_world_size=1,
        action_order=1,
        transition_id="fsdp:r0:gblock0:u0",
    )
    residency_free = FSDPResidencyEvent(
        group_id="block0",
        action="free",
        seq_idx=40,
        phase="forward",
        num_bytes=256,
        pp_stage=0,
        pp_mb_idx=0,
        comp_type="RESHARD",
        parent_compute_instance_id="s0_RESHARD_mb0",
        shard_world_size=1,
        action_order=9,
        transition_id="fsdp:r0:gblock0:u0",
    )

    plan = build_schedule_plan(
        step_templates={"s0_F": StepGraph("s0_F", "F", {})},
        rank_table=_RankTable(pp=1),
        comm_events=[],
        fsdp_residency_events=[residency_alloc, residency_free],
        timeline_events=[timeline],
        pipeline_schedule="ZBVZeroBubble",
        rank=0,
    )

    assert [action.action_type for action in plan.actions] == [
        "UNSHARD",
        "COMPUTE",
        "RESHARD",
    ]
    unshard = plan.actions[0]
    assert unshard.is_noop
    assert unshard.annotations["parent_compute_instance_id"] == "s0_F_mb0"
    assert unshard.annotations["residency_comp_type"] == "F"


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


def test_group_local_shard_size_controls_unshard_noop() -> None:
    allgather = OpNode(
        op_id=920,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.allgather"},
    )
    templates = {
        "s0_F": StepGraph("s0_F", "F", {920: allgather}),
        "s0_B": StepGraph("s0_B", "B", {}),
    }
    comm_events = [
        CommEvent(
            event_id="dense-allgather",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=920,
            comm_layer="L2",
            p2p_stage=0,
            p2p_mb_idx=0,
            seq_idx=12,
            comp_type="F",
        ),
    ]
    residency = [
        FSDPResidencyEvent(
            group_id="expert-efsdp",
            action="alloc",
            seq_idx=11,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=1,
        ),
        FSDPResidencyEvent(
            group_id="dense",
            action="alloc",
            seq_idx=12,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=2,
        ),
        FSDPResidencyEvent(
            group_id="dense",
            action="free",
            seq_idx=18,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=2,
        ),
        FSDPResidencyEvent(
            group_id="expert-efsdp",
            action="free",
            seq_idx=19,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=1,
        ),
    ]

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[
            _timeline(0, "F", 0, 10, 20),
            _timeline(0, "B", 0, 30, 40),
        ],
        pipeline_schedule="1F1B",
        rank=0,
    )

    unshards = [action for action in plan.actions if action.action_type == "UNSHARD"]
    dense = next(
        action for action in unshards if action.annotations["fsdp_group_id"] == "dense"
    )
    expert = next(
        action for action in unshards if action.annotations["fsdp_group_id"] == "expert-efsdp"
    )
    assert dense.comm is not None and dense.comm.primitive == "allgather"
    assert not dense.is_noop
    assert expert.is_noop
    assert expert.comm is not None and expert.comm.is_noop
    assert not expert.consumes and not expert.produces


def test_fsdp_transition_id_matches_allgather_without_order_guessing() -> None:
    nodes = {
        op_id: OpNode(
            op_id=op_id,
            op_type="allgather",
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=[],
            successors=[],
            annotations={"raw_op_type": "comm.allgather"},
        )
        for op_id in (930, 931)
    }
    templates = {"s0_F": StepGraph("s0_F", "F", nodes)}
    comm_events = [
        CommEvent(
            event_id="group-b",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(64,),
            dtype="bfloat16",
            volume_bytes=128,
            op_id=931,
            comm_layer="L2",
            p2p_stage=0,
            comp_type="F",
            seq_idx=11,
            fsdp_group_id="group-b",
            fsdp_transition_id="transition-b",
        ),
        CommEvent(
            event_id="group-a",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=930,
            comm_layer="L2",
            p2p_stage=0,
            comp_type="F",
            seq_idx=99,
            fsdp_group_id="group-a",
            fsdp_transition_id="transition-a",
        ),
    ]
    residency = []
    for order, (group_id, transition_id) in enumerate(
        (("group-a", "transition-a"), ("group-b", "transition-b"))
    ):
        residency.extend(
            [
                FSDPResidencyEvent(
                    group_id=group_id,
                    action="alloc",
                    seq_idx=20 + order,
                    phase="forward",
                    num_bytes=256,
                    pp_stage=0,
                    pp_mb_idx=0,
                    comp_type="F",
                    parent_compute_instance_id="s0_F_mb0",
                    shard_world_size=2,
                    transition_id=transition_id,
                    action_order=order,
                ),
                FSDPResidencyEvent(
                    group_id=group_id,
                    action="free",
                    seq_idx=30 + order,
                    phase="forward",
                    num_bytes=256,
                    pp_stage=0,
                    pp_mb_idx=0,
                    comp_type="F",
                    parent_compute_instance_id="s0_F_mb0",
                    shard_world_size=2,
                    transition_id=transition_id,
                    action_order=order + 10,
                ),
            ]
        )

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[_timeline(0, "F", 0, 10, 40)],
        pipeline_schedule="custom",
        rank=0,
    )

    matched = {
        action.annotations["fsdp_group_id"]: action.comm_op_id
        for action in plan.actions
        if action.action_type == "UNSHARD"
    }
    assert matched == {"group-a": 930, "group-b": 931}
