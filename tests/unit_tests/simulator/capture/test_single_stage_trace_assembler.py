# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.communication_ownership import (
    normalize_communication_ownership,
)
from torchtitan_npu.simulator.capture.schedule_assemblers import (
    ActionSpec,
    CapturedTraceAssembler,
)
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_plan
from torchtitan_npu.simulator.capture.schedule_validation import (
    replay_1f1b_readiness,
    validate_1f1b_transfer_pairs,
)
from torchtitan_npu.simulator.capture.workload_builder import build_workload_graph
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
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


def _fsdp_marker(
    op_id: int,
    seq_idx: int,
    marker: str,
    group_id: str,
    module_fqn: str,
) -> OpNode:
    return OpNode(
        op_id=op_id,
        op_type=f"fsdp_{marker}",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=seq_idx,
        annotations={
            "raw_op_type": f"sim.fsdp_{marker}",
            "fsdp_marker": marker,
            "fsdp_group_id": group_id,
            "fsdp_module_fqn": module_fqn,
        },
    )


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


def test_dualpipe_overlap_intent_groups_observed_compute_children() -> None:
    overlap_intent = {
        "event_kind": "schedule_action",
        "action_type": "OVERLAP_F_B",
        "pp_stage": -1,
        "pp_mb_idx": -1,
        "seq_idx": 10,
        "action_order": 4,
        "sub_actions": [
            {"action_type": "F", "pp_stage": 0, "pp_mb_idx": 3},
            {"action_type": "B", "pp_stage": 3, "pp_mb_idx": 1},
        ],
    }
    forward = _timeline(0, "F", 3, 20, 21)
    forward["action_order"] = 5
    backward = _timeline(3, "B", 1, 30, 31)
    backward["action_order"] = 6

    plan = build_schedule_plan(
        step_templates={
            "s0_F": StepGraph("s0_F", "F", {}),
            "s3_B": StepGraph("s3_B", "B", {}),
        },
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=[overlap_intent, forward, backward],
        pipeline_schedule="DualPipeV",
        rank=0,
        captured_trace_primary=True,
    )

    overlap = next(
        action
        for action in plan.actions
        if action.action_type == "OVERLAP_F_B"
    )
    assert [(child.stage, child.mb_idx, child.comp_type) for child in overlap.sub_actions] == [
        (0, 3, "F"),
        (3, 1, "B"),
    ]
    assert all(
        child.schedule_order == overlap.schedule_order
        for child in overlap.sub_actions
    )
    assert not any(
        action.action_type == "COMPUTE"
        for action in plan.actions
    )


def test_runtime_fsdp_intents_without_fsdp_module_are_removed_from_l2() -> None:
    forward = _timeline(0, "F", 0, 20, 21)
    forward["action_order"] = 1
    plan = build_schedule_plan(
        step_templates={"s0_F": StepGraph("s0_F", "F", {})},
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=[
            {
                "event_kind": "schedule_action",
                "action_type": "UNSHARD",
                "pp_stage": 0,
                "pp_mb_idx": -1,
                "seq_idx": 10,
                "action_order": 0,
            },
            {
                "event_kind": "schedule_action",
                "action_type": "F",
                "pp_stage": 0,
                "pp_mb_idx": 0,
                "seq_idx": 19,
                "action_order": 1,
            },
            forward,
            {
                "event_kind": "schedule_action",
                "action_type": "RESHARD",
                "pp_stage": 0,
                "pp_mb_idx": -1,
                "seq_idx": 30,
                "action_order": 2,
            },
        ],
        pipeline_schedule="DualPipeV",
        rank=0,
        captured_trace_primary=True,
    )

    assert [action.action_type for action in plan.actions] == ["COMPUTE"]
    assert all(
        slot.kind not in {"param_full", "param_shard"}
        for slot in plan.data_slots.values()
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


def test_pipeline_p2p_is_extracted_from_stage_compute_template() -> None:
    p2p_node = OpNode(
        op_id=70,
        op_type="p2p_send",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[71],
        annotations={"raw_op_type": "comm.p2p_send"},
    )
    compute_node = OpNode(
        op_id=71,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[70],
        successors=[],
    )
    send = _p2p(
        "f-send",
        "forward_send",
        stage=0,
        mb_idx=0,
        seq_idx=20,
        peer_rank=1,
    )
    send.op_id = 70
    plan = build_schedule_plan(
        step_templates={
            "s0_F": StepGraph(
                "s0_F",
                "F",
                {70: p2p_node, 71: compute_node},
            )
        },
        rank_table=_RankTable(),
        comm_events=[send],
        timeline_events=[_timeline(0, "F", 0, 10, 11)],
        pipeline_schedule="1F1B",
        rank=0,
    )

    assert set(plan.step_templates["s0_F"].nodes) == {71}
    assert plan.step_templates["s0_F"].nodes[71].predecessors == []
    fragment = plan.step_templates["s0_PP_SEND_F"]
    assert set(fragment.nodes) == {70}
    assert (
        fragment.nodes[70].annotations["communication_owner"]
        == "L2_PIPELINE"
    )


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

    optimizer = next(
        action for action in plan.actions if action.action_type == "OPTIMIZER"
    )
    weight = next(action for action in plan.actions if action.comp_type == "W")

    assert not any(
        action.action_type == "REDUCE_GRAD" for action in plan.actions
    )
    assert any(
        plan.data_slots[slot_id].producer_action_id == weight.action_id
        for slot_id in optimizer.consumes
    )
    assert plan.annotations["communication_ownership"][
        "removed_noop_gradient_intents"
    ] == 1


def test_compute_local_size_one_fsdp_transition_is_removed_from_l2() -> None:
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

    assert [action.action_type for action in plan.actions] == ["COMPUTE"]
    assert plan.annotations["communication_ownership"] == {
        "internal_fsdp_transitions": 1,
        "external_fsdp_prefetches": 0,
        "generated_l1_templates": [],
        "internal_gradient_reductions": 0,
        "external_gradient_reductions": 0,
        "removed_noop_gradient_intents": 0,
        "stage_owned_collectives": 0,
    }


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
    ]
    reshard = next(action for action in plan.actions if action.action_type == "RESHARD")
    reshard_slot = plan.data_slots[reshard.consumes[0]]
    assert reshard.comm is None
    assert reshard_slot.kind == "control"
    assert plan.action_map[reshard_slot.producer_action_id].comp_type == "F"

    assert not any(
        action.action_type == "REDUCE_GRAD" for action in plan.actions
    )
    assert reduce_scatter.annotations["communication_owner"] == "L1_STAGE"
    ownership = plan.annotations["communication_ownership"]
    assert ownership["internal_gradient_reductions"] == 1
    assert ownership["external_gradient_reductions"] == 0


def test_standalone_gradient_reduction_remains_in_l2() -> None:
    reduce_scatter = OpNode(
        op_id=905,
        op_type="reduce_scatter",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.reduce_scatter"},
    )
    event = CommEvent(
        event_id="standalone-reduce",
        comm_primitive="reduce_scatter",
        group_name="fsdp",
        world_size=2,
        tensor_shape=(128,),
        dtype="float32",
        volume_bytes=512,
        op_id=905,
        comm_layer="L2",
        p2p_stage=0,
        p2p_mb_idx=0,
        seq_idx=80,
        comp_type="REDUCE_GRAD",
    )
    plan = build_schedule_plan(
        step_templates={
            "s0_B": StepGraph("s0_B", "B", {}),
            "s0_REDUCE_GRAD": StepGraph(
                "s0_REDUCE_GRAD",
                "REDUCE_GRAD",
                {905: reduce_scatter},
            ),
        },
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=[event],
        timeline_events=[_timeline(0, "B", 0, 60, 70)],
        pipeline_schedule="1F1B",
        rank=0,
    )

    reduction = next(
        action
        for action in plan.actions
        if action.action_type == "REDUCE_GRAD"
    )
    assert reduction.annotations["communication_owner"] == "L2_STANDALONE"
    ownership = plan.annotations["communication_ownership"]
    assert ownership["internal_gradient_reductions"] == 0
    assert ownership["external_gradient_reductions"] == 1


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


def test_state_unshards_keep_observed_order_after_initial_prefetch_intent() -> None:
    initial_prefetch = ActionSpec(
        action_type="UNSHARD",
        stage=0,
        mb_idx=0,
        seq_idx=10,
        order_key=(10, 0, 0),
        annotations={
            "capture_action_order": 10,
            "fsdp_schedule_source": "intent",
        },
    )
    later_microbatch = ActionSpec(
        action_type="UNSHARD",
        stage=0,
        mb_idx=1,
        seq_idx=40,
        order_key=(40, 0, 1),
        annotations={
            "capture_action_order": 40,
            "fsdp_schedule_source": "state",
        },
    )

    specs = CapturedTraceAssembler._bind_residency_intents(
        [initial_prefetch, later_microbatch],
        [
            {
                "action_type": "UNSHARD",
                "pp_stage": 0,
                "pp_mb_idx": -1,
                "action_order": 5,
                "seq_idx": 5,
            }
        ],
    )

    assert initial_prefetch.order_key == (5, 0, 0)
    assert initial_prefetch.annotations["capture_schedule_intent"]
    assert later_microbatch.order_key == (40, 0, 1)
    assert "capture_schedule_intent" not in later_microbatch.annotations
    assert specs == [initial_prefetch, later_microbatch]


def test_prefetch_stays_l2_while_later_unshard_uses_l1_variant() -> None:
    allgather = OpNode(
        op_id=915,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        comm_bytes=256,
        annotations={"raw_op_type": "comm.allgather"},
    )
    compute = OpNode(
        op_id=916,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=16,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    wait = _fsdp_marker(
        917,
        15,
        "unshard_wait",
        "block0",
        "layers.0",
    )
    release = _fsdp_marker(
        918,
        18,
        "reshard_release",
        "block0",
        "layers.0",
    )
    templates = {
        "s0_UNSHARD": StepGraph("s0_UNSHARD", "UNSHARD", {915: allgather}),
        "s0_F": StepGraph(
            "s0_F",
            "F",
            {916: compute, 917: wait, 918: release},
        ),
    }
    comm_events = [
        CommEvent(
            event_id="prefetch-allgather",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=915,
            comm_layer="L2",
            p2p_stage=0,
            comp_type="UNSHARD",
            fsdp_group_id="block0",
            fsdp_transition_id="prefetch-transition",
        ),
        CommEvent(
            event_id="inline-allgather",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=0,
            comm_layer="L2",
            p2p_stage=0,
            comp_type="F",
            fsdp_group_id="block0",
            fsdp_transition_id="inline-transition",
            fsdp_module_fqn="layers.0",
        ),
    ]
    residency = [
        FSDPResidencyEvent(
            group_id="block0",
            action="alloc",
            seq_idx=5,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=-1,
            comp_type="UNSHARD",
            shard_world_size=2,
            transition_id="prefetch-transition",
            action_order=0,
            schedule_source="intent",
        ),
        FSDPResidencyEvent(
            group_id="block0",
            action="alloc",
            seq_idx=11,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=2,
            transition_id="prefetch-transition",
            action_order=2,
            schedule_source="state",
            module_fqn="layers.0",
        ),
        FSDPResidencyEvent(
            group_id="block0",
            action="free",
            seq_idx=20,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=0,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb0",
            shard_world_size=2,
            transition_id="prefetch-transition",
            action_order=4,
            schedule_source="state",
            module_fqn="layers.0",
        ),
        FSDPResidencyEvent(
            group_id="block0",
            action="alloc",
            seq_idx=25,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=1,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb1",
            shard_world_size=2,
            transition_id="inline-transition",
            action_order=5,
            schedule_source="state",
        ),
        FSDPResidencyEvent(
            group_id="block0",
            action="free",
            seq_idx=35,
            phase="forward",
            num_bytes=256,
            pp_stage=0,
            pp_mb_idx=1,
            comp_type="F",
            parent_compute_instance_id="s0_F_mb1",
            shard_world_size=2,
            transition_id="inline-transition",
            action_order=7,
            schedule_source="state",
        ),
    ]
    first = _timeline(0, "F", 0, 10, 19)
    first["action_order"] = 3
    second = _timeline(0, "F", 1, 30, 34)
    second["action_order"] = 6

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[first, second],
        pipeline_schedule="custom",
        num_micro_batches=2,
        rank=0,
    )

    assert [action.action_type for action in plan.actions] == [
        "UNSHARD",
        "COMPUTE",
        "RESHARD",
        "COMPUTE",
    ]
    compute_actions = [
        action for action in plan.actions if action.action_type == "COMPUTE"
    ]
    assert compute_actions[0].template_ref == "s0_F"
    assert compute_actions[1].template_ref == "s0_F__comm_v1"
    assert all(
        node.annotations.get("communication_owner") != "L1_STAGE"
        for node in plan.step_templates["s0_F"].nodes.values()
    )
    variant_nodes = plan.step_templates["s0_F__comm_v1"].nodes.values()
    internal_allgathers = [
        node
        for node in variant_nodes
        if node.annotations.get("communication_owner") == "L1_STAGE"
    ]
    assert len(internal_allgathers) == 1
    assert internal_allgathers[0].annotations["fsdp_group_id"] == "block0"
    assert internal_allgathers[0].annotations["ownership_placement"] == "layer_jit"
    assert internal_allgathers[0].successors == [916]
    assert 915 not in plan.step_templates["s0_F__comm_v1"].nodes
    assert all(
        not node.annotations.get("fsdp_marker")
        for template in plan.step_templates.values()
        for node in template.nodes.values()
    )
    assert plan.annotations["communication_ownership"] == {
        "internal_fsdp_transitions": 1,
        "external_fsdp_prefetches": 1,
        "generated_l1_templates": ["s0_F__comm_v1"],
        "internal_gradient_reductions": 0,
        "external_gradient_reductions": 0,
        "removed_noop_gradient_intents": 0,
        "stage_owned_collectives": 1,
    }


@pytest.mark.parametrize(
    ("prefetch_source_fqn", "expected_placement"),
    [
        ("", "layer_jit"),
        ("layers.0", "layer_prefetch"),
    ],
)
def test_fsdp_allgather_is_anchored_to_its_parameter_group(
    prefetch_source_fqn: str,
    expected_placement: str,
) -> None:
    layer0 = OpNode(
        op_id=100,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[200],
        seq_idx=20,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    layer1 = OpNode(
        op_id=200,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[100],
        successors=[],
        seq_idx=40,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    templates = {
        "s0_F": StepGraph(
            "s0_F",
            "F",
            {
                10: _fsdp_marker(
                    10, 10, "unshard_wait", "group0", "layers.0"
                ),
                11: _fsdp_marker(
                    11, 30, "reshard_release", "group0", "layers.0"
                ),
                12: _fsdp_marker(
                    12, 35, "unshard_wait", "group1", "layers.1"
                ),
                13: _fsdp_marker(
                    13, 50, "reshard_release", "group1", "layers.1"
                ),
                100: layer0,
                200: layer1,
            },
        ),
        "s0_UNSHARD": StepGraph(
            "s0_UNSHARD",
            "UNSHARD",
            {
                300: OpNode(
                    op_id=300,
                    op_type="allgather",
                    inputs=[],
                    outputs=[],
                    attrs={},
                    predecessors=[],
                    successors=[],
                    comm_bytes=128,
                    annotations={"raw_op_type": "comm.allgather"},
                ),
                301: OpNode(
                    op_id=301,
                    op_type="allgather",
                    inputs=[],
                    outputs=[],
                    attrs={},
                    predecessors=[],
                    successors=[],
                    comm_bytes=128,
                    annotations={"raw_op_type": "comm.allgather"},
                ),
            },
        ),
    }
    compute = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=0,
        seq_idx=1,
        order_key=(1, 0, 0),
        comp_type="F",
        template_ref="s0_F",
        annotations={"compute_instance_id": "s0_F_mb0"},
    )
    unshards = [
        ActionSpec(
            action_type="UNSHARD",
            stage=0,
            mb_idx=0,
            seq_idx=2 + index,
            order_key=(2 + index, 0, index),
            annotations={
                "fsdp_schedule_source": "state",
                "fsdp_transition_id": f"transition{index}",
                "fsdp_group_id": f"group{index}",
                "fsdp_module_fqn": f"layers.{index}",
                "fsdp_prefetch_source_fqn": (
                    prefetch_source_fqn if index == 1 else ""
                ),
                "fsdp_prefetch_type": (
                    "BACKWARD" if index == 1 and prefetch_source_fqn else ""
                ),
                "parent_compute_instance_id": "s0_F_mb0",
                "residency_comp_type": "F",
                "shard_world_size": 2,
            },
        )
        for index in range(2)
    ]
    comm_events = []
    for index in range(2):
        comm_events.extend([
            CommEvent(
                event_id=f"canonical{index}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                op_id=300 + index,
                p2p_stage=0,
                comp_type="UNSHARD",
                fsdp_group_id=f"group{index}",
                fsdp_transition_id=f"canonical{index}",
                fsdp_module_fqn=f"layers.{index}",
            ),
            CommEvent(
                event_id=f"semantic{index}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                p2p_stage=0,
                comp_type="F",
                fsdp_group_id=f"group{index}",
                fsdp_transition_id=f"transition{index}",
                fsdp_module_fqn=f"layers.{index}",
                fsdp_prefetch_source_fqn=(
                    prefetch_source_fqn if index == 1 else ""
                ),
            ),
        ])

    result = normalize_communication_ownership(
        step_templates=templates,
        specs=[compute, *unshards],
        comm_events=comm_events,
    )

    assert result.specs == [compute]
    graph = templates["s0_F"]
    allgathers = {
        node.annotations["fsdp_group_id"]: node
        for node in graph.nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    }
    group0 = allgathers["group0"]
    group1 = allgathers["group1"]
    assert 100 in group0.successors
    assert group1.successors == [200]
    assert group0.op_id in graph.nodes[100].predecessors
    assert group1.op_id in graph.nodes[200].predecessors
    assert group1.annotations["ownership_placement"] == expected_placement
    if prefetch_source_fqn:
        launch_nodes = [
            node
            for node in graph.nodes.values()
            if node.op_type == "FSDP_PREFETCH_LAUNCH"
        ]
        assert len(launch_nodes) == 1
        launch = launch_nodes[0]
        assert launch.predecessors == [group0.op_id]
        assert group1.predecessors == [launch.op_id]
        assert launch.op_id in graph.nodes[100].predecessors
        assert (
            launch.flops,
            launch.peak_mem,
            launch.param_mem,
            launch.comm_bytes,
        ) == (0, 0, 0, 0)
        assert launch.annotations["zero_cost"] is True
    else:
        assert group1.predecessors == [100]
        assert all(
            node.op_type != "FSDP_PREFETCH_LAUNCH"
            for node in graph.nodes.values()
        )
    assert all(
        not node.annotations.get("fsdp_marker")
        for node in graph.nodes.values()
    )


@pytest.mark.parametrize(
    ("comp_type", "layer_order", "prefetch_type"),
    [
        ("F", (0, 1, 2), "FORWARD"),
        ("B", (2, 1, 0), "BACKWARD"),
    ],
)
def test_fsdp_prefetch_launch_blocks_recursive_layer_runahead(
    comp_type: str,
    layer_order: tuple[int, ...],
    prefetch_type: str,
) -> None:
    template_id = f"s0_{comp_type}"
    compute_ids = [100 + position for position in range(3)]
    allgather_ids = [500 + position for position in range(3)]
    nodes: dict[int, OpNode] = {}
    for position, layer in enumerate(layer_order):
        compute_id = compute_ids[position]
        predecessor = compute_ids[position - 1] if position else None
        nodes[10 + 2 * position] = _fsdp_marker(
            10 + 2 * position,
            10 + 20 * position,
            "unshard_wait",
            f"group{layer}",
            f"layers.{layer}",
        )
        nodes[11 + 2 * position] = _fsdp_marker(
            11 + 2 * position,
            30 + 20 * position,
            "reshard_release",
            f"group{layer}",
            f"layers.{layer}",
        )
        nodes[compute_id] = OpNode(
            op_id=compute_id,
            op_type="matmul",
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=[] if predecessor is None else [predecessor],
            successors=(
                []
                if position == len(layer_order) - 1
                else [compute_ids[position + 1]]
            ),
            seq_idx=20 + 20 * position,
            annotations={"raw_op_type": "aten.mm.default"},
        )

    templates = {
        template_id: StepGraph(template_id, comp_type, nodes),
        "s0_UNSHARD": StepGraph(
            "s0_UNSHARD",
            "UNSHARD",
            {
                op_id: OpNode(
                    op_id=op_id,
                    op_type="allgather",
                    inputs=[],
                    outputs=[],
                    attrs={},
                    predecessors=[],
                    successors=[],
                    comm_bytes=128,
                    annotations={"raw_op_type": "comm.allgather"},
                )
                for op_id in allgather_ids
            },
        ),
    }
    compute = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=0,
        seq_idx=1,
        order_key=(1, 0, 0),
        comp_type=comp_type,
        template_ref=template_id,
        annotations={"compute_instance_id": f"{template_id}_mb0"},
    )
    unshards: list[ActionSpec] = []
    comm_events: list[CommEvent] = []
    for position, layer in enumerate(layer_order):
        source_fqn = (
            "" if position == 0 else f"layers.{layer_order[position - 1]}"
        )
        transition_id = f"transition{position}"
        unshards.append(
            ActionSpec(
                action_type="UNSHARD",
                stage=0,
                mb_idx=0,
                seq_idx=2 + position,
                order_key=(2 + position, 0, position),
                annotations={
                    "fsdp_schedule_source": "state",
                    "fsdp_transition_id": transition_id,
                    "fsdp_group_id": f"group{layer}",
                    "fsdp_module_fqn": f"layers.{layer}",
                    "fsdp_prefetch_source_fqn": source_fqn,
                    "fsdp_prefetch_type": prefetch_type if source_fqn else "",
                    "parent_compute_instance_id": f"{template_id}_mb0",
                    "residency_comp_type": comp_type,
                    "shard_world_size": 2,
                },
            )
        )
        comm_events.extend([
            CommEvent(
                event_id=f"canonical{position}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                op_id=allgather_ids[position],
                p2p_stage=0,
                comp_type="UNSHARD",
                fsdp_group_id=f"group{layer}",
                fsdp_transition_id=f"canonical{position}",
                fsdp_module_fqn=f"layers.{layer}",
            ),
            CommEvent(
                event_id=f"semantic{position}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                p2p_stage=0,
                comp_type=comp_type,
                fsdp_group_id=f"group{layer}",
                fsdp_transition_id=transition_id,
                fsdp_module_fqn=f"layers.{layer}",
                fsdp_prefetch_source_fqn=source_fqn,
                fsdp_prefetch_type=prefetch_type if source_fqn else "",
            ),
        ])

    normalize_communication_ownership(
        step_templates=templates,
        specs=[compute, *unshards],
        comm_events=comm_events,
    )

    workload = build_workload_graph(
        schedule_graph=ScheduleGraph(
            schedule_id="fsdp-prefetch-launch",
            workload_type="train",
            step_templates=templates,
            instances=[],
        ),
        step_templates=templates,
        local_batch_size=1,
        seq_len=128,
    )
    graph = workload.step_templates[template_id]
    allgathers = {
        node.annotations["fsdp_group_id"]: node
        for node in graph.nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    }
    launches = {
        node.annotations["fsdp_group_id"]: node
        for node in graph.nodes.values()
        if node.op_type == "FSDP_PREFETCH_LAUNCH"
    }
    assert len(launches) == 2
    assert graph.is_acyclic
    for position in (1, 2):
        source_layer = layer_order[position - 1]
        target_layer = layer_order[position]
        launch = launches[f"group{target_layer}"]
        source_allgather = allgathers[f"group{source_layer}"]
        target_allgather = allgathers[f"group{target_layer}"]
        assert source_allgather.op_id in launch.predecessors
        assert launch.op_id in graph.nodes[compute_ids[position - 1]].predecessors
        assert target_allgather.predecessors == [launch.op_id]
        assert target_allgather.op_id in graph.nodes[compute_ids[position]].predecessors
        assert (
            launch.flops,
            launch.peak_mem,
            launch.param_mem,
            launch.comm_bytes,
        ) == (0, 0, 0, 0)
    assert compute_ids[0] in launches[f"group{layer_order[2]}"].predecessors


def test_fsdp_prefetch_skips_a_source_gate_that_would_form_a_cycle() -> None:
    source_entry = OpNode(
        op_id=100,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[300],
        seq_idx=20,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    source_allgather = OpNode(
        op_id=300,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[100],
        successors=[150],
        seq_idx=25,
        comm_bytes=128,
        annotations={"raw_op_type": "comm.allgather"},
    )
    source_exit = OpNode(
        op_id=150,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[300],
        successors=[],
        seq_idx=27,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    target_entry = OpNode(
        op_id=200,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=50,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    target_allgather = OpNode(
        op_id=301,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=45,
        comm_bytes=128,
        annotations={"raw_op_type": "comm.allgather"},
    )
    templates = {
        "s0_B": StepGraph(
            "s0_B",
            "B",
            {
                10: _fsdp_marker(
                    10, 10, "unshard_wait", "group0", "layers.0"
                ),
                11: _fsdp_marker(
                    11, 30, "reshard_release", "group0", "layers.0"
                ),
                12: _fsdp_marker(
                    12, 40, "unshard_wait", "group1", "layers.1"
                ),
                13: _fsdp_marker(
                    13, 60, "reshard_release", "group1", "layers.1"
                ),
                100: source_entry,
                150: source_exit,
                200: target_entry,
                300: source_allgather,
                301: target_allgather,
            },
        )
    }
    compute = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=0,
        seq_idx=1,
        order_key=(1, 0, 0),
        comp_type="B",
        template_ref="s0_B",
        annotations={"compute_instance_id": "s0_B_mb0"},
    )
    unshards = [
        ActionSpec(
            action_type="UNSHARD",
            stage=0,
            mb_idx=0,
            seq_idx=2 + index,
            order_key=(2 + index, 0, index),
            annotations={
                "fsdp_schedule_source": "state",
                "fsdp_transition_id": f"transition{index}",
                "fsdp_group_id": f"group{index}",
                "fsdp_module_fqn": f"layers.{index}",
                "fsdp_prefetch_source_fqn": "layers.0" if index == 1 else "",
                "fsdp_prefetch_type": "BACKWARD" if index == 1 else "",
                "parent_compute_instance_id": "s0_B_mb0",
                "residency_comp_type": "B",
                "shard_world_size": 2,
            },
        )
        for index in range(2)
    ]
    comm_events = [
        CommEvent(
            event_id=f"allgather{index}",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(64,),
            dtype="bfloat16",
            volume_bytes=128,
            op_id=300 + index,
            p2p_stage=0,
            comp_type="B",
            fsdp_group_id=f"group{index}",
            fsdp_transition_id=f"transition{index}",
            fsdp_module_fqn=f"layers.{index}",
            fsdp_prefetch_source_fqn="layers.0" if index == 1 else "",
            fsdp_prefetch_type="BACKWARD" if index == 1 else "",
        )
        for index in range(2)
    ]

    normalize_communication_ownership(
        step_templates=templates,
        specs=[compute, *unshards],
        comm_events=comm_events,
    )

    graph = templates["s0_B"]
    launch = next(
        node
        for node in graph.nodes.values()
        if node.op_type == "FSDP_PREFETCH_LAUNCH"
    )
    allgathers = {
        node.annotations["fsdp_group_id"]: node
        for node in graph.nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    }
    assert graph.is_acyclic
    assert launch.predecessors == [allgathers["group0"].op_id]
    assert allgathers["group1"].predecessors == [launch.op_id]
    assert launch.op_id not in graph.nodes[100].predecessors
    assert launch.annotations["fsdp_prefetch_source_gate_skipped_entries"] == [100]


def test_non_pipeline_fsdp_prefetch_launch_survives_workload_build() -> None:
    layer0 = OpNode(
        op_id=100,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[200],
        seq_idx=20,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    layer1 = OpNode(
        op_id=200,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[100],
        successors=[],
        seq_idx=60,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    allgather0 = OpNode(
        op_id=300,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=5,
        comm_bytes=128,
        annotations={"raw_op_type": "comm.allgather"},
    )
    allgather1 = OpNode(
        op_id=301,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=45,
        comm_bytes=128,
        annotations={"raw_op_type": "comm.allgather"},
    )
    templates = {
        "s0_F": StepGraph(
            "s0_F",
            "F",
            {
                10: _fsdp_marker(
                    10, 10, "unshard_wait", "group0", "layers.0"
                ),
                11: _fsdp_marker(
                    11, 30, "reshard_release", "group0", "layers.0"
                ),
                12: _fsdp_marker(
                    12, 50, "unshard_wait", "group1", "layers.1"
                ),
                13: _fsdp_marker(
                    13, 70, "reshard_release", "group1", "layers.1"
                ),
                100: layer0,
                200: layer1,
                300: allgather0,
                301: allgather1,
            },
        ),
        "s0_B": StepGraph("s0_B", "B", {}),
    }
    residency = [
        FSDPResidencyEvent(
            group_id=f"group{index}",
            action="alloc",
            seq_idx=6 + 40 * index,
            phase="forward",
            num_bytes=128,
            transition_id=f"transition{index}",
            module_fqn=f"layers.{index}",
            prefetch_source_fqn="layers.0" if index == 1 else "",
            prefetch_type="FORWARD" if index == 1 else "",
        )
        for index in range(2)
    ]
    comm_events = [
        CommEvent(
            event_id=f"allgather{index}",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(64,),
            dtype="bfloat16",
            volume_bytes=128,
            op_id=300 + index,
            p2p_stage=0,
            p2p_mb_idx=0,
            comp_type="F",
            fsdp_group_id=f"group{index}",
            fsdp_transition_id=f"transition{index}",
            fsdp_module_fqn=f"layers.{index}",
            fsdp_prefetch_source_fqn="layers.0" if index == 1 else "",
            fsdp_prefetch_type="FORWARD" if index == 1 else "",
        )
        for index in range(2)
    ]

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=[],
        pipeline_schedule="none",
        rank=0,
    )
    workload = build_workload_graph(
        schedule_graph=ScheduleGraph(
            schedule_id="non-pipeline-fsdp",
            workload_type="train",
            step_templates=templates,
            instances=[],
        ),
        step_templates=templates,
        local_batch_size=1,
        seq_len=128,
        schedule_plan=plan,
    )

    forward_ref = next(
        action.template_ref
        for action in plan.actions
        if action.comp_type == "F"
    )
    launches = [
        node
        for node in workload.step_templates[forward_ref].nodes.values()
        if node.op_type == "FSDP_PREFETCH_LAUNCH"
    ]
    assert len(launches) == 1
    assert launches[0].annotations["zero_cost"] is True


def test_fsdp_backward_sync_waits_prior_rs_but_not_hsdp_allreduce() -> None:
    layer_order = (2, 1, 0)
    nodes: dict[int, OpNode] = {}
    allgather_ids = [500, 501, 502]
    reduce_scatter_ids = [600, 601, 602]
    allreduce_ids = [700, 701, 702]
    for position, layer in enumerate(layer_order):
        compute_id = 100 + position
        predecessor = 100 + position - 1 if position else None
        nodes[10 + 2 * position] = _fsdp_marker(
            10 + 2 * position,
            10 + 40 * position,
            "unshard_wait",
            f"group{layer}",
            f"layers.{layer}",
        )
        nodes[11 + 2 * position] = _fsdp_marker(
            11 + 2 * position,
            30 + 40 * position,
            "reshard_release",
            f"group{layer}",
            f"layers.{layer}",
        )
        nodes[compute_id] = OpNode(
            op_id=compute_id,
            op_type="matmul",
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=[] if predecessor is None else [predecessor],
            successors=[],
            seq_idx=20 + 40 * position,
            annotations={"raw_op_type": "aten.mm.default"},
        )
        nodes[reduce_scatter_ids[position]] = OpNode(
            op_id=reduce_scatter_ids[position],
            op_type="reduce_scatter",
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=[compute_id],
            successors=[allreduce_ids[position]],
            seq_idx=35 + 40 * position,
            annotations={"raw_op_type": "comm.reduce_scatter"},
        )
        nodes[allreduce_ids[position]] = OpNode(
            op_id=allreduce_ids[position],
            op_type="allreduce",
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=[reduce_scatter_ids[position]],
            successors=[],
            seq_idx=36 + 40 * position,
            annotations={"raw_op_type": "comm.allreduce"},
        )

    templates = {
        "s0_B": StepGraph("s0_B", "B", nodes),
        "s0_UNSHARD": StepGraph(
            "s0_UNSHARD",
            "UNSHARD",
            {
                op_id: OpNode(
                    op_id=op_id,
                    op_type="allgather",
                    inputs=[],
                    outputs=[],
                    attrs={},
                    predecessors=[],
                    successors=[],
                    comm_bytes=128,
                    annotations={"raw_op_type": "comm.allgather"},
                )
                for op_id in allgather_ids
            },
        ),
    }
    compute = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=0,
        seq_idx=1,
        order_key=(1, 0, 0),
        comp_type="B",
        template_ref="s0_B",
        annotations={"compute_instance_id": "s0_B_mb0"},
    )
    unshards: list[ActionSpec] = []
    comm_events: list[CommEvent] = []
    for position, layer in enumerate(layer_order):
        source_fqn = (
            "" if position == 0 else f"layers.{layer_order[position - 1]}"
        )
        transition_id = f"transition{position}"
        unshards.append(
            ActionSpec(
                action_type="UNSHARD",
                stage=0,
                mb_idx=0,
                seq_idx=2 + position,
                order_key=(2 + position, 0, position),
                annotations={
                    "fsdp_schedule_source": "state",
                    "fsdp_transition_id": transition_id,
                    "fsdp_group_id": f"group{layer}",
                    "fsdp_module_fqn": f"layers.{layer}",
                    "fsdp_prefetch_source_fqn": source_fqn,
                    "fsdp_prefetch_type": "BACKWARD" if source_fqn else "",
                    "parent_compute_instance_id": "s0_B_mb0",
                    "residency_comp_type": "B",
                    "shard_world_size": 2,
                },
            )
        )
        comm_events.extend([
            CommEvent(
                event_id=f"canonical{position}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                op_id=allgather_ids[position],
                p2p_stage=0,
                comp_type="UNSHARD",
                fsdp_group_id=f"group{layer}",
                fsdp_transition_id=f"canonical{position}",
                fsdp_module_fqn=f"layers.{layer}",
            ),
            CommEvent(
                event_id=f"semantic{position}",
                comm_primitive="allgather",
                group_name="fsdp",
                world_size=2,
                tensor_shape=(64,),
                dtype="bfloat16",
                volume_bytes=128,
                p2p_stage=0,
                comp_type="B",
                fsdp_group_id=f"group{layer}",
                fsdp_transition_id=transition_id,
                fsdp_module_fqn=f"layers.{layer}",
                fsdp_prefetch_source_fqn=source_fqn,
                fsdp_prefetch_type="BACKWARD" if source_fqn else "",
            ),
        ])

    normalize_communication_ownership(
        step_templates=templates,
        specs=[compute, *unshards],
        comm_events=comm_events,
    )

    graph = templates["s0_B"]
    syncs = sorted(
        (
            node
            for node in graph.nodes.values()
            if node.op_type == "FSDP_POST_BACKWARD_SYNC"
        ),
        key=lambda node: node.seq_idx,
    )
    assert len(syncs) == 2
    assert graph.is_acyclic
    assert reduce_scatter_ids[0] in syncs[0].predecessors
    assert allreduce_ids[0] not in syncs[0].predecessors
    assert syncs[0].op_id in graph.nodes[reduce_scatter_ids[1]].predecessors
    assert syncs[0].op_id in graph.nodes[102].predecessors
    assert reduce_scatter_ids[1] in syncs[1].predecessors
    assert allreduce_ids[1] not in syncs[1].predecessors
    assert syncs[1].op_id in graph.nodes[reduce_scatter_ids[2]].predecessors
    assert all(sync.annotations["zero_cost"] is True for sync in syncs)


def test_fsdp_backward_sync_handles_overlapping_param_group_regions() -> None:
    from torchtitan_npu.simulator.capture.communication_ownership import (
        _FSDPGroupRegion,
        _add_fsdp_backward_reduction_syncs,
        _refresh_graph_topology,
    )

    def node(
        op_id: int,
        seq_idx: int,
        *,
        predecessors: list[int] | None = None,
        raw_op_type: str = "aten.mm.default",
    ) -> OpNode:
        return OpNode(
            op_id=op_id,
            op_type=raw_op_type.removeprefix("comm."),
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=list(predecessors or []),
            successors=[],
            seq_idx=seq_idx,
            annotations={"raw_op_type": raw_op_type},
        )

    graph = StepGraph(
        "s0_B",
        "B",
        {
            100: node(100, 10),
            600: node(
                600,
                35,
                predecessors=[100],
                raw_op_type="comm.reduce_scatter",
            ),
            200: node(200, 60, predecessors=[100]),
            201: node(201, 80, predecessors=[200]),
            601: node(
                601,
                95,
                predecessors=[201],
                raw_op_type="comm.reduce_scatter",
            ),
            300: node(300, 120, predecessors=[201]),
            602: node(
                602,
                135,
                predecessors=[300],
                raw_op_type="comm.reduce_scatter",
            ),
        },
    )
    regions = [
        _FSDPGroupRegion(
            "prior",
            "layers.2",
            1,
            30,
            (100,),
            (100,),
            (),
            0,
            1,
        ),
        _FSDPGroupRegion(
            "outer",
            "layers.1",
            40,
            90,
            (200,),
            (201,),
            (100,),
            0,
            2,
        ),
        _FSDPGroupRegion(
            "inner",
            "layers.1",
            50,
            70,
            (200,),
            (200,),
            (100,),
            1,
            2,
        ),
        _FSDPGroupRegion(
            "next",
            "layers.0",
            110,
            130,
            (300,),
            (300,),
            (201,),
            0,
            1,
        ),
    ]

    _, sync_count = _add_fsdp_backward_reduction_syncs(graph, regions, -1)
    _refresh_graph_topology(graph)

    syncs = sorted(
        (
            item
            for item in graph.nodes.values()
            if item.op_type == "FSDP_POST_BACKWARD_SYNC"
        ),
        key=lambda item: item.seq_idx,
    )
    assert sync_count == 2
    assert graph.is_acyclic
    assert syncs[0].seq_idx == 70
    assert set(syncs[0].predecessors) == {200, 600}
    assert syncs[0].op_id in graph.nodes[201].predecessors
    assert syncs[0].op_id not in graph.nodes[200].predecessors
    assert set(syncs[1].predecessors) == {300, 601}
    assert syncs[1].op_id in graph.nodes[602].predecessors


def test_cross_action_fsdp_prefetch_belongs_to_launch_compute() -> None:
    prefetch = OpNode(
        op_id=300,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        comm_bytes=128,
        seq_idx=20,
        annotations={"raw_op_type": "comm.allgather"},
    )
    target_compute = OpNode(
        op_id=200,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=50,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    source_compute = OpNode(
        op_id=100,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        seq_idx=20,
        annotations={"raw_op_type": "aten.mm.default"},
    )
    templates = {
        "s0_B": StepGraph(
            "s0_B",
            "B",
            {
                1: _fsdp_marker(
                    1, 10, "unshard_wait", "group0", "layers.0"
                ),
                2: _fsdp_marker(
                    2, 30, "reshard_release", "group0", "layers.0"
                ),
                100: source_compute,
                300: prefetch,
            },
        ),
        "s0_F": StepGraph(
            "s0_F",
            "F",
            {
                10: _fsdp_marker(
                    10, 40, "unshard_wait", "group1", "layers.1"
                ),
                11: _fsdp_marker(
                    11, 60, "reshard_release", "group1", "layers.1"
                ),
                200: target_compute,
            },
        ),
    }
    backward = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=0,
        seq_idx=10,
        order_key=(10, 0, 0),
        comp_type="B",
        template_ref="s0_B",
        annotations={
            "compute_instance_id": "s0_B_mb0",
            "capture_start_seq": 10,
            "capture_end_seq": 30,
        },
    )
    forward = ActionSpec(
        action_type="COMPUTE",
        stage=0,
        mb_idx=1,
        seq_idx=40,
        order_key=(40, 0, 0),
        comp_type="F",
        template_ref="s0_F",
        annotations={"compute_instance_id": "s0_F_mb1"},
    )
    unshard = ActionSpec(
        action_type="UNSHARD",
        stage=0,
        mb_idx=1,
        seq_idx=50,
        order_key=(40, -2, 0),
        annotations={
            "fsdp_schedule_source": "state",
            "fsdp_transition_id": "transition1",
            "fsdp_group_id": "group1",
            "fsdp_module_fqn": "layers.1",
            "fsdp_prefetch_source_fqn": "layers.0",
            "fsdp_prefetch_type": "BACKWARD",
            "parent_compute_instance_id": "s0_F_mb1",
            "residency_comp_type": "F",
            "shard_world_size": 2,
        },
    )
    event = CommEvent(
        event_id="cross-action-prefetch",
        comm_primitive="allgather",
        group_name="fsdp",
        world_size=2,
        tensor_shape=(64,),
        dtype="bfloat16",
        volume_bytes=128,
        op_id=300,
        p2p_stage=0,
        p2p_mb_idx=0,
        seq_idx=20,
        comp_type="B",
        fsdp_group_id="group1",
        fsdp_transition_id="transition1",
        fsdp_module_fqn="layers.1",
        fsdp_prefetch_source_fqn="layers.0",
        fsdp_prefetch_type="BACKWARD",
    )

    result = normalize_communication_ownership(
        step_templates=templates,
        specs=[backward, forward, unshard],
        comm_events=[event],
    )

    assert result.specs == [backward, forward]
    backward_allgathers = [
        node
        for node in templates["s0_B"].nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    ]
    assert len(backward_allgathers) == 1
    assert (
        backward_allgathers[0].annotations["ownership_placement"]
        == "cross_action_prefetch"
    )
    assert (
        backward_allgathers[0].annotations[
            "fsdp_target_compute_instance_id"
        ]
        == "s0_F_mb1"
    )
    launch_nodes = [
        node
        for node in templates["s0_B"].nodes.values()
        if node.op_type == "FSDP_PREFETCH_LAUNCH"
    ]
    assert len(launch_nodes) == 1
    launch = launch_nodes[0]
    assert launch.op_id in templates["s0_B"].nodes[100].predecessors
    assert backward_allgathers[0].predecessors == [launch.op_id]
    assert all(
        node.annotations.get("raw_op_type") != "comm.allgather"
        for node in templates["s0_F"].nodes.values()
    )
    assert all(
        not node.annotations.get("fsdp_marker")
        for node in templates["s0_F"].nodes.values()
    )


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
            successors=[932],
            annotations={"raw_op_type": "comm.allgather"},
        )
        for op_id in (930, 931)
    }
    nodes[932] = OpNode(
        op_id=932,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[930, 931],
        successors=[],
        annotations={"raw_op_type": "aten.mm.default"},
    )
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

    assert all(
        action.action_type not in {"UNSHARD", "RESHARD"}
        for action in plan.actions
    )
    fsdp_nodes = [
        node
        for node in plan.step_templates["s0_F"].nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    ]
    assert {node.annotations["fsdp_group_id"] for node in fsdp_nodes} == {
        "group-a",
        "group-b",
    }
    assert all(
        not set(node.predecessors).intersection({930, 931})
        for node in fsdp_nodes
    )
    assert plan.step_templates["s0_F"].nodes[932].predecessors == [930, 931]
    assert set(plan.step_templates["s0_F"].entry_nodes) == {930, 931}
    assert plan.step_templates["s0_F"].exit_nodes == [932]
    assert plan.step_templates["s0_F"].is_acyclic


def test_repeated_compute_local_fsdp_transition_reuses_l1_template() -> None:
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
        for op_id in (940, 941)
    }
    templates = {"s0_F": StepGraph("s0_F", "F", nodes)}
    comm_events = [
        CommEvent(
            event_id=f"allgather-{mb_idx}",
            comm_primitive="allgather",
            group_name="fsdp",
            world_size=2,
            tensor_shape=(128,),
            dtype="bfloat16",
            volume_bytes=256,
            op_id=940 + mb_idx,
            comm_layer="L2",
            p2p_stage=0,
            p2p_mb_idx=mb_idx,
            comp_type="F",
            seq_idx=10 + mb_idx,
            fsdp_group_id="shared-group",
            fsdp_transition_id=f"transition-{mb_idx}",
        )
        for mb_idx in range(2)
    ]
    residency = []
    for mb_idx, (alloc_order, free_order) in enumerate(((0, 3), (1, 5))):
        residency.extend(
            [
                FSDPResidencyEvent(
                    group_id="shared-group",
                    action="alloc",
                    seq_idx=10 + mb_idx,
                    phase="forward",
                    num_bytes=256,
                    pp_stage=0,
                    pp_mb_idx=mb_idx,
                    comp_type="F",
                    parent_compute_instance_id=f"s0_F_mb{mb_idx}",
                    shard_world_size=2,
                    transition_id=f"transition-{mb_idx}",
                    action_order=alloc_order,
                ),
                FSDPResidencyEvent(
                    group_id="shared-group",
                    action="free",
                    seq_idx=20 + mb_idx,
                    phase="forward",
                    num_bytes=256,
                    pp_stage=0,
                    pp_mb_idx=mb_idx,
                    comp_type="F",
                    parent_compute_instance_id=f"s0_F_mb{mb_idx}",
                    shard_world_size=2,
                    transition_id=f"transition-{mb_idx}",
                    action_order=free_order,
                ),
            ]
        )
    timelines = [
        _timeline(0, "F", 0, 30, 31),
        _timeline(0, "F", 1, 40, 41),
    ]
    timelines[0]["action_order"] = 2
    timelines[1]["action_order"] = 4

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=comm_events,
        fsdp_residency_events=residency,
        timeline_events=timelines,
        pipeline_schedule="custom",
        rank=0,
    )

    assert all(
        action.action_type not in {"UNSHARD", "RESHARD"}
        for action in plan.actions
    )
    compute = [
        action for action in plan.actions if action.action_type == "COMPUTE"
    ]
    assert [action.template_ref for action in compute] == ["s0_F", "s0_F"]
    fsdp_nodes = [
        node
        for node in plan.step_templates["s0_F"].nodes.values()
        if node.annotations.get("communication_owner") == "L1_STAGE"
    ]
    assert len(fsdp_nodes) == 1
    assert fsdp_nodes[0].annotations["fsdp_group_id"] == "shared-group"


def test_free_only_fsdp_transition_does_not_create_l2_residency_action() -> None:
    timeline = _timeline(0, "F", 0, 10, 20)
    timeline["action_order"] = 0
    plan = build_schedule_plan(
        step_templates={"s0_F": StepGraph("s0_F", "F", {})},
        rank_table=_RankTable(pp=1, dp_shard=2),
        comm_events=[],
        fsdp_residency_events=[
            FSDPResidencyEvent(
                group_id="cleanup-group",
                action="free",
                seq_idx=30,
                phase="backward",
                num_bytes=256,
                pp_stage=0,
                pp_mb_idx=0,
                comp_type="W",
                parent_compute_instance_id="",
                shard_world_size=2,
                transition_id="transition-outside-step",
                action_order=1,
                schedule_source="state",
            )
        ],
        timeline_events=[timeline],
        pipeline_schedule="custom",
        rank=0,
    )

    assert all(
        action.action_type not in {"UNSHARD", "RESHARD"}
        for action in plan.actions
    )
