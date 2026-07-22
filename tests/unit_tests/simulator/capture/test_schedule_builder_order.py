# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_plan
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph


class _RankTable:
    def __init__(self, *, dp_shard: int = 1) -> None:
        self.dim_degrees = {"pp": 2, "dp_shard": dp_shard}

    @staticmethod
    def to_dict() -> dict:
        return {}


def _runtime_action(
    comp_type: str,
    stage: int,
    microbatch: int | None,
    *,
    sub_actions: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        computation_type=SimpleNamespace(value=comp_type),
        stage_index=stage,
        microbatch_index=microbatch,
        sub_actions=sub_actions,
    )


def _real_unshard_capture() -> tuple[dict[str, StepGraph], list[CommEvent]]:
    op = OpNode(
        op_id=900,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "comm.allgather"},
    )
    event = CommEvent(
        event_id="unshard",
        comm_primitive="allgather",
        group_name="fsdp",
        world_size=2,
        tensor_shape=(128,),
        dtype="bfloat16",
        volume_bytes=256,
        op_id=900,
        comm_layer="L2",
        p2p_stage=0,
    )
    return {"s0_UNSHARD": StepGraph("s0_UNSHARD", "UNSHARD", {900: op})}, [event]


def test_runtime_plan_order_is_independent_from_captured_l0_seq_idx() -> None:
    runtime_actions = [
        _runtime_action("F", 0, 0),
        _runtime_action("SEND_F", 0, 0),
        _runtime_action("F", 0, 1),
    ]
    schedule = SimpleNamespace(
        pipeline_order_with_comms={0: runtime_actions},
        stage_index_to_group_rank={0: 0},
    )
    timeline = [
        {"pp_stage": 0, "pp_mb_idx": 0, "comp_type": "F", "seq_idx": 100},
        {"pp_stage": 0, "pp_mb_idx": 1, "comp_type": "F", "seq_idx": 200},
    ]

    plan = build_schedule_plan(
        step_templates={},
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=timeline,
        pp_schedule_obj=schedule,
        pipeline_schedule="test",
        num_micro_batches=2,
        rank=0,
    )

    assert [action.action_type for action in plan.actions] == [
        "COMPUTE",
        "SEND_F",
        "COMPUTE",
    ]
    assert [action.schedule_order for action in plan.actions] == [0, 1, 2]
    assert [action.seq_idx for action in plan.actions] == [100, 1, 200]


def test_noop_fsdp_actions_do_not_create_blocking_slots() -> None:
    runtime_actions = [
        _runtime_action("UNSHARD", 0, None),
        _runtime_action("F", 0, 0),
        _runtime_action("RESHARD", 0, None),
    ]
    schedule = SimpleNamespace(
        pipeline_order_with_comms={0: runtime_actions},
        stage_index_to_group_rank={0: 0},
    )

    plan = build_schedule_plan(
        step_templates={},
        rank_table=_RankTable(),
        comm_events=[],
        timeline_events=[
            {"pp_stage": 0, "pp_mb_idx": 0, "comp_type": "F", "seq_idx": 10},
        ],
        pp_schedule_obj=schedule,
        rank=0,
    )

    fsdp_actions = [action for action in plan.actions if action.action_type in {"UNSHARD", "RESHARD"}]
    assert all(action.is_noop for action in fsdp_actions)
    assert all(action.comm is not None and action.comm.is_noop for action in fsdp_actions)
    assert all(not action.consumes and not action.produces for action in fsdp_actions)
    assert plan.data_slots == {}


def test_real_reshard_uses_overlap_sub_action_as_producer() -> None:
    overlap = _runtime_action(
        "OVERLAP_F_B",
        -1,
        None,
        sub_actions=[_runtime_action("B", 0, 0)],
    )
    schedule = SimpleNamespace(
        pipeline_order_with_comms={
            0: [_runtime_action("UNSHARD", 0, None), overlap, _runtime_action("RESHARD", 0, None)]
        },
        stage_index_to_group_rank={0: 0},
    )
    templates, comm_events = _real_unshard_capture()

    plan = build_schedule_plan(
        step_templates=templates,
        rank_table=_RankTable(dp_shard=2),
        comm_events=comm_events,
        timeline_events=[
            {"pp_stage": 0, "pp_mb_idx": 0, "comp_type": "B", "seq_idx": 10},
        ],
        pp_schedule_obj=schedule,
        rank=0,
    )

    overlap_action = plan.actions[1]
    backward = overlap_action.sub_actions[0]
    reshard = next(action for action in plan.actions if action.action_type == "RESHARD")
    slot = plan.data_slots[reshard.consumes[0]]
    assert slot.kind == "control"
    assert slot.volume_bytes == 0
    assert slot.producer_action_id == backward.action_id
    assert slot.slot_id in backward.produces


def test_real_reshard_without_compute_producer_fails_fast() -> None:
    schedule = SimpleNamespace(
        pipeline_order_with_comms={
            0: [
                _runtime_action("UNSHARD", 0, None),
                _runtime_action("RESHARD", 0, None),
                _runtime_action("F", 0, 0),
            ]
        },
        stage_index_to_group_rank={0: 0},
    )
    templates, comm_events = _real_unshard_capture()

    with pytest.raises(RuntimeError, match="no preceding compute producer"):
        build_schedule_plan(
            step_templates=templates,
            rank_table=_RankTable(dp_shard=2),
            comm_events=comm_events,
            timeline_events=[
                {"pp_stage": 0, "pp_mb_idx": 0, "comp_type": "F", "seq_idx": 10},
            ],
            pp_schedule_obj=schedule,
            rank=0,
        )


def test_missing_fsdp_comm_event_on_sharded_mesh_fails_fast() -> None:
    schedule = SimpleNamespace(
        pipeline_order_with_comms={0: [_runtime_action("UNSHARD", 0, None)]},
        stage_index_to_group_rank={0: 0},
    )

    with pytest.raises(RuntimeError, match="requires FSDP communication"):
        build_schedule_plan(
            step_templates={},
            rank_table=_RankTable(dp_shard=2),
            comm_events=[],
            pp_schedule_obj=schedule,
            rank=0,
        )
