# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_plan


class _RankTable:
    dim_degrees = {"pp": 2}

    @staticmethod
    def to_dict() -> dict:
        return {}


def _runtime_action(comp_type: str, stage: int, microbatch: int) -> SimpleNamespace:
    return SimpleNamespace(
        computation_type=SimpleNamespace(value=comp_type),
        stage_index=stage,
        microbatch_index=microbatch,
    )


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
