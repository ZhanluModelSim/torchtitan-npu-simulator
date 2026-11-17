# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan_simulator.capture.schedule_validation import replay_pp_readiness
from torchtitan_simulator.ir.schedule_plan import (
    DataSlot,
    ScheduleAction,
    SchedulePlan,
)


def _action(
    action_id: str,
    *,
    action_type: str = "COMPUTE",
    comp_type: str = "",
    consumes: list[str] | None = None,
    produces: list[str] | None = None,
) -> ScheduleAction:
    return ScheduleAction(
        id=0,
        action_id=action_id,
        rank=0,
        stage=0,
        mb_idx=0,
        action_type=action_type,
        comp_type=comp_type,
        schedule_order=3,
        consumes=consumes or [],
        produces=produces or [],
    )


def test_dualpipe_overlap_children_publish_slots_independently() -> None:
    forward = _action(
        "child_f",
        comp_type="F",
        consumes=["input"],
        produces=["activation"],
    )
    backward = _action("child_b", comp_type="B", consumes=["activation"])
    overlap = _action("overlap", action_type="OVERLAP_F_B")
    overlap.sub_actions = [forward, backward]
    plan = SchedulePlan(
        plan_id="dualpipe",
        workload_type="train",
        step_templates={},
        actions=[overlap],
        data_slots={
            "input": DataSlot(
                slot_id="input",
                kind="dataloader_input",
                consumer_action_ids=[forward.action_id],
                external=True,
            ),
            "activation": DataSlot(
                slot_id="activation",
                kind="activation",
                producer_action_id=forward.action_id,
                consumer_action_ids=[backward.action_id],
            ),
        },
        pipeline_schedule="DualPipeV",
    )

    replay_pp_readiness([plan])
