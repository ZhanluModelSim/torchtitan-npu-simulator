# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv

from torchtitan_npu.simulator.ir.schedule_plan import (
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


def test_csv_export_includes_overlap_children_and_parent_relation(tmp_path) -> None:
    forward = _action("child_f", comp_type="F", produces=["activation"])
    backward = _action("child_b", comp_type="B", consumes=["activation"])
    overlap = _action("overlap", action_type="OVERLAP_F_B")
    overlap.sub_actions = [forward, backward]
    plan = SchedulePlan(
        plan_id="dualpipe",
        workload_type="train",
        step_templates={},
        actions=[overlap],
        data_slots={
            "activation": DataSlot(
                slot_id="activation",
                kind="forward_state",
                producer_action_id=forward.action_id,
                consumer_action_ids=[backward.action_id],
            )
        },
    )

    output = tmp_path / "schedule_plan.csv"
    plan.export_schedule_plan_csv(str(output))

    raw_rows = list(csv.reader(output.open(newline="", encoding="utf-8")))
    slots_marker = next(i for i, row in enumerate(raw_rows) if row[0] == "# DataSlots")
    action_rows = [
        dict(zip(raw_rows[0], row, strict=True))
        for row in raw_rows[1:slots_marker]
    ]

    assert [row["action_id"] for row in action_rows] == [
        "overlap",
        "child_f",
        "child_b",
    ]
    assert action_rows[0]["parent_action_id"] == ""
    assert all(
        row["parent_action_id"] == "overlap"
        for row in action_rows[1:]
    )

    exported_action_ids = {row["action_id"] for row in action_rows}
    assert plan.data_slots["activation"].producer_action_id in exported_action_ids
    assert set(plan.data_slots["activation"].consumer_action_ids) <= exported_action_ids
