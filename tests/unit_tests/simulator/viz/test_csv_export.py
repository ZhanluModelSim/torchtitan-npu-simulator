# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.csv_export import export_kernel_summary_csv


def test_kernel_summary_omits_metadata_views():
    view = OpNode(
        op_id="view", op_type="transpose", inputs=[], outputs=[], attrs={},
        predecessors=[], successors=["mm"], annotations={"metadata_view": True},
    )
    matmul = OpNode(
        op_id="mm", op_type="matmul", inputs=[], outputs=[], attrs={},
        predecessors=["view"], successors=[],
    )
    template = StepGraph(
        step_id="tmpl", step_type="forward", nodes={"view": view, "mm": matmul},
    )
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[],
    )
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        export_kernel_summary_csv(workload, os.path.join(tmp, "kernel_summary.csv"))
        with open(os.path.join(tmp, "kernel_summary", "rank_0.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["op_id"] == "mm"
    assert rows[0]["topo_order"] == "0"
