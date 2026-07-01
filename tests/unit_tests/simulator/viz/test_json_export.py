# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.json_export import export_json, workload_graph_to_dict


def _tiny_workload() -> WorkloadGraph:
    node = OpNode(op_id="op1", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"op1": node})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    return WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )


def test_workload_graph_to_dict_is_plain_dict():
    d = workload_graph_to_dict(_tiny_workload())
    assert isinstance(d, dict)
    assert d["workload_id"] == "wl1"
    assert d["step_templates"]["tmpl"]["nodes"]["op1"]["op_type"] == "matmul"


def test_export_json_writes_valid_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        export_json(_tiny_workload(), path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["workload_id"] == "wl1"
