# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.dot_export import export_dot


def test_export_dot_writes_valid_digraph_with_edges():
    node_a = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=["b"])
    node_b = OpNode(op_id="b", op_type="allreduce", inputs=[], outputs=[], attrs={}, predecessors=["a"], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": node_a, "b": node_b})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.dot")
        export_dot(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert content.startswith("digraph ComputeGraph {")
    assert '"a" -> "b"' in content
    assert "fillcolor=gold" in content  # allreduce is a comm op
    assert "fillcolor=lightblue" in content  # matmul is a compute op
