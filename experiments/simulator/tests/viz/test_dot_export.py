# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_simulator.ir.step_graph import StepGraph
from torchtitan_simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_simulator.viz.dot_export import export_dot


def test_export_dot_writes_valid_digraph_with_edges():
    node_a = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=["b"])
    node_b = OpNode(op_id="b", op_type="allreduce", inputs=[], outputs=[], attrs={}, predecessors=["a"], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": node_a, "b": node_b})
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[]
    )
    workload = WorkloadGraph(
        workload_id="wl1",
        workload_type="train",
        step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1),
        num_iterations=1,
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


def test_export_dot_labels_unknown_ops_with_real_raw_op_type():
    # An op_type of "unknown" must never render as the literal string
    # "unknown" -- the real dispatcher op name must be shown instead so the
    # graph stays reviewable against the real execution's op names.
    node = OpNode(
        op_id="a",
        op_type="unknown",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[],
        annotations={"raw_op_type": "aten.embedding.default"},
    )
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": node})
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[]
    )
    workload = WorkloadGraph(
        workload_id="wl1",
        workload_type="train",
        step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1),
        num_iterations=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.dot")
        export_dot(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert 'label="aten.embedding.default"' in content
    assert 'label="unknown"' not in content


def test_export_dot_folds_metadata_views_and_preserves_reachable_edge():
    node_a = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=["v"])
    node_v = OpNode(
        op_id="v",
        op_type="transpose",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=["a"],
        successors=["b"],
        annotations={"metadata_view": True},
    )
    node_b = OpNode(op_id="b", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=["v"], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": node_a, "v": node_v, "b": node_b})
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[]
    )
    workload = WorkloadGraph(
        workload_id="wl1",
        workload_type="train",
        step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1),
        num_iterations=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.dot")
        export_dot(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert '"v" [label=' not in content
    assert '"a" -> "b"' in content


def test_dot_keeps_forward_to_backward_and_optimizer_dependencies(tmp_path):
    forward = OpNode(op_id=1, op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[2])
    backward = OpNode(op_id=2, op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[1], successors=[3])
    optimizer = OpNode(op_id=3, op_type="adamw", inputs=[], outputs=[], attrs={}, predecessors=[2], successors=[])
    templates = {
        "s0_F": StepGraph(step_id="s0_F", step_type="F", nodes={1: forward}),
        "s0_B": StepGraph(step_id="s0_B", step_type="B", nodes={2: backward}),
        "s0_OPTIMIZER": StepGraph(step_id="s0_OPTIMIZER", step_type="OPTIMIZER", nodes={3: optimizer}),
    }
    schedule = ScheduleGraph(schedule_id="schedule", workload_type="train", step_templates=templates, instances=[])
    workload = WorkloadGraph(
        workload_id="capture",
        workload_type="train",
        step_templates=templates,
        iteration=IterationSpec(schedule=schedule, microbatch_count=1),
        num_iterations=1,
    )
    path = tmp_path / "graph.dot"
    export_dot(workload, str(path))
    content = path.read_text()
    assert content.count('"1" -> "2";') == 1
    assert content.count('"2" -> "3";') == 1
