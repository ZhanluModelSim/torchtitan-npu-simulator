# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.html_export import export_html, normalize_module_path, render_html


def test_normalize_module_path_strips_numeric_modulelist_indices():
    assert normalize_module_path("layers.0.attention.wq") == "layers.N.attention.wq"
    assert normalize_module_path("layers.60.mlp.w1") == "layers.N.mlp.w1"
    assert normalize_module_path("gate") == "gate"
    assert normalize_module_path("") == ""


def _workload_with_repeated_layers(num_layers: int) -> WorkloadGraph:
    nodes: dict[str, OpNode] = {}
    for layer_idx in range(num_layers):
        op_id = f"op_{layer_idx}"
        nodes[op_id] = OpNode(
            op_id=op_id, op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[],
            annotations={"module_path": f"layers.{layer_idx}.attention.wq"},
        )
    template = StepGraph(step_id="tmpl", step_type="forward", nodes=nodes)
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[],
        annotations={"rank_table": {"world_size": 384, "dim_degrees": {"ep": 192}}},
    )
    return WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
        data_inputs=[DataFlow(source="dataloader", tensor_shape=(1, 4096), dtype="int64", volume_per_iter=32768)],
    )


def test_render_html_folds_repeated_layers_into_one_group():
    workload = _workload_with_repeated_layers(61)
    page = render_html(workload)
    assert "layers.N.attention.wq" in page
    assert "&times; 61 layers" in page
    # only the representative layer's op should actually be listed
    assert page.count("<li><code>op_") == 1


def test_render_html_includes_rank_table_world_size():
    workload = _workload_with_repeated_layers(2)
    page = render_html(workload)
    assert "world_size=384" in page


def test_export_html_writes_a_file():
    workload = _workload_with_repeated_layers(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.html")
        export_html(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert content.startswith("<!DOCTYPE html>")
