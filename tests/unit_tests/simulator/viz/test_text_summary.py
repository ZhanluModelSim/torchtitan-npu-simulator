# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph, StepInstance
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.text_summary import export_text_summary


def test_export_text_summary_reports_flops_and_unknown_ops():
    known = OpNode(
        op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[], flops=1000,
    )
    unknown = OpNode(
        op_id="b", op_type="unknown", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[],
        annotations={"cost_unknown": True, "raw_op_type": "aten.mystery_op.default"},
    )
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": known, "b": unknown})
    instance = StepInstance(
        instance_id="rank0", step_ref="tmpl", step_type="forward", micro_batch_idx=0,
        pipeline_stage=0, device_ids=[0], dp_group=0,
    )
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[instance],
    )
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    summary = export_text_summary(workload)
    assert "total_flops=1000" in summary
    assert "aten.mystery_op.default" in summary
    assert "Unrecognized op types" in summary


def test_export_text_summary_reports_no_unknown_ops_when_all_recognized():
    known = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": known})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    summary = export_text_summary(workload)
    assert "All captured op types were recognized" in summary
