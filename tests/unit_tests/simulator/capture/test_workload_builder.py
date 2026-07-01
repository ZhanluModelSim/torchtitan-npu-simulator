# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.workload_builder import build_workload_graph
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def test_build_workload_graph_wraps_single_iteration():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template}, local_batch_size=2, seq_len=4096,
    )
    assert graph.num_iterations == 1
    assert graph.warmup_iterations == 0
    assert graph.iteration.schedule is schedule
    assert graph.iteration.microbatch_count == 1


def test_build_workload_graph_data_flow_shapes_and_volume():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template}, local_batch_size=2, seq_len=4096,
    )
    assert graph.data_inputs[0].tensor_shape == (2, 4096)
    assert graph.data_inputs[0].volume_per_iter == 2 * 4096 * 8
    assert graph.data_inputs[0].is_streaming is True
    assert graph.data_outputs[0].source == "labels"


def test_build_workload_graph_respects_microbatch_count():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template},
        local_batch_size=1, seq_len=128, num_micro_batches=4,
    )
    assert graph.iteration.microbatch_count == 4
