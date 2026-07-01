# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph


def _empty_schedule() -> ScheduleGraph:
    return ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={}, instances=[])


def test_data_flow_defaults():
    flow = DataFlow(source="dataloader", tensor_shape=(1, 4096), dtype="int64", volume_per_iter=32768)
    assert flow.is_streaming is False
    assert flow.interleave_strategy == "synced"


def test_iteration_spec_defaults():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    assert spec.iteration_time_est == 0.0


def test_workload_graph_construction_and_defaults():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    graph = WorkloadGraph(
        workload_id="wl1",
        workload_type="train",
        step_templates={},
        iteration=spec,
        num_iterations=1,
    )
    assert graph.warmup_iterations == 0
    assert graph.data_inputs == []
    assert graph.data_outputs == []
    assert graph.cross_iter_passes == []
    assert graph.total_runtime_est == 0.0
    assert graph.total_cost_est == 0.0


def test_workload_graph_data_inputs_independent_between_instances():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    a = WorkloadGraph(workload_id="a", workload_type="train", step_templates={}, iteration=spec, num_iterations=1)
    b = WorkloadGraph(workload_id="b", workload_type="train", step_templates={}, iteration=spec, num_iterations=1)
    a.data_inputs.append(DataFlow(source="x", tensor_shape=(1,), dtype="int64", volume_per_iter=8))
    assert b.data_inputs == []
