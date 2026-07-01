# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.schedule_graph import (
    DataPass,
    ScheduleGraph,
    StepInstance,
    TensorSlot,
)
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def _instance(instance_id: str, step_ref: str = "tmpl") -> StepInstance:
    return StepInstance(
        instance_id=instance_id,
        step_ref=step_ref,
        step_type="forward",
        micro_batch_idx=0,
        pipeline_stage=0,
        device_ids=[0],
        dp_group=0,
    )


def test_step_instance_defaults():
    inst = _instance("rank0")
    assert inst.estimated_runtime == 0.0


def test_tensor_slot_defaults():
    slot = TensorSlot(name="act", src_exit_op="op1", dst_entry_op="op2", shape=(2, 4), dtype="float32", volume_bytes=32)
    assert slot.is_incremental is False


def test_data_pass_defaults():
    slot = TensorSlot(name="act", src_exit_op="op1", dst_entry_op="op2", shape=(2, 4), dtype="float32", volume_bytes=32)
    dp = DataPass(src_instance="rank0", dst_instance="rank1", slots=[slot])
    assert dp.requires_communication is False
    assert dp.comm_primitive == ""
    assert dp.src_device is None


def test_schedule_graph_builds_instance_map_from_instances():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    instances = [_instance("rank0"), _instance("rank1")]
    graph = ScheduleGraph(
        schedule_id="sched1",
        workload_type="train",
        step_templates={"tmpl": template},
        instances=instances,
    )
    assert set(graph.instance_map.keys()) == {"rank0", "rank1"}
    assert graph.instance_map["rank0"] is instances[0]


def test_schedule_graph_respects_explicit_instance_map():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    instances = [_instance("rank0")]
    explicit_map = {"rank0": instances[0]}
    graph = ScheduleGraph(
        schedule_id="sched2",
        workload_type="train",
        step_templates={"tmpl": template},
        instances=instances,
        instance_map=explicit_map,
    )
    assert graph.instance_map is explicit_map


def test_schedule_graph_defaults():
    graph = ScheduleGraph(schedule_id="sched3", workload_type="train", step_templates={}, instances=[])
    assert graph.dp_degree == 1
    assert graph.pipeline_schedule == "none"
    assert graph.annotations == {}
