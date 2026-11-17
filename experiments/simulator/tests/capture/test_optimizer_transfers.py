# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent update-window and state-residency examples."""

import pytest
from torchtitan_simulator.capture.optimizer_transfers import insert_optimizer_transfers
from torchtitan_simulator.capture.step_boundary import build_step_graphs
from torchtitan_simulator.capture.workload_builder import build_workload_graph
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_simulator.ir.schedule_plan import SchedulePlan
from torchtitan_simulator.memory.optimizer_offload import apply_optimizer_offload_budget
from torchtitan_simulator.memory.records import MemoryPlan, RawMemoryEvent, TensorLifetime


def _memory():
    memory = MemoryPlan(optimizer_state_bytes=10, optimizer_state_storage_count=1)
    memory.tensor_lifetimes = [
        TensorLifetime("weight", "parameter", 7, 0, 8, 0, storage_key="weight"),
        TensorLifetime("state", "optimizer_state", 10, 4, 8, 4, storage_key="state"),
        TensorLifetime("temp", "temporary", 17, 7, 8, 7, storage_key="temp"),
    ]
    memory.activation_buffer_lifetimes = [
        TensorLifetime("activation", "activation_transfer_buffer", 5, 1, 2, 1, storage_key="activation")
    ]
    memory.raw_events = [
        RawMemoryEvent(i, i, i, "aten.mm.default", "matmul", phase, "", (), (), phase)
        for i, phase in ((1, "forward"), (2, "backward"), (3, "optimizer"), (8, "optimizer"))
    ]
    return memory


def _workload():
    # Clip/preparation precedes two updates; postprocessing must wait for copies.
    nodes = {}
    for key in range(1, 8):
        comp = "F" if key == 1 else "B" if key == 2 else "OPTIMIZER"
        update = key in (5, 6)
        annotations = {
            "comp_type": comp,
            "phase": comp,
            "stream_id": 0,
            "raw_op_type": "aten._fused_adamw_.default" if update else "aten.mul.Tensor",
        }
        if update:
            annotations["optimizer_work_scope"] = "actual_updated_parameters"
        nodes[key] = OpNode(
            key,
            "adamw_step" if update else "mul",
            [],
            [],
            {},
            [key - 1] if key > 1 else [],
            [key + 1] if key < 7 else [],
            annotations=annotations,
            seq_idx=key,
        )
    steps = build_step_graphs(nodes)
    memory = _memory()
    schedule = ScheduleGraph(
        "schedule",
        "train",
        steps,
        [],
        annotations={
            "native_dispatch_op_ids": list(nodes),
            "memory_plan": memory,
        },
    )
    return build_workload_graph(
        schedule_graph=schedule,
        step_templates=steps,
        schedule_plan=SchedulePlan("plan", "train", steps, [], {}),
        local_batch_size=1,
        seq_len=8,
    )


def _nodes(workload):
    return {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}


def test_full_duplex_window_preserves_updates_and_excludes_pre_and_post_work():
    original = _workload()
    source = _nodes(original)
    result = insert_optimizer_transfers(original)
    nodes = _nodes(result)
    copies = [node for node in nodes.values() if "optimizer_transfer" in node.attrs]
    entry = nodes[copies[0].predecessors[0]]
    exit_node = next(node for node in nodes.values() if {copy.op_id for copy in copies} <= set(node.predecessors))

    assert len(copies) == 2
    assert {node.attrs["optimizer_transfer"]["direction"] for node in copies} == {"h2d", "d2h"}
    assert all(node.attrs["optimizer_transfer"]["bytes"] == 10 for node in copies)
    assert entry.predecessors == [4]
    assert all(node.predecessors == [entry.op_id] for node in copies)
    assert nodes[5].predecessors == [4, entry.op_id]
    assert nodes[6].predecessors == [5, entry.op_id]
    assert exit_node.op_id in nodes[7].predecessors
    assert nodes[4].predecessors == [3]
    assert nodes[5].attrs == source[5].attrs and nodes[6].annotations == source[6].annotations
    assert source[5].predecessors == [4] and source[7].predecessors == [6]
    assert result.iteration.schedule.annotations["native_dispatch_op_ids"] == list(range(1, 8))
    assert all(step.is_acyclic for step in result.step_templates.values())


def test_optimizer_budget_preserves_inventory_activation_buffers_and_raw_lifetimes():
    source = _memory()
    result = apply_optimizer_offload_budget(source)

    assert result.optimizer_state_bytes == source.optimizer_state_bytes == 10
    assert result.tensor_lifetimes is source.tensor_lifetimes
    assert result.raw_events is source.raw_events
    assert result.activation_buffer_lifetimes is source.activation_buffer_lifetimes
    assert source.tensor_lifetimes[1].resident_num_bytes == 10
    assert result.steady_state_forward_peak_active_bytes == 12
    assert result.steady_state_backward_peak_active_bytes == 12
    assert result.steady_state_optimizer_peak_active_bytes == 24
    assert result.steady_state_peak_active_bytes == 24
    state = next(item for item in result.device_budget_lifetimes() if item.kind == "optimizer_state")
    assert state.num_bytes == 10 and state.resident_num_bytes == 0


def test_interleaved_preparation_rejected_instead_of_building_a_cycle():
    workload = _workload()
    nodes = _nodes(workload)
    # The old post node becomes preparation between the two updates.
    nodes[5].successors = [7]
    nodes[7].predecessors = [5]
    nodes[7].successors = [6]
    nodes[6].predecessors = [7]
    nodes[6].successors = []
    with pytest.raises(ValueError, match="interleaved"):
        insert_optimizer_transfers(workload)


def test_missing_state_inventory_cannot_be_treated_as_zero():
    memory = _memory()
    memory.optimizer_state_bytes = None
    with pytest.raises(ValueError, match="inventory"):
        apply_optimizer_offload_budget(memory)


def test_zero_state_inventory_adds_no_transfer_work():
    workload = _workload()
    memory = workload.iteration.schedule.annotations["memory_plan"]
    memory.tensor_lifetimes = [item for item in memory.tensor_lifetimes if item.kind != "optimizer_state"]
    memory.optimizer_state_bytes = memory.optimizer_state_storage_count = 0
    result = insert_optimizer_transfers(workload)
    assert set(_nodes(result)) == set(_nodes(workload))
    assert result.iteration.schedule.annotations["optimizer_transfer_plan"]["modeled_op_ids"] == []
