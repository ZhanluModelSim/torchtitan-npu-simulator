# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native Muon state, persistent workspace and aggregate transfer boundaries."""

import pytest
from torchtitan_simulator.capture import optimizer_transfers
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _nodes(workload):
    return {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}


@pytest.mark.parametrize(
    "world, ep, workspace_sizes",
    [
        (1, 1, [4_194_304]),
        (8, 2, [1_096_704, 2_621_440, 524_288, 2_097_152, 393_216, 786_432, 131_072, 262_144]),
    ],
    ids=["local-scratch", "redistributed-double-buffer"],
)
def test_muon_aggregate_offload_retains_workspace_and_native_dependencies(monkeypatch, world, ep, workspace_sizes):
    original_insert = optimizer_transfers.insert_optimizer_transfers
    inputs = []

    def observe_input(workload):
        inputs.append(workload)
        return original_insert(workload)

    monkeypatch.setattr(optimizer_transfers, "insert_optimizer_transfers", observe_input)

    workload = simulate(
        SimulationConfig(
            optimizer_name="Muon",
            muon_ns_steps=1,
            world_size=world,
            expert_parallel_degree=ep,
            memory_offload_optimizer_states=True,
        )
    )

    assert validate_workload(workload)["graph_acyclic"]
    assert len(inputs) == 1
    source = _nodes(inputs[0])
    nodes = _nodes(workload)
    annotations = workload.iteration.schedule.annotations
    memory = annotations["memory_plan"]
    original_memory = inputs[0].iteration.schedule.annotations["memory_plan"]
    groups = annotations["optimizer_parameter_groups"]
    layouts = annotations["parameter_layouts"]
    # The default debug recipe uses FP32 local state: one momentum for Muon,
    # two moments plus a scalar step for each fallback AdamW parameter.
    expected_state_bytes = sum(
        layouts[name]["storage_bytes"] * (1 if group["algorithm"] == "DistMuon" else 2)
        + (0 if group["algorithm"] == "DistMuon" else 4)
        for group in groups
        for name in group["parameter_names"]
    )
    assert memory.optimizer_state_bytes == original_memory.optimizer_state_bytes == expected_state_bytes
    # Fixed debug geometry's reserved slot capacities, independent of the memory
    # estimator and workspace discovery. They count allocation capacity, not views.
    workspace = [item for item in memory.tensor_lifetimes if item.kind == "optimizer_workspace"]
    assert sorted(item.resident_num_bytes for item in workspace) == sorted(workspace_sizes)
    capture_end = max(event.seq_idx for event in memory.raw_events)
    assert all(item.birth_seq == -1 and item.death_seq >= capture_end for item in workspace)
    assert sum(
        item.resident_num_bytes for item in memory.device_budget_lifetimes() if item.kind == "optimizer_workspace"
    ) == sum(workspace_sizes)
    assert all(
        item.resident_num_bytes == 0 for item in memory.device_budget_lifetimes() if item.kind == "optimizer_state"
    )
    for phase in ("forward", "backward", "optimizer"):
        key = f"steady_state_{phase}_peak_active_bytes"
        assert getattr(original_memory, key) - getattr(memory, key) == expected_state_bytes
        assert getattr(memory, key) >= sum(workspace_sizes)

    plan = annotations["optimizer_transfer_plan"]
    assert plan["window_scope"] == "native_optimizer_step"
    window = set(plan["window_op_ids"])
    assert plan["transfer_bytes_per_direction"] == expected_state_bytes
    assert set(plan["update_op_ids"]) < window
    copies = [node for node in nodes.values() if "optimizer_transfer" in node.attrs]
    assert len(copies) == 2
    entry = nodes[copies[0].predecessors[0]]
    join = next(node for node in nodes.values() if {copy.op_id for copy in copies} <= set(node.predecessors))
    assert {node.attrs["optimizer_transfer"]["direction"] for node in copies} == {"h2d", "d2h"}
    assert all(node.attrs["optimizer_transfer"]["bytes"] == expected_state_bytes for node in copies)
    assert all(node.predecessors == [entry.op_id] for node in copies)
    assert len({node.annotations["stream_id"] for node in copies}) == 2
    assert not {node.annotations["stream_id"] for node in copies} & {
        source[key].annotations.get("stream_id", 0) for key in window
    }
    assert window <= set(join.predecessors)
    assert any(source[key].annotations["raw_op_type"] == "aten.lerp_.Scalar" for key in window)
    assert all("gradient_clip" not in source[key].annotations.get("framework_scopes", ()) for key in window)
    # Adding only a shared entry and exit leaves the entire native partial order
    # unchanged, including no transfer -> native or native -> transfer edge.
    for key, original in source.items():
        assert nodes[key].attrs == original.attrs
        assert nodes[key].annotations == original.annotations
        assert set(nodes[key].predecessors) & source.keys() == set(original.predecessors)
        if key in window:
            assert set(nodes[key].predecessors) - set(original.predecessors) == {entry.op_id}
    assert set(entry.predecessors) == {
        parent for key in window for parent in source[key].predecessors if parent not in window
    }
    alltoalls = [
        node
        for node in source.values()
        if node.op_type == "all_to_all" and node.annotations.get("comp_type") == "OPTIMIZER"
    ]
    assert len(alltoalls) == (0 if world == 1 else 18)
    assert all(node.op_id in window for node in alltoalls)
