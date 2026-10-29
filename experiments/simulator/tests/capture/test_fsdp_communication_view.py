# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import pytest
from torchtitan_simulator.capture.fsdp_communication_view import project_fsdp_communication_view
from torchtitan_simulator.capture.step_boundary import build_step_graphs
from torchtitan_simulator.capture.workload_builder import build_workload_graph
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.schedule_graph import DataPass, ScheduleGraph, TensorSlot
from torchtitan_simulator.ir.schedule_plan import CommDetail, DataSlot, ScheduleAction, SchedulePlan
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _buffer_graph():
    specs = [
        (1, "aten.mm.default", "matmul", [], False),
        (2, "aten.empty.memory_format", "unknown", [], True),
        (3, "fsdp.all_gather_copy_in.default", "unknown", [1, 2], True),
        (4, "c10d._allgather_base_.default", "allgather", [3], True),
        (5, "fsdp.split_with_sizes_copy.default", "unknown", [4], True),
        (6, "simulator.stream_wait", "stream_wait", [4, 5], False),
        (7, "aten.mm.default", "matmul", [5, 6], False),
        (8, "fsdp.chunk_cat.default", "unknown", [7], True),
        (9, "c10d._reduce_scatter_base_.default", "reduce_scatter", [8], True),
        # A wrapper-class FQN must not hide genuine model work.
        (10, "aten.copy_.default", "unknown", [9], False),
        (11, "c10d.allreduce_.default", "allreduce", [9], True),
    ]
    nodes = {
        key: OpNode(
            key,
            canonical,
            [],
            [],
            {},
            parents,
            [],
            seq_idx=key,
            annotations={
                "raw_op_type": raw,
                "comp_type": "F",
                "module_path": "FSDPDeepSeekV4Model",
                **({"framework_scopes": ["fsdp_unshard_scaffold"]} if fsdp else {}),
                **({"stream_predecessors": [5], "event_predecessors": [4]} if key == 6 else {}),
            },
        )
        for key, raw, canonical, parents, fsdp in specs
    }
    for key, node in nodes.items():
        for parent in node.predecessors:
            nodes[parent].successors.append(key)
    templates = build_step_graphs(nodes)
    comm = CommDetail(primitive="allgather", src_exit_op=3, dst_entry_op=5, comm_op_id=4, slot_id="param")
    action = ScheduleAction(0, "unshard", 0, 0, 0, "UNSHARD", comm_op_id=4, comm=comm)
    plan = SchedulePlan(
        "plan", "train", templates, [action], {"param": DataSlot("param", "param_full", src_exit_op=3, dst_entry_op=5)}
    )
    schedule = ScheduleGraph(
        "schedule",
        "train",
        templates,
        [],
        data_passes=[DataPass("source", "destination", [TensorSlot("weight", (), "float32", 4, 3, 5)])],
        annotations={
            "native_dispatch_op_ids": list(nodes),
            "communication_events": [{"op_id": 4, "src_exit_op": 3, "dst_entry_op": 5}],
            "memory_plan": object(),
        },
    )
    workload = build_workload_graph(
        schedule_graph=schedule, step_templates=templates, schedule_plan=plan, local_batch_size=1, seq_len=8
    )
    workload.cross_iter_passes = list(schedule.data_passes)
    return workload


def test_fsdp_view_contracts_buffer_paths_and_preserves_communication_and_control():
    original = _buffer_graph()
    projected = project_fsdp_communication_view(original)
    nodes = {key: node for step in projected.step_templates.values() for key, node in step.nodes.items()}

    assert set(nodes) == {1, 4, 6, 7, 9, 10, 11}
    assert nodes[4].predecessors == [1]
    assert nodes[6].predecessors == [4]
    assert nodes[7].predecessors == [4, 6]
    assert nodes[9].predecessors == [7]
    assert nodes[11].predecessors == [9]
    assert nodes[6].annotations["stream_predecessors"] == [4]
    assert nodes[6].annotations["event_predecessors"] == [4]
    assert nodes[4].successors == [6, 7]
    assert all(step.is_acyclic for step in projected.step_templates.values())
    source_nodes = {key: node for step in original.step_templates.values() for key, node in step.nodes.items()}
    assert source_nodes[4].predecessors == [3]
    assert source_nodes[7].predecessors == [5, 6]
    assert source_nodes[6].annotations["stream_predecessors"] == [5]
    assert projected.captured_workload is original
    assert (
        projected.iteration.schedule.annotations["memory_plan"]
        is original.iteration.schedule.annotations["memory_plan"]
    )


def test_fsdp_view_reanchors_slots_and_communication_descriptors_without_dangling_ids():
    original = _buffer_graph()
    projected = project_fsdp_communication_view(original)
    plan = projected.schedule_plan
    assert plan is not None
    assert plan.step_templates is projected.step_templates
    assert projected.iteration.schedule.step_templates is projected.step_templates
    assert (plan.data_slots["param"].src_exit_op, plan.data_slots["param"].dst_entry_op) == (1, 4)
    assert (plan.actions[0].comm.src_exit_op, plan.actions[0].comm.dst_entry_op) == (1, 4)
    slot = projected.iteration.schedule.data_passes[0].slots[0]
    assert (slot.src_exit_op, slot.dst_entry_op) == (1, 0)
    cross_iteration_slot = projected.cross_iter_passes[0].slots[0]
    assert (cross_iteration_slot.src_exit_op, cross_iteration_slot.dst_entry_op) == (1, 0)
    assert original.cross_iter_passes[0].slots[0].src_exit_op == 3
    assert original.schedule_plan.data_slots["param"].src_exit_op == 3


@pytest.mark.parametrize("config", ["deepseek_v4_smoketest", "kimi_k3_smoketest"], ids=["dsv4", "kimi"])
def test_model_fsdp_view_preserves_memory_and_communication_causality(config, tmp_path):
    workload = simulate(
        SimulationConfig(
            config=config,
            world_size=4,
            data_parallel_replicate_degree=2,
            expert_parallel_degree=2,
            seq_len=128,
            output_dir=str(tmp_path),
        )
    )
    captured = workload.captured_workload
    assert captured is not None
    source = {key: node for step in captured.step_templates.values() for key, node in step.nodes.items()}
    visible = {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}
    removed = set(source) - set(visible)
    assert removed
    assert all(source[key].annotations.get("framework_scopes") for key in removed)
    assert not any(
        source[key].op_type in {"allgather", "reduce_scatter", "allreduce", "all_to_all", "stream_wait"}
        for key in removed
    )
    assert Counter(node.op_type for node in source.values() if node.op_type in {"matmul", "grouped_mm"}) == Counter(
        node.op_type for node in visible.values() if node.op_type in {"matmul", "grouped_mm"}
    )
    expected = validate_workload(captured)
    actual = validate_workload(workload)
    for key, step in workload.step_templates.items():
        original_step = captured.step_templates[key]
        for field in ("total_flops", "comm_volume", "param_mem", "peak_active_mem"):
            assert getattr(step, field) == getattr(original_step, field)
    for field in (
        "gradient_parameters",
        "communication_events",
        "saved_slots",
        "peak_active_bytes",
        "persistent_param_bytes",
    ):
        assert actual[field] == expected[field]
    assert (
        workload.iteration.schedule.annotations["memory_summary"]
        == captured.iteration.schedule.annotations["memory_summary"]
    )

    # Compare communication ancestry for every visible node, using the source
    # graph as an independent reachability oracle across all hidden paths.
    comm_ids = {key for key, node in source.items() if node.comm_bytes}

    def comm_ancestors(nodes):
        ancestors = {}
        for node in sorted(nodes.values(), key=lambda item: item.seq_idx):
            inherited = (
                set().union(*(ancestors[parent] for parent in node.predecessors)) if node.predecessors else set()
            )
            ancestors[node.op_id] = inherited | ({node.op_id} if node.op_id in comm_ids else set())
        return ancestors

    reference = comm_ancestors(source)
    projected = comm_ancestors(visible)
    assert all(projected[key] == reference[key] for key in visible)
