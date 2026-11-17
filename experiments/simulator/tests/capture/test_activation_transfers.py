# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-computed dependency and storage oracles for activation transfer policy."""

from dataclasses import replace

import pytest
from torchtitan_simulator.capture.activation_transfers import insert_activation_transfers
from torchtitan_simulator.capture.fsdp_communication_view import project_fsdp_communication_view
from torchtitan_simulator.capture.step_boundary import build_step_graphs
from torchtitan_simulator.capture.workload_builder import build_workload_graph
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_simulator.ir.schedule_plan import SchedulePlan
from torchtitan_simulator.memory.records import CheckpointTensorRecord, MemoryPlan, RawMemoryEvent, TensorLifetime


def _workload(*, backward_head=True, shared=False):
    specs = [
        (1, "layers.0", "forward", "forward"),
        (2, "layers.0", "forward", "forward"),
        (3, "layers.0", "forward", "forward"),
        (4, "layers.1", "forward", "forward"),
        (5, "layers.2", "forward", "forward"),
        (6, "head", "forward", "backward"),
        (7, "head" if backward_head else "layers.2", "backward", "backward"),
        (8, "layers.2", "backward", "recompute"),
        (9, "layers.2", "backward", "backward"),
        (10, "layers.1", "backward", "backward"),
        (11, "layers.0", "backward", "backward"),
        (12, "", "optimizer", "optimizer"),
    ]
    nodes = {}
    for key, path, phase, execution in specs:
        comm = key == 2
        nodes[key] = OpNode(
            key,
            "allreduce" if comm else "matmul",
            [],
            [],
            {},
            [key - 1] if key > 1 else [],
            [key + 1] if key < 12 else [],
            seq_idx=key,
            annotations={
                "raw_op_type": "c10d.allreduce_.default" if comm else "aten.mm.default",
                "module_path": path,
                "phase": phase,
                "execution_kind": execution,
                "comp_type": {"forward": "F", "backward": "B", "optimizer": "OPTIMIZER"}[phase],
                "stream_id": 0,
            },
        )
    memory = MemoryPlan(
        offload_ac_saved_tensors=True,
        optimizer_state_bytes=0,
        steady_state_optimizer_peak_active_bytes=7,
        optimizer_peak_active_bytes=7,
    )
    memory.tensor_lifetimes = [TensorLifetime("base", "parameter", 7, 0, 12, 0, storage_key="base")]
    for index, (size, birth, consumer) in enumerate(((10, 1, 11), (20, 4, 10), (30, 5, 9))):
        tensor_id = "a0" if shared and index == 1 else f"a{index}"
        if not (shared and index == 1):
            memory.tensor_lifetimes.append(
                TensorLifetime(
                    tensor_id,
                    "activation",
                    size,
                    birth,
                    consumer,
                    birth,
                    modeled_num_bytes=0,
                    storage_key=tensor_id,
                    consumer_ops=[consumer],
                    consumer_seqs=[consumer],
                    consumer_phases=["backward"],
                )
            )
        memory.activation_offload_tensors.append(
            CheckpointTensorRecord(
                f"part0:layers.{index}",
                "module",
                birth,
                tensor_id,
                "input",
                (size,),
                "uint8",
                size,
                0,
                False,
                True,
                "offloaded",
            )
        )
    memory.raw_events = [
        RawMemoryEvent(key, key, key, "aten.mm.default", "matmul", phase, path, (), (), execution)
        for key, path, phase, execution in specs
    ]
    templates = build_step_graphs(nodes)
    schedule = ScheduleGraph(
        "schedule",
        "train",
        templates,
        [],
        annotations={"native_dispatch_op_ids": list(nodes), "communication_events": [], "memory_plan": memory},
    )
    plan = SchedulePlan("plan", "train", templates, [], {})
    return build_workload_graph(
        schedule_graph=schedule, step_templates=templates, schedule_plan=plan, local_batch_size=1, seq_len=8
    )


def _nodes(workload):
    return {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}


def _copies(workload):
    return {
        (node.attrs["activation_transfer"]["direction"], node.attrs["activation_transfer"]["layer"]): node
        for node in _nodes(workload).values()
        if "activation_transfer" in node.attrs
    }


def test_transfers_use_full_dependency_frontiers_and_wait_before_recompute():
    raw = _workload()
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    nodes, copies = _nodes(result), _copies(result)

    assert copies["d2h", "layers.0"].predecessors == [1, 3]
    assert copies["h2d", "layers.2"].predecessors == []
    assert copies["h2d", "layers.1"].predecessors == sorted([7, copies["h2d", "layers.2"].op_id])
    assert any(copies["h2d", "layers.2"].op_id in nodes[parent].predecessors for parent in nodes[8].predecessors)
    assert any(copies["d2h", "layers.0"].op_id in nodes[parent].predecessors for parent in nodes[5].predecessors)
    assert nodes[4].predecessors == [3]
    assert _nodes(raw)[5].predecessors == [4]
    assert all(step.is_acyclic for step in result.step_templates.values())
    assert result.iteration.schedule.annotations["native_dispatch_op_ids"] == list(range(1, 13))


def test_first_backward_consumer_without_head_has_explicit_entry_copy():
    result = insert_activation_transfers(project_fsdp_communication_view(_workload(backward_head=False)))
    nodes, copies = _nodes(result), _copies(result)

    assert copies["h2d", "layers.2"].predecessors == []
    assert any(copies["h2d", "layers.2"].op_id in nodes[parent].predecessors for parent in nodes[7].predecessors)
    assert len(copies) == 6


def test_static_buffers_deduplicate_shared_storage_and_preserve_raw_and_optimizer():
    raw = _workload(shared=True)
    original = raw.iteration.schedule.annotations["memory_plan"]
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    memory = result.iteration.schedule.annotations["memory_plan"]

    # Layers 0/1 share 10 bytes, layer 2 has 30; plus 7 persistent bytes.
    assert memory.steady_state_forward_peak_active_bytes == 47
    assert memory.steady_state_backward_peak_active_bytes == 47
    assert memory.steady_state_optimizer_peak_active_bytes == 7
    assert memory.optimizer_peak_active_bytes == 7
    assert memory.raw_events is original.raw_events
    assert memory.tensor_lifetimes is original.tensor_lifetimes
    assert original.activation_buffer_lifetimes == []
    assert result.iteration.schedule.annotations["memory_summary"] == memory.to_summary_dict()


def test_offload_disabled_leaves_graph_and_memory_unchanged():
    raw = _workload()
    raw.iteration.schedule.annotations["memory_plan"] = replace(
        raw.iteration.schedule.annotations["memory_plan"], offload_ac_saved_tensors=False
    )
    view = project_fsdp_communication_view(raw)

    assert insert_activation_transfers(view) is view


def test_forward_interleaved_backward_does_not_supply_a_backward_prefetch_window():
    raw = _workload(backward_head=False)
    # The head's autograd kernels run inside the F action before the first B layer.
    _nodes(raw)[6].annotations["phase"] = "backward"
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    copy = _copies(result)["h2d", "layers.2"]

    assert copy.predecessors == []
    owners = {key: step.step_type for step in result.step_templates.values() for key in step.nodes}
    assert owners[copy.op_id] == owners[7]
    memory = result.iteration.schedule.annotations["memory_plan"]
    restored = [item for item in memory.activation_buffer_lifetimes if item.storage_key == "a2"]
    assert any(item.birth_seq == 7 for item in restored)


def test_later_out_write_to_offloaded_storage_is_rejected():
    raw = _workload(shared=True)
    memory = raw.iteration.schedule.annotations["memory_plan"]
    memory.raw_events[3] = replace(
        memory.raw_events[3], raw_op_type="aten.add.out", comp_type="F", mutated_storage_keys=("a0",)
    )

    with pytest.raises(ValueError, match="mutated after its layer copy"):
        insert_activation_transfers(project_fsdp_communication_view(raw))


def test_late_alias_does_not_replace_forward_storage_budget():
    raw = _workload()
    memory = raw.iteration.schedule.annotations["memory_plan"]
    memory.tensor_lifetimes.append(TensorLifetime("late_view", "alias", 0, 11, 11, 11, alias_of="a0", storage_key="a0"))
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    budget = result.iteration.schedule.annotations["memory_plan"]

    assert budget.steady_state_forward_peak_active_bytes == 57
    assert budget.steady_state_backward_peak_active_bytes == 57
    assert all(item.birth_seq <= item.death_seq for item in budget.activation_buffer_lifetimes)


def test_late_head_producer_is_a_copy_dependency_and_drains_at_phase_end():
    raw = _workload()
    memory = raw.iteration.schedule.annotations["memory_plan"]
    memory.tensor_lifetimes[-1] = replace(memory.tensor_lifetimes[-1], birth_seq=6, producer_op=6)
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    copy = _copies(result)["d2h", "layers.2"]

    assert 6 in copy.predecessors
    assert copy.op_id not in _nodes(result)[6].predecessors
    assert not any(copy.op_id in _nodes(result)[key].predecessors for key in _nodes(result)[6].predecessors)
    buffers = result.iteration.schedule.annotations["memory_plan"].activation_buffer_lifetimes
    assert any(item.storage_key == "a2" and item.birth_seq == 6 and item.death_seq == 6 for item in buffers)
    assert all(step.is_acyclic for step in result.step_templates.values())


def test_entry_restore_budget_overlaps_unattributed_backward_prelude():
    raw = _workload(backward_head=False)
    _nodes(raw)[7].annotations["module_path"] = ""
    memory = raw.iteration.schedule.annotations["memory_plan"]
    memory.tensor_lifetimes.append(
        TensorLifetime("prelude", "workspace", 100, 7, 7, 7, producer_phase="backward", storage_key="prelude")
    )
    result = insert_activation_transfers(project_fsdp_communication_view(raw))
    budget = result.iteration.schedule.annotations["memory_plan"]

    # Entry restore 30 + a short-lived 100-byte prelude + 7 persistent bytes.
    assert budget.steady_state_backward_peak_active_bytes == 137
    assert any(item.storage_key == "a2" and item.birth_seq == 7 for item in budget.activation_buffer_lifetimes)
