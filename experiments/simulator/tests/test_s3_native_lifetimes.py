# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native DSV4 AC/parallel integration gates in addition to byte oracles."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig, prepare_config
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("topology", ["hsdp-ep", "cp-ep", "hsdp-ep-tp-cp"])
def test_native_ac_lifetimes_and_accumulation_contract(mode, topology):
    options = dict(world_size=8, expert_parallel_degree=2)
    if topology == "hsdp-ep":
        options["data_parallel_replicate_degree"] = 2
    elif topology == "cp-ep":
        options["context_parallel_degree"] = 2
    else:
        options.update(
            world_size=16,
            logical_rank=15,
            data_parallel_replicate_degree=2,
            tensor_parallel_degree=2,
            context_parallel_degree=2,
            seq_len=512,
        )
    workload = simulate(
        SimulationConfig(
            **options,
            num_mtp_modules=1,
            activation_checkpoint_mode=mode,
            global_batch_size=32,
        )
    )
    annotations = workload.iteration.schedule.annotations
    memory = annotations["memory_plan"]
    initial_refs = [ref for ref in memory.initial_tensors if ref.device == "meta"]
    assert initial_refs
    for ref in initial_refs:
        lifetime = next(item for item in memory.tensor_lifetimes if item.storage_key == ref.storage_key)
        assert lifetime.birth_seq == -1
        assert lifetime.num_bytes == ref.storage_bytes
    assert len(annotations["gradient_validation"]) == len(annotations["parameter_layouts"])
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    assert all(predecessor in nodes for node in nodes.values() for predecessor in node.predecessors)
    visited = set()
    pending_nodes = set(nodes)
    while pending_nodes:
        ready = {op_id for op_id in pending_nodes if set(nodes[op_id].predecessors) <= visited}
        assert ready, "Cross-template AC graph contains a dependency cycle"
        visited.update(ready)
        pending_nodes -= ready
    source_counts = Counter(node.annotations.get("source_op_id", node.op_id) for node in nodes.values())
    assert all(source_counts[event["op_id"]] == 1 for event in annotations["communication_events"])
    actual_to_graph = {node.annotations.get("source_op_id", node.op_id): node.op_id for node in nodes.values()}
    pending = [
        node.op_id for node in nodes.values() if node.annotations.get("raw_op_type") == "aten._fused_adamw_.default"
    ]
    ancestors = set()
    while pending:
        op_id = pending.pop()
        if op_id not in ancestors:
            ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    assert all(
        actual_to_graph[event["op_id"]] in ancestors
        for event in annotations["communication_events"]
        if event["gradient_reduction"]
    )
    batch = annotations["batch_configuration"]
    assert batch["global_batch_size"] == 32
    assert batch["gradient_accumulation"] == 32 // batch["data_parallel_degree"]
    assert batch["captured_microbatches"] == 1
    assert memory.capture_metadata["configured_gradient_accumulation"] == batch["gradient_accumulation"]
    assert all(
        item.death_seq >= max([item.birth_seq, *item.consumer_seqs])
        for item in memory.tensor_lifetimes
        if item.alias_of is None
    )
    # Independent interval sweep, including both allocations and final uses
    # at a sequence boundary; the estimator's timeline is not consulted.
    peak = max(
        sum(item.resident_num_bytes for item in memory.tensor_lifetimes if item.birth_seq <= seq <= item.death_seq)
        for seq in {item.birth_seq for item in memory.tensor_lifetimes}
    )
    assert peak == memory.peak_active_bytes
    markers = memory.fsdp_residency_events
    for marker in markers:
        if marker.action == "alloc" and marker.num_bytes:
            assert marker.allocation_storage_bytes
            assert sum(size for key, size in marker.allocation_storage_bytes) == marker.num_bytes
    assert all(
        item.storage_key
        for item in memory.tensor_lifetimes
        if item.resident_num_bytes and item.kind != "fsdp_full_param"
    )
    allocations = {
        f"fsdp_full_param:{marker.group_id}:{marker.seq_idx}": marker.allocation_storage_bytes
        for marker in markers
        if marker.action == "alloc"
    }
    peak_storage_keys = []
    for item in memory.tensor_lifetimes:
        if item.resident_num_bytes and item.birth_seq <= memory.peak_seq_idx <= item.death_seq:
            keys = (
                [key for key, _ in allocations[item.tensor_id]]
                if item.kind == "fsdp_full_param"
                else [item.storage_key]
            )
            peak_storage_keys.extend(key.partition(":")[0] for key in keys)
    assert len(peak_storage_keys) == len(set(peak_storage_keys)), "Peak double-counts an underlying storage"
    stages = [
        item for item in memory.tensor_lifetimes if item.kind == "comm_buffer" and item.reason.startswith("fsdp_")
    ]
    releases = [marker for marker in markers if marker.action == "staging_free"]
    assert stages and len(stages) == len(releases)
    launches = [
        event
        for event in annotations["communication_events"]
        if event["comm_primitive"] == "allgather" and event["fsdp_transition_id"]
    ]
    assert {event["fsdp_transition_id"] for event in launches} == {marker.transition_id for marker in releases}
    assert len(launches) == len(releases)
    for marker in releases:
        lifetime = next(item for item in stages if item.tensor_id.endswith(marker.transition_id))
        assert lifetime.death_seq == marker.seq_idx
        assert lifetime.num_bytes == marker.num_bytes
    assert not any("open_at_capture_end" in item.reason for item in memory.tensor_lifetimes)
    kinds = {event.execution_kind for event in memory.raw_events}
    assert ("recompute" in kinds) == (mode != "none")
    cached = [slot for slot in memory.autograd_saved_tensors if slot.source == "selective_cache"]
    assert bool(cached) == (mode == "selective")
    assert all(slot.unpack_seq >= slot.pack_seq for slot in cached)


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_storage_dtype_and_saved_offload_preserve_logical_graph(mode):
    base = dict(world_size=2, num_mtp_modules=1, activation_checkpoint_mode=mode)
    resident = simulate(SimulationConfig(**base)).iteration.schedule.annotations
    offloaded = simulate(
        SimulationConfig(
            **base,
            memory_parameter_storage_dtype="bfloat16",
            memory_offload_ac_saved_tensors=True,
        )
    ).iteration.schedule.annotations
    assert resident["gradient_validation"] == offloaded["gradient_validation"]
    assert resident["parameter_layouts"] == offloaded["parameter_layouts"]
    reference, modeled = resident["memory_plan"], offloaded["memory_plan"]
    assert modeled.persistent_param_bytes * 2 == reference.persistent_param_bytes
    assert modeled.model_active_bytes_peak < reference.model_active_bytes_peak
    assert modeled.activation_offload_tensors or modeled.checkpoint_tensors
    assert [(event.raw_op_type, event.execution_kind) for event in modeled.raw_events] == [
        (event.raw_op_type, event.execution_kind) for event in reference.raw_events
    ]


@pytest.mark.parametrize("selection", [["default"], ["gmm"], ["quant-mm"], ["comm"], ["none"]])
def test_native_selective_policy_config_builds_without_global_patch(selection):
    from torchtitan.distributed.activation_checkpoint import SelectiveAC
    from torchtitan_simulator.selective_ac import SimulatorSelectiveAC

    original = SelectiveAC.get_save_ops
    prepared = prepare_config(SimulationConfig(activation_checkpoint_mode="selective", selective_ac_save_ops=selection))
    policy = prepared.trainer.activation_checkpoint.build()
    assert isinstance(policy, SimulatorSelectiveAC)
    chosen = policy.get_save_ops()
    assert bool(chosen) == (selection != ["none"])
    assert SelectiveAC.get_save_ops is original


@pytest.mark.parametrize("selection", ["default", "gmm", "quant-mm", "comm", "none"])
def test_native_selective_groups_capture_actual_selected_values(selection, caplog):
    workload = simulate(
        SimulationConfig(
            world_size=4,
            expert_parallel_degree=2,
            num_mtp_modules=1,
            activation_checkpoint_mode="selective",
            selective_ac_save_ops=[selection],
        )
    )
    annotations = workload.iteration.schedule.annotations
    memory = annotations["memory_plan"]
    first_producer = {}
    for event in memory.raw_events:
        for ref in event.outputs:
            first_producer.setdefault(ref.storage_key, event.raw_op_type)
    cached_ops = {
        first_producer[slot.storage_key] for slot in memory.autograd_saved_tensors if slot.source == "selective_cache"
    }
    if selection == "gmm":
        assert cached_ops == {"aten._grouped_mm.default"}
    elif selection == "comm":
        # Default spmd_types uses eager ProcessGroup exchanges, which are
        # outside native SAC's canonical functional-collective save set.
        assert not cached_ops
        assert "eager SPMD exchanges follow native recomputation" in caplog.text
        assert any(
            event.raw_op_type == "c10d.alltoall_.default" and event.execution_kind == "recompute"
            for event in memory.raw_events
        )
    elif selection in {"none", "quant-mm"}:
        # BF16 has no quantized matmul. MXFP8 operator execution belongs to S4.
        assert not cached_ops
    else:
        assert cached_ops
    assert annotations["gradient_validation"]
