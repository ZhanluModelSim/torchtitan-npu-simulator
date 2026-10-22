# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate captured graph references and independently reconcile memory bytes."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from torchtitan_simulator.capture.op_mapping import communication_primitive
from torchtitan_simulator.capture.schedule_validation import validate_schedule_plan
from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes

if TYPE_CHECKING:
    from torchtitan_simulator.ir.workload_graph import WorkloadGraph


def _require(condition, message):
    if not condition:
        raise RuntimeError(f"Capture validation failed: {message}")


def validate_workload(workload: WorkloadGraph) -> dict:
    """Check the recorded single-microbatch execution, without expanding ranks."""
    schedule = workload.iteration.schedule
    annotations = schedule.annotations
    plan = workload.schedule_plan
    _require(plan is not None, "missing authoritative SchedulePlan")
    if plan is None:
        raise RuntimeError("Missing SchedulePlan")
    validate_schedule_plan(plan)
    coordinates = annotations["capture_coordinates"]
    worker_rank = coordinates["capture_process_rank"]
    _require(plan.annotations["capture_process_rank"] == worker_rank, "plan capture worker differs")
    _require(schedule.annotations["capture_process_rank"] == worker_rank, "schedule capture worker differs")
    for action in plan.action_map.values():
        _require(action.annotations["capture_process_rank"] == worker_rank, "action capture worker differs")
        _require(action.rank == coordinates["logical_rank"], "action logical rank differs")
    templates = workload.step_templates
    _require(set(templates) == set(plan.step_templates) == set(schedule.step_templates), "template sets differ")
    nodes = {}
    for key, template in templates.items():
        _require(key == template.step_id, "template key/id differs")
        for op_id, node in template.nodes.items():
            _require(op_id == node.op_id and op_id not in nodes, "duplicate or mismatched op ID")
            nodes[op_id] = node
    _require(bool(nodes), "empty graph")
    degree = {}
    successors = defaultdict(list)
    for op_id, node in nodes.items():
        predecessors = set(node.predecessors)
        _require(predecessors <= nodes.keys(), f"missing predecessor for {op_id}")
        degree[op_id] = len(predecessors)
        for predecessor in predecessors:
            successors[predecessor].append(op_id)
    ready = [op_id for op_id, count in degree.items() if count == 0]
    visited = 0
    while ready:
        op_id = ready.pop()
        visited += 1
        for successor in successors[op_id]:
            degree[successor] -= 1
            if degree[successor] == 0:
                ready.append(successor)
    _require(visited == len(nodes), "cross-template dependency cycle")
    for op_id, node in nodes.items():
        _require(set(node.successors) == set(successors[op_id]), f"nonreciprocal graph edges for {op_id}")
    for action in plan.action_map.values():
        if action.template_ref:
            _require(action.template_ref in templates, "action references missing template")
        for slot_id in action.consumes:
            _require(action.action_id in plan.data_slots[slot_id].consumer_action_ids, "nonreciprocal consumer")
        for slot_id in action.produces:
            _require(plan.data_slots[slot_id].producer_action_id == action.action_id, "nonreciprocal producer")
        if action.comm_op_id:
            _require(action.comm_op_id in nodes, "action references missing communication op")
        if action.comm is not None and action.comm.comm_op_id:
            _require(action.comm.comm_op_id in nodes, "communication descriptor references missing op")
    instances = {instance.instance_id: instance for instance in schedule.instances}
    _require(len(instances) == len(schedule.instances), "duplicate StepInstance")
    for instance in instances.values():
        _require(instance.step_ref in templates, "instance references missing template")
    for data_pass in schedule.data_passes:
        _require(data_pass.src_instance in instances and data_pass.dst_instance in instances, "invalid DataPass")
    sources = Counter(node.annotations.get("source_op_id", node.op_id) for node in nodes.values())
    native_ids = annotations["native_dispatch_op_ids"]
    _require(all(sources[op_id] == 1 for op_id in native_ids), "native dispatch node lost or duplicated")
    _require(
        all(
            not entry["scope"].startswith(("fsdp_", "replicate_"))
            for entry in annotations["suppressed_dispatch_counts"]
        ),
        "native FSDP dispatch was suppressed",
    )
    communication = annotations["communication_events"]
    for event in communication:
        _require(sources[event["op_id"]] == 1, "communication source missing or duplicated")
        _require(bool(event["comm_ranks"]), "communication group lost its ranks")
        _require(
            any(coordinates["logical_rank"] in ranks for ranks in event["comm_ranks"]),
            "communication group excludes logical rank",
        )
    expected_dp_reduction = any(
        any(
            name in {"fsdp", "efsdp", "dp_replicate"} and size > 1
            for name, size in zip(layout["mesh_dim_names"], layout["mesh_shape"], strict=True)
        )
        for layout in annotations["parameter_layouts"].values()
    )
    if expected_dp_reduction:
        _require(any(event["gradient_reduction"] for event in communication), "missing actual DP gradient reduction")
    actual_to_graph = {node.annotations.get("source_op_id", node.op_id): node.op_id for node in nodes.values()}
    for event in communication:
        node = nodes[actual_to_graph[event["op_id"]]]
        _require(
            communication_primitive(node.annotations["raw_op_type"]) == event["comm_primitive"],
            "communication descriptor differs from native operator",
        )
        _require(
            not node.annotations["raw_op_type"].startswith("comm."), "synthetic transport replaced native dispatch"
        )
        _require(node.attrs["communication_input_bytes"] == event["volume_bytes"], "communication payload bytes differ")
        if event["comm_dim"] == "tp":
            _require(not event["gradient_reduction"], "TP reduction classified as DP reduction")
    if any(
        any(
            name == "ep" and size > 1 for name, size in zip(layout["mesh_dim_names"], layout["mesh_shape"], strict=True)
        )
        for layout in annotations["parameter_layouts"].values()
    ):
        _require(
            any(event["comm_primitive"] == "all_to_all" for event in communication), "missing native EP all-to-all"
        )
    ownership = plan.annotations["communication_ownership"]
    _require(
        ownership["internal_gradient_reductions"] + ownership["external_gradient_reductions"]
        == sum(event["gradient_reduction"] for event in communication),
        "gradient reduction ownership count differs",
    )
    optimizers = [
        node for node in nodes.values() if node.annotations.get("raw_op_type") == "aten._fused_adamw_.default"
    ]
    _require(bool(optimizers), "missing native optimizer")
    gradients = annotations["gradient_validation"]
    clipped = [
        node
        for node in nodes.values()
        if node.annotations.get("raw_op_type") == "aten.mul_.Tensor"
        and node.annotations.get("comp_type") == "OPTIMIZER"
    ]
    _require(len(clipped) == len(gradients), "clip set differs from gradient set")
    _require(
        sum(node.annotations["optimizer_updated_parameter_count"] for node in optimizers) == len(gradients),
        "optimizer update set differs from gradient set",
    )
    ancestors = set()
    pending = [node.op_id for node in optimizers]
    while pending:
        op_id = pending.pop()
        if op_id not in ancestors:
            ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    for event in communication:
        if event.get("gradient_reduction"):
            _require(actual_to_graph[event["op_id"]] in ancestors, "gradient reduction disconnected from optimizer")
    _require(all(node.op_id in ancestors for node in clipped), "clip disconnected from optimizer")
    memory = annotations["memory_plan"]
    _require(not any("marker imbalance" in note for note in memory.notes), "unbalanced FSDP residency")
    markers = memory.fsdp_residency_events
    _require(
        Counter(event.group_id for event in markers if event.action == "alloc")
        == Counter(event.group_id for event in markers if event.action == "free"),
        "unreleased full parameters",
    )
    _require(
        memory.persistent_param_bytes
        == sum(
            tensor_volume_bytes(tuple(item["local_shape"]), memory.parameter_storage_dtype)
            if memory.parameter_storage_dtype
            else item["storage_bytes"]
            for item in annotations["parameter_layouts"].values()
        ),
        "local persistent parameter byte mismatch",
    )
    allocations = {
        f"fsdp_full_param:{event.group_id}:{event.seq_idx}": event.allocation_storage_bytes
        for event in memory.fsdp_residency_events
        if event.action == "alloc"
    }
    inventory = [
        item
        for item in memory.tensor_lifetimes
        if item.resident_num_bytes and item.birth_seq <= memory.peak_seq_idx <= item.death_seq
    ]
    deltas = Counter()
    for item in memory.tensor_lifetimes:
        _require(item.resident_num_bytes >= 0 and item.birth_seq <= item.death_seq, "invalid lifetime interval")
        deltas[item.birth_seq] += item.resident_num_bytes
        deltas[item.death_seq + 1] -= item.resident_num_bytes
    active = 0
    peak = 0
    for seq in sorted(deltas):
        active += deltas[seq]
        peak = max(peak, active)
    _require(peak == memory.peak_active_bytes, "lifetime sweep peak differs from estimator")
    _require(sum(item.resident_num_bytes for item in inventory) == memory.peak_active_bytes, "peak byte mismatch")
    keys = [
        key.partition(":")[0]
        for item in inventory
        for key in ([item.storage_key] if item.storage_key else [key for key, _ in allocations[item.tensor_id]])
    ]
    _require(len(keys) == len(set(keys)), "duplicate peak storage")
    refs = defaultdict(set)
    for ref in memory.initial_tensors:
        refs[ref.tensor_id].add((ref.storage_key, ref.shape, ref.dtype))
    for event in memory.raw_events:
        for ref in (*event.inputs, *event.outputs):
            refs[ref.tensor_id].add((ref.storage_key, ref.shape, ref.dtype))
    slots = memory.autograd_saved_tensors
    _require(bool(slots), "missing actual autograd saves")
    for slot in slots:
        _require(slot.release_seq >= slot.pack_seq, "saved slot not released")
        if slot.unpack_seq >= 0:
            _require(slot.pack_seq <= slot.unpack_seq <= slot.release_seq, "invalid saved slot interval")
            _require(
                (slot.storage_key, slot.shape, slot.dtype) in refs[slot.tensor_id],
                f"saved value lacks captured provenance: {slot}; refs={refs[slot.tensor_id]}",
            )
    return {
        "graph_nodes": len(nodes),
        "native_dispatch_nodes": len(native_ids),
        "templates": len(templates),
        "actions": len(plan.action_map),
        "data_slots": len(plan.data_slots),
        "communication_events": len(communication),
        "gradient_parameters": len(gradients),
        "saved_slots": len(slots),
        "peak_active_bytes": memory.peak_active_bytes,
        "persistent_param_bytes": memory.persistent_param_bytes,
        "peak_storage_unique": True,
        "graph_acyclic": True,
        "scope": "Meta single-microbatch F/B/clip/optimizer; no hardware numeric or timing validation",
    }
