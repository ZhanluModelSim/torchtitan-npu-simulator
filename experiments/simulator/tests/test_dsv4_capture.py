# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict
from math import prod

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 1},
        {"world_size": 2, "data_parallel_replicate_degree": 2, "data_parallel_shard_degree": 1},
        {"world_size": 2},
        {"world_size": 4, "data_parallel_replicate_degree": 2},
        {"world_size": 4, "expert_parallel_degree": 2},
        {"world_size": 4, "tensor_parallel_degree": 2},
        {"world_size": 4, "context_parallel_degree": 2},
        {"world_size": 8, "data_parallel_replicate_degree": 2, "expert_parallel_degree": 2},
        {"world_size": 8, "tensor_parallel_degree": 2, "expert_parallel_degree": 2},
        {"world_size": 8, "context_parallel_degree": 2, "expert_parallel_degree": 2},
        {
            "world_size": 16,
            "data_parallel_replicate_degree": 2,
            "tensor_parallel_degree": 2,
            "context_parallel_degree": 2,
            "expert_parallel_degree": 2,
        },
    ],
    ids=["single", "ddp", "fsdp", "hsdp", "ep-efsdp", "tp-sp", "cp", "hsdp-ep", "ep-tp", "ep-cp", "hsdp-ep-tp-cp"],
)
def test_native_dsv4_mtp_capture_has_gradients_closed_graph_and_memory(kwargs):
    workload = simulate(SimulationConfig(num_mtp_modules=1, **kwargs))
    from torchtitan_simulator.validation import validate_workload

    validation = validate_workload(workload)
    assert validation["graph_acyclic"] and validation["peak_storage_unique"]

    annotations = workload.iteration.schedule.annotations
    gradients = annotations["gradient_validation"]
    for parameter in (
        "layers.0.attention.wq_a.weight",
        "layers.2.attention.indexer.compressor.wkv.weight",
        "layers.0.moe.routed_experts.inner_experts.w1_EFD",
        "mtp_layers.0.e_proj.weight",
    ):
        assert parameter in gradients
        assert gradients[parameter]["dtype"] == "torch.float32"
    nodes = {op_id: node for template in workload.step_templates.values() for op_id, node in template.nodes.items()}
    assert all(template.is_acyclic for template in workload.step_templates.values())
    assert all(predecessor in nodes for node in nodes.values() for predecessor in node.predecessors)
    successors = defaultdict(list)
    degree = {op_id: len(set(node.predecessors)) for op_id, node in nodes.items()}
    for op_id, node in nodes.items():
        for predecessor in set(node.predecessors):
            successors[predecessor].append(op_id)
    ready = [op_id for op_id, count in degree.items() if count == 0]
    visited = set()
    while ready:
        op_id = ready.pop()
        visited.add(op_id)
        for successor in successors[op_id]:
            degree[successor] -= 1
            if degree[successor] == 0:
                ready.append(successor)
    assert visited == set(nodes), "Cross-template graph contains a dependency cycle"
    assert any(node.annotations["raw_op_type"] == "aten._fused_adamw_.default" for node in nodes.values())
    fused = [node for node in nodes.values() if node.annotations["raw_op_type"] == "aten._fused_adamw_.default"]
    expected_shapes = Counter(tuple(item["original_shape"]) for item in annotations["parameter_layouts"].values())
    assert (
        Counter(tuple(shape) for node in fused for shape in node.annotations["optimizer_logical_parameter_shapes"])
        == expected_shapes
    )
    assert sum(node.annotations["optimizer_updated_parameter_count"] for node in fused) == len(gradients)
    source_op_ids = {node.op_id for node in nodes.values()}
    source_counts = Counter(node.op_id for node in nodes.values())
    for event in annotations["communication_events"]:
        assert event["op_id"] in source_op_ids, event
        assert source_counts[event["op_id"]] == 1, event
        if event["comm_dim"] == "tp":
            assert not event["gradient_reduction"], event
    expected_reductions = sum(event["gradient_reduction"] for event in annotations["communication_events"])
    ownership = workload.schedule_plan.annotations["communication_ownership"]
    assert ownership["internal_gradient_reductions"] + ownership["external_gradient_reductions"] == expected_reductions
    actual_to_graph = {node.op_id: node.op_id for node in nodes.values()}
    optimizer_ancestors = set()
    pending = [node.op_id for node in nodes.values() if node.annotations["raw_op_type"] == "aten._fused_adamw_.default"]
    while pending:
        op_id = pending.pop()
        if op_id not in optimizer_ancestors:
            optimizer_ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    clipped = [
        node
        for node in nodes.values()
        if node.annotations["raw_op_type"] == "aten.mul_.Tensor" and node.annotations.get("comp_type") == "OPTIMIZER"
    ]
    assert len(clipped) == len(gradients)
    assert all(node.op_id in optimizer_ancestors for node in clipped)
    clip_ancestors = set()
    pending = [node.op_id for node in clipped]
    while pending:
        op_id = pending.pop()
        if op_id not in clip_ancestors:
            clip_ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    for event in annotations["communication_events"]:
        if event.get("gradient_reduction"):
            assert actual_to_graph[event["op_id"]] in clip_ancestors, event
            assert actual_to_graph[event["op_id"]] in optimizer_ancestors, event
    raw_ops = {node.annotations["raw_op_type"] for node in nodes.values()}
    assert {
        "npu.npu_rms_norm.default",
        "npu.npu_rms_norm_backward.default",
        "npu.npu_moe_token_permute.default",
        "npu.npu_moe_token_permute_grad.default",
    } <= raw_ops
    memory = annotations["memory_plan"]
    assert memory.autograd_saved_tensors
    expert_slots = [
        slot for slot in memory.autograd_saved_tensors if "inner_experts" in slot.module_path and slot.unpack_seq >= 0
    ]
    assert expert_slots, "Native grouped experts must retain actual forward values for backward"
    captured_refs = defaultdict(list)
    for raw in memory.raw_events:
        for ref in (*raw.inputs, *raw.outputs):
            captured_refs[ref.tensor_id].append(ref)
    for slot in expert_slots:
        assert any(
            ref.storage_key == slot.storage_key and ref.shape == slot.shape and ref.dtype == slot.dtype
            for ref in captured_refs[slot.tensor_id]
        ), slot
    assert not any("FSDP residency marker imbalance" in note for note in memory.notes)
    assert memory.persistent_param_bytes > 0
    fused_memory = [raw for raw in memory.raw_events if raw.raw_op_type == "aten._fused_adamw_.default"]
    assert sum(ref.num_bytes for raw in fused_memory for ref in raw.inputs) == (
        4 * memory.persistent_param_bytes + 4 * len(gradients)
    )
    assert sum(prod(tensor.shape) for node in fused for tensor in node.inputs) >= (memory.persistent_param_bytes // 4)
    assert memory.peak_active_bytes >= memory.model_active_bytes_peak >= memory.persistent_param_bytes
    assert memory.forward_peak_active_bytes <= memory.peak_active_bytes
    if kwargs == {"world_size": 4, "data_parallel_replicate_degree": 2}:
        post = [
            node
            for node in nodes.values()
            if "fsdp_post_backward_scaffold" in node.annotations.get("framework_scopes", ())
        ]
        assert {"aten._to_copy.default", "aten.add_.Tensor", "fsdp.chunk_cat.default"} <= {
            node.annotations["raw_op_type"] for node in post
        }
        post_ids = {node.op_id for node in post}
        casts = [
            event
            for event in memory.raw_events
            if event.op_id in post_ids
            and event.raw_op_type == "aten._to_copy.default"
            and event.inputs[0].dtype != event.outputs[0].dtype
        ]
        assert casts
        storages = {item.storage_key for item in memory.tensor_lifetimes}
        assert all(event.outputs[0].storage_key in storages for event in casts)
        assert any(event["gradient_reduction"] for event in annotations["communication_events"])
    assert not any(
        lifetime.kind == "gradient_accumulator" and lifetime.reason == "missing_meta_autograd"
        for lifetime in memory.tensor_lifetimes
    )

    if kwargs.get("expert_parallel_degree", 1) > 1:
        pending_buffers = [
            lifetime for lifetime in memory.tensor_lifetimes if lifetime.reason == "fsdp_prefetch_finalize_release"
        ]
        assert pending_buffers
        assert all(lifetime.death_seq > lifetime.birth_seq and lifetime.num_bytes > 0 for lifetime in pending_buffers)


@pytest.mark.parametrize("logical_rank", [1, 8, 15])
def test_nonzero_rank_keeps_logical_coordinates_separate_from_pipeline_stage(logical_rank):
    workload = simulate(
        SimulationConfig(
            world_size=16,
            logical_rank=logical_rank,
            seq_len=512,
            num_mtp_modules=1,
            data_parallel_replicate_degree=2,
            tensor_parallel_degree=2,
            context_parallel_degree=2,
            expert_parallel_degree=2,
        )
    )
    annotations = workload.iteration.schedule.annotations
    assert annotations["capture_coordinates"]["logical_rank"] == logical_rank
    assert annotations["capture_coordinates"]["capture_process_rank"] == 0
    assert annotations["capture_process_rank"] == 0
    assert workload.schedule_plan.annotations["capture_process_rank"] == 0
    assert workload.schedule_plan.annotations["logical_rank"] == logical_rank
    assert all(
        action.rank == logical_rank and action.annotations["capture_process_rank"] == 0
        for action in workload.schedule_plan.action_map.values()
    )
    assert annotations["capture_coordinates"]["target_device_type"] == "non_a5"
    assert set(workload.step_templates) == {"s0_F", "s0_B", "s0_OPTIMIZER"}
    assert annotations["gradient_validation"]
    layouts = annotations["parameter_layouts"]
    for name, tp_shards in (("layers.0.attention.wq_a.weight", 1), ("layers.0.attention.wq_b.weight", 2)):
        layout = layouts[name]
        rows, columns = layout["original_shape"]
        assert layout["local_shape"] == (rows // (4 * tp_shards), columns)
        assert layout["mesh_dim_names"] == ("dp_replicate", "dp_shard_cp", "tp")
        assert layout["mesh_shape"] == (2, 4, 2)
    experts = layouts["layers.0.moe.routed_experts.inner_experts.w1_EFD"]
    num_experts, hidden, dim = experts["original_shape"]
    assert experts["mesh_dim_names"] == ("dp_replicate", "efsdp", "ep")
    assert experts["mesh_shape"] == (2, 4, 2)
    expected_experts = (num_experts // 2, hidden // 4, dim) if num_experts < 8 else (num_experts // 8, hidden, dim)
    assert experts["local_shape"] == expected_experts
    assert annotations["memory_plan"].persistent_param_bytes == sum(item["storage_bytes"] for item in layouts.values())
    assert all(template.is_acyclic for template in workload.step_templates.values())
    # Every collective must identify the group containing this logical rank.
    for event in annotations["communication_events"]:
        assert event["capture_process_rank"] == 0
        assert event["comm_ranks"], event
        assert any(logical_rank in group for group in event["comm_ranks"]), event


def test_native_main_loss_without_mtp_keeps_the_same_gradient_and_optimizer_contract():
    workload = simulate(SimulationConfig(world_size=2, num_mtp_modules=0))
    annotations = workload.iteration.schedule.annotations
    gradients = annotations["gradient_validation"]
    assert gradients and not any(name.startswith("mtp_layers.") for name in gradients)
    assert "layers.2.attention.indexer.compressor.wkv.weight" in gradients
    assert {"s0_F", "s0_B", "s0_OPTIMIZER"} == set(workload.step_templates)
    assert all(template.is_acyclic for template in workload.step_templates.values())
    assert annotations["memory_plan"].persistent_param_bytes > 0
