# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native MX training operands, saved-code identity and byte-level oracles."""

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.cli import main
from torchtitan_simulator.memory.estimator import estimate_static_memory


@pytest.mark.parametrize("flag", ["--mxfp8-fqns", "--simulation.mxfp8-fqns"])
def test_legacy_fqn_cli_override_is_unavailable(flag):
    with pytest.raises(SystemExit) as error:
        main([flag, "layers.*.attention.wq_a"])
    assert error.value.code == 2


@pytest.mark.parametrize("primitive,saved_bytes", [("mm", 8448), ("bmm", 16896), ("grouped_mm", 17152)])
def test_native_mx_saved_operands_match_the_forward_and_manual_byte_oracle(primitive, saved_bytes):
    from torchao_npu.ops.mx_ops import to_mx_then_bmm, to_mx_then_grouped_mm, to_mx_then_mm
    from torchao_npu.quantization.quant_configs import MXQuantizeConfig

    if primitive == "bmm":
        shape_a = shape_b = (2, 64, 64)
    elif primitive == "grouped_mm":
        shape_a, shape_b = (128, 64), (2, 64, 64)
    else:
        shape_a = shape_b = (64, 64)
    a = torch.empty(shape_a, dtype=torch.bfloat16, device="meta", requires_grad=True)
    b = torch.empty(shape_b, dtype=torch.bfloat16, device="meta", requires_grad=True)
    groups = torch.tensor([64, 128], dtype=torch.int32)
    config = MXQuantizeConfig()
    phase = ["forward"]
    capture = OpDispatchCapture(record_memory=True, phase_provider=lambda: phase[0])
    capture.record_initial_memory_tensors((a, b, groups))
    with capture, AutogradSavedTensorCapture():
        if primitive == "grouped_mm":
            output = to_mx_then_grouped_mm(a, b, groups, config, config)
        elif primitive == "bmm":
            output = to_mx_then_bmm(a, b, config, config)
        else:
            output = to_mx_then_mm(a, b, config, config)
        capture.record_synthetic_op("test.forward_backward_gap", [], [])
        gap_seq = capture.memory_events()[-1].seq_idx
        phase[0] = "backward"
        output.sum().backward()
        capture.record_synthetic_op("test.after_backward", [], [])
        after_seq = capture.memory_events()[-1].seq_idx
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(),
        initial_tensors=capture.initial_memory_tensors(),
        storage_lifetimes=capture.storage_lifetime_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
        counted_device_type="meta",
    )
    forward_quant = [event for event in plan.raw_events if event.phase == "forward" and "mx_quant" in event.raw_op_type]
    if primitive == "grouped_mm":
        assert len(forward_quant) == 3
        expected_refs = [*forward_quant[1].outputs, *forward_quant[2].outputs[:2]]
    else:
        assert len(forward_quant) == 2
        expected_refs = [*forward_quant[0].outputs[2:], *forward_quant[1].outputs[:2]]
    slots = [slot for slot in plan.autograd_saved_tensors if slot.dtype in {"float8_e4m3fn", "uint8"}]
    assert len(slots) == 4
    assert {slot.storage_key for slot in slots} == {ref.storage_key for ref in expected_refs}
    assert sum(slot.storage_bytes for slot in slots) == saved_bytes
    # For 64x64 MM, two FP8 code arrays cost 2*4096 and their paired
    # E8M0 scales cost 2*128. BMM doubles this. Grouped scales reserve
    # [M//64+E,K,2] bytes: 8192+512+8192+256 = 17152.
    for slot in slots:
        lifetime = next(item for item in plan.tensor_lifetimes if item.storage_key == slot.storage_key)
        assert lifetime.birth_seq < gap_seq < lifetime.death_seq < after_seq
        assert slot.pack_seq < gap_seq < slot.unpack_seq <= slot.release_seq
        assert any(
            event.phase == "backward" and any(ref.storage_key == slot.storage_key for ref in event.inputs)
            for event in plan.raw_events
        )
    kernel = "npu.npu_grouped_matmul.default" if primitive == "grouped_mm" else "npu.npu_quant_matmul.default"
    assert sum(event.raw_op_type == kernel for event in plan.raw_events) == 3
    assert a.grad is not None and a.grad.shape == a.shape and a.grad.dtype == a.dtype
    assert b.grad is not None and b.grad.shape == b.shape and b.grad.dtype == b.dtype


def test_meta_swiglu_bridge_preserves_actual_saved_operands_and_restores_binding():
    from torchtitan_simulator.compat.patches import PatchScope
    from torchtitan_simulator.compat.swiglu import install_meta_swiglu_autograd

    from torchtitan_npu.override.common.swiglu_group import ascendc

    original = ascendc.swiglu_group_activation
    x = torch.empty(2, 64, device="meta", requires_grad=True)
    scores = torch.empty(2, dtype=torch.float32, device="meta", requires_grad=True)
    capture = OpDispatchCapture(record_memory=True)
    capture.record_initial_memory_tensors((x, scores))
    scores_key = capture.initial_memory_tensors()[1].storage_key
    with PatchScope() as scope:
        assert install_meta_swiglu_autograd(scope) in {"native", "bridge_vendor_gradient_arity"}
        with capture, AutogradSavedTensorCapture():
            ascendc.swiglu_group_activation(x, 10.0, routed_scores=scores).sum().backward()
    slots = capture.autograd_saved_tensor_events()
    assert any(slot.tensor_id == capture.tensor_id(x) for slot in slots)
    assert any(slot.storage_key == scores_key for slot in slots)
    assert x.grad is not None and x.grad.shape == x.shape
    assert scores.grad is not None and scores.grad.shape == scores.shape
    assert any(event.raw_op_type == "cann_ops_nn.swiglu_group_backward.default" for event in capture.memory_events())
    assert ascendc.swiglu_group_activation is original


@pytest.mark.parametrize("selective", [False, True], ids=["full", "selective-quant-mm"])
def test_checkpoint_releases_forward_mx_codes_and_tracks_recomputed_operands(selective):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts
    from torchao_npu.ops.mx_ops import to_mx_then_mm
    from torchao_npu.quantization.quant_configs import MXQuantizeConfig
    from torchtitan_simulator.capture.checkpoint_execution import install_checkpoint_execution_tracking

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16, device="meta"))

        def forward(self, x):
            config = MXQuantizeConfig()
            return to_mx_then_mm(x, self.weight, config, config)

    kwargs = {}
    if selective:
        kwargs["context_fn"] = lambda: create_selective_checkpoint_contexts([torch.ops.npu.npu_quant_matmul.default])
    model = checkpoint_wrapper(Block(), early_stop=False, preserve_rng_state=False, **kwargs)
    install_checkpoint_execution_tracking([model])
    x = torch.empty(64, 64, dtype=torch.bfloat16, device="meta", requires_grad=True)
    phase = ["forward"]
    capture = OpDispatchCapture(record_memory=True, phase_provider=lambda: phase[0])
    capture.record_initial_memory_tensors(x)
    with capture, AutogradSavedTensorCapture():
        output = model(x)
        torch.empty(0, device="meta")
        gap = capture.memory_events()[-1].seq_idx
        phase[0] = "backward"
        output.sum().backward()
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(),
        model_parts=[model],
        initial_tensors=capture.initial_memory_tensors(),
        storage_lifetimes=capture.storage_lifetime_events(),
        checkpoint_boundary_events=capture.checkpoint_boundary_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
        counted_device_type="meta",
    )
    # HP input, master weight and returned HP result are each 64*64*2.
    # Original forward codes/scales must have gone; SAC keeps the same result.
    active = [item for item in plan.tensor_lifetimes if item.birth_seq <= gap <= item.death_seq]
    assert sum(item.resident_num_bytes for item in active) == 3 * 8192
    original_quant = [
        event
        for event in plan.raw_events
        if event.execution_kind == "original_forward" and "mx_quant" in event.raw_op_type
    ]
    assert len(original_quant) == 2
    keys = {ref.storage_key for event in original_quant for ref in event.outputs}
    assert all(item.death_seq < gap for item in plan.tensor_lifetimes if item.storage_key in keys)
    recomputed = [
        event for event in plan.raw_events if event.execution_kind == "recompute" and "mx_quant" in event.raw_op_type
    ]
    assert len(recomputed) == 2
    for event in recomputed:
        for ref in event.outputs:
            uses = [
                raw.seq_idx
                for raw in plan.raw_events
                if raw.execution_kind == "backward"
                and any(value.storage_key == ref.storage_key for value in raw.inputs)
            ]
            if uses:
                lifetime = next(item for item in plan.tensor_lifetimes if item.storage_key == ref.storage_key)
                assert lifetime.death_seq >= max(uses)
    assert x.grad is not None and x.grad.shape == x.shape
    assert model.weight.grad is not None and model.weight.grad.shape == model.weight.shape


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("topology", ["hsdp-ep", "cp-ep", "hsdp-ep-tp-cp"])
def test_native_mxfp8_parallel_ac_preserves_quantized_storage_gradients_and_graph(mode, topology):
    from collections import Counter

    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

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
            enable_quantized_training=True,
            num_mtp_modules=1,
            activation_checkpoint_mode=mode,
            global_batch_size=32,
        )
    )
    annotations = workload.iteration.schedule.annotations
    quantization = annotations["quantization_configuration"]
    assert quantization["enabled"] and quantization["recipe"] == "all_mxfp8"
    selected = quantization["quantized_parameters"]
    assert len(selected) == 56
    assert "layers.2.attention.indexer.wq_b.weight" in selected
    assert "mtp_layers.0.attention.wo_a.weight" in selected
    assert "mtp_layers.0.e_proj.weight" not in selected
    assert len(annotations["gradient_validation"]) == len(annotations["parameter_layouts"]) == 143
    memory = annotations["memory_plan"]
    quant_ops = [event for event in memory.raw_events if "mx_quant" in event.raw_op_type]
    assert quant_ops and all(event.op_type == "quantize" for event in quant_ops)
    slots = [slot for slot in memory.autograd_saved_tensors if slot.dtype in {"float8_e4m3fn", "uint8"}]
    assert slots
    for slot in slots:
        assert any(any(ref.storage_key == slot.storage_key for ref in event.outputs) for event in quant_ops)
        lifetime = next(item for item in memory.tensor_lifetimes if item.storage_key == slot.storage_key)
        assert lifetime.birth_seq <= slot.pack_seq <= slot.unpack_seq <= slot.release_seq <= lifetime.death_seq
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    visited = set()
    pending = set(nodes)
    while pending:
        ready = {key for key in pending if set(nodes[key].predecessors) <= visited}
        assert ready
        visited.update(ready)
        pending -= ready
    sources = Counter(node.annotations.get("source_op_id", node.op_id) for node in nodes.values())
    assert all(sources[event["op_id"]] == 1 for event in annotations["communication_events"])
    graph_for_source = {node.annotations.get("source_op_id", node.op_id): key for key, node in nodes.items()}
    pending_updates = [
        key for key, node in nodes.items() if node.annotations.get("raw_op_type") == "aten._fused_adamw_.default"
    ]
    assert pending_updates
    ancestors = set()
    while pending_updates:
        key = pending_updates.pop()
        if key not in ancestors:
            ancestors.add(key)
            pending_updates.extend(nodes[key].predecessors)
    assert all(
        graph_for_source[event["op_id"]] in ancestors
        for event in annotations["communication_events"]
        if event["gradient_reduction"]
    )
    markers = memory.fsdp_residency_events
    assert Counter(marker.group_id for marker in markers if marker.action == "alloc") == Counter(
        marker.group_id for marker in markers if marker.action == "free"
    )
    assert not any("open_at_capture_end" in item.reason for item in memory.tensor_lifetimes)
    peak = max(
        sum(item.resident_num_bytes for item in memory.tensor_lifetimes if item.birth_seq <= seq <= item.death_seq)
        for seq in {item.birth_seq for item in memory.tensor_lifetimes}
    )
    assert peak == memory.peak_active_bytes
    allocations = {
        f"fsdp_full_param:{marker.group_id}:{marker.seq_idx}": marker.allocation_storage_bytes
        for marker in markers
        if marker.action == "alloc"
    }
    keys = [
        key.partition(":")[0]
        for item in memory.tensor_lifetimes
        if item.resident_num_bytes and item.birth_seq <= memory.peak_seq_idx <= item.death_seq
        for key in ([item.storage_key] if item.storage_key else [key for key, _ in allocations[item.tensor_id]])
    ]
    assert len(keys) == len(set(keys))
    assert Counter(marker.transition_id for marker in markers if marker.action == "staging_free") == Counter(
        event["fsdp_transition_id"]
        for event in annotations["communication_events"]
        if event["comm_primitive"] == "allgather" and event["fsdp_transition_id"]
    )


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "preset,layers,dim,experts", [("flash", 43, 4096, 256), ("pro", 61, 7168, 384), ("pro_20t", 96, 12288, 2048)]
)
def test_presets_build_native_geometry_and_recipe_without_fqn_override(preset, layers, dim, experts, precision):
    from torchao_npu.wrapper_tensors.mx_wrapper_tensor import MXTrainingWeightWrapperTensor
    from torchtitan_simulator.config import SimulationConfig, build_meta_model, prepare_config
    from torchtitan_simulator.runtime import meta_runtime

    source = SimulationConfig(config=f"deepseek_v4_{preset}_baseline_{precision}", num_mtp_modules=1)
    prepared = prepare_config(source)
    assert source.enable_quantized_training is False
    assert prepared.simulation.enable_quantized_training == (precision == "mxfp8")
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        model = build_meta_model(prepared)
        assert len(model.layers) == layers and len(model.mtp_layers) == 1
        assert model.tok_embeddings.weight.shape == (129280, dim)
        weight = model.layers["0"].moe.routed_experts.inner_experts.w1_EFD
        assert weight.shape[0] == experts
        assert isinstance(weight, MXTrainingWeightWrapperTensor) == (precision == "mxfp8")
        assert all(parameter.device.type == "meta" for parameter in model.parameters())
