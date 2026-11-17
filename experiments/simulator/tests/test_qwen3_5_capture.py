# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 text-capture contracts: virtual GDN/flex fusions and memory plans."""

from collections import Counter

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.compat.qwen3_5 import flex_attention, gated_delta_rule
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Native qwen35_debugmodel geometry, text-only (vision encoder detached):
# dim=256, vocab=248320, 8 layers (6 GDN + 2 full attention at ids 3/7),
# n_heads=4, n_kv_heads=2, head_dim=64, hidden=512 (gated w1/w2/w3),
# GDN key_dim=128, value_dim=256, conv kernel=4, FP32 parameters.
_D, _V, _HID = 256, 248320, 512
_KD, _VD, _KVH = 128, 256, 4
_GDN_LAYER = (
    2 * (_KD * _D)  # in_proj_q/k
    + 3 * (_VD * _D)  # in_proj_v/z + out_proj
    + 2 * (_KVH * _D)  # in_proj_a/b
    + 2 * _KD * 4  # conv_q/k depthwise
    + _VD * 4  # conv_v
    + 2 * _KVH  # A_log + dt_bias
    + 64  # RMSNormGated
    + 2 * _D  # attention_norm + ffn_norm
    + 3 * _HID * _D  # w1/w2/w3
)
_FULL_LAYER = (
    _HID * _D  # wq: out = 2 * n_heads * head_dim = 512
    + 2 * (2 * 64 * _D)  # wk + wv: n_kv_heads * head_dim = 128
    + _D * _D  # wo
    + 2 * 64  # q_norm + k_norm
    + 2 * _D  # attention_norm + ffn_norm
    + 3 * _HID * _D  # w1/w2/w3
)
_TOTAL_PARAMS = 2 * _V * _D + _D + 6 * _GDN_LAYER + 2 * _FULL_LAYER
# GDN layer: q/k/v/z/a/b proj + out_proj + 3 convs + A_log + dt_bias + gated
# norm + 2 block norms + w1/w2/w3 = 18; full layer: qkv/wo + 2 head norms +
# 2 block norms + w1/w2/w3 = 11; plus embeddings, head and final norm.
_TOTAL_PARAM_TENSORS = 6 * 18 + 2 * 11 + 3


def test_qwen3_5_param_formula_matches_native_geometry():
    assert _TOTAL_PARAMS == 132_412_336
    assert _TOTAL_PARAM_TENSORS == 133


@pytest.mark.parametrize(
    "world,replicate,rank,ac",
    [
        (1, 1, 0, "none"),
        (2, 1, 0, "none"),
        (2, 1, 1, "none"),
        (4, 2, 0, "none"),
        (1, 1, 0, "full"),
        (1, 1, 0, "selective"),
    ],
    ids=["single", "fsdp-first", "fsdp-last", "hsdp", "full-ac", "selective-ac"],
)
def test_qwen3_5_capture_reaches_iteration_with_virtual_gdn_and_flex_boundaries(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="qwen3_5_debugmodel",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            data_parallel_replicate_degree=replicate,
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    for op in (
        "qwen3_5_sim.gated_delta_rule.default",
        "qwen3_5_sim.gated_delta_rule_grad.default",
        "qwen3_5_sim.flex_attention.default",
        "qwen3_5_sim.flex_attention_grad.default",
    ):
        assert counts[op] > 0, op
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS
    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    local_params = _TOTAL_PARAMS if degree == 1 else _TOTAL_PARAMS // degree
    assert memory.persistent_param_bytes == local_params * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter.
    assert memory.optimizer_state_bytes == local_params * 8 + _TOTAL_PARAM_TENSORS * 4
    custom = [n for n in nodes if "tensor_semantics" in n.attrs]
    assert {n.annotations["raw_op_type"] for n in custom} == {
        "qwen3_5_sim.gated_delta_rule.default",
        "qwen3_5_sim.gated_delta_rule_grad.default",
        "qwen3_5_sim.flex_attention.default",
        "qwen3_5_sim.flex_attention_grad.default",
    }
    assert all(
        "tensor_semantics" not in n.attrs
        for n in nodes
        if n.annotations["raw_op_type"].startswith(("aten.", "npu.", "cann_ops_transformer."))
    )
    source_nodes = {
        node.op_id: node
        for graph in workload.captured_workload.step_templates.values()
        for node in graph.nodes.values()
    }
    for node in custom:
        assert node.attrs["tensor_semantics"] == source_nodes[node.op_id].attrs["tensor_semantics"]
        for key, tensors in (("inputs", node.inputs), ("outputs", node.outputs)):
            assert [item["index"] for item in node.attrs["tensor_semantics"][key]] == list(range(len(tensors)))
            for item in node.attrs["tensor_semantics"][key]:
                assert len(item["axes"]) == len(tensors[item["index"]].shape)
    gdn_forward = next(
        n
        for n in custom
        if n.annotations["raw_op_type"] == "qwen3_5_sim.gated_delta_rule.default"
    )
    roles = [item["role"] for item in gdn_forward.attrs["tensor_semantics"]["inputs"]]
    # The fixed-length path still materializes a per-document reset mask from
    # cu_seqlens; the direct-dispatch test covers the absent-reset branch.
    assert roles == ["query", "key", "value", "decay_gate", "update_gate", "state_reset_mask"]
    assert tuple(gdn_forward.inputs[5].shape) == (1, 16)


@pytest.mark.parametrize(
    "op,backward_name,shapes,attrs",
    [
        (
            gated_delta_rule,
            "gated_delta_rule_grad",
            [(1, 8, 2, 64), (1, 8, 2, 64), (1, 8, 2, 64), (1, 8, 2), (1, 8, 2)],
            (None,),
        ),
        (flex_attention, "flex_attention_grad", [(1, 4, 8, 64), (1, 2, 8, 64), (1, 2, 8, 64)], (0.125, True)),
    ],
    ids=["gdn", "flex"],
)
def test_qwen3_5_virtual_fusion_saves_actual_inputs_and_preserves_backward_identity(op, backward_name, shapes, attrs):
    leaves = [torch.empty(shape, device="meta", requires_grad=True) for shape in shapes]
    capture = OpDispatchCapture()
    with capture, AutogradSavedTensorCapture():
        inputs = [leaf.sin() for leaf in leaves]
        output = op(*inputs, *attrs)
        dy = torch.empty_like(output).sin()
        output.backward(dy)
    saved = [value for value in inputs if isinstance(value, torch.Tensor)]
    for leaf, value in zip(leaves, inputs, strict=True):
        assert leaf.grad is not None and leaf.grad.shape == value.shape
        assert leaf.grad.dtype == value.dtype
    forward_name = str(op._opoverload)
    backward_raw = f"qwen3_5_sim.{backward_name}.default"
    memory = capture.memory_events()
    forward = next(event for event in memory if event.raw_op_type == forward_name)
    backward = next(event for event in memory if event.raw_op_type == backward_raw)
    expected_ids = [capture.tensor_id(value) for value in saved]
    assert [t.tensor_id for t in forward.inputs if t.tensor_id in expected_ids] == expected_ids
    backward_inputs = [t.alias_of if t.alias_of is not None else t.tensor_id for t in backward.inputs]
    for value_id in expected_ids:
        assert value_id in backward_inputs
    assert backward.inputs[-1].tensor_id == capture.tensor_id(dy)
    assert [tuple(t.shape) for t in backward.outputs[: len(saved)]] == [tuple(t.shape) for t in saved]
    assert [t.dtype for t in backward.outputs[: len(saved)]] == [str(t.dtype).removeprefix("torch.") for t in saved]


def test_qwen3_5_fusion_semantics_declare_roles_and_skip_absent_reset():
    import json

    query = torch.empty(1, 8, 2, 64, device="meta", requires_grad=True)
    key = torch.empty(1, 8, 2, 64, device="meta", requires_grad=True)
    value = torch.empty(1, 8, 2, 64, device="meta", requires_grad=True)
    gate = torch.empty(1, 8, 2, device="meta", requires_grad=True)
    beta = torch.empty(1, 8, 2, device="meta", requires_grad=True)
    capture = OpDispatchCapture()
    with capture:
        output = gated_delta_rule(query.sin(), key.sin(), value.sin(), gate.sin(), beta.sin(), None)
        output.backward(torch.empty_like(output))
        q = torch.empty(1, 4, 8, 64, device="meta", requires_grad=True)
        k = torch.empty(1, 2, 8, 64, device="meta", requires_grad=True)
        v = torch.empty(1, 2, 8, 64, device="meta", requires_grad=True)
        flex_out = flex_attention(q.sin(), k.sin(), v.sin(), 0.125, True)
        flex_out.backward(torch.empty_like(flex_out))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    gdn = nodes["qwen3_5_sim.gated_delta_rule.default"]
    gdn_backward = nodes["qwen3_5_sim.gated_delta_rule_grad.default"]
    expected_gdn_inputs = (
        {"index": 0, "role": "query", "axes": ("batch", "sequence", "head", "key_head_dim")},
        {"index": 1, "role": "key", "axes": ("batch", "sequence", "head", "key_head_dim")},
        {"index": 2, "role": "value", "axes": ("batch", "sequence", "head", "value_head_dim")},
        {"index": 3, "role": "decay_gate", "axes": ("batch", "sequence", "head")},
        {"index": 4, "role": "update_gate", "axes": ("batch", "sequence", "head")},
    )
    assert gdn.attrs["tensor_semantics"] == {
        "version": 1,
        "inputs": expected_gdn_inputs,
        "outputs": ({"index": 0, "role": "attention_output", "axes": ("batch", "sequence", "head", "value_head_dim")},),
    }
    assert gdn_backward.attrs["tensor_semantics"]["inputs"] == (
        *expected_gdn_inputs,
        {"index": 5, "role": "output_gradient", "axes": ("batch", "sequence", "head", "value_head_dim")},
    )
    assert [item["role"] for item in gdn_backward.attrs["tensor_semantics"]["outputs"]] == [
        "query_gradient",
        "key_gradient",
        "value_gradient",
        "decay_gate_gradient",
        "update_gate_gradient",
    ]
    flex = nodes["qwen3_5_sim.flex_attention.default"]
    flex_backward = nodes["qwen3_5_sim.flex_attention_grad.default"]
    expected_flex_inputs = (
        {"index": 0, "role": "query", "axes": ("batch", "head", "q_len", "head_dim")},
        {"index": 1, "role": "key", "axes": ("batch", "head", "kv_len", "head_dim")},
        {"index": 2, "role": "value", "axes": ("batch", "head", "kv_len", "head_dim")},
    )
    assert flex.attrs["tensor_semantics"]["inputs"] == expected_flex_inputs
    assert flex.attrs["tensor_semantics"]["outputs"] == (
        {"index": 0, "role": "attention_output", "axes": ("batch", "head", "q_len", "head_dim")},
    )
    assert [item["role"] for item in flex_backward.attrs["tensor_semantics"]["outputs"]] == [
        "query_gradient",
        "key_gradient",
        "value_gradient",
    ]
    assert all(
        "tensor_semantics" not in node.attrs for raw, node in nodes.items() if raw.startswith("aten.")
    )
    json.dumps(gdn.attrs["tensor_semantics"], allow_nan=False)
    json.dumps(flex.attrs["tensor_semantics"], allow_nan=False)


def test_qwen3_5_default_overrides_reach_gdn_kernel_and_moe_dispatcher():
    from torchtitan.models.common.linear import Linear
    from torchtitan_simulator.config import build_meta_model, prepare_config
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.models.qwen3_5._fla_compat import ensure_qwen3_5_importable

    ensure_qwen3_5_importable()
    from torchtitan.models.qwen3_5.model import GatedDeltaKernel

    from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher
    from torchtitan_npu.override.qwen3_5.gated_delta import TritonGatedDeltaKernel

    prepared = prepare_config(
        SimulationConfig(
            config="qwen3_5_debugmodel_moe",
            world_size=2,
            data_parallel_shard_degree=2,
            expert_parallel_degree=2,
        )
    )
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        model = build_meta_model(prepared)
    configured = {fqn for fqn, _, _, _ in prepared.trainer.model_spec.model.traverse(Linear.Config)}
    # Structural config→module renames in Qwen35TransformerBlock/VisionEncoder
    # (config fields `attention`/`delta_net` both build as module `attn`), and
    # the vision tower is detached by the text-capture parallelizer.
    configured = {
        fqn.replace(".delta_net.", ".attn.").replace(".attention.", ".attn.")
        for fqn in configured
        if not fqn.startswith("vision_encoder")
    }
    actual = {
        fqn
        for fqn, module in model.named_modules()
        if isinstance(module, Linear) and not fqn.startswith("vision_encoder")
    }
    assert configured == actual
    kernels = [m for m in model.modules() if isinstance(m, GatedDeltaKernel)]
    assert kernels and all(isinstance(m, TritonGatedDeltaKernel) for m in kernels)
    # The qwen3.5 MoE block constructs its dispatcher inside the routed
    # experts module, so the override contract is asserted on the derived
    # config tree rather than on built submodule types.
    model_config = prepared.trainer.model_spec.model
    dispatcher_cfgs = [
        layer.moe.routed_experts.token_dispatcher for layer in model_config.layers
    ]
    assert dispatcher_cfgs and all(
        isinstance(cfg, AscAllToAllTokenDispatcher.Config) for cfg in dispatcher_cfgs
    )


def test_qwen3_5_rejects_mtp_modules():
    with pytest.raises(ValueError, match="no MTP"):
        simulate(SimulationConfig(config="qwen3_5_debugmodel", num_mtp_modules=1))
