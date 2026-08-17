# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.converters.kernels.kimi_k3_moe import (
    NpuKimiRMSNormGated,
)
from torchtitan_npu.models.kimi_k3.attention import (
    KimiDeltaAttention,
    KimiGatedMLA,
    RMSNormGated,
)
from torchtitan_npu.models.kimi_k3.feed_forward import KimiMLP
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.kda_converter import (
    apply_kimi_k3_shims,
)


def _build_attention() -> KimiDeltaAttention:
    return KimiDeltaAttention(
        KimiDeltaAttention.Config(dim=32, num_heads=2, head_dim=16)
    )


def test_kda_binding_preserves_module_identity_and_hooks():
    attention = _build_attention()
    hook = attention.register_forward_hook(lambda module, args, output: None)
    module_id = id(attention)
    hook_ids = set(attention._forward_hooks)

    apply_kimi_k3_shims(attention)
    apply_kimi_k3_shims(attention)

    assert id(attention) == module_id
    assert set(attention._forward_hooks) == hook_ids
    assert attention._simulator_kda_shim_installed is True
    hook.remove()


def test_kda_core_records_only_fused_forward_and_backward_nodes():
    attention = _build_attention()
    apply_kimi_k3_shims(attention)
    tensors = [
        torch.empty(
            1,
            8,
            2,
            16,
            device="meta",
            requires_grad=True,
        )
        for _ in range(5)
    ]
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = attention._chunk_kda(*tensors)
        phase["value"] = "backward"
        output.sum().backward()

    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    assert raw_names.count("triton_ascend_kernels.chunk_kda") == 1
    assert raw_names.count("triton_ascend_kernels.chunk_kda_grad") == 1
    assert "aten.empty_like.default" not in raw_names
    assert output.shape == tensors[2].shape


def test_kimi_gated_rms_norm_uses_new_module_level_autograd_shim():
    module = NpuKimiRMSNormGated(RMSNormGated(16))
    apply_kimi_k3_shims(module)
    apply_kimi_k3_shims(module)
    x = torch.empty((2, 3, 16), device="meta", requires_grad=True)
    gate = torch.empty_like(x, requires_grad=True)
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = module(x, gate)
        phase["value"] = "backward"
        output.sum().backward()

    raw_names = [
        node.annotations["raw_op_type"]
        for node in capture.build_nodes().values()
    ]
    assert raw_names.count("npu.npu_rms_norm.default") == 1
    assert raw_names.count("npu.npu_rms_norm_backward.default") == 1
    assert module._simulator_gated_rms_norm_shim_installed is True
    assert x.grad is not None
    assert gate.grad is not None


def test_kimi_gated_mla_records_one_virtual_fused_op_per_pass():
    module = KimiGatedMLA(
        KimiGatedMLA.Config(
            dim=16,
            n_heads=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
        )
    ).to("meta")
    apply_kimi_k3_shims(module)
    x = torch.empty((2, 3, 16), device="meta", requires_grad=True)
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = module(x)
        phase["value"] = "backward"
        output.sum().backward()

    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    assert raw_names.count("fusion_attention") == 1
    assert raw_names.count("fusion_attention_gard") == 1
    assert "aten.scaled_dot_product_attention.default" not in raw_names
    assert raw_names.count("aten.mm.default") >= 6
    assert "aten.sigmoid.default" in raw_names
    assert module._simulator_mla_shim_installed is True
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())
    fused_forward = next(node for node in nodes if node.annotations["raw_op_type"] == "fusion_attention")
    fused_backward = next(node for node in nodes if node.annotations["raw_op_type"] == "fusion_attention_gard")
    assert fused_forward.op_type == fused_backward.op_type == "fusion_attention"
    assert [meta.shape for meta in fused_forward.inputs] == [(2, 2, 3, 6)] * 3
    assert [meta.shape for meta in fused_forward.outputs] == [(2, 2, 3, 6)]
    assert fused_forward.attrs == fused_backward.attrs == {"num_heads": 2, "layout": "BNSD"}


def test_kimi_shared_expert_situ_glu_records_one_fused_activation_per_pass():
    module = KimiMLP(KimiMLP.Config(hidden_size=16, intermediate_size=32)).to("meta")
    apply_kimi_k3_shims(module)
    x = torch.empty((2, 3, 16), device="meta", requires_grad=True)
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = module(x)
        phase["value"] = "backward"
        output.sum().backward()

    raw_names = [node.annotations["raw_op_type"] for node in capture.build_nodes().values()]
    assert raw_names.count("situ_glu") == 1
    assert raw_names.count("situ_glu_backward") == 1
    assert "aten.tanh.default" not in raw_names
    assert "aten.sigmoid.default" not in raw_names
    assert module._simulator_mlp_shim_installed is True
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())
