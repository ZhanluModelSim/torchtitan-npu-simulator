# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One NPU: real vision QKV/RoPE layouts, joint FX and variable patch replay."""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.decorators import mark_unbacked
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced
from torchtitan.models.common.linear import Linear

import torchtitan_npu  # noqa: F401
from torchtitan_npu.models.deepseek_v4_1.vision.encoder import VisionAttention, _vision_rope
from torchtitan_npu.override.deepseek_v4_1.vision import AscUnmaskedVisionSDPA


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_vision_fused_attention_joint_graph_replays_variable_patches(dtype):
    if not torch.npu.is_available():
        pytest.skip("Requires one NPU")
    torch.npu.set_device(0)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(42)
        attention = VisionAttention.Config(dim=1024, num_heads=16).build(unmasked_sdpa=AscUnmaskedVisionSDPA.Config())
        attention.init_states(buffer_device=torch.device("cpu"))
    for module in attention.modules():
        if isinstance(module, Linear):
            module.inplace_wgrad_accum = False
    attention.requires_grad_(False)
    attention = attention.to(device="npu", dtype=dtype)

    def joint(x, cos, sin, valid):
        output = attention(x, cos, sin, valid)
        gradient = torch.autograd.grad(output.float().sum(), x)[0]
        return output, gradient

    runner = None
    for height, width in ((39, 39), (37, 55)):
        patches = height * width
        generator = torch.Generator().manual_seed(42 + patches)
        x = torch.randn(patches, 1024, generator=generator).to(device="npu", dtype=dtype).requires_grad_()
        cos, sin = _vision_rope(height, width, 32, 10000.0, x.device)
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        valid = torch.ones(patches, dtype=torch.bool, device=x.device)
        captured = []

        def capture_attention(_module, inputs, output, captured=captured):
            captured.append((*inputs, output))

        eager_x = x.detach().requires_grad_()
        with attention.unmasked_sdpa.register_forward_hook(capture_attention):
            eager_output = attention(eager_x, cos, sin, valid)
        q, k, v, fused_output = captured[0]
        assert all(not value.is_contiguous() for value in (q, k, v))
        assert q.stride(-2) == k.stride(-2) == attention.num_heads * attention.head_dim
        assert v.stride(-2) == 3 * attention.num_heads * attention.head_dim
        eager_gradient = torch.autograd.grad(eager_output.float().sum(), eager_x, retain_graph=True)[0]
        fused_gradients = torch.autograd.grad(fused_output.float().sum(), (q, k, v))

        reference_qkv = tuple(value.detach().cpu().float().requires_grad_() for value in (q, k, v))
        reference_output = F.scaled_dot_product_attention(*reference_qkv, dropout_p=0.0)
        reference_gradients = torch.autograd.grad(reference_output.sum(), reference_qkv)
        output_atol, output_rtol = (0.002, 0.02) if dtype == torch.bfloat16 else (1e-4, 1e-3)
        grad_atol, grad_rtol = (0.003, 0.03) if dtype == torch.bfloat16 else (1e-4, 1e-3)
        torch.testing.assert_close(fused_output.cpu().float(), reference_output, atol=output_atol, rtol=output_rtol)
        for actual, expected in zip(fused_gradients, reference_gradients, strict=True):
            torch.testing.assert_close(actual.cpu().float(), expected, atol=grad_atol, rtol=grad_rtol)
        for value in (x, cos, sin, valid):
            mark_unbacked(value, 0)
        if runner is None:
            traced = minimal_fx_tracer(joint, module=attention)(x, cos, sin, valid)
            targets = {str(node.target) for node in traced.gm.graph.nodes if node.op == "call_function"}
            assert "npu.npu_fusion_attention.default" in targets
            assert "npu.npu_fusion_attention_grad.default" in targets
            assert "npu.npu_fusion_attention_v3.default" not in targets
            assert "npu.npu_fusion_attention_grad_v3.default" not in targets
            runner = run_traced(traced, module=attention)
        actual_output, actual_gradient = runner(x, cos, sin, valid)
        assert actual_output.dtype == actual_gradient.dtype == dtype
        torch.testing.assert_close(actual_output, eager_output, atol=0, rtol=0)
        # Allow BF16 accumulation rounding in the full-layer backward; the
        # fused kernel's three input gradients are checked separately above.
        torch.testing.assert_close(actual_gradient, eager_gradient, atol=grad_atol, rtol=grad_rtol)
        torch.npu.synchronize()
