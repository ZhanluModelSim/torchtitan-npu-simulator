# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Official Attention Residual FP32 score order, including BF16 model inputs."""

import pytest
import torch

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_attention_residual_matches_official_score_order_output_and_all_gradients(dtype):
    config = kimi_k3_debugmodel().model_spec.model.output_attn_res
    with torch.random.fork_rng():
        torch.manual_seed(499)
        module = config.build().to(dtype=dtype)
        module.init_states()
        with torch.no_grad():
            module.norm.weight.copy_(torch.linspace(0.7, 1.3, 256))
            module.proj.weight.copy_(torch.randn(1, 256) * 0.1)
        prefix = torch.randn(3, 256, dtype=dtype, requires_grad=True)
        residual = torch.randn(3, 4, 256, dtype=dtype, requires_grad=True)
        dy = torch.randn_like(prefix)
    ref_prefix = prefix.detach().clone().requires_grad_()
    ref_residual = residual.detach().clone().requires_grad_()
    gamma = module.norm.weight.detach().clone().requires_grad_()
    projection = module.proj.weight.detach().clone().requires_grad_()

    actual = module(prefix, residual)
    # moonshotai/Kimi-K3 f831ab6, modeling_kimi_linear.py::_apply_attn_res.
    values = torch.cat((ref_residual, ref_prefix.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    variance = values_fp32.pow(2).mean(-1, keepdim=True)
    keys = values_fp32 * torch.rsqrt(variance + config.norm.eps)
    score_weight = gamma.float() * projection.squeeze(0).float()
    scores = (keys * score_weight).sum(-1)
    expected = torch.matmul(scores.softmax(-1).unsqueeze(1), values_fp32).squeeze(1).to(dtype)
    actual.backward(dy)
    expected.backward(dy)

    for left, right in (
        (actual, expected),
        (prefix.grad, ref_prefix.grad),
        (residual.grad, ref_residual.grad),
        (module.norm.weight.grad, gamma.grad),
        (module.proj.weight.grad, projection.grad),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
