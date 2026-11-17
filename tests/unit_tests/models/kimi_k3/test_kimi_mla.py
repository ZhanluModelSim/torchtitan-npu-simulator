# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi MLA projections, low-rank normalization and padded attention boundary."""

import copy
import math

import torch
import torch.nn.functional as F

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


def test_mla_matches_official_eager_attention_without_value_padding_and_all_gradients():
    config = kimi_k3_debugmodel().model_spec.model.layers[3].attention
    with torch.random.fork_rng():
        torch.manual_seed(557)
        module = config.build()
        module.init_states()
        reference = copy.deepcopy(module)
        x = (torch.randn(2, 8, 256) * 0.01).requires_grad_()
        dy = torch.randn_like(x)
    ref_x = x.detach().clone().requires_grad_()

    actual = module(x)

    # Official KimiMLAAttention constructs these two KimiRMSNorms with
    # the default 1e-6, even though the text config's global epsilon is 1e-5.
    def norm(value, weight):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return weight * normalized.to(value.dtype)

    q_low = norm(F.linear(ref_x, reference.q_a_proj.weight), reference.q_a_layernorm.weight)
    q = F.linear(q_low, reference.q_b_proj.weight).reshape(2, 8, 8, 32).transpose(1, 2)
    compressed = F.linear(ref_x, reference.kv_a_proj_with_mqa.weight)
    latent, key_extra = compressed.split((64, 16), dim=-1)
    latent = norm(latent, reference.kv_a_layernorm.weight)
    kv = F.linear(latent, reference.kv_b_proj.weight).reshape(2, 8, 8, 32).transpose(1, 2)
    key_base, value = kv.split((16, 16), dim=-1)
    key_extra = key_extra.unsqueeze(1).expand(2, 8, 8, 16)
    key = torch.cat((key_base, key_extra), dim=-1)
    scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(32)
    mask = torch.ones(8, 8, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    output = torch.matmul(scores.float().softmax(-1).to(q.dtype), value)
    output = output.transpose(1, 2).reshape(2, 8, 128)
    output = output * F.linear(ref_x, reference.g_proj.weight).sigmoid()
    expected = F.linear(output, reference.o_proj.weight)
    actual.backward(dy)
    expected.backward(dy)

    assert module.q_a_layernorm.eps == module.kv_a_layernorm.eps == 1e-6
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-5, atol=2e-6)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=1e-5, atol=2e-6, msg=name)
