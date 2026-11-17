# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of Kimi's bounded activation and native MoE composition."""

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention, KimiGatedMLA
from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel, kimi_k3_full


def _reference_activation(gate, up, beta, linear_beta):
    # Moonshot f831ab6, modeling_kimi_linear.py, SituAndMul.
    gate, up = gate.float(), up.float()
    value = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return value * up


@pytest.mark.parametrize("linear_beta", [None, 25.0], ids=["unbounded-up", "bounded-up"])
def test_dense_config_matches_official_situ_output_and_gradients(linear_beta):
    config = kimi_k3_debugmodel().model_spec.model.layers[0].feed_forward
    config.linear_beta = linear_beta
    with torch.random.fork_rng():
        torch.manual_seed(173)
        module = config.build()
        module.init_states()
        reference = copy.deepcopy(module)
        x = (torch.randn(2, 3, 256) * 2).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()

    actual = module(x)
    gate = F.linear(ref_x, reference.w1.weight)
    up = F.linear(ref_x, reference.w3.weight)
    expected = F.linear(_reference_activation(gate, up, config.beta, linear_beta), reference.w2.weight)
    actual.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-7, rtol=1e-4)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=1e-7, rtol=1e-4)


def test_latent_moe_native_dispatch_matches_tokenwise_post_projection_reference(monkeypatch):
    config = kimi_k3_debugmodel().model_spec.model.layers[1].moe
    config.router._debug_force_load_balance = False
    with torch.random.fork_rng():
        torch.manual_seed(197)
        module = config.build()
        module.init_states()
        reference = copy.deepcopy(module)
        x = torch.randn(1, 4, 256, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()

    # CPU supports the projections; this seam implements grouped-MM using
    # per-expert matmul, retaining the production BF16 conversions and gradients.
    def cpu_grouped_mm(*, A, B_t, offs):
        starts = [0, *offs.tolist()]
        return torch.cat([A[starts[e] : starts[e + 1]] @ B_t[e] for e in range(B_t.shape[0])])

    monkeypatch.setattr(module.routed_experts.inner_experts, "_grouped_mm", cpu_grouped_mm)
    actual = module(x)

    scores = torch.sigmoid(F.linear(ref_x.float(), reference.router.gate.weight.float()))
    ids = (scores + reference.expert_bias_E).topk(config.router.top_k, dim=-1, sorted=False).indices
    selected = scores.gather(-1, ids)
    selected = selected / (selected.sum(-1, keepdim=True) + 1e-20) * config.router.route_scale
    latent = F.linear(ref_x, reference.down_proj.weight)
    experts = reference.routed_experts.inner_experts
    rows = []
    for token in range(x.shape[1]):
        contributions = []
        for slot in range(ids.shape[-1]):
            e = ids[0, token, slot].item()
            inp = latent[0, token].bfloat16()
            gate = F.linear(inp, experts.w1_EFD[e].bfloat16())
            up = F.linear(inp, experts.w3_EFD[e].bfloat16())
            hidden = _reference_activation(
                gate, up, config.routed_experts.inner_experts.beta, config.routed_experts.inner_experts.linear_beta
            ).bfloat16()
            out = F.linear(hidden, experts.w2_EDF[e].bfloat16()).float()
            contributions.append(out * selected[0, token, slot])
        rows.append(torch.stack(contributions).sum(0))
    combined = torch.stack(rows).unsqueeze(0)
    combined = F.rms_norm(combined, (combined.shape[-1],), reference.norm.weight, reference.norm.eps)
    routed = F.linear(combined, reference.up_proj.weight)
    shared = reference.shared_experts
    gate, up = F.linear(ref_x, shared.w1.weight), F.linear(ref_x, shared.w3.weight)
    expected = routed + F.linear(
        _reference_activation(gate, up, shared.act_fn.beta, shared.act_fn.linear_beta), shared.w2.weight
    )
    actual.square().mean().backward()
    expected.square().mean().backward()

    assert not module.routed_experts.token_dispatcher.absorb_router_scores
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=5e-4, rtol=0.03)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=8e-4, rtol=0.05, msg=name)


def test_full_config_matches_frozen_official_text_geometry():
    # moonshotai/Kimi-K3 f831ab6 config.json, text_config.
    config = kimi_k3_full().model_spec.model
    assert (config.dim, config.vocab_size, len(config.layers), config.attn_res_block_size) == (7168, 163840, 93, 12)
    kda = [i for i, layer in enumerate(config.layers) if isinstance(layer.attention, KimiDeltaAttention.Config)]
    mla = [i for i, layer in enumerate(config.layers) if isinstance(layer.attention, KimiGatedMLA.Config)]
    assert mla == [*range(3, 92, 4), 92]
    assert len(kda) == 69 and len(mla) == 24
    assert config.layers[0].feed_forward.w1.out_features == 33792
    for layer in config.layers[1:]:
        moe = layer.moe
        assert (moe.num_experts, moe.router.top_k) == (896, 16)
        assert (moe.down_proj.in_features, moe.down_proj.out_features) == (7168, 3584)
        assert (moe.routed_experts.inner_experts.dim, moe.routed_experts.inner_experts.hidden_dim) == (3584, 3072)
        assert moe.shared_experts.w1.out_features == 6144
        assert moe.routed_experts.inner_experts.beta == 4.0
        assert moe.routed_experts.inner_experts.linear_beta == 25.0
