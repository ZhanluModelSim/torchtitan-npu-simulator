# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gate-B numerics: upstream GQAttention (GQA + QK-norm + half-split RoPE) vs a reference.

The fused v3 kernel has no CPU backend, so the inner attention runs a causal
SDPA shim; everything Qwen3-specific around the kernel -- the per-KV-group
interleaved fused QKV layout, per-head QK RMSNorm with learnable weight BEFORE
RoPE, the half-split rotary with float32 upcast, GQA broadcast and the output
fold, plus SwiGLU FFN -- is checked against this hand-written reference
following the HF Qwen3 conventions. The AscVarlenAttention TND packaging
itself is covered structurally by the capture tests (see docs/qwen3-integration.md).
"""

import torch
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import CosSinRoPE

B, S, DIM, HQ, HKV, HD = 2, 16, 256, 8, 2, 128
EPS = 1e-6
THETA = 1e6


class _CausalSDPAInner(torch.nn.Module):
    """CPU-runnable stand-in for the fused kernel inside GQAttention."""

    def forward(self, q, k, v, *, attention_masks=None, scale=None, enable_gqa=False, **kwargs):
        del attention_masks, kwargs
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2).contiguous()


def _reference_attention(x, qkv_weight, o_weight, q_gamma, k_gamma):
    """Independent GQA attention: fused QKV (per-KV-group interleave), QK RMSNorm, half-split RoPE."""
    hpk = HQ // HKV
    r_dim = hpk + 2
    qkv = (x @ qkv_weight.t()).view(B, S, HKV, r_dim, HD)
    q = qkv[..., :hpk, :].reshape(B, S, HQ, HD)
    k = qkv[..., hpk, :]
    v = qkv[..., hpk + 1, :]

    def head_norm(t, gamma):
        return (t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + EPS)) * gamma

    q, k = head_norm(q, q_gamma), head_norm(k, k_gamma)

    pos = torch.arange(S, dtype=torch.float32)
    inv = 1.0 / THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD)
    angles = torch.outer(pos, inv)  # (S, HD/2)
    cos_full = torch.cat([angles.cos(), angles.cos()], -1).view(1, S, 1, HD)
    sin_full = torch.cat([angles.sin(), angles.sin()], -1).view(1, S, 1, HD)

    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat((-t[..., half:], t[..., :half]), -1)

    def apply_rope(t):
        return t * cos_full + rotate_half(t) * sin_full

    q, k = apply_rope(q), apply_rope(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    groups = HQ // HKV
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    scores = q @ k.transpose(-2, -1) * (HD**-0.5)
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out = scores.masked_fill(causal, float("-inf")).softmax(-1) @ v
    out = out.transpose(1, 2).reshape(B, S, HQ * HD)
    return out @ o_weight.t()


def _build_modules():
    torch.manual_seed(42)
    attention = make_gqa_config(
        dim=DIM,
        n_heads=HQ,
        n_kv_heads=HKV,
        head_dim=HD,
        wqkv_param_init={"weight": torch.nn.init.normal_},
        wo_param_init={"weight": torch.nn.init.normal_},
        inner_attention=VarlenAttention.Config(),
        fuse_qkv=True,
        rope=CosSinRoPE.Config(dim=HD, max_seq_len=4096, theta=THETA),
        qk_norm=RMSNorm.Config(normalized_shape=HD, eps=EPS),
    ).build()
    attention.inner_attention = _CausalSDPAInner()
    ffn = make_ffn_config(
        dim=DIM,
        hidden_dim=3 * DIM,
        w1_param_init={"weight": torch.nn.init.normal_},
        w2w3_param_init={"weight": torch.nn.init.normal_},
    ).build()
    with torch.no_grad():
        for module in (attention, ffn):
            for p in module.parameters():
                p.uniform_(-0.02, 0.02)
    return attention, ffn


def test_qwen3_swiglu_ffn_matches_reference():
    attention, ffn = _build_modules()
    del attention
    torch.manual_seed(7)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        w1, w3, w2 = ffn.w1.weight, ffn.w3.weight, ffn.w2.weight
        expected = (torch.nn.functional.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        assert (ffn(x) - expected).abs().max().item() < 1e-5


def test_qwen3_gqa_qk_norm_rope_matches_reference():
    attention, _ = _build_modules()
    torch.manual_seed(7)
    x = torch.randn(B, S, DIM)
    positions = torch.arange(S).expand(B, -1).contiguous()
    with torch.no_grad():
        got = attention(x, None, positions)
        expected = _reference_attention(
            x,
            attention.qkv_linear.wqkv.weight,
            attention.wo.weight,
            attention.q_norm.weight,
            attention.k_norm.weight,
        )
    assert (got - expected).abs().max().item() < 1e-5
