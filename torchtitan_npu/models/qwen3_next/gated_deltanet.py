# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet linear attention for Qwen3-Next.

Fork reason: Upstream torchtitan has no Qwen3-Next support. Gated DeltaNet is
the O(n) token mixer used by 3 out of every 4 Qwen3-Next layers.

The core ``chunk_gated_delta_rule`` is a pure-torch chunked implementation
ported line-by-line from the MindSpeed-LLM eager fallback (itself aligned
with the FLA library's fallback path). It runs open on meta tensors so the
simulator captures the full internal aten op stream, matching reference
eager measurements.

Reference: MindSpeed-LLM
mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py
"""

import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .attention import RMSNormGated


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization aligned with the FLA library implementation."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Chunked gated delta rule (pure torch, meta-device friendly).

    Args:
        query: ``(B, S, H_K, K_DIM)`` queries.
        key: ``(B, S, H_K, K_DIM)`` keys.
        value: ``(B, S, H_V, V_DIM)`` values.
        g: ``(B, S, H_V)`` cumulative log decay gates.
        beta: ``(B, S, H_V)`` insertion strength in ``[0, 1]``.

    Returns:
        Core attention output ``(B, S, H_V, V_DIM)``.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(
        batch_size, num_heads, k_head_dim, v_head_dim
    ).to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
            mask, 0
        )
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2)
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet: gated delta rule linear attention token mixer.

    Projects hidden states into q/k/v/z/b/a, applies a causal depthwise
    convolution with SiLU activation, runs the chunked gated delta rule, then
    normalizes with an SiLU-gated RMSNorm before the output projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 2048
        num_v_heads: int = 32
        num_k_heads: int = 16
        head_k_dim: int = 128
        head_v_dim: int = 128
        conv_kernel_size: int = 4
        norm_eps: float = 1e-6
        gdn_chunk_size: int = 64

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_v_heads = config.num_v_heads
        self.num_k_heads = config.num_k_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gdn_chunk_size = config.gdn_chunk_size

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by "
                f"num_k_heads ({self.num_k_heads})"
            )

        # QKV + gate (z) fused projection, plus b/a projections.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = Linear.Config(
            in_features=config.dim,
            out_features=projection_size_qkvz,
            bias=False,
        ).build()
        self.in_proj_ba = Linear.Config(
            in_features=config.dim,
            out_features=projection_size_ba,
            bias=False,
        ).build()

        # Causal depthwise convolution over the concatenated q/k/v.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=self.conv_dim,
            padding=config.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.norm_eps)
        self.out_proj = Linear.Config(
            in_features=self.value_dim,
            out_features=config.dim,
            bias=False,
        ).build()

    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused qkvz/ba projections into q, k, v, z, b, a."""
        batch_size, seq_len = mixed_qkvz.shape[:2]
        ratio = self.num_v_heads // self.num_k_heads

        mixed_qkvz = mixed_qkvz.view(
            batch_size,
            seq_len,
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * ratio,
        )
        mixed_ba = mixed_ba.view(batch_size, seq_len, self.num_k_heads, 2 * ratio)

        query, key, value, z = torch.split(
            mixed_qkvz,
            [
                self.head_k_dim,
                self.head_k_dim,
                ratio * self.head_v_dim,
                ratio * self.head_v_dim,
            ],
            dim=3,
        )
        b, a = torch.split(mixed_ba, [ratio, ratio], dim=3)

        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cache: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del rope_cache, positions  # linear attention does not use RoPE
        # Under TP the mixer input is a replicated DTensor and the fused
        # projections produce per-rank head shards; the sharded GDN core
        # (local head counts) runs on plain local tensors, and the
        # row-parallel out_proj re-wraps the local output for the residual
        # stream.
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        batch_size, seq_len, _ = hidden_states.shape

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )

        # Causal depthwise conv over the concatenated q/k/v channels.
        mixed_qkv = torch.cat(
            (
                query.reshape(batch_size, seq_len, -1),
                key.reshape(batch_size, seq_len, -1),
                value.reshape(batch_size, seq_len, -1),
            ),
            dim=-1,
        ).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        # Compute in float32: in low precision, A_log.exp() may overflow to -inf.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        core_attn_out = self._chunk_gated_delta_rule(query, key, value, g, beta)

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    def _chunk_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Core gated delta rule computation.

        Runs the MindSpeed-aligned pure-torch chunked implementation so the
        internal aten ops (including the per-chunk loop) are captured as-is,
        matching the eager torch fallback used in reference measurements.
        """
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=self.gdn_chunk_size,
            use_qk_l2norm_in_kernel=True,
        )
