# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next attention modules.

Fork reason: Upstream torchtitan has no Qwen3-Next support. Qwen3-Next uses a
hybrid layout where most layers replace full attention with Gated DeltaNet
linear attention (see gated_deltanet.py), and the remaining full-attention
layers add an output gate on top of GQA.

Reference: MindSpeed-LLM
mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.protocols.module import Module


class RMSNormGated(nn.Module):
    """RMSNorm gated by SiLU activation: norm(x) * silu(gate).

    Used as the output norm of GatedDeltaNet. Note that this differs from
    Kimi K3's RMSNormGated which uses a sigmoid gate.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        return (x * F.silu(gate.float())).to(input_dtype)


class GatedAttention(nn.Module):
    """GQA with QK-Norm and an output gate (Qwen3-Next full attention layers).

    The query projection produces both the queries and a per-head output
    gate: ``attn_output *= sigmoid(gate)`` before the output projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 2048
        n_heads: int = 16
        n_kv_heads: int = 2
        head_dim: int = 256
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        # The query projection outputs queries and gates interleaved as two
        # halves of the head dimension: (n_heads * head_dim * 2).
        self.q_proj = Linear.Config(
            in_features=config.dim,
            out_features=config.n_heads * config.head_dim * 2,
            bias=False,
        ).build()
        self.k_proj = Linear.Config(
            in_features=config.dim,
            out_features=config.n_kv_heads * config.head_dim,
            bias=False,
        ).build()
        self.v_proj = Linear.Config(
            in_features=config.dim,
            out_features=config.n_kv_heads * config.head_dim,
            bias=False,
        ).build()
        self.o_proj = Linear.Config(
            in_features=config.n_heads * config.head_dim,
            out_features=config.dim,
            bias=False,
        ).build()

        # Head-local QK normalization (RMSNorm on head_dim).
        self.q_norm = RMSNorm.Config(
            normalized_shape=config.head_dim,
            eps=config.norm_eps,
        ).build()
        self.k_norm = RMSNorm.Config(
            normalized_shape=config.head_dim,
            eps=config.norm_eps,
        ).build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cache: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q_and_gate = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.n_heads, self.head_dim * 2
        )
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        query = self.q_norm(query)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, seq_len, self.n_kv_heads, self.head_dim
            )
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )

        query, key = apply_rotary_emb_cos_sin(query, key, rope_cache, positions)

        attn_output = self._attention_core(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        attn_output = (
            attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        )
        attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)

    def _attention_core(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Head-major GQA core. Replaced by the fused ``fusion_attention``
        virtual operator in simulator mode (see qwen3_next_converter)."""
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
