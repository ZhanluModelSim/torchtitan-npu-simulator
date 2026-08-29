# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 attention modules: KDA (Kimi Delta Attention) and Gated MLA.

Fork reason: Upstream torchtitan has no K3 support. KDA is a novel linear
attention mechanism (delta rule) not present in any upstream model.
Reference: MindSpeed-MM mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module


class ShortConvolution(nn.Module):
    """Causal short convolution with optional activation (kernel_size=4 for KDA)."""

    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        x_t = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        weight = self.conv.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        x_t = F.conv1d(
            x_t,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=x_t.shape[1],
        )[..., : x.shape[1]]
        x = x_t.transpose(1, 2)
        if self.activation == "silu":
            x = F.silu(x)
        return x


class RMSNormGated(nn.Module):
    """RMSNorm with sigmoid gating: norm(x) * sigmoid(gate)."""

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
        return x * F.sigmoid(gate.float()).to(input_dtype)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention (KDA) — linear attention via delta rule.

    69 of 93 layers in Kimi K3 use this O(n) attention variant.
    Core computation: chunk_kda from triton-ascend-kernels (fused) or naive fallback.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 7168
        num_heads: int = 96
        head_dim: int = 128
        conv_kernel_size: int = 4
        gate_lower_bound: float | None = -5.0
        use_full_rank_gate: bool = True
        norm_eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        projection_size = self.num_heads * self.head_dim

        self.q_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=projection_size,
            bias=False,
        ).build()
        self.k_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=projection_size,
            bias=False,
        ).build()
        self.v_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=projection_size,
            bias=False,
        ).build()

        self.q_conv1d = ShortConvolution(projection_size, config.conv_kernel_size, activation="silu")
        self.k_conv1d = ShortConvolution(projection_size, config.conv_kernel_size, activation="silu")
        self.v_conv1d = ShortConvolution(projection_size, config.conv_kernel_size, activation="silu")

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(projection_size))

        # Gate: f_a_proj -> f_b_proj produces per-head gate
        self.f_a_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=self.head_dim,
            bias=False,
        ).build()
        self.f_b_proj = Linear.Config(
            in_features=self.head_dim,
            out_features=projection_size,
            bias=False,
        ).build()

        # Beta projection
        self.b_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
        ).build()

        # Output gate
        self.use_full_rank_gate = config.use_full_rank_gate
        if self.use_full_rank_gate:
            self.g_proj = Linear.Config(
                in_features=self.hidden_size,
                out_features=projection_size,
                bias=False,
            ).build()
        else:
            self.g_a_proj = Linear.Config(
                in_features=self.hidden_size,
                out_features=self.head_dim,
                bias=False,
            ).build()
            self.g_b_proj = Linear.Config(
                in_features=self.head_dim,
                out_features=projection_size,
                bias=False,
            ).build()

        self.o_norm = RMSNormGated(self.head_dim, eps=config.norm_eps)
        self.o_proj = Linear.Config(
            in_features=projection_size,
            out_features=self.hidden_size,
            bias=False,
        ).build()

        self.gate_lower_bound = config.gate_lower_bound

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: Any | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        # Gate and beta
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Core KDA computation
        o = self._chunk_kda(q, k, v, g, beta)

        # Output gate + norm
        if self.use_full_rank_gate:
            out_g = self.g_proj(hidden_states)
        else:
            out_g = self.g_b_proj(self.g_a_proj(hidden_states))
        out_g = out_g.view(batch_size, seq_len, self.num_heads, self.head_dim)

        o = self.o_norm(o, out_g)
        o = o.reshape(batch_size, seq_len, -1)
        return self.o_proj(o)

    def _chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Chunk-wise KDA forward via triton-ascend-kernels fused operator."""
        from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda

        A_log = (
            self.A_log.to_local()
            if isinstance(self.A_log, DTensor)
            else self.A_log
        )
        dt_bias = (
            self.dt_bias.to_local()
            if isinstance(self.dt_bias, DTensor)
            else self.dt_bias
        )
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=None,
        )
        return o


class KimiGatedMLA(nn.Module):
    """Gated Multi-head Latent Attention for Kimi K3.

    24 of 93 layers use this. Same as DSv3 MLA but with an output gate:
    attn_output *= sigmoid(g_proj(hidden_states))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 7168
        n_heads: int = 96
        q_lora_rank: int = 1536
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        norm_eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)

        # Query projection with LoRA
        self.q_a_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=self.q_lora_rank,
            bias=False,
        ).build()
        self.q_a_layernorm = RMSNorm.Config(
            normalized_shape=self.q_lora_rank,
            eps=config.norm_eps,
        ).build()
        self.q_b_proj = Linear.Config(
            in_features=self.q_lora_rank,
            out_features=self.num_heads * self.q_head_dim,
            bias=False,
        ).build()

        # KV projection with MQA compression
        self.kv_a_proj_with_mqa = Linear.Config(
            in_features=self.hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        ).build()
        self.kv_a_layernorm = RMSNorm.Config(
            normalized_shape=self.kv_lora_rank,
            eps=config.norm_eps,
        ).build()
        self.kv_b_proj = Linear.Config(
            in_features=self.kv_lora_rank,
            out_features=self.num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        ).build()

        # Output gate (K3-specific)
        self.g_proj = Linear.Config(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.v_head_dim,
            bias=False,
        ).build()
        self.o_proj = Linear.Config(
            in_features=self.num_heads * self.v_head_dim,
            out_features=self.hidden_size,
            bias=False,
        ).build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: Any | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Query
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(k_compressed))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Kimi K3 sets mla_use_nope=True, so the legacy q_rope/k_rope slices
        # participate in attention without rotary position encoding.
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope_expanded], dim=-1)

        # Pad v to match q_head_dim for flash attention compatibility
        if self.v_head_dim != self.q_head_dim:
            v_padded = F.pad(v, [0, self.q_head_dim - self.v_head_dim])
        else:
            v_padded = v

        # Scaled dot-product attention
        query = query.transpose(1, 2)  # (B, H, S, D)
        key = key.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        attn_output = self._attention_core(query, key, v_padded)
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)

        # Trim padding
        if self.v_head_dim != self.q_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output gate (K3-specific)
        gate = torch.sigmoid(self.g_proj(hidden_states))
        attn_output = attn_output * gate

        return self.o_proj(attn_output)

    def _attention_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MLA attention core after all projection/layout work."""
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
