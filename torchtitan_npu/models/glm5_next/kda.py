# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention (KDA) linear-attention layers for GLM-5.3-Flash.

Mirrors ``Glm5NextTextLinearAttention`` + ``Glm5NextTextForgetGate`` +
``Glm5NextTextRMSNormGated`` in MindSpeed-MM ``modeling_glm5_next.py``.
Production kernels: ascendc ``chunk_kda`` (via fla_npu) + triton
``causal_conv1d``. The 0.3.0 capture is ATen level: the eager chunk-free
fallback (recurrent scan) provides the same math through plain ATen ops,
so the meta graphs record the real dataflow. Its op count scales with the
sequence length, so capture flavors keep ``seq_len`` modest.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.protocols.module import Module

from .common import RMSNormGated, kda_dt_bias_init, linear_config


class ForgetGate(Module):
    """Low-rank forget gate with the safe lower bound (linear_lower_bound=-5)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_heads: int = 64
        head_dim: int = 128
        lower_bound: float = -5.0
        f_a_proj: Linear.Config | None = None
        f_b_proj: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.f_a_proj is None:
                self.f_a_proj = linear_config(self.dim, self.head_dim)
            if self.f_b_proj is None:
                self.f_b_proj = linear_config(self.head_dim, self.num_heads * self.head_dim)
            # FLA init via the safe gate: the production config sets
            # linear_lower_bound=-5, so A_log is zeros (decay_rate = 1) and
            # dt_bias follows the FLA inverse-softplus recipe.
            self.param_init = {"A_log": nn.init.zeros_, "dt_bias": kda_dt_bias_init}

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.lower_bound = config.lower_bound
        self.f_a_proj = config.f_a_proj.build()
        self.f_b_proj = config.f_b_proj.build()
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))

    def raw(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.f_b_proj(self.f_a_proj(hidden_states))

    def decay(self, raw: torch.Tensor) -> torch.Tensor:
        """Safe-gated per-token decay ``a`` in [e^lower_bound, 1) per channel.

        The ascendc chunk_kda kernel consumes the RAW gate logits plus
        A_log/dt_bias (use_gate_in_kernel + safe_gate); the eager path folds
        the safe-gate transform itself."""
        hidden_shape = (*raw.shape[:2], -1, self.head_dim)
        g = (raw.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay_rate = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))
        return self.lower_bound * torch.sigmoid(decay_rate * g)


class KDALinearAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_heads: int = 64
        head_dim: int = 128
        conv_kernel_size: int = 4
        norm_eps: float = 1e-6
        lower_bound: float = -5.0
        qkv_proj: Linear.Config | None = None
        conv1d: Conv1d.Config | None = None
        forget_gate: ForgetGate.Config | None = None
        b_proj: Linear.Config | None = None
        g_a_proj: Linear.Config | None = None
        g_b_proj: Linear.Config | None = None
        o_norm: RMSNormGated.Config | None = None
        o_proj: Linear.Config | None = None

        def __post_init__(self) -> None:
            qkv_dim = self.num_heads * self.head_dim
            if self.qkv_proj is None:
                self.qkv_proj = linear_config(self.dim, qkv_dim * 3)
            if self.conv1d is None:
                self.conv1d = Conv1d.Config(
                    in_channels=qkv_dim * 3,
                    out_channels=qkv_dim * 3,
                    kernel_size=self.conv_kernel_size,
                    groups=qkv_dim * 3,
                    padding=self.conv_kernel_size - 1,
                    bias=False,
                )
            if self.forget_gate is None:
                self.forget_gate = ForgetGate.Config(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    lower_bound=self.lower_bound,
                )
            if self.b_proj is None:
                self.b_proj = linear_config(self.dim, self.num_heads)
            if self.g_a_proj is None:
                self.g_a_proj = linear_config(self.dim, self.head_dim)
            if self.g_b_proj is None:
                self.g_b_proj = linear_config(self.head_dim, qkv_dim)
            if self.o_norm is None:
                self.o_norm = RMSNormGated.Config(hidden_size=self.head_dim, eps=self.norm_eps)
            if self.o_proj is None:
                self.o_proj = linear_config(qkv_dim, self.dim)

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv_dim = self.head_dim * self.num_heads
        self.conv_kernel_size = config.conv_kernel_size
        self.layer_norm_epsilon = config.norm_eps

        self.qkv_proj = config.qkv_proj.build()
        self.conv1d = config.conv1d.build()
        self.forget_gate = config.forget_gate.build()
        self.b_proj = config.b_proj.build()
        self.g_a_proj = config.g_a_proj.build()
        self.g_b_proj = config.g_b_proj.build()
        self.o_norm = config.o_norm.build()
        self.o_proj = config.o_proj.build()

    def _eager_chunk_kda(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """FLA chunked KDA (chunk_size=64): faithful port of the MindSpeed
        eager fallback ``chunk_kimi_delta_attention``. ``g`` is the log-space
        safe-gated decay (values in [lower_bound, 0)); the scan applies it
        through exp/cumsum, never as a linear multiplier. Meta-safe: every
        loop bound is static and all control flow is shape-only.

        Shapes: q/k/v [B, S, H, D], g [B, S, H, D], beta [B, S, H]."""
        chunk_size = 64
        initial_dtype = q.dtype
        q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]

        def l2norm(x):
            return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        q, k = l2norm(q), l2norm(k)

        batch_size, num_heads, sequence_length, k_head_dim = k.shape
        v_head_dim = v.shape[-1]
        scale = 1 / (q.shape[-1] ** 0.5)
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        total_sequence_length = sequence_length + pad_size

        q = F.pad(q, (0, 0, 0, pad_size)) * scale
        k = F.pad(k, (0, 0, 0, pad_size))
        v = F.pad(v, (0, 0, 0, pad_size))
        g = F.pad(g, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)

        chunk_shape = (batch_size, num_heads, -1, chunk_size)
        q, k, v, g, k_beta, v_beta = [x.reshape(*chunk_shape, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta)]
        beta = beta.reshape(batch_size, num_heads, -1, chunk_size)

        # Intra chunk: per-head cumulative gate and the decayed triangular
        # system solved by forward substitution (static 64 iterations).
        g = g.cumsum(dim=-2)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
        strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strict_upper.unsqueeze(-1), float("-inf")).exp()
        attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)
        v = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp())

        state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=v.dtype, device=v.device)
        core_attn_out = torch.zeros_like(v)

        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        for i in range(total_sequence_length // chunk_size):
            q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
            # Inter chunk: state contribution decayed to this chunk.
            attn_inter = (q_i * g_i.exp()) @ state
            # Intra chunk: decayed causal attention inside the chunk.
            attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
            # Corrected value target: remove the state part k_cumdecay already carries.
            v_prime = k_cumdecay[:, :, i] @ state
            v_new = v_i - v_prime
            core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
            state = (
                state * g_i[:, :, -1].exp().unsqueeze(-1)
                + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
            )

        core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks=None,
        positions=None,
    ) -> torch.Tensor:
        # Under TP the mixer input may arrive as a replicated DTensor; the
        # mask multiply and the sharded core run on plain local tensors and
        # the row-parallel o_proj re-wraps the output (qwen3_5 GDN pattern).
        if isinstance(hidden_states, torch.distributed.tensor.DTensor):
            hidden_states = hidden_states.to_local()
        if attention_masks is not None:
            mask = attention_masks
            if isinstance(mask, torch.distributed.tensor.DTensor):
                mask = mask.to_local()
            if mask.dtype != torch.bool:
                mask = mask.bool()
            hidden_states = (hidden_states * mask[:, :, None]).to(hidden_states.dtype)
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = self.qkv_proj(hidden_states)
        mixed_qkv = self.conv1d(mixed_qkv.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)

        query, key, value = torch.split(mixed_qkv, [self.qkv_dim] * 3, dim=-1)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        fg_raw = self.forget_gate.raw(hidden_states)
        g_raw = fg_raw.view(hidden_shape)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        decay = self.forget_gate.decay(fg_raw)
        core_attn_out = self._eager_chunk_kda(query, key, value, decay, beta)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)
