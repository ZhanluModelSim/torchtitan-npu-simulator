# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview attention: modality-grouped MHA with RoPE and learned sinks.

Fork reason: Upstream torchtitan has no MAGI-2 support. MAGI-2 attention
combines modality-grouped projections (3 per-modality expert weights),
per-head q/k RMSNorm, partial RoPE (first 96 of 128 head dims), non-causal
varlen attention, and one learned sink logit per head that acts as an extra
zero-valued key in the softmax.
Reference: MAGI-2-preview official inference/model/magi2_preview.py (Attention)
"""

import logging
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.protocols.module import Module

from .grouped_linear import GroupedLinear
from .norms import MultiModalityRMSNorm

logger = logging.getLogger(__name__)


def _apply_rotary_emb(
    x: torch.Tensor,
    sin_emb: torch.Tensor,
    cos_emb: torch.Tensor,
) -> torch.Tensor:
    """Non-interleaved half-split RoPE on the leading rotary dims.

    Args:
        x: (T, num_heads, head_dim) queries or keys in original token order.
        sin_emb, cos_emb: (T, rotary_dim / 2) each (halves of the rope embed).

    Returns:
        Tensor of the same shape: the first rotary_dim dims are rotated
        (x1*cos - x2*sin, x2*cos + x1*sin over the two halves), the trailing
        dims pass through unchanged.
    """
    rotary_dim = cos_emb.shape[-1] * 2
    cos = cos_emb.unsqueeze(1)  # (T, 1, rotary_dim/2), broadcast over heads
    sin = sin_emb.unsqueeze(1)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


class Magi2Attention(Module):
    """Modality-grouped multi-head self-attention with sinks (MAGI-2-preview).

    Pipeline (mirrors the official Attention.forward):
    1. pre_norm + modality-grouped gate/qkv projections (sorted token order);
    2. per-head q/k RMSNorm with fp32 output;
    3. inverse-permute q/k/v to the original token order, where RoPE and the
       (non-causal, per packed segment) attention run;
    4. each head gets one learned sink logit appended as an extra zero-valued
       key column before the softmax (equivalent to the reference flash-attn
       LSE correction);
    5. permute back to sorted order, gate by sigmoid(linear_g), project out.

    Submodule names (pre_norm/linear_g/linear_qkv/linear_proj/q_norm/k_norm
    and the ``sinks`` parameter) match the official checkpoint keys exactly.
    Meta-init safe: only parameters are materialized in ``__init__``; values
    are filled by the model-level ``init_weights``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 3072
        head_dim: int = 128
        num_modality: int = 3
        norm_eps: float = 1e-6
        sink_token_num: int = 1

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.hidden_size // config.head_dim
        self.num_modality = config.num_modality
        self.softmax_scale = self.head_dim ** -0.5

        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, eps=config.norm_eps, num_modality=config.num_modality
        )
        self.linear_g = GroupedLinear(
            config.hidden_size, self.num_heads, num_experts=config.num_modality
        )
        self.linear_qkv = GroupedLinear(
            config.hidden_size,
            3 * self.num_heads * self.head_dim,
            num_experts=config.num_modality,
        )
        self.linear_proj = GroupedLinear(
            config.hidden_size, config.hidden_size, num_experts=config.num_modality
        )
        # One learned sink logit per head; kept fp32 like the official model.
        self.sinks = nn.Parameter(
            torch.empty(max(1, config.sink_token_num), self.num_heads, dtype=torch.float32)
        )
        self.q_norm = MultiModalityRMSNorm(
            config.head_dim,
            eps=config.norm_eps,
            num_modality=config.num_modality,
            out_dtype=torch.float32,
        )
        self.k_norm = MultiModalityRMSNorm(
            config.head_dim,
            eps=config.norm_eps,
            num_modality=config.num_modality,
            out_dtype=torch.float32,
        )

    def forward(
        self,
        x_sorted: torch.Tensor,
        rope: torch.Tensor,
        m_splits: list[int],
        sort_idx: torch.Tensor,
        inv_sort_idx: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the attention sublayer on modality-sorted packed tokens.

        Args:
            x_sorted: (T, hidden_size) stream slice in sorted (modality-major)
                token order.
            rope: (T, rotary_dim) Fourier RoPE embed in ORIGINAL token order
                (as produced by the pre-adapter).
            m_splits: per-modality token counts of the sorted rows.
            sort_idx: (T,) long indices mapping original -> sorted order.
            inv_sort_idx: (T,) long indices mapping sorted -> original order.
            cu_seqlens: optional (S+1,) int cumulative segment boundaries for
                packed varlen attention; None means one whole-sequence segment.

        Returns:
            (T, hidden_size) attention output in sorted token order.
        """
        T = x_sorted.shape[0]
        h = self.pre_norm(x_sorted, m_splits)
        g = self.linear_g(h, m_splits)  # (T, num_heads)
        qkv = self.linear_qkv(h, m_splits)  # (T, 3 * num_heads * head_dim)
        q, k, v = qkv.view(T, 3, self.num_heads, self.head_dim).unbind(1)

        # Per-head RMSNorm over the last dim; fp32 output.
        q = self.q_norm(q, m_splits)
        k = self.k_norm(k, m_splits)

        # RoPE and attention run in the original token order.
        q = q[inv_sort_idx]
        k = k[inv_sort_idx]
        v = v[inv_sort_idx]
        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = _apply_rotary_emb(q, sin_emb, cos_emb)
        k = _apply_rotary_emb(k, sin_emb, cos_emb)

        # fp32 accumulation throughout the attention core.
        v = v.to(q.dtype)
        out = self._segment_attention_with_sinks(q, k, v, cu_seqlens)

        # Back to sorted order, gate per head, then the grouped out projection.
        out = out[sort_idx]
        out = out * torch.sigmoid(g).view(T, self.num_heads, 1)
        out = out.reshape(T, self.num_heads * self.head_dim)
        return self.linear_proj(out, m_splits)

    def _segment_attention_with_sinks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Non-causal per-segment attention with a learned sink key per head.

        Each segment attends only to its own keys; the sink logit of every
        head is appended as an extra zero-valued key column, so it only
        renormalizes the softmax without contributing value mass.

        Args:
            q, k, v: (T, num_heads, head_dim) in original token order.
            cu_seqlens: optional (S+1,) boundaries; None = single segment.

        Returns:
            (T, num_heads, head_dim) attention output in original order.
        """
        T = q.shape[0]
        if cu_seqlens is None:
            bounds = [0, T]
        elif isinstance(cu_seqlens, torch.Tensor):
            bounds = cu_seqlens.tolist()
        else:
            bounds = list(cu_seqlens)

        num_sinks = self.sinks.shape[0]
        sink_logits = self.sinks.t()  # (num_heads, num_sinks)

        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            if end <= start:
                continue
            qs = q[start:end]
            ks = k[start:end]
            vs = v[start:end]
            L = end - start
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * self.softmax_scale
            sink_cols = sink_logits.unsqueeze(1).expand(qs.shape[1], L, num_sinks)
            scores_ext = torch.cat([scores, sink_cols], dim=-1)
            probs = torch.softmax(scores_ext, dim=-1)
            outputs.append(torch.einsum("hlm,mhd->lhd", probs[..., :L], vs))
        return torch.cat(outputs, dim=0)
