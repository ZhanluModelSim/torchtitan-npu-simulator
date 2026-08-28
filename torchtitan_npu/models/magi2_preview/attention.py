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

Ulysses context parallelism: ``Magi2Attention.forward`` accepts a CP
context (``cp_context``) and/or carries one installed by
``Magi2UlyssesAttentionCP`` (see ``cp_ulysses.py``). In CP mode the
attention core runs on the full sequence with a head subset per rank: the
sequence<->head all-to-all swaps and the RoPE gather are applied as hooks
on the parameter-free ``attn_core`` submodule (or inline when a
``cp_context`` is passed to forward directly), so RoPE is applied after
the swap and every attention backend stays usable under CP.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from .cp_ulysses import CpContext

from .grouped_linear import GroupedLinear
from .norms import MultiModalityRMSNorm

logger = logging.getLogger(__name__)

# Supported attention backends for Magi2Attention.Config.attn_backend.
ATTN_BACKENDS = ("sdpa", "flex")


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


class _AttentionCore(nn.Module):
    """Parameter-free RoPE + attention-core stage of ``Magi2Attention``.

    Exists as its own module so ``Magi2UlyssesAttentionCP`` can hook the
    Ulysses all-to-all swaps around it: the pre-hook replaces q/k/v with
    their (full sequence, head subset) all-to-all images (and gathers the
    RoPE embed to the full sequence), the post-hook swaps the output back.
    Without CP the hooks are simply absent and this runs the original
    inline math. Holds no parameters/buffers, so state-dict keys of the
    owning attention module are unchanged; the owner reference is stored
    as a plain attribute (``object.__setattr__``) because registering the
    parent as a submodule would create a module-tree cycle.
    """

    def __init__(self, owner: "Magi2Attention") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rope: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply RoPE then run the attention core in original token order.

        Args:
            q, k, v: (T, num_heads, head_dim) — full heads without CP, or
                the swapped (full T, local head group) layout under CP.
            rope: (T, rotary_dim) RoPE embed matching q/k rows (gathered
                to the full sequence under CP by the pre-hook).
            cu_seqlens: optional packed-segment boundaries (full sequence).

        Returns:
            (T, num_heads, head_dim) attention output, original order.
        """
        attention = self._owner
        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = _apply_rotary_emb(q, sin_emb, cos_emb)
        k = _apply_rotary_emb(k, sin_emb, cos_emb)

        # fp32 accumulation throughout the attention core.
        v = v.to(q.dtype)
        if attention.attn_backend == "flex":
            return attention._flex_attention_with_sinks(q, k, v, cu_seqlens)
        return attention._segment_attention_with_sinks(q, k, v, cu_seqlens)


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

    Attention core backends (``config.attn_backend``):
    - ``"sdpa"`` (default): the reference python per-segment softmax loop
      (``_segment_attention_with_sinks``), kept for parity checks.
    - ``"flex"``: one attention call over keys extended by per-segment
      zero-valued sink keys (no python segment loop). On accelerator devices
      this runs ``torch.nn.attention.flex_attention`` with a segment
      ``create_block_mask`` and a score_mod that injects the learned sink
      logits; on CPU (eager flex_attention has no CPU backward kernel as of
      torch 2.12) the same sink-extended math runs as a single masked SDPA
      call instead. Both mechanisms are numerically equivalent to "sdpa".

    Ulysses CP mode: ``self.cp_context`` is installed by
    ``Magi2UlyssesAttentionCP`` (None = CP disabled, the default). In CP
    mode ``forward`` runs in head-subset/full-sequence mode: hooks on
    ``self.attn_core`` all-to-all q/k/v from ``(T/cp, H, D)`` to
    ``(T, H/cp, D)``, gather the RoPE embed to the full sequence, and
    swap the core output back; ``sinks`` is head-sharded by the style.
    ``forward`` also accepts an explicit ``cp_context`` argument for
    hook-free CP invocations (used by single-process tests).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 3072
        head_dim: int = 128
        num_modality: int = 3
        norm_eps: float = 1e-6
        sink_token_num: int = 1
        # "sdpa": reference per-segment softmax path. "flex": sink-extended
        # single-call path (flex_attention on accelerators, masked SDPA on
        # CPU); numerically equivalent to "sdpa" (see the unit tests).
        attn_backend: str = "sdpa"

    def __init__(self, config: Config):
        super().__init__()
        if config.attn_backend not in ATTN_BACKENDS:
            raise ValueError(
                f"Magi2Attention.Config.attn_backend must be one of "
                f"{ATTN_BACKENDS}, got {config.attn_backend!r}"
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.hidden_size // config.head_dim
        self.num_modality = config.num_modality
        self.softmax_scale = self.head_dim ** -0.5
        self.attn_backend = config.attn_backend

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
        # RoPE + attention-core stage; hook target of the Ulysses CP style.
        self.attn_core = _AttentionCore(self)
        # Ulysses CP state (CpContext), installed by Magi2UlyssesAttentionCP;
        # None keeps the non-CP path (the default).
        self.cp_context: "CpContext | None" = None

    def forward(
        self,
        x_sorted: torch.Tensor,
        rope: torch.Tensor,
        m_splits: list[int],
        sort_idx: torch.Tensor,
        inv_sort_idx: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cp_context: "CpContext | None" = None,
    ) -> torch.Tensor:
        """Run the attention sublayer on modality-sorted packed tokens.

        Args:
            x_sorted: (T, hidden_size) stream slice in sorted (modality-major)
                token order; T is the local sequence shard under CP.
            rope: (T, rotary_dim) Fourier RoPE embed in ORIGINAL token order
                (as produced by the pre-adapter); local shard under CP.
            m_splits: per-modality token counts of the sorted rows.
            sort_idx: (T,) long indices mapping original -> sorted order.
            inv_sort_idx: (T,) long indices mapping sorted -> original order.
            cu_seqlens: optional (S+1,) int cumulative segment boundaries for
                packed varlen attention; None means one whole-sequence
                segment. Under CP this stays the FULL-sequence boundary
                tensor: the attention core runs on the all-to-all'd full
                sequence.
            cp_context: optional Ulysses CP context for hook-free CP
                invocations (single-process tests). When ``self.cp_context``
                is set by the parallelize wiring, its hooks already perform
                the swaps inside ``attn_core`` and this argument is ignored.

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
        if self.cp_context is not None:
            # CP hooks on attn_core swap q/k/v to (full T, local head
            # group), gather the RoPE embed to the full sequence, and
            # swap the core output back to the local shard.
            out = self.attn_core(q, k, v, rope, cu_seqlens)
        elif cp_context is not None:
            # Hook-free CP invocation: the same Ulysses swaps, applied
            # inline around the attention core.
            from .cp_ulysses import (
                cp_post_attention_swap,
                cp_pre_attention_swap,
            )

            q, k, v, rope = cp_pre_attention_swap(q, k, v, rope, cp_context)
            out = self.attn_core(q, k, v, rope, cu_seqlens)
            out = cp_post_attention_swap(out, cp_context)
        else:
            out = self.attn_core(q, k, v, rope, cu_seqlens)

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

    def _flex_attention_with_sinks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Non-causal packed attention with sinks via sink-extended keys.

        Instead of looping over packed segments in python, the key/value
        sequence is extended by ``num_segments * num_sinks`` zero-valued sink
        keys (one group per segment) and attention runs in a single call
        restricted to same-segment pairs. Mechanism selection:

        - accelerator devices: ``flex_attention`` with a segment block mask
          (``doc_id[q_idx] == kv_doc[kv_idx]``) and a score_mod that
          replaces the sink columns' scores with the learned per-head sink
          logits (score_mod receives already-scaled scores, so the logits
          enter exactly as in the "sdpa" path);
        - CPU: eager flex_attention has no backward kernel there (as of
          torch 2.12), so the same sink-extended math runs as one masked
          SDPA call (``_flex_attention_masked_sdpa``).

        Both mechanisms are numerically equivalent to
        ``_segment_attention_with_sinks``; see
        tests/unit_tests/models/test_magi2_attention_backend.py.

        Args:
            q, k, v: (T, num_heads, head_dim) in original token order.
            cu_seqlens: optional (S+1,) boundaries; None = single segment.

        Returns:
            (T, num_heads, head_dim) attention output in original order.
        """
        if q.device.type == "cpu":
            return self._flex_attention_masked_sdpa(q, k, v, cu_seqlens)
        return self._flex_attention_kernel(q, k, v, cu_seqlens)

    def _sink_key_layout(
        self,
        seq_len: int,
        device: torch.device,
        cu_seqlens: torch.Tensor | list[int] | None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Derive segment ids for the sink-extended KV layout.

        Args:
            seq_len: packed token count T.
            device: device for the created index tensors.
            cu_seqlens: optional (S+1,) cumulative segment boundaries.

        Returns:
            (num_segments, doc_id, kv_doc) where doc_id is (T,) int32
            (segment id per token) and kv_doc is (T + S * num_sinks,) int32:
            real keys carry their token's segment id and each segment's
            num_sinks appended sink keys carry that segment's id.
        """
        if cu_seqlens is None:
            lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
        else:
            if not isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
            cu = cu_seqlens.to(device=device, dtype=torch.int32)
            lengths = cu[1:] - cu[:-1]
        num_segments = lengths.numel()
        segments = torch.arange(num_segments, dtype=torch.int32, device=device)
        doc_id = torch.repeat_interleave(segments, lengths)
        num_sinks = self.sinks.shape[0]
        kv_doc = torch.cat(
            [doc_id, torch.repeat_interleave(segments, num_sinks)]
        )
        return num_segments, doc_id, kv_doc

    def _flex_attention_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """flex_attention mechanism of the "flex" backend (accelerators).

        The block mask is created per forward because packed segment
        boundaries vary per batch; under ``torch.compile`` this call is a
        graph break (correct, not fused) — hoisting BlockMask creation out
        of the compiled region is left as a perf follow-up.
        """
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        T, num_heads, head_dim = q.shape
        num_sinks = self.sinks.shape[0]
        num_segments, doc_id, kv_doc = self._sink_key_layout(
            T, q.device, cu_seqlens
        )
        num_sink_keys = num_segments * num_sinks
        k_ext = torch.cat([k, k.new_zeros(num_sink_keys, num_heads, head_dim)])
        v_ext = torch.cat([v, v.new_zeros(num_sink_keys, num_heads, head_dim)])

        def mask_mod(b, h, q_idx, kv_idx):
            return doc_id[q_idx] == kv_doc[kv_idx]

        # The sink columns' qk scores are meaningless (sink keys are zero);
        # replace them with the learned sink logits. score_mod receives
        # already-scaled scores, so the logits are used as-is, exactly like
        # the sink columns of the "sdpa" path.
        sink_logits = self.sinks.t().to(q.dtype)  # (num_heads, num_sinks)

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(
                kv_idx >= T, sink_logits[h, (kv_idx - T) % num_sinks], score
            )

        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=T,
            KV_LEN=T + num_sink_keys,
            device=q.device,
            # Mask creation is O(T + S); keep it eager for portability.
            _compile=False,
        )
        out = flex_attention(
            q.unsqueeze(0).transpose(1, 2),
            k_ext.unsqueeze(0).transpose(1, 2),
            v_ext.unsqueeze(0).transpose(1, 2),
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).squeeze(0)

    def _flex_attention_masked_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked-SDPA mechanism of the "flex" backend (CPU fallback).

        Eager flex_attention has no CPU backward kernel (torch 2.12), so on
        CPU the sink-extended math runs as one ``scaled_dot_product_attention``
        call: sink keys are zero-valued, hence their scaled qk scores are 0
        and the additive attention mask carries the learned sink logits on
        sink columns while blocking cross-segment pairs.
        """
        T, num_heads, head_dim = q.shape
        num_sinks = self.sinks.shape[0]
        num_segments, doc_id, kv_doc = self._sink_key_layout(
            T, q.device, cu_seqlens
        )
        num_sink_keys = num_segments * num_sinks
        k_ext = torch.cat([k, k.new_zeros(num_sink_keys, num_heads, head_dim)])
        v_ext = torch.cat([v, v.new_zeros(num_sink_keys, num_heads, head_dim)])

        # Per-(head, kv-col) additive score contribution: 0 for real keys,
        # the learned sink logit for sink keys (their qk score is 0).
        sink_logits = self.sinks.t().to(q.dtype)  # (num_heads, num_sinks)
        sink_col_idx = torch.arange(num_sink_keys, device=q.device) % num_sinks
        col_bias = torch.cat(
            [sink_logits.new_zeros(num_heads, T), sink_logits[:, sink_col_idx]],
            dim=-1,
        )
        allowed = (doc_id.unsqueeze(1) == kv_doc.unsqueeze(0)).unsqueeze(0)
        blocked = q.new_full((), torch.finfo(q.dtype).min)
        attn_mask = torch.where(allowed, col_bias.unsqueeze(1), blocked)

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k_ext.unsqueeze(0).transpose(1, 2),
            v_ext.unsqueeze(0).transpose(1, 2),
            attn_mask=attn_mask.unsqueeze(0),
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).squeeze(0)
