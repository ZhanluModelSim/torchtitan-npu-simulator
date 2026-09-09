# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ar_llm hybrid attention: CSA / HCA / KDA dispatched by layer position.

Contract: see MODEL_CONTRACT.md. Layer types repeat with period ``unit_size``:
``[CSA, HCA, KDA, KDA, KDA, KDA]``. CSA and HCA share the MLA-style low-rank
Q/KV projections and the grouped low-rank O projection; KDA owns its own
projections and a delta-rule linear-attention core. All cores are written so
that they only depend on tensor shapes (no ``.item()``/``.nonzero()`` on
activation values), which keeps them executable on meta tensors.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm

if TYPE_CHECKING:
    from .model import ArLlmModel


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [B, S, H, D]; cos/sin: [S, D]."""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def stable_softmax(scores: torch.Tensor, dim: int = -1, clamp_val: float | None = None) -> torch.Tensor:
    if clamp_val is not None:
        scores = scores.clamp(-clamp_val, clamp_val)
    return F.softmax(scores.float(), dim=dim).to(scores.dtype)


def swiglu(up: torch.Tensor, gate: torch.Tensor, clamp_val: float | None = None) -> torch.Tensor:
    if clamp_val is not None:
        gate = gate.clamp(max=clamp_val)
    out = up * F.silu(gate)
    if clamp_val is not None:
        out = out.clamp(-clamp_val, clamp_val)
    return out


class QLowRankProjection(nn.Module):
    """MLA-style Q projection: d -> q_lora -> (nope | rope) per head."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.nope_dim = model_args.qk_nope_head_dim
        self.rope_dim = model_args.qk_rope_head_dim
        self.q_down = Linear.Config(
            in_features=model_args.dim,
            out_features=model_args.q_lora_rank,
            bias=False,
        ).build()
        self.q_norm = RMSNorm.Config(normalized_shape=model_args.q_lora_rank, eps=model_args.norm_eps).build()
        self.q_up_nope = Linear.Config(
            in_features=model_args.q_lora_rank,
            out_features=model_args.n_heads * self.nope_dim,
            bias=False,
        ).build()
        self.q_up_rope = Linear.Config(
            in_features=model_args.q_lora_rank,
            out_features=model_args.n_heads * self.rope_dim,
            bias=False,
        ).build()
        self.q_up_nope_norm = RMSNorm.Config(normalized_shape=self.nope_dim, eps=model_args.norm_eps).build()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = hidden_states.shape
        q_c = F.silu(self.q_norm(self.q_down(hidden_states)))
        q_nope = self.q_up_nope(q_c).view(b, s, self.n_heads, self.nope_dim)
        q_nope = self.q_up_nope_norm(q_nope)
        q_rope = self.q_up_rope(q_c).view(b, s, self.n_heads, self.rope_dim)
        return q_nope, q_rope


class KVLowRankProjection(nn.Module):
    """MLA-style KV projection: d -> (kv_lora | rope) -> (k_nope | v) per head."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.kv_lora_rank = model_args.kv_lora_rank
        self.rope_dim = model_args.qk_rope_head_dim
        self.head_dim = model_args.head_dim
        self.kv_down = Linear.Config(
            in_features=model_args.dim,
            out_features=model_args.kv_lora_rank + self.rope_dim,
            bias=False,
        ).build()
        self.kv_norm = RMSNorm.Config(
            normalized_shape=model_args.kv_lora_rank + self.rope_dim, eps=model_args.norm_eps
        ).build()
        self.kv_latent_norm = RMSNorm.Config(normalized_shape=self.kv_lora_rank, eps=model_args.norm_eps).build()
        self.k_up_nope = Linear.Config(
            in_features=model_args.kv_lora_rank,
            out_features=model_args.n_heads * model_args.qk_nope_head_dim,
            bias=False,
        ).build()
        self.v_up = Linear.Config(
            in_features=model_args.kv_lora_rank,
            out_features=model_args.n_heads * self.head_dim,
            bias=False,
        ).build()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = hidden_states.shape
        kv = self.kv_norm(self.kv_down(hidden_states))
        kv_c = self.kv_latent_norm(kv[..., : self.kv_lora_rank])
        k_rope = kv[..., self.kv_lora_rank :].view(b, s, 1, self.rope_dim)
        k_nope = self.k_up_nope(kv_c).view(b, s, self.n_heads, -1)
        v = self.v_up(kv_c).view(b, s, self.n_heads, self.head_dim)
        return k_nope, k_rope, v


class GroupedOProjection(nn.Module):
    """Grouped low-rank O projection: per head-group d_g -> o_lora -> d/G.

    Weights are batched over ``o_groups`` so TP can shard the group (= head)
    dimension directly. Output feature slices per group are disjoint, so the
    TP boundary all-gathers the feature dim instead of all-reducing.
    """

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        heads_per_group = model_args.n_heads // model_args.o_groups
        self.o_groups = model_args.o_groups
        self.heads_per_group = heads_per_group
        self.hidden_size = model_args.dim
        self.o_down = nn.Parameter(
            torch.empty(model_args.o_groups, model_args.o_lora_rank, heads_per_group * model_args.head_dim)
        )
        self.o_up = nn.Parameter(
            torch.empty(model_args.o_groups, model_args.dim // model_args.o_groups, model_args.o_lora_rank)
        )

    def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
        b, s = attn_out.shape[:2]
        o_down = self.o_down.to_local() if isinstance(self.o_down, DTensor) else self.o_down
        o_up = self.o_up.to_local() if isinstance(self.o_up, DTensor) else self.o_up
        x = attn_out.reshape(b, s, self.o_groups, self.heads_per_group * attn_out.shape[-1])
        h = torch.einsum("bsgh,goh->bsgo", x.to(o_down.dtype), o_down)
        h = F.silu(h)
        out = torch.einsum("bsgo,gdo->bsgd", h.to(o_up.dtype), o_up)
        return out.reshape(b, s, self.hidden_size)


class CompressedSparseAttention(nn.Module):
    """CSA core: sliding-window local attention + stride-compressed global attention."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.compress_ratio = model_args.csa_compress_ratio
        self.window_size = model_args.csa_window_size
        self.scale = 1.0 / math.sqrt(model_args.head_dim)
        self.softmax_clamp = model_args.attn_softmax_clamp

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, nh, hd = q.shape
        pos = torch.arange(s, device=q.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal = dist >= 0
        in_window = dist.abs() <= (self.window_size // 2)
        local_mask = torch.where(causal & in_window, 0.0, -1e9).to(q.dtype)

        local_scores = torch.einsum("bshd,bthd->bhst", q, k) * self.scale
        if sink_bias is not None:
            local_scores = local_scores + sink_bias.view(1, -1, 1, 1)
        local_scores = local_scores + local_mask.unsqueeze(0).unsqueeze(0)

        k_c = k[:, :: self.compress_ratio]
        v_c = v[:, :: self.compress_ratio]
        cl = k_c.shape[1]
        global_scores = torch.einsum("bshd,bthd->bhst", q, k_c) * self.scale
        if sink_bias is not None:
            global_scores = global_scores + sink_bias.view(1, -1, 1, 1)
        q_pos = pos.unsqueeze(1)
        k_pos = torch.arange(cl, device=q.device).unsqueeze(0) * self.compress_ratio
        global_mask = torch.where(k_pos <= q_pos, 0.0, -1e9).to(q.dtype)
        global_scores = global_scores + global_mask.unsqueeze(0).unsqueeze(0)

        local_out = torch.einsum("bhst,bthd->bshd", stable_softmax(local_scores, clamp_val=self.softmax_clamp), v)
        global_out = torch.einsum("bhst,bthd->bshd", stable_softmax(global_scores, clamp_val=self.softmax_clamp), v_c)
        return local_out + global_out


class HeavilyCompressedAttention(nn.Module):
    """HCA core: indexer top-k selection over compressed KV + attention on selected KV."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.compress_ratio = model_args.hca_compress_ratio
        self.topk = model_args.indexer_topk
        self.scale = 1.0 / math.sqrt(model_args.head_dim)
        self.softmax_clamp = model_args.attn_softmax_clamp
        idx_dim = model_args.indexer_n_heads * model_args.indexer_head_dim
        self.idx_dim = idx_dim
        self.indexer_q = Linear.Config(in_features=model_args.dim, out_features=idx_dim, bias=False).build()
        self.indexer_k = Linear.Config(in_features=model_args.dim, out_features=idx_dim, bias=False).build()
        self.indexer_q_norm = RMSNorm.Config(normalized_shape=idx_dim, eps=model_args.norm_eps).build()
        self.indexer_k_norm = RMSNorm.Config(normalized_shape=idx_dim, eps=model_args.norm_eps).build()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        sink_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, nh, hd = q.shape
        k_c = k[:, :: self.compress_ratio]
        v_c = v[:, :: self.compress_ratio]
        cl = k_c.shape[1]

        idx_q = self.indexer_q_norm(self.indexer_q(hidden_states))
        idx_k = self.indexer_k_norm(self.indexer_k(hidden_states))
        idx_k_c = idx_k[:, :: self.compress_ratio]

        scores = torch.bmm(idx_q.reshape(b, s, -1), idx_k_c.reshape(b, cl, -1).transpose(1, 2))
        scores = scores / math.sqrt(self.idx_dim)
        q_pos = torch.arange(s, device=q.device).unsqueeze(1)
        k_pos = torch.arange(0, s, self.compress_ratio, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, -1e9)

        actual_topk = min(self.topk, cl)
        topk_scores, topk_indices = torch.topk(scores, actual_topk, dim=-1)

        bidx = torch.arange(b, device=q.device).view(b, 1, 1).expand(-1, s, actual_topk)
        k_sel = k_c[bidx, topk_indices].view(b, s, actual_topk, nh, hd)
        v_sel = v_c[bidx, topk_indices].view(b, s, actual_topk, nh, hd)

        attn_scores = torch.einsum("bshd,bskhd->bhsk", q, k_sel) * self.scale
        if sink_bias is not None:
            attn_scores = attn_scores + sink_bias.view(1, -1, 1, 1)
        attn_scores = attn_scores + topk_scores.unsqueeze(1).to(attn_scores.dtype)
        attn_probs = stable_softmax(attn_scores, clamp_val=self.softmax_clamp)
        return torch.einsum("bhsk,bskhd->bshd", attn_probs, v_sel)


class KimiDeltaAttentionCore(nn.Module):
    """KDA core: linear attention with gated delta-rule state update.

    State ``S ∈ [B, H, d_state, d_state]``; per token
    ``S_t = (1 - beta_erase_t) ⊙_row S_{t-1} + beta_write_t ⊗ (v_t - S_{t-1} k_t) k_tᵀ``
    and ``o_t = alpha_t ⊙ (S_t q_t)``. ``d_k``/``d_v`` are zero-padded up to
    ``d_state`` when smaller (raw reference semantics).

    ``_chunk_kda`` is the production-kernel dispatch point: the simulator binds
    a shape-only shim recording ``triton_ascend_kernels.chunk_kda[_grad]`` and a
    future fused kernel plugs in at the same seam. The sequential reference
    below is the exact-math fallback.
    """

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        d = model_args.dim
        self.n_heads = model_args.n_heads
        self.d_state = model_args.kda_d_state
        self.d_k = model_args.kda_d_k
        self.d_v = model_args.kda_d_v
        self.q_proj = Linear.Config(in_features=d, out_features=model_args.n_heads * self.d_k, bias=False).build()
        self.k_proj = Linear.Config(in_features=d, out_features=model_args.n_heads * self.d_k, bias=False).build()
        self.v_proj = Linear.Config(in_features=d, out_features=model_args.n_heads * self.d_v, bias=False).build()
        self.o_proj = Linear.Config(in_features=model_args.n_heads * self.d_v, out_features=d, bias=False).build()
        self.alpha_gate = Linear.Config(in_features=d, out_features=model_args.n_heads * self.d_k, bias=False).build()
        # Two head-major gate projections (one linear each) so TP can colwise
        # shard the head dim; a fused [2, H, ds] output layout would shard
        # across the erase/write dim instead.
        self.erase_gate = Linear.Config(
            in_features=d, out_features=model_args.n_heads * self.d_state, bias=False
        ).build()
        self.write_gate = Linear.Config(
            in_features=d, out_features=model_args.n_heads * self.d_state, bias=False
        ).build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        h, dk, dv, ds = self.n_heads, self.d_k, self.d_v, self.d_state
        q = self.q_proj(hidden_states).view(b, s, h, dk)
        k = self.k_proj(hidden_states).view(b, s, h, dk)
        v = self.v_proj(hidden_states).view(b, s, h, dv)
        alpha = torch.sigmoid(self.alpha_gate(hidden_states).view(b, s, h, dk))
        beta_erase = torch.sigmoid(self.erase_gate(hidden_states).view(b, s, h, ds))
        beta_write = torch.sigmoid(self.write_gate(hidden_states).view(b, s, h, ds))
        out = self._chunk_kda(q, k, v, alpha, beta_erase, beta_write)
        return self.o_proj(out.reshape(b, s, h * dv))

    def _chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta_erase: torch.Tensor,
        beta_write: torch.Tensor,
    ) -> torch.Tensor:
        return self._sequential_kda(q, k, v, alpha, beta_erase, beta_write)

    @staticmethod
    def _fit_state_dim(x: torch.Tensor, state_dim: int) -> torch.Tensor:
        dim = x.shape[-1]
        if dim < state_dim:
            return F.pad(x, (0, state_dim - dim))
        if dim > state_dim:
            return x[..., :state_dim]
        return x

    def _sequential_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta_erase: torch.Tensor,
        beta_write: torch.Tensor,
    ) -> torch.Tensor:
        b, s, h, dk = q.shape
        dv, ds = self.d_v, self.d_state
        state = torch.zeros(b, h, ds, ds, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(s):
            k_state = self._fit_state_dim(k[:, t], ds)
            v_state = self._fit_state_dim(v[:, t], ds)
            q_state = self._fit_state_dim(q[:, t], ds)
            erase = beta_erase[:, t]
            write = beta_write[:, t]

            read_prev = torch.einsum("bhij,bhj->bhi", state, k_state)
            delta = v_state - read_prev
            update = torch.einsum("bhi,bhj->bhij", write * delta, k_state)
            state = state * (1 - erase).unsqueeze(-1) + update

            out = torch.einsum("bhij,bhj->bhi", state, q_state)
            out = self._fit_state_dim(alpha[:, t], ds) * out
            outputs.append(out[..., :dv])
        return torch.stack(outputs, dim=1)


class ArLlmAttention(nn.Module):
    """Per-layer attention module dispatching to CSA / HCA / KDA."""

    def __init__(self, model_args: "ArLlmModel.Config", layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = model_args.layer_type(layer_idx)
        self.n_heads = model_args.n_heads

        if self.layer_type == "kda":
            self.q_proj = None
            self.kv_proj = None
            self.o_proj = None
            self.core = None
            self.kda = KimiDeltaAttentionCore(model_args)
            self.attn_sink = None
            return

        self.kda = None
        self.q_proj = QLowRankProjection(model_args)
        self.kv_proj = KVLowRankProjection(model_args)
        self.o_proj = GroupedOProjection(model_args)
        if self.layer_type == "csa":
            self.core = CompressedSparseAttention(model_args)
        else:
            self.core = HeavilyCompressedAttention(model_args)
        if model_args.use_attn_sink:
            self.attn_sink = nn.Parameter(torch.zeros(model_args.n_heads))
        else:
            self.attn_sink = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.layer_type == "kda":
            return self.kda(hidden_states)

        b, s, _ = hidden_states.shape
        rope_cos = rope_cos[:s]
        rope_sin = rope_sin[:s]
        q_nope, q_rope = self.q_proj(hidden_states)
        k_nope, k_rope, v = self.kv_proj(hidden_states)

        sink_bias = self.attn_sink
        if sink_bias is not None and isinstance(sink_bias, torch.distributed.tensor.DTensor):
            sink_bias = sink_bias.to_local()

        k_rope_exp = k_rope.expand(-1, -1, self.n_heads, -1)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope_exp, rope_cos, rope_sin)
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        if self.layer_type == "csa":
            out = self.core(q_full, k_full, v, sink_bias)
        else:
            out = self.core(q_full, k_full, v, hidden_states, sink_bias)
        return self.o_proj(out)
