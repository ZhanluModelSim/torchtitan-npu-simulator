# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek Sparse Attention (DSA) layers for GLM-5.3-Flash.

Mirrors ``Glm5NextTextAttention`` + ``Glm5NextTextIndexer`` in MindSpeed-MM
``modeling_glm5_next.py``: MLA (q/kv LoRA) + a k-pool compressed lightning
indexer selecting top-k token pools, with the absorbed-MLA sparse attention
(``npu_sparse_flash_attention`` in production). The whole indexer runs
under ``torch.no_grad()`` (frozen by design — its parameters get no
gradients in production). The 0.3.0 capture is ATen level: the eager
absorbed-attention fallback computes the same math through explicit
gathers and stays meta-safe.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm
from torchtitan.protocols.module import Module

from .common import linear_config


class DSAIndexer(Module):
    """K-pool compressed lightning indexer (frozen: no_grad forward)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        q_lora_rank: int = 1536
        index_n_heads: int = 32
        index_head_dim: int = 128
        index_topk: int = 2048
        index_kpool: int = 16
        kpool_always_select_tail: bool = False
        wq_b: Linear.Config | None = None
        wk: Linear.Config | None = None
        k_norm: LayerNorm.Config | None = None
        weights_proj: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.wq_b is None:
                self.wq_b = linear_config(self.q_lora_rank, self.index_n_heads * self.index_head_dim)
            if self.wk is None:
                self.wk = linear_config(self.dim, self.index_head_dim)
            if self.k_norm is None:
                self.k_norm = LayerNorm.Config(normalized_shape=self.index_head_dim, eps=1e-6)
            if self.weights_proj is None:
                self.weights_proj = linear_config(self.dim, self.index_n_heads)
            # Both compression tensors ship as exact zeros in the reference.
            self.param_init = {
                "index_kpool_compress_ape": nn.init.zeros_,
                "index_kpool_compress_gate": nn.init.zeros_,
            }

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.index_kpool = config.index_kpool
        self.kpool_always_select_tail = config.kpool_always_select_tail
        self.softmax_scale = config.index_head_dim**-0.5

        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(config.index_kpool, config.index_head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(config.index_head_dim, config.dim))

    def _gather_rows(self, source: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
        return source.index_select(0, flat_index)

    def _pooled_states(self, packed_states: torch.Tensor):
        keys, gate_scores, valid_keys = torch.split(packed_states, [self.head_dim, self.head_dim, 1], dim=-1)
        valid_keys = valid_keys.bool().squeeze(-1)
        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool

        # Pooling starts at the first valid token; every op below is
        # shape/position arithmetic (no value reads), so it stays meta-safe.
        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=keys.device),
        )
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=keys.device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets

        safe_indices = pool_indices.clamp(0, seq_len - 1)
        flat_idx = (safe_indices + torch.arange(batch_size, device=keys.device)[:, None, None] * seq_len).reshape(-1)
        pool_shape = (batch_size, number_of_pools, self.index_kpool)

        grouped_keys = self._gather_rows(keys.reshape(batch_size * seq_len, -1), flat_idx).view(*pool_shape, -1)
        grouped_gate_scores = self._gather_rows(gate_scores.reshape(batch_size * seq_len, -1), flat_idx).view(
            *pool_shape, -1
        )
        grouped_valid_keys = self._gather_rows(valid_keys.reshape(-1), flat_idx).view(*pool_shape)

        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = torch.where(grouped_valid_keys, pool_indices, -1)

        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = torch.where(grouped_valid_keys[..., None], logits, float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)

        keep = pool_valid.any(0)
        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def _lightning_score_eager(self, q, pool_keys, weights, valid_candidates):
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * self.softmax_scale)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        return torch.where(valid_candidates, index_scores, torch.finfo(index_scores.dtype).min)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        pool_keys, pool_indices, pool_valid = self._pooled_states(packed_states)
        # With the all-valid uniform mask every pool whose end token is in
        # range is visible to every query; pool_end is meta-safe arithmetic.
        kv_len = seq_len
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = pool_end.ge(0)
        valid_candidates = pool_visible & pool_valid[:, None]

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = self._lightning_score_eager(q, pool_keys, weights, valid_candidates)

        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        # Expand the per-pool validity to the query grid explicitly: meta's
        # gather shape check rejects the broadcast form.
        selected_valid = (
            valid_candidates.to(torch.int8).expand(-1, index_scores.shape[1], -1).gather(-1, selected).bool()
        )
        num_pools = pool_indices.shape[1]
        flat_sel = (
            selected + torch.arange(batch_size, device=hidden_states.device)[:, None, None] * num_pools
        ).reshape(-1)
        selected_indices = self._gather_rows(
            pool_indices.reshape(batch_size * num_pools, self.index_kpool), flat_sel
        ).view(*selected.shape, self.index_kpool)

        topk_indices = selected_indices.flatten(-2)
        selected_valid_flat = selected_valid[..., None].expand_as(selected_indices).flatten(-2)
        topk_indices = torch.where(selected_valid_flat, topk_indices, -1)

        # The sfa kernel consumes front-packed indices; pad to index_topk
        # with invalid (-1) entries.
        output_width = self.index_topk
        pad_width = output_width - topk_indices.shape[-1]
        if pad_width > 0:
            padded = topk_indices.new_full((*topk_indices.shape[:-1], output_width), -1)
            padded[..., : topk_indices.shape[-1]] = topk_indices
            topk_indices = padded
        topk_indices = topk_indices[..., :output_width]
        topk_indices = torch.where(attention_mask[..., None], topk_indices, torch.zeros_like(topk_indices))
        return topk_indices.long()


class DSAAttention(Module):
    """MLA + DSA indexer with the absorbed sparse-attention path."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_attention_heads: int = 64
        q_lora_rank: int = 1536
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 256
        # GLM-5.3-Flash DSA is NoPE: qk_rope_head_dim must be 0 (validated by
        # the reference config: "Expecting NoPE for the DSA attention
        # layers"), so qk_head_dim == qk_nope_head_dim and the kv_a_proj
        # output is the bare latent.
        qk_rope_head_dim: int = 0
        v_head_dim: int = 256
        norm_eps: float = 1e-6
        indexer: DSAIndexer.Config = field(default_factory=DSAIndexer.Config)
        q_a_proj: Linear.Config | None = None
        q_a_layernorm: RMSNorm.Config | None = None
        q_b_proj: Linear.Config | None = None
        kv_a_proj_with_mqa: Linear.Config | None = None
        kv_a_layernorm: RMSNorm.Config | None = None
        kv_b_proj: Linear.Config | None = None
        o_proj: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.qk_rope_head_dim != 0:
                raise ValueError(
                    f"GLM-5.3-Flash DSA layers are NoPE; qk_rope_head_dim must be 0, got {self.qk_rope_head_dim}"
                )
            if self.indexer.dim != self.dim:
                self.indexer.dim = self.dim
            if self.indexer.q_lora_rank != self.q_lora_rank:
                self.indexer.q_lora_rank = self.q_lora_rank
            if self.q_a_proj is None:
                self.q_a_proj = linear_config(self.dim, self.q_lora_rank)
            if self.q_a_layernorm is None:
                self.q_a_layernorm = RMSNorm.Config(normalized_shape=self.q_lora_rank, eps=self.norm_eps)
            if self.q_b_proj is None:
                self.q_b_proj = linear_config(
                    self.q_lora_rank, self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
                )
            if self.kv_a_proj_with_mqa is None:
                self.kv_a_proj_with_mqa = linear_config(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
            if self.kv_a_layernorm is None:
                self.kv_a_layernorm = RMSNorm.Config(normalized_shape=self.kv_lora_rank, eps=self.norm_eps)
            if self.kv_b_proj is None:
                self.kv_b_proj = linear_config(
                    self.kv_lora_rank,
                    self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
                )
            if self.o_proj is None:
                self.o_proj = linear_config(self.num_attention_heads * self.v_head_dim, self.dim)

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scaling = self.qk_head_dim ** (-0.5)

        self.q_a_proj = config.q_a_proj.build()
        self.q_a_layernorm = config.q_a_layernorm.build()
        self.q_b_proj = config.q_b_proj.build()
        self.kv_a_proj_with_mqa = config.kv_a_proj_with_mqa.build()
        self.kv_a_layernorm = config.kv_a_layernorm.build()
        self.kv_b_proj = config.kv_b_proj.build()
        self.o_proj = config.o_proj.build()
        self.indexer = config.indexer.build()
        # The lightning indexer is frozen by design: its forward runs under
        # torch.no_grad() and production never trains these parameters.
        self.indexer.requires_grad_(False)

    def _absorbed_weights(self):
        """Split kv_b_proj.weight into W_UK / W_UV ([N, kv_lora, *] views)."""
        weight = self.kv_b_proj.weight
        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.to_local()
        weight = weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        return weight[:, : self.qk_nope_head_dim, :], weight[:, self.qk_nope_head_dim :, :]

    def _eager_absorbed_attention(self, query_states, k_pass, topk_indices):
        """Meta-safe absorbed MLA fallback: gather selected latents per query
        and run explicit absorbed attention. query_states [B, S, N, qk_nope];
        k_pass [B, S, kv_lora]."""
        _, s, _, _ = query_states.shape
        w_uk, w_uv = self._absorbed_weights()
        q_exp = torch.einsum("bsnd,ndc->bsnc", query_states, w_uk.to(query_states.dtype))

        idx = topk_indices.clamp(0, s - 1)
        latent_sel = (
            k_pass[:, None].expand(-1, s, -1, -1).gather(2, idx[..., None].expand(-1, -1, -1, k_pass.shape[-1]))
        )  # [B, S, K, C]: latent_sel[b, i, k, :] = k_pass[b, idx[b, i, k], :]

        scores = torch.einsum("bsnc,bskc->bsnk", q_exp, latent_sel)
        attn = torch.softmax(scores * self.scaling, dim=-1)
        out = torch.einsum("bsnk,bskc->bsnc", attn, latent_sel)
        return torch.einsum("bsnc,nvc->bsnv", out, w_uv.to(out.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks=None,
        positions=None,
        prev_topk_indices=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Under TP the mixer input may arrive as a replicated DTensor; the
        # mask arithmetic and the sharded core run on plain local tensors
        # (qwen3_5 GDN pattern).
        if isinstance(hidden_states, torch.distributed.tensor.DTensor):
            hidden_states = hidden_states.to_local()
        if attention_masks is not None and isinstance(attention_masks, torch.distributed.tensor.DTensor):
            attention_masks = attention_masks.to_local()
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape)

        # NoPE: the kv_a_proj output is the bare compressed latent.
        k_pass = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))

        mask = attention_masks
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        if mask is None:
            mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=hidden_states.device)

        topk_indices = self.indexer(hidden_states, q_resid, mask)
        attn_output = self._eager_absorbed_attention(query_states, k_pass, topk_indices)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        return self.o_proj(attn_output), topk_indices
