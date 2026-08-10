# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .metadata import CompressedKernelContract


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class Compressor(Module):
    """Document-packed key compression.

    Consumes the ``CompressedBlockLayout`` from the mask handler: gathers the
    complete-block tokens of every document (never across documents), pools
    each block (with CSA overlap for ``ratio == 4``), applies RoPE at the
    document-relative block starts, and scatters the blocks back into the
    ``[B, S // ratio, D]`` container grid (unused slots stay zero).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        head_dim: int
        rope_head_dim: int
        compress_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()
        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        # ``ape`` is a plain score bias on the compression block, not a
        # projection; own it directly like the upstream implementation.  Its
        # param_init is declared on the Config (registry-side, ``_APE_INIT``).
        self.ape = nn.Parameter(
            torch.empty(cfg.compress_ratio, self.wkv.out_features, dtype=torch.float32)
        )

    def forward(self, x, positions, attention_masks):
        """Gather complete-block tokens, project, pool, scatter to container."""

        if not isinstance(attention_masks, CompressedKernelContract):
            raise TypeError(
                "DSV4 compression requires a CompressedKernelContract (the "
                "model-dir CompressedVarlenMetadata or the NPU slim type), "
                f"got {type(attention_masks)}."
            )
        metadata = attention_masks
        if x.shape[:2] != (metadata.batch_size, metadata.seq_len):
            raise ValueError(
                "DeepSeek-V4 packing requires local_batch_size == 1; the batch "
                f"shape is {tuple(x.shape[:2])} but the container grid is "
                f"[{metadata.batch_size}, {metadata.seq_len}]. Raise seq_len "
                "instead of local_batch_size."
            )
        batch_size, seqlen, _ = x.shape
        ratio = self.compress_ratio
        plan = metadata.plans.get(ratio)
        if plan is None or plan.gather_indices is None:
            raise ValueError(f"No CompressedBlockLayout for ratio={ratio}")

        dtype = x.dtype
        rd = self.rope_head_dim
        nope_dim = self.head_dim - rd

        # -- gather complete-block tokens from the flat token stream --
        flat_x = x.flatten(0, 1).float()
        block_tokens = flat_x[plan.gather_indices]
        n_blocks = plan.gather_indices.numel() // ratio
        if n_blocks == 0:
            return x.new_zeros((batch_size, seqlen // ratio, self.head_dim))
        block_tokens = block_tokens.reshape(n_blocks, ratio, -1)

        # -- document-local block starts and overlap validity (B=1 contract:
        #    derived per layer from the packed stream, like the removed
        #    layout fields) --
        bids = torch.arange(n_blocks, device=x.device)
        seq_ids = torch.searchsorted(
            plan.cu_seqlens_cmp_k[1:],  # pyrefly: ignore [unsupported-operation]
            bids,
            right=True,
        )
        block_local = (
            bids
            - plan.cu_seqlens_cmp_k[seq_ids]  # pyrefly: ignore [unsupported-operation]
        )
        block_positions = (block_local * ratio).to(torch.int32)
        overlap_valid = block_local != 0

        # -- project gathered tokens (BF16 weights, FP32 compute via autocast) --
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(block_tokens)
            score = self.wgate(block_tokens) + self.ape

        # -- overlap (ratio=4 only) --
        if self.overlap:
            head_dim = self.head_dim
            overlap_kv = kv.new_zeros(n_blocks, 2 * ratio, head_dim)
            overlap_score = score.new_full(
                (n_blocks, 2 * ratio, head_dim), float("-inf")
            )
            overlap_kv[:, ratio:] = kv[:, :, head_dim:]
            overlap_score[:, ratio:] = score[:, :, head_dim:]
            prev_idx = (torch.arange(n_blocks, device=x.device) - 1).clamp_min(0)
            valid = overlap_valid.view(-1, 1, 1)  # pyrefly: ignore [missing-attribute]
            overlap_kv[:, :ratio] = torch.where(
                valid, kv[prev_idx, :, :head_dim], overlap_kv[:, :ratio]
            )
            overlap_score[:, :ratio] = torch.where(
                valid, score[prev_idx, :, :head_dim], overlap_score[:, :ratio]
            )
            kv, score = overlap_kv, overlap_score

        # -- softmax pool + norm + RoPE --
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [nope_dim, rd], dim=-1)
        kv_rope = (
            self.rope(
                kv_rope.unsqueeze(0).unsqueeze(2),
                positions=block_positions.unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(1)
        )
        compressed = torch.cat([kv_nope, kv_rope], dim=-1)

        # -- scatter the packed blocks into the container grid --
        container = compressed.new_zeros(
            (batch_size * (seqlen // ratio), compressed.shape[-1])
        )
        container[:n_blocks] = compressed
        return container.view(batch_size, seqlen // ratio, -1)


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: "Compressor.Config"
        num_index_heads: int
        index_head_dim: int
        rope_head_dim: int

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, dtype=x.dtype, device=x.device)
        return F.linear(x, H) * (d**-0.5)

    def forward(self, x, qr, *, positions, attention_masks):
        """Project raw indexer queries, keys, and per-head weights.

        Returns:
            idx_q: Indexer queries ``[B, L, num_index_heads, index_head_dim]``
                with RoPE applied and Hadamard-rotated.
            idx_k: Indexer compressed keys in the container grid
                ``[B, L // ratio, index_head_dim]``, Hadamard-rotated.
            idx_w: Per-head indexer weights ``[B, L, num_index_heads]``.
        """
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(idx_q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.rope(q_rope, positions=positions)
        idx_q = torch.cat([q_nope, q_rope], dim=-1)
        idx_q = self._rotate_activation(idx_q)
        idx_k = self.compressor(
            x,
            positions=positions,
            attention_masks=attention_masks,
        )
        idx_k = self._rotate_activation(idx_k)
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w

    @staticmethod
    def select(
        idx_q,
        idx_k,
        idx_w,
        dense_mask: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the top-k compressed container slots per query.

        Scores are masked by the precomputed attendability dense mask (same
        document and causally reachable), so all returned indices are
        attendable.  Unused slots hold ``-1``.

        Returns:
            topk_indices: ``[B, L, K]`` container-grid slots, ``K =
                min(topk, S // ratio)``.
            index_scores: ``[B, L, S // ratio]`` masked indexer scores.
        """
        index_score = torch.einsum("bshd,btd->bsht", idx_q.float(), idx_k.float())
        index_score = index_score.relu_() * idx_w.float().unsqueeze(-1)
        index_score = index_score.sum(dim=2)

        k = min(topk, idx_k.shape[1])
        index_score = index_score.where(dense_mask.squeeze(1), float("-inf"))
        topk_scores, topk_indices = index_score.topk(k, dim=-1)
        return topk_indices.where(topk_scores.isfinite(), -1), index_score
