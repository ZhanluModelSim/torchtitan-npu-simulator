# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .metadata import CompressedKernelContract
from .token_dispatcher import CPTokenDispatcher


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

    Consumes the unified per-ratio compressor contract from the metadata's
    ``plans[ratio]`` — ``gather_indices`` / ``block_positions`` /
    ``first_indices`` — and returns the pooled key stream ``[n_blocks, D]``.
    The contract is provided identically under context parallel and
    without: the projection (``wkv`` / ``wgate``) is per-token and runs on
    the local stream, and the compressor's own token dispatcher gathers
    the plan-block rows — without context parallel a plain local gather
    (the doc-major ``gather_indices`` over the local stream), under
    context parallel the remote gather + permute (the exchange plus the
    pooled-order ``gather_indices``).  The container packing and the CP
    strip are the call sites' concern.
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
        # The CP token dispatcher (the RoutedExperts mirror): owned by the
        # compressor, wired once by ``Compressor.parallelize``.
        token_dispatcher: CPTokenDispatcher.Config = field(default_factory=CPTokenDispatcher.Config)

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
        self.token_dispatcher = cfg.token_dispatcher.build()
        # ``ape`` is a plain score bias on the compression block, not a
        # projection; own it directly like the upstream implementation.  Its
        # param_init is declared on the Config (registry-side, ``_APE_INIT``).
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features, dtype=torch.float32))

    def parallelize(self, parallel_dims) -> None:
        """Wire the compressor's own CP dispatcher (invoked automatically
        by the framework's ``Module.parallelize`` recursion — the
        compressor is a Module child of the Attention / Indexer)."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    @staticmethod
    def _overlap_transform(
        state: torch.Tensor,
        first_indices: torch.Tensor,
        *,
        value: float,
    ) -> torch.Tensor:
        """The C4A overlap window of one projected state.

        The window's left half is the previous plan block's a-series, rolled
        one block; the right half is the current block's b-series (the
        paper's C^a/C^b halves).  The first-block rows of the left half are
        filled with ``value``: the score state passes ``-inf`` (exactly zero
        softmax weight — which also annihilates the rolled-in block-0 rows),
        the kv state passes ``0`` (defensive, like the reference
        implementations: its masked rows are multiplied by those exact-zero
        weights anyway).  (The CP borrow exchange is a different mechanism —
        the token-level augmented stream.)
        """
        assert state.size(-1) % 2 == 0, "the overlap window needs the 2*head_dim split"
        head_dim = state.size(-1) // 2
        state_a = torch.roll(state[:, :, :head_dim], 1, 0)
        state_a[first_indices] = value
        state_b = state[:, :, head_dim:]
        return torch.cat([state_a, state_b], dim=1)

    def forward(self, x, attention_masks):
        """Project kv/score locally, gather the plan-block rows through the
        dispatcher, pool, and RoPE.

        Returns the pooled key stream ``[n_blocks, head_dim]`` (all plan
        blocks, borrow-source blocks included — the strip and the container
        packing happen at the call sites).  ``first_indices`` are the
        doc/segment-first block positions whose borrowed (previous-block)
        rows are filled by ``_overlap_transform`` (``-inf`` on the score —
        exactly zero softmax weight — and ``0`` on the kv).  The projection
        is per-token, so it commutes with the exchange (bitwise): without
        context parallel the dispatcher's gather is a plain local gather,
        under context parallel it exchanges the projected rows of the
        plan's ``[A, B)`` block region.
        """
        if not isinstance(attention_masks, CompressedKernelContract):
            raise TypeError(
                "DSV4 compression requires a CompressedKernelContract (the "
                "model-dir CompressedVarlenMetadata or the NPU slim type), "
                f"got {type(attention_masks)}."
            )
        ratio = self.compress_ratio
        plan = attention_masks.plans.get(ratio)
        if plan is None or plan.gather_indices is None:
            raise ValueError(f"No compressor plan for ratio={ratio}")

        # -- project the local stream (BF16 weights, FP32 compute via
        #    autocast); the dispatcher's gather then collects the
        #    plan-block rows --
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv_rows = self.token_dispatcher.gather(self.wkv(x), plan).flatten(0, 1)
            score_rows = self.token_dispatcher.gather(self.wgate(x), plan).flatten(0, 1)
        if kv_rows.numel() == 0:
            return x.new_zeros((0, self.head_dim))
        n_blocks = kv_rows.shape[0] // ratio
        kv = kv_rows.reshape(n_blocks, ratio, -1)
        score = score_rows.reshape(n_blocks, ratio, -1) + self.ape
        first_indices = plan.first_indices
        block_positions = plan.block_positions
        assert first_indices is not None and block_positions is not None, (
            "the compressor contract requires first_indices and block_positions"
        )
        rd = self.rope_head_dim
        nope_dim = self.head_dim - rd

        # -- overlap (ratio=4 only) --
        if self.overlap:
            score = self._overlap_transform(score, first_indices, value=float("-inf"))
            kv = self._overlap_transform(kv, first_indices, value=0.0)

        # -- softmax pool + norm + RoPE --
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(x.dtype))
        kv_nope, kv_rope = torch.split(kv, [nope_dim, rd], dim=-1)
        kv_rope = (
            self.rope(
                kv_rope.unsqueeze(0).unsqueeze(2),
                positions=block_positions.unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(1)
        )
        return torch.cat([kv_nope, kv_rope], dim=-1)


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

        ``x`` is the caller's **local** stream (never the augmented
        stream): ``idx_q`` / ``idx_w`` derive from the local ``qr`` and
        the local rows, and the indexer's compressor borrows its own
        block rows internally.

        Returns:
            idx_q: Indexer queries ``[B, L, num_index_heads, index_head_dim]``
                with RoPE applied and Hadamard-rotated.
            idx_k: Indexer compressed keys in the container grid
                ``[B, L // ratio, index_head_dim]``, Hadamard-rotated.
            idx_w: Per-head indexer weights ``[B, L, num_index_heads]``
                (the local rows).
        """
        bsz, seqlen, _ = qr.size()
        rd = self.rope_head_dim
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(idx_q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.rope(q_rope, positions=positions)
        idx_q = torch.cat([q_nope, q_rope], dim=-1)
        idx_q = self._rotate_activation(idx_q)
        idx_k = self.compressor(x, attention_masks)
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
