# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_gym.sparse import lightning_indexer
from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.models.common.metadata_extension import LightningIndexerKernelConfig

from .token_dispatcher import CPTokenDispatcher


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class CompressorImplementation(Configurable):
    """Select the compressor execution while keeping child overrides composable."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config

    def forward(self, compressor, x, attention_masks):
        return compressor._forward(x, attention_masks)


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
        implementation: "CompressorImplementation.Config" = field(
            default_factory=lambda: CompressorImplementation.Config()
        )
        # The CP token dispatcher (the RoutedExperts mirror): owned by the
        # compressor, wired once by ``Compressor.parallelize``.
        token_dispatcher: CPTokenDispatcher.Config = field(default_factory=CPTokenDispatcher.Config)

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.implementation = cfg.implementation.build()
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
        n_blocks = state.size(0)
        # Functional roll/fill (gather + ``torch.where``) instead of
        # ``torch.roll`` + in-place scatter: the in-place write is dropped by
        # aot_autograd recompute and its data-dependent guard trips make_fx.
        prev_idx = (torch.arange(n_blocks, device=state.device) - 1).clamp_min(0)
        state_a = state[prev_idx, :, :head_dim]
        first_mask = torch.zeros(n_blocks, dtype=torch.bool, device=state.device).scatter(0, first_indices, True)
        state_a = torch.where(first_mask.view(-1, 1, 1), state.new_full((1,), value), state_a)
        state_b = state[:, :, head_dim:]
        return torch.cat([state_a, state_b], dim=1)

    def forward(self, x, attention_masks):
        return self.implementation.forward(self, x, attention_masks)

    def _forward(self, x, attention_masks):
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
        # The contract is checked by the attribute the compressor reads rather than by
        # ``isinstance``: this protocol is runtime-checkable, and inside a dynamo region the
        # static path rejects metadata that carries every member the check is about.
        if getattr(attention_masks, "plans", None) is None:
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
        # Block count from the plan's (dynamic) gather_indices, not the
        # gathered rows' runtime shape, which would specialize the trace.
        # An empty plan flows through reshape(0, ...) without a numel() guard.
        n_blocks = plan.gather_indices.numel() // ratio
        # Explicit last dim: a ``-1`` fold trips a data-dependent guard.
        kv = kv_rows.reshape(n_blocks, ratio, kv_rows.size(-1))
        score = score_rows.reshape(n_blocks, ratio, score_rows.size(-1)) + self.ape
        first_indices = plan.first_indices
        block_positions = plan.block_positions
        assert first_indices is not None and block_positions is not None, (
            "the compressor contract requires first_indices and block_positions"
        )

        # -- overlap (ratio=4 only) --
        if self.overlap:
            score = self._overlap_transform(score, first_indices, value=float("-inf"))
            kv = self._overlap_transform(kv, first_indices, value=0.0)

        # -- softmax pool + norm + RoPE (RoPE owns the [nope | rope] split) --
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(x.dtype))
        return (
            self.rope(
                kv.unsqueeze(0).unsqueeze(2),
                positions=block_positions.unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(1)
        )


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
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        idx_q = self.rope(idx_q, positions=positions)
        idx_q = self._rotate_activation(idx_q)
        idx_k = self.compressor(x, attention_masks)
        idx_k = self._rotate_activation(idx_k)
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w


def select_compressed_slots(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """The reference selection: Attention Gym's packed ``lightning_indexer``.

    The operator applies each query's own document's causal limit itself and numbers the picks
    inside that document -- the numbering ``gather_attn`` reads -- so nothing rebases them here.
    Every row it cannot fill comes back as ``-1``, and the packed gather only reads entries
    inside the query's own document pool, so an empty pool can never name its neighbour's row.
    Its score is
    ``sum_h w * relu(q . k) / sqrt(H * D)``, and the caller's weights already carry the model's
    own scale, so multiplying them by ``sqrt(H * D)`` cancels that division exactly.  ``topk``
    is clamped to the pool's width, as this model's selection has always been.
    """
    bsz, _, heads, head_dim = idx_q.shape
    if bsz != 1:
        raise ValueError(f"the packed selection addresses one row, got batch size {bsz}.")
    return lightning_indexer(
        idx_q.float(),
        idx_k.float(),
        idx_w.float() * math.sqrt(heads * head_dim),
        min(topk, idx_k.size(1)),
        causal=True,
        compress_ratio=4,
        cu_seqlens=cu_seqlens.contiguous(),
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        impl="reference",
    )


def student_logits_at_picks(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """The per-head scores at a document-local selection: the pass that carries gradient.

    The selection itself is discrete, so only the picks can be scored differentiably -- and a
    selection that returns indices alone leaves nothing to gather them from, which is why this
    is a pass of its own rather than a by-product of the selection.  ``topk_indices`` is
    renumbered onto the row-wide entries the pooled keys are stored in; a ``-1`` slot is
    gathered harmlessly and marked ``-inf``, the value a softmax drops.  The row is the batch,
    so the key axis is squeezed.  Returns ``[B, L, K]``, head-summed and in the model's own
    score scale (``select_compressed_slots`` cancels the operator's internal division; this
    pass never divides).

    No production call site reads this pass yet: the reference recipe's indexer loss lives in
    the golden override, and the fused one runs inside its SLIG bridge.  It is the reference
    teacher pass that the LI/teacher binding is meant to consume, and until that lands it is
    pinned by the unit tests only.
    """
    pool_start_L1, _, _ = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=idx_q.size(1))
    picked_BLKDi = idx_k.squeeze(0)[(topk_indices + pool_start_L1).squeeze(0).clamp_min(0)].unsqueeze(0)
    scores_BLHK = torch.einsum("blhd,blkd->blhk", idx_q, picked_BLKDi)
    weighted_BLHK = scores_BLHK.relu() * idx_w.unsqueeze(-1)
    return weighted_BLHK.sum(dim=2).masked_fill(topk_indices < 0, -torch.inf)


class LightningIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int
        li_kernel_config: LightningIndexerKernelConfig = field(default_factory=LightningIndexerKernelConfig)

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk
        self.li_kernel_config = config.li_kernel_config

    def forward(self, idx_q, idx_k, idx_w, *, attention_masks):
        plan = attention_masks.plans.get(4)
        if plan is None or plan.cu_seqlens_cmp_k is None:
            raise ValueError("LightningIndexer requires a ratio-4 plan carrying a compressed pool.")
        return select_compressed_slots(
            idx_q,
            idx_k,
            idx_w,
            attention_masks.varlen.cu_seq_q,
            plan.cu_seqlens_cmp_k,
            self.index_topk,
        )
