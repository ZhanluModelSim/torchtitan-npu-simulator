# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream compressor references and explicit NPU geometry/carry adapters."""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from attn_gym.sparse import lightning_indexer
from torchtitan.config import Configurable
from torchtitan.models.base.module import Module
from torchtitan.models.deepseek_v4.compressor import (
    Compressor as UpstreamCompressor,
)
from torchtitan.models.deepseek_v4.compressor import (
    Indexer as UpstreamIndexer,
)
from torchtitan.models.deepseek_v4.compressor import (
    _hadamard,  # noqa: F401 -- preserve the existing helper import path
)

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.models.common.document_intervals import document_intervals


class CompressorImplementation(Configurable):
    """Select the compressor execution while keeping child overrides composable."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config

    def forward(
        self,
        compressor: "Compressor",
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compressor._forward(x, positions, cu_seqlens)


class NPUCompressor(UpstreamCompressor):
    """NPU document-packed key compression with geometry and carry handling.

    The compressor sees one local token stream, that stream's own document-relative positions and
    (for the ratio-4 overlap only) the **row's** document boundaries.  Pooling is a flat fold of
    ``ratio`` consecutive tokens, and the overlap's crossing-block state is masked at the first
    block of every segment.

    Under context parallelism the stream it is handed is this rank's chunks while the pooling it
    must reproduce is the row's: the module therefore owns both movements -- the restoration onto
    this rank's window (``attention_metadata.key_stream``, the identity at one degree) and the
    ratio-4 carry
    (:class:`~torchtitan_npu.models.common.carry_exchange.CarryExchange`) -- and returns the row's
    blocks with the carry's own block trimmed off.  A caller hands over the stream exactly as the
    layout produced it and never learns that either step happened.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(UpstreamCompressor.Config):
        # The ratio-4 overlap's cross-rank block.  The compressor owns it, so the modules that
        # call this one never learn that the overlap continues on another rank.
        carry_exchange: CarryExchange.Config
        implementation: "CompressorImplementation.Config" = field(
            default_factory=lambda: CompressorImplementation.Config()
        )

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config
        self.implementation = cfg.implementation.build()
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        # The CP mesh: wired by the framework's ``Module.parallelize`` recursion, exactly like
        # the exchange's own.  ``None`` at one degree, where nothing moves.
        self.group = None
        # ``ape`` is a plain score bias on the compression block, not a
        # projection; own it directly like the upstream implementation.  Its
        # param_init is declared on the Config (registry-side, ``_APE_INIT``).
        if self.ape.dtype != torch.float32:
            self.ape = nn.Parameter(self.ape.to(dtype=torch.float32))
        self.carry_exchange = cfg.carry_exchange.build()

    def _parallelize(self, parallelism_context) -> None:
        """Take the CP mesh the framework hands down to every module."""
        self.group = parallelism_context.get_optional_mesh("cp")
        super()._parallelize(parallelism_context)

    @staticmethod
    def _npu_overlap_transform(
        state: torch.Tensor,
        first_indices: torch.Tensor,
        *,
        value: float,
    ) -> torch.Tensor:
        """The C4A overlap window of one projected state.

        The window's left half is the previous block's a-series, rolled
        one block; the right half is the current block's b-series (the
        paper's C^a/C^b halves).  The first-block rows of the left half are
        filled with ``value``: the score state passes ``-inf`` (exactly zero
        softmax weight -- which also annihilates the rolled-in block-0 rows),
        the kv state passes ``0`` (defensive, like the reference
        implementations: its masked rows are multiplied by those exact-zero
        weights anyway).  The first block of every ``cu_seqlens`` segment is
        such a first block, which is what keeps a document -- or the CP
        wrapper's carry segment -- from reading across its own start.
        """
        assert state.size(-1) % 2 == 0, "the overlap window needs the 2*head_dim split"
        head_dim = state.size(-1) // 2
        n_blocks = state.size(1)
        # Functional roll/fill (gather + ``torch.where``) instead of
        # ``torch.roll`` + in-place scatter: the in-place write is dropped by
        # aot_autograd recompute and its data-dependent guard trips make_fx.
        prev_idx = (torch.arange(n_blocks, device=state.device) - 1).clamp_min(0)
        state_a = state[:, prev_idx, :, :head_dim]
        first_mask = torch.zeros(n_blocks, dtype=torch.bool, device=state.device).scatter(0, first_indices, True)
        state_a = torch.where(first_mask.view(1, -1, 1, 1), state.new_full((1,), value), state_a)
        state_b = state[:, :, :, head_dim:]
        return torch.cat([state_a, state_b], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        *,
        attention_metadata=None,
    ) -> torch.Tensor:
        """Compress one stream into the row's blocks.

        ``cu_seqlens`` and ``attention_metadata`` are keyword-only: a module with a sharding
        config is called through the framework's positional contract, which requires every
        positional-or-keyword parameter to be passed positionally.

        Args:
            x: The stream as the layout produced it, ``[T, D]``: the row at one degree, this
                rank's chunks under context parallelism.
            positions: The **key stream's** own document-relative positions, ``[T]`` (the
                entry's ``position_k``: the input's at one degree, this rank's window under CP).
                The pooled blocks rotate at ``positions[..., ::ratio]``.
            cu_seqlens: The **row's** document boundaries (``[n + 1]``), required by the ratio-4
                overlap: the first block of every segment has no predecessor block, and the carry
                window is clipped from the same grid.  The non-overlap pooling ignores it.
            attention_metadata: The entry describing this rank's geometry.  ``None`` means an
                unsharded caller (a test or a reference forward): nothing is restored and no
                carry is prepended.

        Returns:
            The row's pooled key blocks ``[T // ratio, D]``, the carry block already trimmed.
        """
        # The aligned dataloader and trainer validate the grid before model execution.
        trim = 0
        if attention_metadata is not None:
            # The window this rank's pooling must reproduce: identity at one degree, the restored
            # slab under CP.  The carry then completes the document the window starts inside.
            x = attention_metadata.key_stream(x, group=self.group)
            if self.overlap:
                carry = self.carry_exchange(x.unsqueeze(0), positions.unsqueeze(0), cu_seqlens)
                x, positions, cu_seqlens = carry.x.squeeze(0), carry.positions.squeeze(0), carry.cu_seqlens
                trim = carry.trim
        # Implementations consume one packed stream whose boundaries already include every
        # batch row. Restore the container shape only after pooling, including RoPE.
        pooled = self.implementation.forward(
            self,
            x.unsqueeze(0),
            positions.reshape(1, -1),
            cu_seqlens=cu_seqlens,
        )
        pooled = pooled.squeeze(0)
        # The carry's own block is the head of the pooled stream: dropping it leaves exactly the
        # blocks this rank holds of the row.
        return pooled.narrow(0, trim, pooled.size(0) - trim) if trim else pooled

    def _forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project, fold, softmax-gate, normalize and rotate the local stream.

        This is the **unsharded** reference implementation of the pooling itself: the wrapper
        already restored the stream and, for a ratio-4 geometry, prepended the carry block, so
        what arrives here is the one stream to pool.  Returns the pooled stream ``[1, L // ratio, D]``.
        """
        ratio = self.compress_ratio
        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(f"the compressor addresses one packed row, got shape {tuple(x.shape)}.")
        length = x.size(1)
        n_blocks = length // ratio
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x).unflatten(1, (n_blocks, ratio))
            score = self.wgate(x).unflatten(1, (n_blocks, ratio)) + self.ape

        # -- overlap (ratio=4 only): the crossing block's a-series, masked at every segment start --
        if self.overlap:
            if cu_seqlens is None:
                raise ValueError(
                    "the ratio-4 overlap needs the local stream's cu_seqlens: the first block of "
                    "every document has no predecessor block to borrow."
                )
            first_indices = (cu_seqlens[:-1] // ratio).to(torch.int64)
            score = self._npu_overlap_transform(score, first_indices, value=float("-inf"))
            kv = self._npu_overlap_transform(kv, first_indices, value=0.0)

        # -- softmax pool + norm + RoPE (RoPE owns the [nope | rope] split) --
        # The gate is a softmax over the block's pooled rows: ``ratio`` of them, or ``2 * ratio``
        # once the overlap prepended the previous block's a-series.
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(x.dtype))
        return self.rope(kv.squeeze(0).unsqueeze(1), positions=positions.reshape(-1)[::ratio]).squeeze(1).unsqueeze(0)


class NPUIndexer(UpstreamIndexer):
    """Upstream indexer projections with NPU RoPE and key-stream metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(UpstreamIndexer.Config):
        compressor: NPUCompressor.Config  # pyrefly: ignore [bad-override]

    def forward(self, x, qr, *, positions, cu_seqlens=None, attention_metadata=None):
        """Project raw indexer queries, keys, and per-head weights.

        ``x`` is the frame the layout produced (this rank's chunks under CP) and ``qr`` its
        latent, so ``idx_q`` / ``idx_w`` derive from the same rows the queries do and rotate at
        the positions the input layout carried with them.  The keys are the pooled row's, and the
        compressor restores and carries for them exactly as it does for the attention's own
        compressed stream.

        Returns:
            idx_q: Indexer queries ``[T, num_index_heads, index_head_dim]``
                with RoPE applied and Hadamard-rotated.
            idx_k: Indexer compressed keys in the container grid
                ``[num_blocks, index_head_dim]``, Hadamard-rotated.
            idx_w: Per-head indexer weights ``[T, num_index_heads]``
                (the local rows).
        """
        assert attention_metadata is not None
        num_tokens = qr.size(0)
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(num_tokens, self.num_index_heads, self.head_dim)
        idx_q = self.rope(idx_q, positions=positions)
        idx_q = self._rotate_activation(idx_q)
        # The compressor restores this rank's window and, for a ratio-4 geometry, prepends the
        # predecessor's block before pooling; the keys that come back are the row's blocks.  The
        # keys' own positions are the entry's, and the row's grid says which blocks they belong to.
        idx_k = self.compressor(
            x,
            attention_metadata.position_k,
            cu_seqlens=attention_metadata.swa_k.cu_seqlens,
            attention_metadata=attention_metadata,
        )
        idx_k = self._rotate_activation(idx_k)
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w


# Compatibility names remain tied to the optional NPU backend. The default
# reference model uses UpstreamCompressor/UpstreamIndexer directly.
Compressor = NPUCompressor
Indexer = NPUIndexer


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
    _, heads, head_dim = idx_q.shape
    return lightning_indexer(
        idx_q.float().unsqueeze(0),
        idx_k.float().unsqueeze(0),
        idx_w.float().unsqueeze(0) * math.sqrt(heads * head_dim),
        min(topk, idx_k.size(0)),
        causal=True,
        compress_ratio=4,
        cu_seqlens=cu_seqlens.contiguous(),
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        impl="reference",
    ).squeeze(0)


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
    gathered harmlessly and marked ``-inf``, the value a softmax drops. The reference path
    requires ``B == 1``, so the key axis is squeezed. Returns ``[1, L, K]``, head-summed and in
    the model's own score scale (``select_compressed_slots`` cancels the operator's internal
    division; this pass never divides).

    This is the pass ``LightningIndexer`` returns beside the selection, and the fused
    selector's backward consumes the same quantity: at the picks, the student logits are the
    only differentiable edge of the selection.
    """
    pool_start_T1, _, _ = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=idx_q.size(0))
    picked_TKDi = idx_k[(topk_indices + pool_start_T1).clamp_min(0)]
    scores_THK = torch.einsum("thd,tkd->thk", idx_q, picked_TKDi)
    weighted_THK = scores_THK.relu() * idx_w.unsqueeze(-1)
    return weighted_THK.sum(dim=1).masked_fill(topk_indices < 0, -torch.inf)


class LightningIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        """Select the compressed slots and score them differentiably.

        Returns ``(topk_indices, topk_scores)``:

        - ``topk_indices`` ``[B, L, K]`` document-local with ``-1`` for an unused slot.  The
          reference path keeps the selection operator's order: ``gather_attn`` sums the
          selected rows in that order, and re-sorting here would perturb the reference
          numerics (the checked-in golden is exact).  The fused selector canonicalises to
          descending order inside its Function, because there the saved indices must pair
          slot-for-slot with the teacher;
        - ``topk_scores`` ``[B, L, K]``: the student logits at those picks, differentiable in
          ``idx_q``/``idx_k``/``idx_w`` and ``-inf`` on every ``-1`` slot.  This is the value
          contract the reference path exposes; the fused override returns an unread carrier
          on the same edge because its teacher arrives on the gradient.
        """
        # The selector runs on a layer of its own ratio and reads that layer's entry; a
        # geometry without a compressed pool is the only shape it refuses.
        if attention_metadata.ratio <= 1:
            raise ValueError("LightningIndexer requires a compressed geometry (ratio > 1).")
        cu_seqlens = attention_metadata.q.cu_seqlens
        cmp_frame = attention_metadata.cmp_k
        if cmp_frame is None:
            raise ValueError("a compressed geometry needs the entry's compressed frame")
        cu_seqlens_k = cmp_frame.cu_seqlens
        # Selection is discrete: it carries no gradient.  The operator's order is kept as-is
        # for the reference path; the fused selector emits the canonical descending order
        # (see the module docstring), so the two paths agree on the *set* and on ``-1``.
        with torch.no_grad():
            topk_indices = select_compressed_slots(
                idx_q,
                idx_k,
                idx_w,
                cu_seqlens,
                cu_seqlens_k,
                self.index_topk,
            )
        topk_scores = student_logits_at_picks(
            idx_q,
            idx_k,
            idx_w,
            topk_indices,
            cu_seqlens,
            cu_seqlens_k,
        )
        return topk_indices, topk_scores
