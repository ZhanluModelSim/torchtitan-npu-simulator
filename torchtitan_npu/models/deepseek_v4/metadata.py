# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed attention metadata for DeepSeek-V4.

The contract is the V4.1 one: **one** dataclass with **one** set of fields describes a geometry at
every context-parallel degree, and only the values differ.  An entry is built once per microbatch
per geometry by the attention backend's config (the map key is that backend's class plus the
layer's compression ratio) and handed to every layer of the geometry, exactly as upstream #4897's
seam does.

The container grid is ``[B, S]``: ``B`` is the query container's leading dimension and ``S`` its
per-sample sequence length, so at CP=1 the flattened microbatch has ``B * S`` tokens.  The fused
path supports ``B > 1`` at CP=1; CP>1 currently requires a single batch row.  The frames below are
**document**-granular, so the container's row count has no carrier among them and is stated once
as :attr:`DeepSeekV4Metadata.batch_size`.

The row is described by one :class:`~torchtitan_npu.models.common.metadata.KernelFrame` per
addressed tensor -- the queries, the sliding-window keys and, for a geometry that pools, the
compressed keys.  :func:`build_v4_metadata` derives them from the row's boundaries, so the
reference core and a fused port grid the same frames; a vendor backend attaches only the AI_CPU
records its kernels launch.  A kernel that cannot read these frames as they are -- arch22's TND
kernels ignore ``seqused`` and anchor their mask at the frame's end -- restates them at its own call
site (``override/a3_workaround.py``) rather than here: nothing about the row changes.

Under context parallelism the queries are **chunk ``r`` of every document** while the KV side
stays the **whole row**.  The input layout (``models/common/restore.py``) moves the batch's
tokens, positions and labels onto the chunk frame once per batch, so ``q.cu_seqlens`` is the
rank-local chunk grid and the KV frames address the row; a layer that projects keys restores its
hidden states onto the slab with the entry's :attr:`DeepSeekV4Metadata.cp`, which is what removes
the per-layer query movement and the output un-permutation.  There are no ``cu_seq_q`` /
``cu_seq_k`` aliases: read the grid from the frame that addresses it (row grid =
``swa_k.cu_seqlens``, query grid = ``q.cu_seqlens``, compressed grid = ``cmp_k.cu_seqlens``).

Nothing here describes the compressor's *internal* stream: under the ratio-4 overlap the carry
block a rank prepends lives in ``models/common/carry_exchange.CarryExchange``, which builds the
augmented boundaries at the call site and hands them to the compressor as an argument.  A
compressor is therefore CP-agnostic: it sees one local stream, one set of positions and, for the
overlap, one set of ``cu_seqlens``.
"""

from dataclasses import dataclass, replace

import torch
from torchtitan.models.common.attention import create_varlen_metadata_for_document

from torchtitan_npu.models.common.alignment import require_document_alignment
from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.common.restore import (
    DOC_CHUNK_LAYOUT,
    RestorePlan,
    chunk_plan,
    slab_positions,
)

__all__ = [
    "CPDeepSeekMetadata",
    "DeepSeekV4Metadata",
    "KernelFrame",
    "build_v4_metadata",
    "chunk_query_positions",
    "doc_chunk_entry",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class CPDeepSeekMetadata:
    """One geometry's doc-chunk layout: what the key side restores, and nothing else.

    The plan is a function of the row's boundaries, the degree and the rank -- no operand, no
    layer -- so it is derived once per microbatch with the entry and read by every layer of the
    geometry.  At one degree the whole object is ``None``: the slab *is* the frame.
    """

    restore: RestorePlan
    """doc-chunk -> slab: the movement the key projections read.

    The positions of the stream it produces are the entry's ``position_k``, so the movement and
    the geometry stay separate fields even though both are derived from the same boundaries.
    """


@dataclass(kw_only=True, slots=True)
class DeepSeekV4Metadata:
    """One geometry's DeepSeek-V4 attention contract (the V4.1 shape).

    Handed to each DSA layer through the decoder's metadata map, **one entry per geometry**: the
    map key is the attention config's own class plus the layer's compression ratio, so a layer
    reads the entry built for the axis its kernels grid.  A vendor backend extends it with its own
    records (the AscendC entry adds the AI_CPU metadata).
    """

    ratio: int
    """The compression ratio this entry describes.  Every consumer grids this ratio: a layer reads
    the entry built for its own, so no consumer checks another geometry's plan."""

    q: KernelFrame
    """The query frame: the row's boundaries at one degree, this rank's chunks under context
    parallelism, with ``seqused`` the per-document query count."""

    swa_k: KernelFrame
    """The sliding-window keys' frame: the whole row at every degree, with ``seqused`` how far
    these queries reach into each document.  Its ``cu_seqlens`` are the **row's** grid -- the one
    a consumer reads when it needs the whole row rather than its own frame."""

    cmp_k: KernelFrame | None
    """The compressed pool's frame (``None`` for a geometry that does not pool): the
    ``ratio``-folded row, with ``seqused`` the reach on the compressed grid and ``residual`` its
    trailing partial group."""

    position_k: torch.Tensor
    """The key stream's own document-relative positions (``[C]``, ``C`` the row under one degree).

    They belong to the stream the key projections read, so they are a property of the geometry, not
    of any module: at one degree they are the input's ``positions``; under context parallelism they
    are ``slab_positions(...)`` -- the restored window's -- which is why a consumer never derives
    them and never learns that the stream moved.
    """

    cp_layout = DOC_CHUNK_LAYOUT
    """The layout these frames describe (``--parallelism.context-parallel-load-balancer``).

    A rank-local entry of this stack exists for the doc-chunk arrangement only, so the CP seam can
    resolve an unnamed layout to it instead of guessing.
    """

    batch_size: int = 1
    """The query container's leading dimension.

    The frames describe documents, not rows, so the container's shape is the one thing they
    cannot carry; the builder states it once and every degree keeps it.
    """

    slots: torch.Tensor | None = None
    """The reference selection for a geometry that does not run a selector.

    A geometry that does not select (any ratio but four) reaches every causally complete
    compressed group of the query's own document, and that set is a function of the row's boundaries
    and this entry's ratio alone -- no operand, no layer.  It is therefore built with the entry
    (``[1, seq_len, n_cmp]``, document-local numbering with ``-1`` for an unreachable slot), so
    the reference forward reads it instead of deriving it per layer.  A ratio-4 geometry's
    selection is the selector's, and a fused entry never reads this field."""

    cp: CPDeepSeekMetadata | None = None
    """The doc-chunk layout under context parallelism, ``None`` at one degree.

    It carries the movement, not the frames and not the positions: those are the same fields at
    every degree (``position_k`` already describes the stream this rank's keys read), and the
    entry's grids already describe the degree it was built for."""

    def key_stream(self, x: torch.Tensor, *, group) -> torch.Tensor:
        """The stream this entry's key projections read: the input's, or this rank's window.

        At one degree the layout left the row in place, so the input *is* the stream; under
        context parallelism it is the window this rank's chunks are restored onto.  A consumer
        hands over the stream as the layout produced it and never branches on the degree.
        """
        if self.cp is None:
            return x
        return self.cp.restore.apply(x.unsqueeze(0), group=group).squeeze(0)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: "DeepSeekV4Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "DeepSeekV4Metadata":
        """Refuse context parallel: only the fused stack has a rank-local form.

        The row is the whole packed sequence and this entry hands one grid to both sides of the
        attention, so a rank that holds a chunk has no metadata a reference consumer could read.
        A CP run therefore stops here, before any layer runs, and the fused entry
        (``AscV4Metadata``) overrides this with the frames and records rebuilt from the rank's own
        chunks.
        """
        del attention_metadata, permutation
        raise ValueError(
            "the DeepSeek-V4 reference stack does not support context parallel: its metadata "
            "describes one contiguous document stream, so it cannot describe a rank-local row. "
            "The current eager V4 upgrade requires CP=1. Legacy geometry backends "
            "require explicit configuration and separate validation."
        )


def _hca_slots(cu_seq_q: torch.Tensor, cu_seq_k: torch.Tensor, ratio: int) -> torch.Tensor:
    """The HCA selection: every causally complete compressed group of the query's document.

    The reference counterpart of a selector's picks: a geometry without a selector does not
    search, it reaches its whole visible prefix, so the slots follow from the two grids and
    the ratio.
    Numbering is document-local with ``-1`` for an unreachable slot, the coordinate system
    ``gather_attn`` reads.
    """
    seqlen = int(cu_seq_q[-1])
    n_cmp = int(cu_seq_k[-1])
    pool_start_L1, pool_end_L1, local_L1 = document_intervals(cu_seq_q, cu_seq_k, num_tokens=seqlen)
    ends_L1 = torch.minimum(pool_end_L1, pool_start_L1 + (local_L1 + 1) // ratio)
    slots_1N = torch.arange(n_cmp, device=cu_seq_q.device).unsqueeze(0)
    local_LN = slots_1N.expand(seqlen, -1) - pool_start_L1
    visible_LN = (slots_1N >= pool_start_L1) & (slots_1N < ends_L1)
    return torch.where(visible_LN, local_LN, -1).unsqueeze(0)


def _frames_from_row(
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    ratio: int,
) -> tuple[KernelFrame, KernelFrame, KernelFrame | None]:
    """The frames of a plain row: boundary differences as the prefixes."""
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    q = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
    swa_k = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
    cmp_k = None
    if ratio > 1:
        # ``cu_seq_k`` was folded at this entry's own ratio by the backend that built it.
        cmp_k = KernelFrame(
            cu_seqlens=cu_seq_k.to(torch.int32),
            seqused=torch.diff(cu_seq_k).to(torch.int32),
            residual=(lengths % ratio).to(torch.int32),
        )
    return q, swa_k, cmp_k


def _frames_from_chunks(
    cu_seq_q: torch.Tensor,
    ratio: int,
    *,
    cp_size: int,
    cp_rank: int,
) -> tuple[KernelFrame, KernelFrame, KernelFrame | None]:
    """The frames of a rank's doc-chunk query frame over the whole-row KV frame.

    The queries are chunk ``cp_rank`` of every document, so their grid is the row's folded at the
    degree and their ``seqused`` is one chunk.  The keys stay the row, and their ``seqused`` is
    how far these queries reach into each document -- ``(cp_rank + 1)`` chunks -- which is what
    keeps the kernels' end-aligned window and causal limit anchored at this rank's chunk.
    """
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    chunk = (lengths // cp_size).to(torch.int32)
    reach = ((cp_rank + 1) * chunk).to(torch.int32)
    q = KernelFrame(cu_seqlens=(cu_seq_q // cp_size).to(torch.int32), seqused=chunk)
    swa_k = KernelFrame(cu_seqlens=cu_seq_q.to(torch.int32), seqused=reach)
    cmp_k = None
    if ratio > 1:
        cmp_k = KernelFrame(
            cu_seqlens=(cu_seq_q // ratio).to(torch.int32),
            seqused=(reach // ratio).to(torch.int32),
            residual=(reach % ratio).to(torch.int32),
        )
    return q, swa_k, cmp_k


def build_v4_metadata(
    positions: torch.Tensor,
    ratio: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    build_slots: bool = True,
) -> DeepSeekV4Metadata:
    """Build one geometry's entry from the pre-shard row positions.

    Called once per microbatch per geometry by the attention backend's config: the row's
    boundaries, this layer's ratio (the only per-geometry part of the grid) and the frames that
    address them.  The vendor records (the AscendC kernel tensors) are added by that backend's own
    builder.

    Args:
        positions: Per-token positions (``[T]``), resetting at every document start. Upstream
            ``create_varlen_metadata_for_document`` reads the boundaries from them.
        ratio: The compression ratio this entry describes -- the layer's own.
        cu_seq_q: The row's boundaries, when the caller already has them.  ``None`` derives them
            from ``positions``.
        build_slots: Whether to precompute the HCA selection for a geometry without a selector
            (``ratio`` above four).  The reference core reads it, so its builder keeps the
            default ``True``; the fused core never reads it and passes ``False``, which keeps
            the ``S x (S / ratio)`` int64 matrix off the fused path.
    """
    if ratio < 1:
        raise ValueError(f"a compression ratio must be positive, got {ratio}.")
    if positions.dim() != 1:
        raise ValueError(f"positions must be [T], got shape {tuple(positions.shape)}.")
    if cu_seq_q is None:
        cu_seq_q = create_varlen_metadata_for_document(positions).cu_seq_q
    cu_seq_q = cu_seq_q.to(torch.int32)
    require_document_alignment(cu_seq_q, ratio=ratio)
    if int(cu_seq_q[0]) != 0 or int(cu_seq_q[-1]) != positions.numel():
        raise ValueError(
            f"the boundaries describe a different row than the positions: cu_seq_q spans "
            f"[{int(cu_seq_q[0])}, {int(cu_seq_q[-1])}) but the positions hold {positions.numel()} tokens."
        )
    cu_seq_k = cu_seq_q if ratio == 1 else (cu_seq_q // ratio).to(torch.int32)
    q, swa_k, cmp_k = _frames_from_row(cu_seq_q, cu_seq_k, ratio)
    # A geometry without a selector (ratio above four) has a fixed selection: the reference
    # builds it here, once per microbatch and geometry, so its forward only reads it.  The
    # fused core never reads ``slots`` (its HCA branch runs full-causal on the compressed pool),
    # so its builder passes ``build_slots=False`` and keeps this matrix off the fused path.
    slots = None
    if build_slots and ratio not in (1, 4):
        slots = _hca_slots(cu_seq_q, cu_seq_k, ratio)
    return DeepSeekV4Metadata(
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        position_k=positions,
        batch_size=1,
        slots=slots,
    )


def chunk_query_positions(cu_seq_q: torch.Tensor, cp_rank: int) -> torch.Tensor:
    """The document-relative positions of one rank's chunk queries (``[C]``).

    Chunk ``r`` of a document starts at ``r * length / cp_size``, so its queries carry that offset
    plus ``0 .. chunk - 1``.  Device arithmetic over the rank-local query frame alone: the CP
    transform has the boundaries, not the row's positions, and the caller that needs them (the
    compact-only selector's proof) only ever asks about this rank's own axis.
    """
    lengths = torch.diff(cu_seq_q)
    starts = torch.repeat_interleave(cu_seq_q[:-1], lengths)
    offsets = torch.repeat_interleave(cp_rank * lengths, lengths)
    return torch.arange(starts.numel(), device=cu_seq_q.device) - starts + offsets


def doc_chunk_entry(
    attention_metadata: DeepSeekV4Metadata,
    *,
    cp_size: int,
    cp_rank: int,
    slab_len: int,
) -> DeepSeekV4Metadata:
    """This rank's entry: the doc-chunk query frame, the row's KV frames and the restore.

    ``attention_metadata`` is the row's entry (built from the pre-shard positions); ``slab_len`` is
    the **slab** width -- the shape of the rows this rank holds, not of the row.  The frames and
    the plan are derived from the row's boundaries alone, so the entry crosses no collective.
    """
    row = attention_metadata.swa_k.cu_seqlens
    ratio = attention_metadata.ratio
    require_document_alignment(row, cp_size=cp_size, ratio=ratio)
    q, swa_k, cmp_k = _frames_from_chunks(row, ratio, cp_size=cp_size, cp_rank=cp_rank)
    restore = RestorePlan(exchange=chunk_plan(row, cp_size, cp_rank, slab_len))
    # ``slots`` is the reference selection the fused stack never reads; a rank's entry has no
    # row-wide selection to carry, so it is dropped rather than left describing the row.
    return replace(
        attention_metadata,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        position_k=slab_positions(row, cp_size, cp_rank, slab_len).squeeze(0),
        slots=None,
        cp=CPDeepSeekMetadata(restore=restore),
    )
