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

Nothing here describes the compressor's *internal* stream: the ratio-4 carry belongs to the
context-parallel module replacement, not to the eager compressor.  At one degree a compressor
stays CP-agnostic: it sees one local stream, one set of positions and, for the overlap, one set
of ``cu_seqlens``.
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import VarlenAttentionMetadata, create_varlen_metadata_for_document

from torchtitan_npu.models.common.alignment import require_document_alignment
from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.models.common.metadata import KernelFrame

__all__ = [
    "DeepSeekV4Metadata",
    "KernelFrame",
    "build_v4_metadata",
]


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
    of any module: they are the input's ``positions``.
    """

    documents: VarlenAttentionMetadata | None = None
    """TorchTitan offsets used by the projections, including reserved empty documents."""

    @property
    def cu_seq_q(self) -> torch.Tensor:
        return self.q.cu_seqlens if self.documents is None else self.documents.cu_seq_q

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
