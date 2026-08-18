# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DSV4 reference-attention metadata tier and its extension.

The model-dir default attention (``CompressedSparseInnerAttention``) and the
eager golden reference consume a reference tier on the common metadata: the
per-token document ids and positions, the per-ratio dense attendability mask,
the container-slot scatter, and the static attention block listing.  The tier
is **no-CP-shaped** (contiguous documents): the reference build enforces
``cu_seq_q == cu_seq_k``.

Per the tiered contract, the tier is *not* part of the common metadata build
(``build_compressed_varlen_metadata``) — it is delivered through the
``metadata_extension`` seam.  The default ``metadata_extension`` config is
``ReferenceMetadataExtension`` (the golden path and the model-dir default
attention get the tier); the AscendC fused path replaces it with the AscendC
extension, which materializes no reference tier at all.
"""

from dataclasses import dataclass
from typing import cast

import torch

from torchtitan_npu.models.common.metadata_extension import MetadataExtension

from .metadata import CompressedBlockLayout, CompressedVarlenMetadata


@dataclass(kw_only=True, slots=True)
class ReferenceRatioLayout:
    """Reference-attention tensors for one compression ratio.

    None of these are read by the kernels or the fused path; the reference
    tier exists for the model-dir reference attention (and, for the per-token
    document ids, the eager golden reference).
    """

    dense_mask: torch.Tensor | None = None
    """``[B, 1, S, S // ratio]`` boolean attendability over the container
    grid: ``True`` iff the slot holds a compressed block of the query token's
    document and the block is causally reachable.  ``None`` for ratio-1
    plans."""

    doc_of_block: torch.Tensor | None = None
    """``[B, S // ratio]`` document id of each container slot (``-1`` for
    unused slots).  Feeds the reference attention core's mask_mod."""

    block_local: torch.Tensor | None = None
    """``[B, S // ratio]`` document-local block index of each container slot
    (``-1`` for unused slots).  Feeds the reference attention core's mask_mod."""

    static_blocks: torch.Tensor | None = None
    """``[1, 1, n_q_blocks, n_kv_blocks]`` int32 block listing with the
    static parts of the reference attention mask: the sliding window, the
    sink block, and (for ratio > 1) the full compressed range.  The CSA
    top-k blocks are scattered on top per layer."""


@dataclass(kw_only=True, slots=True)
class ReferenceLayout:
    """The model-dir reference-attention tier.

    Explicitly no-CP-shaped (contiguous documents): under context parallel
    it re-derives from the segment structure — or the golden reference stays
    no-CP-only.
    """

    doc_of_token: torch.Tensor
    """``[B, S]`` document id of every token (int32)."""

    pos_in_doc: torch.Tensor
    """``[B, S]`` document-relative position of every token (int32)."""

    ratios: dict[int, ReferenceRatioLayout]
    """Per-ratio reference tensors (dense mask, container scatter, static
    block listing)."""


def _build_dense_mask(
    doc_of_block: torch.Tensor,
    block_local: torch.Tensor,
    doc_of_token: torch.Tensor,
    pos_in_doc: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Attendability over the container grid: same document and causally
    reachable block (``block_local < (pos_in_doc + 1) // ratio``)."""
    same_doc = doc_of_block.unsqueeze(1) == doc_of_token.unsqueeze(2)
    causal_limit = torch.div(pos_in_doc + 1, ratio, rounding_mode="floor").unsqueeze(2)
    causal = block_local.unsqueeze(1) < causal_limit
    return (same_doc & causal).unsqueeze(1)


def _build_static_blocks(
    seq_len: int,
    n_cmp: int,
    ratio: int,
    window_size: int,
    block_size: int | tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Static part of the reference attention block listing.

    ``[1, 1, n_q_blocks, n_kv_blocks]`` int32 with the sliding-window blocks,
    the sink block, and (for ratio > 1) the full compressed range.  The CSA
    top-k blocks are scattered on top per layer.  Shape-only (no document
    boundaries) — the one derivation that is unchanged under context
    parallel.
    """
    bq, bk = block_size if isinstance(block_size, tuple) else (block_size, block_size)
    assert seq_len % bq == 0, f"seq_len ({seq_len}) must be divisible by {bq}"
    kv_len = seq_len + n_cmp + 1
    n_kv_blocks = (kv_len + bk - 1) // bk
    n_q_blocks = seq_len // bq
    sink_idx = seq_len + n_cmp

    bm = torch.zeros(1, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device)
    q0 = (torch.arange(n_q_blocks, device=device) * bq).unsqueeze(1)
    kv_ids = torch.arange(n_kv_blocks, device=device).unsqueeze(0)

    first_window_block = (q0 - window_size + 1).clamp_min(0) // bk
    last_window_block = (q0 + bq - 1) // bk
    window_blocks = (kv_ids >= first_window_block) & (kv_ids <= last_window_block)
    bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | window_blocks.to(torch.int32)

    if ratio > 1:
        first_cmp_block = seq_len // bk
        last_cmp_block = (seq_len + n_cmp - 1) // bk
        cmp_blocks = (kv_ids >= first_cmp_block) & (kv_ids <= last_cmp_block)
        bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | cmp_blocks.to(torch.int32)

    bm[:, 0, :, sink_idx // bk] = 1
    return bm


def derive_reference_layout(
    cu_seq_q: torch.Tensor,
    plans: dict[int, "CompressedBlockLayout"],
    batch_size: int,
    seq_len: int,
    window_size: int,
    block_size: int | tuple[int, int],
    device: torch.device,
) -> ReferenceLayout:
    """The model-dir reference-attention tier (no-CP-shaped, contiguous docs).

    Per-token document ids and in-document positions, the per-ratio dense
    attendability mask, the container-slot scatter, and the static block
    listing.  Under context parallel this re-derives from the segment
    structure — or the golden reference stays no-CP-only.
    """
    total_tokens = int(cu_seq_q[-1].item())
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    doc_of_token_flat = torch.repeat_interleave(torch.arange(len(lengths), device=device, dtype=torch.int32), lengths)
    pos_in_doc_flat = (torch.arange(total_tokens, device=device) - cu_seq_q[doc_of_token_flat.long()]).to(torch.int32)
    doc_of_token = doc_of_token_flat.view(batch_size, seq_len)
    pos_in_doc = pos_in_doc_flat.view(batch_size, seq_len)

    ratios: dict[int, ReferenceRatioLayout] = {}
    for ratio, plan in plans.items():
        if ratio == 1:
            ratios[1] = ReferenceRatioLayout(
                static_blocks=_build_static_blocks(seq_len, 0, 1, window_size, block_size, device),
            )
            continue
        container_width = seq_len // ratio
        cu_cmp = plan.cu_seqlens_cmp_k
        assert cu_cmp is not None, "ratio > 1 plans must carry cu_seqlens_cmp_k"
        n_blocks = int(cu_cmp[-1].item())
        if n_blocks == 0:
            empty_slots = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, container_width)
            doc_of_block = empty_slots
            block_local = empty_slots
            dense_mask = _build_dense_mask(
                empty_slots,
                empty_slots,
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        else:
            bids = torch.arange(n_blocks, device=device, dtype=torch.int64)
            seq_ids = torch.searchsorted(cu_cmp[1:], bids, right=True)
            local_idx = bids - cu_cmp[seq_ids]
            doc_of_block = torch.full((batch_size * container_width,), -1, dtype=torch.int32, device=device)
            block_local = torch.full((batch_size * container_width,), -1, dtype=torch.int32, device=device)
            doc_of_block[:n_blocks] = seq_ids.to(torch.int32)
            block_local[:n_blocks] = local_idx.to(torch.int32)
            dense_mask = _build_dense_mask(
                doc_of_block.view(batch_size, container_width),
                block_local.view(batch_size, container_width),
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        ratios[ratio] = ReferenceRatioLayout(
            dense_mask=dense_mask,
            doc_of_block=doc_of_block.view(batch_size, container_width),
            block_local=block_local.view(batch_size, container_width),
            static_blocks=_build_static_blocks(seq_len, seq_len // ratio, ratio, window_size, block_size, device),
        )

    return ReferenceLayout(
        doc_of_token=doc_of_token,
        pos_in_doc=pos_in_doc,
        ratios=ratios,
    )


@dataclass(kw_only=True, slots=True)
class ReferenceCompressedVarlenMetadata(CompressedVarlenMetadata):
    """The common contract plus the reference tier (added by the extension)."""

    reference: ReferenceLayout


class ReferenceMetadataExtension(MetadataExtension):
    """The reference tier post-process (the default ``metadata_extension``).

    Wraps the common metadata (``build_compressed_varlen_metadata``'s
    output) with the reference-attention tensors.  The tier is no-CP-shaped
    (contiguous documents — ``cu_seq_q == cu_seq_k``); under context
    parallel the fused AscendC path is required (the reference tier and the
    golden stay no-CP-only).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MetadataExtension.Config):
        block_size: int | tuple[int, int] = (128, 128)
        """Reference attention block size (model-config constant)."""

    def __call__(self, metadata) -> ReferenceCompressedVarlenMetadata:
        if not isinstance(metadata, CompressedVarlenMetadata):
            raise TypeError(
                "the reference tier requires the model-dir common metadata "
                f"(CompressedVarlenMetadata), got {type(metadata)}."
            )
        if not torch.equal(metadata.varlen.cu_seq_q, metadata.varlen.cu_seq_k):
            raise ValueError(
                "the reference tier requires cu_seq_q == cu_seq_k (contiguous "
                "documents); under context parallel the fused AscendC path is "
                "required — the reference tier and the golden are no-CP-only."
            )
        cfg = cast("ReferenceMetadataExtension.Config", self.config)
        reference = derive_reference_layout(
            metadata.varlen.cu_seq_q,
            metadata.plans,
            metadata.batch_size,
            metadata.seq_len,
            cfg.window_size,
            cfg.block_size,
            metadata.varlen.cu_seq_q.device,
        )
        return ReferenceCompressedVarlenMetadata(
            varlen=metadata.varlen,
            plans=metadata.plans,
            reference=reference,
        )
