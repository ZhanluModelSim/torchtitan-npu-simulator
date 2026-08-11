# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed compression metadata for DeepSeek-V4.

Built once per batch by ``CompressedBlockMaskHandler`` from a
``VarlenMetadata`` stream.  The container grid is ``[1, S]`` (the DSV4 packed
scenario runs with ``local_batch_size == 1``; raise ``seq_len`` instead of
``local_batch_size``), so ``batch_size == 1`` and ``seq_len`` is the total
token count ``cu_seq_q[-1]``.  Later this handler is expected to derive the
metadata from ``CPVarlenMetadata`` instead.

The contract is tiered so each consumer materializes exactly what it reads:

- the **kernel contract** (``plans``: ``cu_seqlens_cmp_k`` /
  ``block_remainder`` / ``gather_indices`` per ratio) — consumed by the
  Compressor, the kernels, and every attention path;
- the **reference tier** (``reference``: per-token document ids and
  positions, the dense attendability mask, the container-slot scatter, and
  the static attention block listing) — consumed only by the model-dir
  reference attention (and, for its document ids, the eager golden
  reference).  It is no-CP-shaped (contiguous documents) and required on
  ``CompressedVarlenMetadata``; the NPU fused path uses a separate slim
  metadata type that carries no reference tier at all.

CP-readiness (the kernel contract is the single unified path for plain
and ``CPVarlenMetadata``-shaped streams):

- ``build_kernel_layout`` derives per segment over ``cu_seq_q`` *and*
  ``cu_seq_k`` separately (``p0 = seqlen_k - seg_len``, ``cu_seqs =
  cumsum(seqlen_k // r)``) and gathers through the stream's
  ``k_global_gather_indices`` (the identity when absent) — the contiguous
  per-document form is the degenerate case without context parallel;
- the model-dir reference tier is non-CP-only: the reference build
  enforces ``cu_seq_q == cu_seq_k`` (contiguous documents);
- ``cu_seqlens_ori_kv`` — under CP it becomes the per-segment window-pack
  ori ranges, not a bare ``cu_seq_k`` pass-through;
- ``static_blocks`` is the only derivation that is CP-proof (shape-only);
  the reference tier's document-contiguity assumption re-derives from the
  segment structure under CP (or the golden reference stays no-CP-only).

Documents never span rows, and complete blocks never cross documents: a row's
compressed region is the concatenation of its documents' complete blocks,
padded to ``S // ratio`` slots.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.patches.torchtitan.models.common.mask_handler import BaseMaskHandler

__all__ = [
    "CompressedBlockLayout",
    "CompressedBlockMaskHandler",
    "CompressedKernelContract",
    "CompressedVarlenMetadata",
    "ReferenceLayout",
    "ReferenceRatioLayout",
    "build_compressed_varlen_metadata",
    "build_kernel_layout",
    "derive_reference_layout",
]


@dataclass(kw_only=True, slots=True)
class CompressedBlockLayout:
    """Kernel contract for one compression ratio (the key of ``plans``).

    All tensors are built once per batch by the mask handler.  Their leading
    dimensions are marked dynamic by the fused path so ``torch.compile``
    does not specialize on batch contents.
    """

    cu_seqlens_cmp_k: torch.Tensor | None
    """Cumulative compressed-block lengths over the packed stream (int32),
    ``[n_seqs + 1]``.  Entry ``i`` is the global start index of sequence
    ``i``'s compressed blocks.  ``None`` for ratio-1 plans."""

    block_remainder: torch.Tensor | None
    """Per-sequence block remainder (int32), ``[n_seqs]``: ``len[i] % ratio``
    trailing tokens of sequence ``i`` fall short of one full block and
    produce no compressed KV entry.  ``None`` for ratio-1 plans."""

    gather_indices: torch.Tensor | None
    """Int64 indices into ``x.flatten(0, 1)`` with shape
    ``[n_blocks, ratio]``, in document-major block order.  ``None`` for
    ratio-1 plans.

    For CP-shaped streams the positions are the per-segment complete-block
    kgather slices, which degenerate to this contiguous doc-major form
    without context parallel."""


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


@dataclass(kw_only=True, slots=True)
class CompressedVarlenMetadata:
    """The DeepSeek-V4 varlen attention contract.

    Carried as ``attention_masks`` through the DSA layers.  Built by
    ``CompressedBlockMaskHandler.post_process`` from a ``VarlenMetadata``
    stream: the kernel contract (``plans``) plus the required reference tier
    (``reference``).  The NPU fused path carries a separate slim type
    (``CANNCompressedVarlenMetadata``) with the kernel contract and the CANN
    metadata kernels but no reference tier.

    The container grid is ``[1, S]`` with ``S`` equal to the total token
    count (``local_batch_size == 1``).  Without context parallel,
    ``cu_seq_q`` and ``cu_seq_k`` are the same tensor, so ``cu_seqlens_q``
    and ``cu_seqlens_ori_kv`` coincide.
    """

    varlen: VarlenMetadata
    """Token-stream boundaries (``cu_seq_q`` / ``cu_seq_k``)."""

    batch_size: int
    """Container batch size (``1`` for the current packed scenario)."""

    seq_len: int
    """Container sequence length (the total token count)."""

    plans: dict[int, CompressedBlockLayout]
    """Kernel contract for each ratio present in the model.  Ratio-1 plans
    describe no compressed region."""

    reference: ReferenceLayout
    """The model-dir reference-attention tier (always present on this type)."""

    @property
    def cu_seqlens_q(self) -> torch.Tensor:
        return self.varlen.cu_seq_q

    @property
    def cu_seqlens_ori_kv(self) -> torch.Tensor:
        # CP seam: under context parallel this becomes the per-segment
        # window-pack ori ranges (end-aligned oriLen), not the bare cu_seq_k.
        return self.varlen.cu_seq_k


@runtime_checkable
class CompressedKernelContract(Protocol):
    """The kernel-contract shape shared by the model-dir metadata and the
    NPU slim type: the container shape plus the per-ratio plans.

    Consumers that read only the kernel contract (e.g. the Compressor)
    accept this protocol instead of the concrete metadata type, so the
    fused path's slim metadata (which carries no reference tier) satisfies
    the contract without the model directory knowing about it.
    """

    batch_size: int
    seq_len: int
    plans: dict[int, CompressedBlockLayout]


def _derive_segments(
    varlen: VarlenMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    """The per-segment structure over the (possibly CP-shaped) stream.

    Returns ``(cu_seq_q, cu_seq_k, kgather, segments)`` where ``segments``
    is a list of ``(seg_len, seqlen_k, p0, k_start)`` for the non-empty
    segments.  ``kgather`` is the stream's ``k_global_gather_indices`` or
    the identity mapping when the stream is plain (no context parallel) —
    the only capability difference between the two input shapes.
    """
    cu_q, cu_k = varlen.cu_seq_q, varlen.cu_seq_k
    kgather = getattr(varlen, "k_global_gather_indices", None)
    if kgather is None:
        kgather = torch.arange(
            int(cu_k[-1].item()), device=cu_k.device, dtype=torch.int64
        )
    segments: list[tuple[int, int, int, int]] = []
    for s in range(len(cu_q) - 1):
        seg_len = int(cu_q[s + 1]) - int(cu_q[s])
        if seg_len == 0:
            continue
        seqlen_k = int(cu_k[s + 1]) - int(cu_k[s])
        segments.append((seg_len, seqlen_k, seqlen_k - seg_len, int(cu_k[s])))
    return cu_q, cu_k, kgather, segments


def _derive_ratio_contract(
    cu_k: torch.Tensor,
    kgather: torch.Tensor,
    segments: list[tuple[int, int, int, int]],
    ratio: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel contract for one ratio over the per-segment structure.

    ``cu_seqlens_cmp_k = cumsum(seqlen_k // ratio)``, ``block_remainder =
    seqlen_k % ratio`` per segment, and the flat gather of the complete
    local blocks (``b_first .. b_last``) through the segment's own kgather
    slice.  Without context parallel (identity kgather, ``p0 = 0``) this is
    exactly the contiguous doc-major form.
    """
    c_lens = [seqlen_k // ratio for _, seqlen_k, _, _ in segments]
    remainder = [seqlen_k % ratio for _, seqlen_k, _, _ in segments]
    cu_seqs = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.tensor(c_lens, dtype=torch.int32, device=device).cumsum(
                0, dtype=torch.int32
            ),
        ]
    )
    pieces = []
    for seg_len, seqlen_k, p0, k_start in segments:
        b_first = (p0 + ratio - 1) // ratio
        b_last = (p0 + seg_len) // ratio - 1
        for b in range(b_first, b_last + 1):
            start = k_start + b * ratio
            pieces.append(kgather[start : start + ratio])
    gather = (
        torch.stack(pieces, dim=0)
        if pieces
        else torch.empty((0, ratio), dtype=torch.int64, device=device)
    )
    return cu_seqs, torch.tensor(remainder, dtype=torch.int32, device=device), gather


def build_kernel_layout(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> tuple[int, int, dict[int, CompressedBlockLayout]]:
    """The kernel-contract tier: container shape + per-ratio plans.

    Called by the mask handler and composed by the NPU fused handler (the
    CANN path materializes no reference tier).  The container grid is
    ``[1, S]`` with ``S`` equal to the total token count, so the layout is
    derived purely from the varlen stream and the ratios.

    The single path for both plain and context-parallel streams: the
    per-segment derivations use ``cu_seq_q`` and ``cu_seq_k`` separately
    (``p0 = seqlen_k - seg_len``) and gather through the stream's
    ``k_global_gather_indices`` (the identity when absent), so the contract
    degenerates exactly to the contiguous per-document form without context
    parallel.  The contiguous-document requirement lives only on the
    reference tier (see ``build_compressed_varlen_metadata``).
    """
    if not hasattr(varlen, "cu_seq_q"):
        raise TypeError(
            f"build_kernel_layout expects a varlen stream, got {type(varlen)}."
        )

    cu_seq_q = varlen.cu_seq_q
    total_tokens = int(cu_seq_q[-1].item())
    if int(cu_seq_q[0].item()) != 0:
        raise ValueError(
            f"varlen stream must start at token 0, got cu_seq_q[0]={cu_seq_q[0]}."
        )
    # The packed scenario runs with local_batch_size == 1; raise seq_len
    # instead of local_batch_size.
    batch_size, seq_len = 1, total_tokens

    _, cu_k, kgather, segments = _derive_segments(varlen)
    distinct_ratios = sorted({int(r) for r in compress_ratios})
    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in distinct_ratios:
        if ratio == 1:
            plans[1] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
        elif ratio == 4 or ratio == 128:
            cu_seqs, remainder, gather = _derive_ratio_contract(
                cu_k, kgather, segments, ratio, cu_seq_q.device
            )
            plans[ratio] = CompressedBlockLayout(
                cu_seqlens_cmp_k=cu_seqs,
                block_remainder=remainder,
                gather_indices=gather,
            )
        else:
            raise NotImplementedError(
                f"CompressedBlockLayout does not support ratio={ratio}; "
                "expected 1, 4, or 128."
            )
    return batch_size, seq_len, plans


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
    plans: dict[int, CompressedBlockLayout],
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
    doc_of_token_flat = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32), lengths
    )
    pos_in_doc_flat = (
        torch.arange(total_tokens, device=device) - cu_seq_q[doc_of_token_flat.long()]
    ).to(torch.int32)
    doc_of_token = doc_of_token_flat.view(batch_size, seq_len)
    pos_in_doc = pos_in_doc_flat.view(batch_size, seq_len)

    ratios: dict[int, ReferenceRatioLayout] = {}
    for ratio, plan in plans.items():
        if ratio == 1:
            ratios[1] = ReferenceRatioLayout(
                static_blocks=_build_static_blocks(
                    seq_len, 0, 1, window_size, block_size, device
                ),
            )
            continue
        container_width = seq_len // ratio
        n_blocks = int(
            plan.cu_seqlens_cmp_k[-1].item()  # pyrefly: ignore [unsupported-operation]
        )
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
            seq_ids = torch.searchsorted(
                plan.cu_seqlens_cmp_k[1:],  # pyrefly: ignore [unsupported-operation]
                bids,
                right=True,
            )
            local_idx = (
                bids
                - plan.cu_seqlens_cmp_k[  # pyrefly: ignore [unsupported-operation]
                    seq_ids
                ]
            )
            doc_of_block = torch.full(
                (batch_size * container_width,), -1, dtype=torch.int32, device=device
            )
            block_local = torch.full(
                (batch_size * container_width,), -1, dtype=torch.int32, device=device
            )
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
            static_blocks=_build_static_blocks(
                seq_len, seq_len // ratio, ratio, window_size, block_size, device
            ),
        )

    return ReferenceLayout(
        doc_of_token=doc_of_token,
        pos_in_doc=pos_in_doc,
        ratios=ratios,
    )


def build_compressed_varlen_metadata(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
    *,
    window_size: int,
    block_size: int | tuple[int, int],
) -> CompressedVarlenMetadata:
    """Build the DSV4 varlen contract for one rank-local token stream.

    Called once per batch by the mask handler: the kernel contract
    (``build_kernel_layout``) plus the required reference tier
    (``derive_reference_layout``).

    Args:
        varlen: Rank-local ``VarlenMetadata``.  ``cu_seq_q`` is
            authoritative for document boundaries.
        compress_ratios: Compression ratios present in the model.
        window_size: Sliding-window size (model-config constant).
        block_size: Reference attention block size (model-config constant).

    The reference tier is contiguous-document (non-CP) only: this entry
    enforces ``cu_seq_q == cu_seq_k``.  The unified kernel contract
    (``build_kernel_layout``) has no such requirement and serves
    context-parallel streams for the fused path.
    """
    if not torch.equal(varlen.cu_seq_q, varlen.cu_seq_k):
        raise ValueError(
            "the model-dir reference tier requires cu_seq_q == cu_seq_k "
            "(contiguous documents); the unified kernel contract supports "
            "context-parallel streams, but the reference tier is "
            "non-CP-only."
        )
    batch_size, seq_len, plans = build_kernel_layout(varlen, compress_ratios)
    reference = derive_reference_layout(
        varlen.cu_seq_q,
        plans,
        batch_size,
        seq_len,
        window_size,
        block_size,
        varlen.cu_seq_q.device,
    )
    return CompressedVarlenMetadata(
        varlen=varlen,
        batch_size=batch_size,
        seq_len=seq_len,
        plans=plans,
        reference=reference,
    )


class CompressedBlockMaskHandler(BaseMaskHandler):
    """Derive the DSV4 compression layout from a rank-local varlen stream.

    The handler runs once per batch in ``Trainer.post_dataloading_process``.
    The model fills the model-config constants (``compress_ratios``,
    ``window_size``, ``block_size``) into the config at build time.  The
    NPU fused handler extends this class but composes only the kernel
    contract (``build_kernel_layout``), so the reference tier is not
    materialized on the fused path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseMaskHandler.Config):
        compress_ratios: tuple[int, ...]
        window_size: int
        block_size: int | tuple[int, int] = (128, 128)

    def __init__(self, config: Config):
        super().__init__(config)
        self.compress_ratios = tuple(config.compress_ratios)
        self.window_size = config.window_size
        self.block_size = config.block_size

    def post_process(  # pyrefly: ignore [bad-override]
        self, masks: VarlenMetadata
    ) -> CompressedVarlenMetadata:
        return build_compressed_varlen_metadata(
            masks,
            self.compress_ratios,
            window_size=self.window_size,
            block_size=self.block_size,
        )
