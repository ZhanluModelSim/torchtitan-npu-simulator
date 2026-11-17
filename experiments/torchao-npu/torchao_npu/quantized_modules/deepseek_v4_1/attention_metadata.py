# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The quantized attention stack's AI_CPU metadata, built from one geometry's frames.

The sliding window's pair, the MXFP4 selector's two schedules, and the per-row candidate count the
sparse search is bound by: ``sparse_attn.fwd``/``.bwd``, ``lightning_indexer.fwd``/``.fwd_sparse``/
``.bwd`` and ``.sparse_length``.  All of them are operator outputs of the frames alone, so a stack
builds them once per microbatch instead of once per layer.

This module is the *provider*: it takes the frames and the numbers and returns one geometry's whole
set.  Where they ride -- which node, which microbatch boundary, which rank -- is the host's
business, so nothing here imports the host package; frames are read through their attributes.  The
host that calls it is the quantized attention's config
(``torchtitan_npu/extensions/adaptor/torchao/converter.py``), which owns the quantized stack's
entry: the quantized attention is a backend of its own, so it builds its entry here instead of
extending the unquantized one.

The SMLA pair is a copy of the unquantized backend's call -- this package replaces that kernel, so
it restates what the kernel is called with -- and its options are built once, in
``smla_kernel_options`` below, which the kernel module launches through as well: a launch and the
metadata it is handed cannot drift while they read one dict.

The count is the part that cannot be approximated.  QLI writes it once per row when it
publishes a candidate prefix, as ``min(ceil(visible / block_size), topk_blocks)``, and QSLI is
bound by it *exactly*: the schedule places worker boundaries by it and the kernel bounds each
row's tiles by it, so a count that is merely an upper bound makes the two disagree about which
row is complete -- the row is then merged out of workspace slots that were never written, with
no error at all.  It is therefore derived here, from the lengths alone, instead of being read
back off QLI's output: the output only exists after the forward has run, and the whole point of
the metadata is to exist before it.
"""

from __future__ import annotations

__all__ = [
    "LAYOUT",
    "MASK_MODE",
    "QUANT_MODE",
    "QuantAttentionMetadata",
    "QuantLightningIndexerMetadata",
    "QuantSparseAttentionMetadata",
    "build_attention_metadata",
    "candidate_sparse_length",
    "kernel_options",
    "slikg_metadata",
    "smla_kernel_options",
    "smla_metadata",
]

from dataclasses import dataclass

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch
import torch_npu

# The packed-MXFP4 ABI is the only one the quantized pair accepts; the number labels the storage
# format (two E2M1 values per byte plus uE8M0 scales), it is not a precision knob.
QUANT_MODE = 1
try:
    _IS_A5 = torch_npu.npu.get_device_name().startswith("Ascend950")
except (AttributeError, TypeError):
    _IS_A5 = False
# This training adapter packs both Q and K as TND with separate scale tensors.
LAYOUT = "TND"
# Right-down causal masking matches the BF16 selector and the SLIKG backward.  Although QLI and
# QSLI also support mode 0, switching requires coordinated changes to model visibility,
# cmp_residual_k, metadata, and backward semantics.
MASK_MODE = 3


@dataclass(frozen=True, kw_only=True, slots=True)
class QuantSparseAttentionMetadata:
    """The sliding window's kernel metadata: one tensor per direction."""

    fwd: torch.Tensor
    bwd: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class QuantLightningIndexerMetadata:
    """The quantized selector's schedules, and the count a sparse search is bound by."""

    fwd: torch.Tensor
    """QLI's metadata: what any layer that scores visible entries launches, whether it publishes
    the pool (the width is a launch argument) or not."""

    fwd_sparse: torch.Tensor | None
    """QSLI's metadata for a layer that searches the pool, sized by ``sparse_length``; absent when
    the geometry declares no pool."""

    bwd: torch.Tensor
    """SLIKG's metadata, the teacher edge's: BF16 in both precision paths."""

    sparse_length: torch.Tensor | None
    """The valid candidate-block count per packed query row (``int32``, ``[T, 1]``), or ``None``
    with no pool."""


@dataclass(frozen=True, kw_only=True, slots=True)
class QuantAttentionMetadata:
    """One geometry's quantized attention metadata, grouped by the kernels that read it."""

    sparse_attn: QuantSparseAttentionMetadata
    """The sliding window's pair: every geometry has one."""

    lightning_indexer: QuantLightningIndexerMetadata | None
    """The selector's schedules, absent on a geometry that runs no selector."""


def kernel_options(*, ratio: int, q, cmp_k) -> dict[str, object]:
    """The kernel options, read off one frame set.

    The metadata call and the launch take these, and they must agree option-for-option;
    reading one source is what makes that structural rather than a convention two call sites
    have to keep in step.  ``q`` and ``cmp_k`` are frames -- anything with ``cu_seqlens``,
    ``seqused`` and (for a compressed axis) ``residual``.
    """
    if cmp_k is None:
        raise ValueError(
            "the MXFP4 selector reads a compressed pool's frame, which only a layer that pools "
            f"owns: this geometry has none (compress_ratio={ratio})."
        )
    return {
        "cu_seqlens_q": q.cu_seqlens,
        "cu_seqlens_k": cmp_k.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_k`` its reach into the gathered keys.
        "seqused_q": q.seqused,
        "seqused_k": cmp_k.seqused,
        "cmp_residual_k": cmp_k.residual,
        "layout_q": LAYOUT,
        "layout_k": LAYOUT,
        "mask_mode": MASK_MODE,
        "cmp_ratio": ratio,
    }


def candidate_sparse_length(
    *,
    ratio: int,
    q,
    cmp_k,
    num_rows: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """The valid candidate-block count per packed query row, in the kernels' own arithmetic.

    ``visible`` is the row's visible Key length over the compressed axis, the same integer the
    AI_CPU kernel computes from the same options: under the right-down causal mask a group counts
    once the query's own document has produced it, which is
    ``min(actual, (actual * ratio + residual - used_q + offset + active) // ratio)``.  Every row
    that is not addressed (``active == 0``) sees nothing, and a row that sees nothing gets a count
    of zero rather than one block.

    ``num_rows`` is the number of packed query rows this frame set addresses.  It is a shape, so
    nothing here reads a device scalar on the host.
    """
    bounds = q.cu_seqlens
    device = bounds.device
    rows = torch.arange(num_rows, device=device)
    # The row -> document mapping at the row's own numbering: the grid is per document, and a
    # query's offset inside its own document is what its causal window is measured against.
    documents = torch.searchsorted(bounds[1:], rows, right=True)
    offset = rows - bounds.index_select(0, documents).long()
    used_q = q.seqused.index_select(0, documents).long()
    actual = cmp_k.seqused.index_select(0, documents).long()
    active = (used_q - offset) > 0
    if MASK_MODE == 3:
        # A ratio of one addresses the row itself, so its frame carries no trailing group.
        residual = cmp_k.residual
        rem = 0 if residual is None else residual.index_select(0, documents).long()
        visible = (actual * ratio + rem - used_q + offset + active.long()) // ratio
        visible = torch.minimum(visible, actual).clamp_min(0)
    else:
        visible = actual
    visible = torch.where(active, visible, torch.zeros_like(visible))
    # Ceiling, then the cap, then zero: the trailing partial block is always retained, and the cap
    # applies to the block count rather than to the length it was computed from.
    blocks = (visible + block_size - 1) // block_size
    count = torch.where(visible > 0, blocks.clamp(max=topk_blocks), torch.zeros_like(blocks))
    return count.to(torch.int32).view(-1, 1).contiguous()


def smla_kernel_options(*, ratio: int, q, swa_k, cmp_k, window_size: int) -> dict[str, object]:
    """The sliding window's kernel options, read off one frame set.

    The metadata call and the main call take these, and they must agree option-for-option; one
    builder both read is what makes that structural rather than a convention two call sites have
    to keep in step.  The kernel takes the window's frame as well as the compressed one, which is
    what the selector's options below do not: it addresses the compressed axis only.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_ori_kv=swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if cmp_k is None else cmp_k.cu_seqlens,
        seqused_q=q.seqused,
        seqused_ori_kv=swa_k.seqused,
        seqused_cmp_kv=None if cmp_k is None else cmp_k.seqused,
        cmp_residual_kv=None if cmp_k is None else cmp_k.residual,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        # A5 without a compressed axis addresses the window only; elsewhere the compressed mask is
        # the causal one.
        cmp_mask_mode=0 if _IS_A5 and cmp_k is None else 3,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q=LAYOUT,
        layout_kv=LAYOUT,
    )


def smla_metadata(*, ratio: int, q, swa_k, cmp_k, window_size: int, num_heads: int, head_dim: int, index_topk: int):
    """The sliding window's AI_CPU metadata pair, for these frames.

    The hand-written copy of the AscendC attention's own builder, kept here because this package
    owns the quantized attention's entry: the two paths read one interface, each building it in its
    own package.  The geometry is head-major (``num_heads``, ``head_dim``) and the selection width
    sizes the compressed axis' top-k.
    """
    options = smla_kernel_options(ratio=ratio, q=q, swa_k=swa_k, cmp_k=cmp_k, window_size=window_size)
    has_cmp = cmp_k is not None
    geometry = dict(
        ori_topk=0,
        cmp_topk=index_topk if has_cmp else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    return (
        torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
            num_heads,
            1,
            head_dim,
            ori_topk_length=None,
            cmp_topk_length=None,
            **options,
            **geometry,
        ),
        torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
            num_heads,
            1,
            head_dim,
            **options,
            **geometry,
        ),
    )


def slikg_metadata(*, ratio: int, q, cmp_k, index_num_heads: int, index_head_dim: int, index_topk: int):
    """SLIKG's metadata: the teacher edge's, BF16 in both precision paths.

    The quantized selector's backward launches the same BF16 kernel the unquantized one does, so
    this is a copy of the host's call rather than a quantization of it.
    """
    options = kernel_options(ratio=ratio, q=q, cmp_k=cmp_k)
    return torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
        num_heads_q=index_num_heads,
        num_heads_k=1,
        head_dim=index_head_dim,
        topk=index_topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
        **options,
    )


def build_attention_metadata(
    *,
    ratio: int,
    q,
    swa_k,
    cmp_k,
    window_size: int,
    num_heads: int,
    head_dim: int,
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    topk_blocks: int,
    block_size: int,
    num_rows: int,
) -> QuantAttentionMetadata:
    """This geometry's whole AI_CPU metadata set, built once per microbatch.

    One entry per geometry: the sliding window's pair, the quantized selector's two forward
    schedules, the teacher edge's backward metadata, and the count the search is bound by.  A
    geometry with no compressed axis carries the pair and nothing else; one with no candidate pool
    keeps its QLI schedule and loses only the search.
    """
    smla_fwd, smla_bwd = smla_metadata(
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        window_size=window_size,
        num_heads=num_heads,
        head_dim=head_dim,
        index_topk=index_topk,
    )
    if cmp_k is None:
        # A window-only geometry runs no selector, so it carries the pair and nothing else.
        return QuantAttentionMetadata(
            sparse_attn=QuantSparseAttentionMetadata(fwd=smla_fwd, bwd=smla_bwd),
            lightning_indexer=None,
        )
    has_pool = topk_blocks > 0 and block_size > 0
    sparse_length = (
        candidate_sparse_length(
            ratio=ratio,
            q=q,
            cmp_k=cmp_k,
            num_rows=num_rows,
            topk_blocks=topk_blocks,
            block_size=block_size,
        )
        if has_pool
        else None
    )
    selector = kernel_options(ratio=ratio, q=q, cmp_k=cmp_k)
    indexer_geometry = dict(
        num_heads_q=index_num_heads,
        num_heads_k=1,
        head_dim=index_head_dim,
        topk=index_topk,
    )
    return QuantAttentionMetadata(
        sparse_attn=QuantSparseAttentionMetadata(fwd=smla_fwd, bwd=smla_bwd),
        lightning_indexer=QuantLightningIndexerMetadata(
            # The candidate writer's width is not an input of this operator, so one schedule serves
            # a pool source and a pool-free selector; the launch states the width.
            fwd=torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
                **indexer_geometry,
                **selector,
            ),
            # The sparse search's schedule is sized by the count, which only a pool has.
            fwd_sparse=(
                torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata(
                    sparse_length,
                    **indexer_geometry,
                    **selector,
                    quant_mode=QUANT_MODE,
                    candidate_block_size=block_size,
                )
                if has_pool
                else None
            ),
            bwd=slikg_metadata(
                ratio=ratio,
                q=q,
                cmp_k=cmp_k,
                index_num_heads=index_num_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
            ),
            sparse_length=sparse_length,
        ),
    )
