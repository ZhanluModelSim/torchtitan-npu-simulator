# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused AscendC TND kernels.

This module holds the AscendC backend: the record tier the fused path consumes
(``AscV4Metadata`` + the ``*_metadata`` fills, built by
``AscCompressedSparseInnerAttention.Config.build_attention_metadata``, upstream
#4897's seam) and the fused TND kernel (``AscCompressedSparseInnerAttention``).
The ``sparse_attn`` package ``__init__`` defines the registration, so the
override path stays ``override.deepseek_v4.sparse_attn.asc``.

The fused path only bridges the AscendC TND kernels' forward/backward with a
slim ``torch.autograd.Function``.  The selector is the separate
``lightning_indexer.asc`` override: this bridge consumes its ``topk_indices``,
and SMLAG's backward emits the teacher on the selector's ``topk_scores`` edge.
All AscendC metadata is precomputed at the backend's own metadata boundary and
carried in the registered ``AscV4Metadata``, so this module never builds or
caches metadata per layer; because the records describe *frames*, the CP
transform rebuilds them for this rank (``prepare_cp_metadata``).  The CP path's
``S(1) -> R`` sharding gathers the KV streams to the whole row at the wrapper
boundary, and the frames' ``seqused`` limits each rank's view.  Every fused
kernel of arch22, however, anchors its mask at the *end* of the key frame it is
given (it reads no ``seqused`` there), so on that SoC the options state the reach
itself as the frame and the wrapper compacts both gathered rows onto it before the
kernel call -- the entry's ``a3_workaround``, built once per microbatch beside the
records (``override/a3_workaround.py``).  The kernels therefore see the reach as
their key frame's end, while the picks and indices keep the model's
document-local numbering.  There is no container or ``_assemble_tnd`` step
anymore.

The kernel inputs follow the training autocast contract: q/k/v are bf16 and
the sink is fp32 — an eager-fp32 caller fails loudly at the kernel's dtype
check.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from torchtitan.models.common.attention.cp_attention import CPInnerAttention

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_CMP_RATIO,
    AicpuGeometry,
    AicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
    CompactSelectorFrameProof,
    lightning_indexer_options,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata as AscV4Metadata,
)

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


# ---------------------------------------------------------------------------
# AscendC metadata layer
# ---------------------------------------------------------------------------
#
# The AscendC backend's metadata is the model-dir row plus its own record
# tier: after the document-packed row is built, this backend fills
# the precomputed AscendC ``*_metadata`` kernel outputs (lightning_indexer,
# sparse_flash_mla, sparse_flash_mla_grad,
# sparse_lightning_indexer_kl_loss_grad) into an ``AscV4Metadata`` wrapper,
# keeping the model-dir metadata NPU-free.  Those kernels take only shape
# information, so their results are layer-invariant and reused by every DSA
# layer and by the backward pass.  ``AscCompressedSparseInnerAttention.Config``
# builds the whole thing (upstream #4897); because a record describes the frames
# it was derived from, ``prepare_cp_metadata`` rebuilds them per rank.


def _selector_records(
    q: KernelFrame,
    key_frame: KernelFrame | None,
    geometry: AicpuGeometry,
    ratio: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The selector's forward record and its SLIG backward record, for one geometry."""
    if ratio != LI_CMP_RATIO or key_frame is None:
        return None, None
    frames = lightning_indexer_options(ratio=ratio, q=q, cmp_k=key_frame)
    li_record = lightning_indexer_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        geometry.index_topk,
        **frames,
    )
    slig_metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        topk=geometry.index_topk,
        **frames,
    )
    return li_record, slig_metadata


def _sparse_attention_options(
    *,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    ratio: int,
    window_size: int,
):
    """The options the SMLA/SMLAG kernels and their metadata builders share.

    ``swa_k`` / ``cmp_k`` are the frames the kernels are addressed with: the entry's own, which on
    arch22 are the per-document reach this rank may read (``a3_workaround``).
    """
    has_cmp = cmp_k is not None
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_ori_kv=swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if cmp_k is None else cmp_k.cu_seqlens,
        seqused_q=q.seqused,
        seqused_ori_kv=swa_k.seqused,
        seqused_cmp_kv=None if cmp_k is None else cmp_k.seqused,
        cmp_residual_kv=None if cmp_k is None else cmp_k.residual,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=_ORI_MASK_MODE,
        cmp_mask_mode=0 if _IS_A5 and not has_cmp else _CMP_MASK_MODE,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q=_LAYOUT,
        layout_kv=_LAYOUT,
    )


def _sparse_attention_records(
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    geometry: AicpuGeometry,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SMLA/SMLAG records for these frames."""
    options = _sparse_attention_options(
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        ratio=ratio,
        window_size=geometry.window_size,
    )
    has_cmp_kv = cmp_k is not None
    geometry_kwargs = dict(
        ori_topk=0,
        # Only the ratio-4 CSA geometry selects compressed entries.  A ratio-128 HCA layer
        # addresses the whole causal prefix, and the kernel rejects a non-zero ``cmp_topk``
        # with a ratio above 4 (the error the A3 integration run surfaced).
        cmp_topk=geometry.index_topk if ratio == LI_CMP_RATIO else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp_kv,
    )
    smla_metadata = sparse_flash_mla_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        ori_topk_length=None,
        cmp_topk_length=None,
        **options,
        **geometry_kwargs,
    )
    smla_grad_metadata = sparse_flash_mla_grad_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        **options,
        **geometry_kwargs,
    )
    return smla_metadata, smla_grad_metadata


def _compact_selector_frame_proof(
    q: KernelFrame,
    key_frame: KernelFrame,
    positions_q: torch.Tensor,
    key_tokens: int,
) -> CompactSelectorFrameProof | None:
    """Validate once at the eager metadata boundary, never inside a layer graph.

    The single scalar read combines compactness, monotonicity and canonical
    endpoints. Noncompact frames remain available to backends that support
    seqused; they simply receive no compact-only Pro proof.
    """
    q_cu, q_used = q.cu_seqlens, q.seqused
    k_cu, k_used = key_frame.cu_seqlens, key_frame.seqused
    if (
        q_cu.ndim != 1
        or k_cu.ndim != 1
        or q_used.ndim != 1
        or k_used.ndim != 1
        or q_cu.numel() < 2
        or q_cu.shape != k_cu.shape
        or q_used.numel() != q_cu.numel() - 1
        or k_used.numel() != k_cu.numel() - 1
    ):
        return None
    q_lengths, k_lengths = torch.diff(q_cu), torch.diff(k_cu)
    valid = (
        (q_cu[0] == 0)
        & (k_cu[0] == 0)
        & (q_cu[-1] == positions_q.numel())
        & (k_cu[-1] == key_tokens)
        & (q_lengths >= 0).all()
        & (k_lengths >= 0).all()
        & (q_used == q_lengths).all()
        & (k_used == k_lengths).all()
    )
    if not bool(valid):
        return None
    tensors = [q_cu, q_used, k_cu, k_used, positions_q]
    try:
        versions = tuple(tensor._version for tensor in tensors)
    except RuntimeError:
        # Inference tensors do not track versions. Do not change another
        # backend's usable frames merely because Pro cannot certify them.
        return None
    return CompactSelectorFrameProof(
        q_cu=q_cu,
        q_seqused=q_used,
        k_cu=k_cu,
        k_seqused=k_used,
        positions_q=positions_q,
        selector_k_gather=None,
        key_tokens=key_tokens,
        versions=versions,
    )


def build_aicpu_metadata(
    metadata: DeepSeekV4Metadata,
    *,
    geometry: AicpuGeometry,
    positions_q: torch.Tensor | None = None,
    include_selector: bool = True,
) -> AscV4Metadata:
    """The AscendC record tier over one geometry's frames (upstream #4897).

    The frames are the entry's own (``q`` / ``swa_k`` / ``cmp_k``); this builder reads them and
    returns the same entry with this geometry's records attached.  Because a record describes the
    frames it was computed for, a re-cut entry -- the CP transform's rank-local chunks -- runs
    this again rather than reusing records of the frame they came from.

    ``positions_q`` is the query axis this build describes -- the row's positions at build time,
    this rank's after the CP transform re-cuts it.  Only the compact-only selector's proof needs
    it; without one the entry simply carries no proof and that backend refuses, which is why it is
    optional rather than read off the entry (the entry no longer stores the row's positions).
    """
    ratio = metadata.ratio
    q = metadata.q
    swa_k = metadata.swa_k
    cmp_k = metadata.cmp_k
    if cmp_k is not None and ratio <= 1:
        cmp_k = None
    cmp_tokens = int(cmp_k.cu_seqlens[-1]) if cmp_k is not None and ratio > 1 else 0
    if cmp_k is not None and ratio > 1 and cmp_tokens == 0:
        raise ValueError(
            f"batch has no complete compression block for ratio={ratio}; doc-packed sequences "
            "must be long enough to produce at least one full block."
        )
    # arch22 anchors its mask at the key frame's end and reads no ``seqused``, so its key frames are
    # the per-document reach this rank may read.  The workaround is built here, once per microbatch
    # (the walk inside it synchronises once), and the entry keeps what it built.
    workaround = None
    if not _IS_A5:
        workaround, swa_k, cmp_k = A3KeyWorkaround.build(swa_k=swa_k, cmp_k=cmp_k)
    li_record, slig_metadata = _selector_records(q, cmp_k, geometry, ratio) if include_selector else (None, None)
    smla_metadata, smla_grad_metadata = _sparse_attention_records(q, swa_k, cmp_k, geometry, ratio)
    record = AicpuMetadata(
        geometry=geometry,
        sparse_attn=AscSparseAttentionMetadata(fwd=smla_metadata, bwd=smla_grad_metadata),
        lightning_indexer=AscLightningIndexerMetadata(fwd=li_record, bwd=slig_metadata),
    )
    # The compact-only Pro selector certifies the exact tensors it reads, so it needs the query
    # axis this build describes.  The entry no longer carries the row's positions -- the frames
    # and the plan say everything else -- so the caller hands them in: the row's at build time,
    # this rank's when the CP transform rebuilds the entry.
    compact_proof = None
    if (
        include_selector
        and positions_q is not None
        and cmp_k is not None
        and ratio == LI_CMP_RATIO
        and geometry.index_num_heads == 64
        and geometry.index_head_dim == 128
    ):
        compact_proof = _compact_selector_frame_proof(q, cmp_k, positions_q, int(cmp_k.cu_seqlens[-1]))
    result = AscV4Metadata(
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        position_k=metadata.position_k,
        batch_size=metadata.batch_size,
        slots=metadata.slots,
        cp=metadata.cp,
        aicpu_metadata=record,
        a3_workaround=workaround,
        compact_selector_frame_proof=compact_proof,
    )
    _mark_dynamic(result)
    return result


def _mark_dynamic(metadata: AscV4Metadata) -> None:
    """Mark batch-dependent leading dimensions so Dynamo does not specialize."""
    tensors: list[torch.Tensor] = [
        metadata.q.cu_seqlens,
        metadata.q.seqused,
        metadata.swa_k.cu_seqlens,
        metadata.swa_k.seqused,
    ]
    if metadata.cmp_k is not None:
        tensors += [metadata.cmp_k.cu_seqlens, metadata.cmp_k.seqused]
        if metadata.cmp_k.residual is not None:
            tensors.append(metadata.cmp_k.residual)
    if not _IS_A5:
        workaround = metadata.a3_workaround
        assert workaround is not None, "an arch22 entry states the key workaround"
        tensors += [gather for gather in (workaround.swa_k_gather, workaround.cmp_k_gather) if gather is not None]
    if metadata.cp is not None:
        # The restore's scatter table moves with the slab width and the batch's document layout;
        # ``position_k`` is the geometry's own field and is marked here with the rest of the
        # tensors whose first dimension follows the batch.
        tensors += [metadata.position_k, metadata.cp.restore.exchange.send_index]
    if metadata.aicpu_metadata.lightning_indexer.fwd is not None:
        tensors.append(metadata.aicpu_metadata.lightning_indexer.fwd)
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)
        # Written via setattr: "_dynamo_unbacked_indices" is the dynamo shape
        # annotation the GraphTrainer tracer reads (torchtitan
        # experiments/graph_trainer/dynamic_shapes.py); the underscore name is
        # deliberate and must stay off the codecheck protected-member list.
        setattr(tensor, "_dynamo_unbacked_indices", {0})  # noqa: B010


class _SparseFlashMLATND(torch.autograd.Function):
    """Bridge SparseFlashMLA forward with SMLAG backward and its teacher by-product.

    The ``cann_ops_transformer`` kernels carry no native autograd registration, so the manual
    Function runs SMLA forward and SMLAG backward.  It does **not** train the indexer: SMLAG's
    backward emits the raw teacher on ``topk_scores``' gradient and the selector's own
    ``_LightningIndexerTND`` backward consumes that edge with SLIKG.  ``topk_scores`` is an
    input here only so an incoming gradient on it exists and is routed to the emitter.

    The kernels are addressed by this geometry's frames, not by a plan: the query frame is this
    rank's chunks and the two KV frames are the row's; on arch22 the options state their reach as
    the frame instead, and the wrapper has already compacted the rows onto it.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        sinks,
        attention_metadata,
        softmax_scale,
        ratio,
        window_size,
        topk_scores,
    ):
        metadata = attention_metadata
        if not isinstance(metadata, AscV4Metadata):
            raise TypeError("the fused core needs AscV4Metadata")
        asc = metadata.aicpu_metadata
        has_compressed = metadata.cmp_k is not None
        options = _sparse_attention_options(
            q=metadata.q,
            swa_k=metadata.swa_k,
            cmp_k=metadata.cmp_k,
            ratio=ratio,
            window_size=window_size,
        )
        result, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            cmp_sparse_indices=(topk_indices if has_compressed else None),
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=asc.sparse_attn.fwd,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            result,
            softmax_lse,
        )
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.window_size = window_size
        # The carrier is an input only while training; its position in the backward's
        # per-input list must still be answered with ``None`` when it was not threaded.
        ctx.has_carrier = topk_scores is not None
        # A dataclass cannot go through ``save_for_backward``; it is a context attribute
        # instead, holding the same frames the graph already keeps for the backward.
        ctx.attention_metadata = metadata
        # The LSE is a teacher signal, never a loss input.
        ctx.mark_non_differentiable(softmax_lse)
        return result, softmax_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse
        (
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            result,
            softmax_lse,
        ) = ctx.saved_tensors
        metadata = ctx.attention_metadata
        has_compressed = metadata.cmp_k is not None
        options = _sparse_attention_options(
            q=metadata.q,
            swa_k=metadata.swa_k,
            cmp_k=metadata.cmp_k,
            ratio=ctx.ratio,
            window_size=ctx.window_size,
        )
        (
            dquery,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(topk_indices if has_compressed else None),
            sinks=sinks,
            metadata=metadata.aicpu_metadata.sparse_attn.bwd,
            softmax_scale=ctx.softmax_scale,
            **options,
        )
        if not has_compressed:
            dcmp_kv = None

        # The teacher edge: SMLAG's raw marginal mass, with every ``-1`` slot zeroed (the
        # kernel may write junk there, and SLIKG sums every slot into ``p_reduce``), scaled by
        # ``softmax_scale`` exactly once here; the selector's backward only divides by the
        # microbatch's query-row count.  Only a layer that threads a carrier has an edge to
        # emit on: an HCA layer's compressed branch has no selector, and an eval forward
        # passes no carrier.
        teacher = None
        if ctx.has_carrier and cmp_softmax_l1 is not None:
            teacher = cmp_softmax_l1.masked_fill(topk_indices < 0, 0.0) * ctx.softmax_scale

        # One gradient per forward input, in order: (q, swa_k, cmp_k, topk_indices, sinks,
        # attention_metadata, softmax_scale, ratio, window_size, topk_scores).
        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,  # topk_indices
            dsinks,
            None,  # attention_metadata
            None,  # softmax_scale
            None,  # ratio
            None,  # window_size
            teacher,  # topk_scores
        )


def run_sparse_flash_mla(
    q,
    swa_k,
    attn_sink,
    *,
    attention_metadata: AscV4Metadata,
    softmax_scale: float,
    compress_ratio: int,
    window_size: int,
    cmp_k=None,
    topk_indices=None,
    teacher_carrier=None,
):
    """Run the existing SMLA bridge and expose its detached TND LSE."""
    if not isinstance(attention_metadata, AscV4Metadata):
        raise TypeError("SMLA requires the AscV4Metadata its config built.")
    if compress_ratio == 4 and (cmp_k is None or topk_indices is None):
        raise ValueError("ratio-4 SMLA requires cmp_k and topk_indices together.")
    metadata = attention_metadata
    if not _IS_A5:
        workaround = metadata.a3_workaround
        assert workaround is not None, "an arch22 entry states the key workaround"
        if workaround.swa_k_gather is not None:
            swa_k = swa_k.index_select(0, workaround.swa_k_gather)
        if cmp_k is not None and workaround.cmp_k_gather is not None:
            cmp_k = cmp_k.index_select(0, workaround.cmp_k_gather)
    swa_k = swa_k.unsqueeze(1).contiguous()
    if cmp_k is not None:
        cmp_k = cmp_k.unsqueeze(1).contiguous()
    indices_tnd = None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1])
    carrier_tnd = None if teacher_carrier is None else teacher_carrier.reshape(-1, 1, teacher_carrier.shape[-1])
    output, lse = _SparseFlashMLATND.apply(
        q,
        swa_k,
        cmp_k,
        indices_tnd,
        attn_sink.float(),
        metadata,
        softmax_scale,
        compress_ratio,
        window_size,
        carrier_tnd,
    )
    return output.reshape(q.shape[0], *output.shape[1:]), lse


class AscCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Run SMLA/SMLAG in a local TND layout, consuming the selector's selection.

    The flow mirrors ``AscSparseInnerAttention`` (dsv3.2): the core reads this geometry's
    frames (queries on this rank's chunks) and calls the AscendC ops directly.  The selector's
    forward/backward is **not** here: it is the separate ``lightning_indexer.asc`` override,
    which emits ``topk_indices`` and the teacher carrier.  This core consumes both, and its
    SMLAG backward writes the raw teacher onto the carrier's gradient.  The container/plan
    assembly is gone: the compressor produces the whole compressed row through the core's
    ``S(1) -> R`` gather, and the frames' ``seqused`` limits each rank's view.  On arch22 both KV
    streams are then compacted onto this rank's reach, because these kernels anchor their mask at
    the key frame's end -- the same frame the selector picked in -- and the gradients scatter back
    through the compaction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        """The fused config: the reference core's fields plus the geometry its records need."""

        num_heads: int  # pyrefly: ignore [bad-override]
        head_dim: int
        """The main attention geometry the SMLA/SMLAG records are sized with."""

        index_num_heads: int
        index_head_dim: int
        """The selector geometry its record and SLIKG backward are sized with."""

        @property
        def geometry(self) -> AicpuGeometry:
            """The numbers the AI_CPU operators are called with, read off this layer's config."""
            return AicpuGeometry(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=self.window_size,
                index_topk=self.index_topk,
                index_num_heads=self.index_num_heads,
                index_head_dim=self.index_head_dim,
            )

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV4Metadata:
            """This backend's per-microbatch metadata (upstream #4897).

            The model-dir plain row plus the record tier: this geometry's four ``*_metadata``
            kernel outputs, computed here once per microbatch for every layer of the geometry.
            """
            del padding_mask, max_num_documents, max_context_length
            from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata

            # The fused core never reads ``slots``: its HCA branch runs full-causal on the
            # compressed pool, so the reference-only S x (S / ratio) selection is skipped here.
            metadata = build_v4_metadata(positions, self.compress_ratio, build_slots=False)
            return build_aicpu_metadata(metadata, geometry=self.geometry, positions_q=positions)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim

    def forward(  # pyrefly: ignore [bad-override]
        self,
        q,
        swa_k,
        attn_sink: torch.Tensor,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        attention_metadata: AscV4Metadata,
    ) -> torch.Tensor:
        output, _lse = run_sparse_flash_mla(
            q,
            swa_k,
            attn_sink,
            attention_metadata=attention_metadata,
            softmax_scale=self.softmax_scale,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            cmp_k=cmp_k,
            topk_indices=topk_indices,
            teacher_carrier=topk_scores if self.training else None,
        )
        return output


class AscV4ContextParallelSparseAttention(CPInnerAttention, AscCompressedSparseInnerAttention):
    """The fused TND backend with its document-chunk CP metadata protocol.

    Forward keeps the existing rank-local query/global key contract: the
    attention wrapper restores the KV stream, and the fused metadata owns
    each rank's query frame and key reach. This class selects that existing
    implementation through TorchTitan's typed CP configuration interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, AscCompressedSparseInnerAttention.Config):
        # Released Flash geometry, also used by the ordinary asc entry.
        num_heads: int = 64
        head_dim: int = 512
        index_num_heads: int = 64
        index_head_dim: int = 128

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        if isinstance(attention_metadata, Mapping):
            return {
                key: AscV4ContextParallelSparseAttention.prepare_cp_metadata(value, permutation=permutation)
                for key, value in attention_metadata.items()
            }
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("V4 context parallel attention requires AscV4Metadata")
        return AscV4Metadata.prepare_cp_metadata(attention_metadata, permutation=permutation)
