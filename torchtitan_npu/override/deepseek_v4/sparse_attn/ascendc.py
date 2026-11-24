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
from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.indexer_loss import scale_fused_teacher
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_CMP_RATIO,
    AicpuGeometry,
    AicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
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


def build_aicpu_metadata(
    metadata: DeepSeekV4Metadata,
    *,
    geometry: AicpuGeometry,
    include_selector: bool = True,
) -> AscV4Metadata:
    """The AscendC record tier over one geometry's frames (upstream #4897).

    The frames are the entry's own (``q`` / ``swa_k`` / ``cmp_k``); this builder reads them and
    returns the same entry with this geometry's records attached.  Because a record describes the
    frames it was computed for, a re-cut entry -- the CP transform's rank-local chunks -- runs
    this again rather than reusing records of the frame they came from.
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


class _SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        indices,
        sinks,
        attention_metadata,
        options,
        softmax_scale,
        topk_scores,
        aux_loss,
        denominator,
        padding_mask_T,
    ):
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=attention_metadata.aicpu_metadata.sparse_attn.fwd,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(q, swa_k, cmp_k, indices, sinks, output, lse, topk_scores, denominator, padding_mask_T)
        ctx.attention_metadata = attention_metadata
        ctx.options = options
        ctx.softmax_scale = softmax_scale
        ctx.aux_loss = aux_loss
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        q, swa_k, cmp_k, indices, sinks, output, lse, scores, denominator, padding_mask_T = ctx.saved_tensors
        dq, dswa_k, dcmp_k, dsinks, _, mass = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=indices,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=ctx.attention_metadata.aicpu_metadata.sparse_attn.bwd,
            softmax_scale=ctx.softmax_scale,
            **ctx.options,
        )
        teacher = None
        if ctx.aux_loss is not None:
            if mass is None:
                raise RuntimeError("SparseFlashMLA backward returned no teacher mass for the indexer.")
            with spmd.no_typecheck():
                teacher = scale_fused_teacher(
                    ctx.aux_loss, mass, scores, indices, denominator=denominator, padding_mask_T=padding_mask_T
                )
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            teacher,
            None,
            None,
            None,
        )


class AscSparseMLAMixin:
    """Translate packed TND operands and run the shared SMLA/SMLAG pair."""

    training: bool
    softmax_scale: float
    aux_loss: SparseIndexerLoss | None
    uses_fused_indexer = True

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        cmp_k,
        topk_indices,
        topk_scores,
        attention_metadata,
        options,
        compact_keys,
        denominator,
        padding_mask_T,
    ):
        aux_loss = self.aux_loss if self.training and self.uses_fused_indexer else None
        scores = None
        if aux_loss is not None:
            if denominator is None:
                raise RuntimeError("SparseIndexerLoss requires aux_loss_denominator.")
            assert topk_scores is not None and topk_indices is not None
            scores = topk_scores.reshape(-1, 1, topk_scores.shape[-1])
        if compact_keys:
            workaround = attention_metadata.a3_workaround
            assert workaround is not None, "an arch22 entry states the key workaround"
            if workaround.swa_k_gather is not None:
                swa_k = swa_k.index_select(0, workaround.swa_k_gather)
            if cmp_k is not None and workaround.cmp_k_gather is not None:
                cmp_k = cmp_k.index_select(0, workaround.cmp_k_gather)
        return _SparseMLA.apply(
            q,
            swa_k.unsqueeze(1).contiguous(),
            None if cmp_k is None else cmp_k.unsqueeze(1).contiguous(),
            None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1]),
            attn_sink.float(),
            attention_metadata,
            options,
            self.softmax_scale,
            scores,
            aux_loss,
            denominator,
            padding_mask_T,
        )


class AscCompressedSparseInnerAttention(AscSparseMLAMixin, CompressedSparseInnerAttention):
    """Run SMLA/SMLAG in a local TND layout, consuming the selector's selection.

    The core reads this geometry's
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

        aux_loss: SparseIndexerLoss.Config | None = None
        uses_fused_indexer: bool = True

        @property
        def attention_metadata_key(self):
            return (type(self), self.compress_ratio, self.uses_fused_indexer)

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
            from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata

            documents = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            with spmd.no_typecheck():
                cu = torch.unique_consecutive(documents.cu_seq_q)
            row = build_v4_metadata(positions, self.compress_ratio, cu_seq_q=cu, build_slots=False)
            fused = build_aicpu_metadata(row, geometry=self.geometry, include_selector=self.uses_fused_indexer)
            return replace(fused, documents=documents)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None
        self.uses_fused_indexer = config.uses_fused_indexer

    def _parallelize(self, parallelism_context):
        if parallelism_context.get_optional_mesh("tp") is not None:
            raise NotImplementedError("DeepSeek-V4 SMLA requires TP=1.")
        if parallelism_context.get_optional_mesh("cp") is not None and not isinstance(self, CPInnerAttention):
            raise NotImplementedError("DeepSeek-V4 SMLA requires a CPInnerAttention backend for CP>1.")
        super()._parallelize(parallelism_context)

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
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        indices = topk_indices
        if indices is not None and indices.size(-1) < self.index_topk:
            indices = torch.nn.functional.pad(indices, (0, self.index_topk - indices.size(-1)), value=-1)
        with spmd.no_typecheck():
            if not isinstance(attention_metadata, AscV4Metadata):
                raise TypeError("SMLA requires the AscV4Metadata its config built.")
            if self.compress_ratio == 4 and (cmp_k is None or indices is None):
                raise ValueError("ratio-4 SMLA requires cmp_k and topk_indices together.")
            options = _sparse_attention_options(
                q=attention_metadata.q,
                swa_k=attention_metadata.swa_k,
                cmp_k=attention_metadata.cmp_k,
                ratio=self.compress_ratio,
                window_size=self.window_size,
            )
            has_pool = attention_metadata.cmp_k is not None
            output, lse = super().forward(
                q,
                swa_k,
                attn_sink,
                cmp_k=cmp_k if has_pool else None,
                topk_indices=indices if has_pool else None,
                topk_scores=topk_scores if self.uses_fused_indexer else None,
                attention_metadata=attention_metadata,
                options=options,
                compact_keys=not _IS_A5,
                denominator=aux_loss_denominator,
                padding_mask_T=padding_mask_T,
            )
            output = output.reshape(q.shape[0], *output.shape[1:])
            if self.training and self.aux_loss is not None and not self.uses_fused_indexer:
                assert cmp_k is not None and topk_indices is not None and topk_scores is not None
                if aux_loss_denominator is None:
                    raise RuntimeError("SparseIndexerLoss requires aux_loss_denominator.")
                output = self.aux_loss(
                    q.detach(),
                    cmp_k.detach(),
                    topk_indices,
                    topk_scores,
                    lse.squeeze(0).detach(),
                    attention_metadata.cu_seq_q,
                    carrier=output,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
        if spmd.is_type_checking():
            spmd.assert_type(output, spmd.get_local_type(q), spmd.get_partition_spec(q))
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
        return type(attention_metadata).prepare_cp_metadata(attention_metadata, permutation=permutation)
