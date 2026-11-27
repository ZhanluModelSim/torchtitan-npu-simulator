# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 SMLA attention and SMLAG-backed indexer distillation."""

from dataclasses import dataclass

import cann_ops_transformer  # noqa: F401  (registers the fused operators)
import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscSparseMLAMixin
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import (
    build_indexer_metadata,
)
from torchtitan_npu.override.deepseek_v4_1.metadata import (
    AicpuGeometry,
    AicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
)


def validate_compressed_frame(ratio: int, cmp_k: KernelFrame | None) -> None:
    """Reject a compressed frame with a partial trailing group.

    Every document must be a whole number of pooling groups, so the compressed axis has no
    partial trailing group and ``cmp_residual_kv`` is zero throughout.  A non-zero value here
    means the loader's document alignment and this layer's ``compress_ratio`` disagree -- the
    kernels would then grid an axis the pooling did not produce -- so it is checked rather than
    passed on.
    """
    if cmp_k is None or cmp_k.residual is None:
        return
    if bool(cmp_k.residual.any()):
        raise ValueError(
            f"compress_ratio {ratio} has a partial trailing group: a document length "
            f"is not a multiple of the pooling ratio ({cmp_k.residual.tolist()}). The "
            "dataloader's document alignment must be a multiple of every ratio the stack pools at."
        )


def kernel_options(
    *,
    ratio: int,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    window_size: int,
):
    """The kernel options, read off the metadata's frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.  ``swa_k`` / ``cmp_k`` are the entry's key
    frames -- on arch22 the per-document reach this rank may read (``a3_workaround``).
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_ori_kv=swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if cmp_k is None else cmp_k.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_ori_kv`` its reach, so the sliding
        # window and the compressed causal limit are anchored at ``seqused_ori_kv - seqused_q``,
        # this rank's chunk offset inside the document.  At ``cp_size = 1`` each of them equals the
        # corresponding boundary difference, so passing them is a no-op there rather than a second
        # code path.
        seqused_q=q.seqused,
        seqused_ori_kv=swa_k.seqused,
        seqused_cmp_kv=None if cmp_k is None else cmp_k.seqused,
        # A3's kernels take the zero-filled empty group; A5 rejects a value at ratio 1.
        cmp_residual_kv=(
            None
            if cmp_k is None
            else cmp_k.residual
            if cmp_k.residual is not None or _IS_A5
            else torch.zeros(cmp_k.seqused.shape, dtype=torch.int32, device=cmp_k.seqused.device)
        ),
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        cmp_mask_mode=0 if _IS_A5 and cmp_k is None else 3,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
    )


# ---------------------------------------------------------------------------
# The entry: this geometry's frames and its AI_CPU metadata
# ---------------------------------------------------------------------------
#
# The AI_CPU metadata operators depend only on the frames of one forward and the layer's static
# geometry, never on the operands, so they are computed once per microbatch and read by every layer
# of the geometry instead of being re-run per layer inside the compiled region.  This module builds
# exactly what its own kernels read: the sparse attention's pair and the BF16 selector's pair.  A
# stack that runs a different backend (the quantized attention) builds its own entry there -- the
# `fwd_sparse` / `sparse_length` fields on this entry stay empty until the non-quantized candidate
# kernels land.
# ``..metadata`` describes the structure; this module owns the entry that carries it.


@dataclass(frozen=True, kw_only=True, slots=True)
class AscV41Metadata(DeepSeekV41Metadata):
    """The model row with this geometry's AscendC records attached.

    One entry per ``(port, ratio)``: the map key is this port's own config plus the layer's
    compression ratio, so every layer reads an entry built for its own geometry and no builder
    has to know the stack's ratio table.  The row and its frames are the model's -- they
    describe the tensors -- and this entry adds exactly what the fused kernels read on top of
    them: ``aicpu_metadata``, the operator outputs computed for those frames at the eager
    boundary once per microbatch and rebuilt per rank under context parallelism.

    Consumers read the fields directly (``cmp_k.seqused``, ``aicpu_metadata.sparse_attn.bwd``):
    one model runs one selection width, so there is nothing to look up.  Nothing here compares the
    metadata's geometry with the layer's own config -- that would need the numbers the reference
    core never sees -- so an override stated with the wrong widths surfaces as the kernels' shape
    error.  Every field is a tensor, a primitive or another registered node, because the class is a
    pytree node (see the registration below).
    """

    aicpu_metadata: AicpuMetadata

    a3_workaround: A3KeyWorkaround | None = None
    """The two key streams as arch22 must be handed them, or ``None`` where the kernels read the
    model's own spelling (arch35).  Built once per microbatch beside the records, so the forward,
    the backward and the selector read one frame and one gather each instead of re-deriving them
    per layer.
    """


# The registration is asymmetric on purpose, and reserved: DSV4 registers its whole masks object
# because the graph_trainer's ``minimal_fx_tracer`` requires ``attention_masks`` to flatten into
# tensor/primitive leaves, while V4.1 has no tracer path today.  The parent
# ``DeepSeekV41Metadata`` therefore stays an opaque pytree leaf -- the golden path's flattening
# behavior is untouched -- and only the Asc node is registered, mirroring DSV4's seam so the
# fused metadata is pytree-transparent from day one when such a path arrives.
if AscV41Metadata not in SUPPORTED_NODES:
    register_dataclass(AscV41Metadata, serialized_type_name=f"{AscV41Metadata.__module__}.{AscV41Metadata.__name__}")


def build_fused_metadata(row: DeepSeekV41Metadata, *, ratio: int, geometry: AicpuGeometry) -> AscV41Metadata:
    """This layer's geometry, built from the row: frames plus the AI_CPU metadata it reads.

    The row is the only input: ``cu_seq_k`` is already folded at this ratio by the reference
    backend, so the compressed frame needs nothing but this rank's per-document lengths for
    ``seqused``/``residual``.  Both the reference core and this port therefore start from the same
    upstream call, and what the port adds is what only it reads: these frames and the operator
    outputs sized by them.  A backend that runs other kernels (the quantized attention) derives its
    own frames from the same row and builds its own entry in its own package.
    """
    cu_seqlens = row.cu_seq_q
    # ``cu_seq_k`` was folded by the backend that built the row, so it must be *this* ratio's
    # folding: a row built for another geometry would pair a boundary set with a ``seqused``
    # derived from a different one.  The tensor is one int per document, so the check is cheap.
    expected = cu_seqlens if ratio <= 1 else (cu_seqlens // ratio).to(torch.int32)
    if not torch.equal(row.cu_seq_k, expected):
        raise ValueError(
            f"the row's cu_seq_k was not folded at compress_ratio {ratio}: build the metadata "
            "through the attention backend that owns this layer's ratio, not from a row built "
            "for another geometry."
        )
    # The frames are the row's: the model metadata already built them for exactly this
    # geometry, so the port only attaches the operator outputs sized by them.
    q, swa_k, cmp_k = row.q, row.swa_k, row.cmp_k
    validate_compressed_frame(ratio, cmp_k)
    workaround = None
    if not _IS_A5:
        workaround, swa_k, cmp_k = A3KeyWorkaround.build(swa_k=swa_k, cmp_k=cmp_k)
    record = build_aicpu_metadata(ratio=ratio, q=q, swa_k=swa_k, cmp_k=cmp_k, geometry=geometry)
    return AscV41Metadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=row.cu_seq_k,
        position_k=row.position_k,
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        aicpu_metadata=record,
        a3_workaround=workaround,
    )


def build_aicpu_metadata(
    *,
    ratio: int,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    geometry: AicpuGeometry,
) -> AicpuMetadata:
    """This geometry's AI_CPU operator outputs for these frames, once per microbatch.

    Only what this backend's own kernels are called with, grouped by the consumer that launches it.
    A stack that runs other kernels (the quantized attention) is another backend: it replaces this
    layer's config and builds its own entry, so nothing here has to know that it exists.
    """
    options = kernel_options(ratio=ratio, q=q, swa_k=swa_k, cmp_k=cmp_k, window_size=geometry.window_size)
    has_cmp = cmp_k is not None
    geometry_kwargs = dict(
        ori_topk=0,
        cmp_topk=geometry.index_topk if has_cmp else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    sparse_attn = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        ori_topk_length=None,
        cmp_topk_length=None,
        **options,
        **geometry_kwargs,
    )
    sparse_attn_grad = torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        **options,
        **geometry_kwargs,
    )
    lightning_indexer = AscLightningIndexerMetadata(fwd=None, bwd=None)
    if has_cmp:
        # The selector's metadata belongs to the selector that replaces it, and rides this entry:
        # the teacher edge crosses both kernels, so the two share one lifetime.
        assert cmp_k is not None, "a geometry with a compressed axis states its frame"
        lightning_indexer = build_indexer_metadata(ratio=ratio, q=q, cmp_k=cmp_k, geometry=geometry)
    return AicpuMetadata(
        geometry=geometry,
        sparse_attn=AscSparseAttentionMetadata(fwd=sparse_attn, bwd=sparse_attn_grad),
        lightning_indexer=lightning_indexer,
    )


def entry_options(metadata: AscV41Metadata, *, ratio: int, window_size: int) -> dict:
    """The two attention kernels' options, read off this entry's frames.

    The forward and the backward both read their options here, so the two kernel calls cannot drift
    apart; on arch22 the entry's key frames are the per-document reach this rank may read
    (``a3_workaround``), and the wrapper compacts the key rows onto them.
    """
    return kernel_options(
        ratio=ratio,
        q=metadata.q,
        swa_k=metadata.swa_k,
        cmp_k=metadata.cmp_k,
        window_size=window_size,
    )


class AscV41SparseAttention(AscSparseMLAMixin, CompressedSparseInnerAttention2):
    """The A5 TND kernel behind the reference forward contract.

    It is the port that carries the indexer's teacher: ``topk_scores`` is not a loss
    input but the message channel between the two kernels' backwards -- SMLAG's returns
    the raw marginal ``p`` on it, and the indexer's SLIKG backward consumes that edge to
    train the indexer.  That is why the tensor keeps threading through every layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention2.Config):
        """The fused config: the reference core's fields plus the geometry its metadata needs.

        That geometry is the *port's*: the reference core neither reads it nor knows it exists,
        and it is stated once by the override's arguments instead of being restated on every
        layer of every recipe.  The metadata built from it are the kernels' only metadata source --
        there is no per-layer fallback call -- and a recipe that states widths its layers do not
        run is a misconfiguration the kernels report, since this config cannot see them.
        """

        window_size: int
        compress_ratio: int
        # The kernel geometry has no defaults here: the released V4.1-Flash shape is stated once,
        # on the override factory a recipe names, and travels to every layer as these arguments.
        num_heads: int
        head_dim: int
        index_topk: int
        index_num_heads: int
        index_head_dim: int

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

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the port's class plus its compression ratio.

            One entry per ratio is what lets the builder precompute exactly the metadata this
            geometry reads -- no stack-wide ratio table anywhere, and no entry shared by layers
            whose kernels grid different axes.
            """
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV41Metadata:
            """This geometry's frames and AI_CPU metadata, built once per microbatch (#4897).

            The row -- and with it ``cu_seq_k``, already folded at this layer's ratio -- comes from
            the reference backend's builder, so both cores grid one set of boundaries.  The
            metadata is the AI_CPU operators' answer for that geometry, computed here at the
            eager boundary instead of per layer inside the compiled region, and rebuilt per rank
            by the context-parallel transform.
            """
            # The parent builder is named explicitly: ``slots=True`` recreates this dataclass, so
            # a zero-argument ``super()`` would bind to the decorated-away class and raise.
            row = CompressedSparseInnerAttention2.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            return build_fused_metadata(row, ratio=self.compress_ratio, geometry=self.geometry)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        q,
        swa_k,
        # Not optional, unlike the reference core's default: the kernel takes the sink
        # logits as part of its softmax denominator, and the model always has them
        # (``Attention.attn_sink`` is a parameter), so ``None`` is not a case to handle.
        attn_sink: torch.Tensor,
        attention_metadata,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ):
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together")
        output, _ = super().forward(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            attention_metadata=attention_metadata,
            options=entry_options(attention_metadata, ratio=self.compress_ratio, window_size=self.window_size),
            compact_keys=not _IS_A5,
            denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
        )
        return output.reshape_as(q)
