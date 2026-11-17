# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The selector's BF16 port: LightningIndexer with its SLIKG edge.

Its two metadata tensors (``build_indexer_metadata``) are sized from the frames and the geometry
the attention override states, and neither is launched from a layer.  A backend that runs another
selector (the quantized one) builds its own entry in its own package instead of extending this one.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.deepseek_v4_1.indexer import Selector
from torchtitan_npu.override.deepseek_v4_1.metadata import (
    AicpuGeometry,
    AscLightningIndexerMetadata,
    KernelFrame,
)

_LAYOUT = "TND"
_MASK_MODE = 3  # right-down causal: entry j is complete for query t iff j < (t+1)//ratio


def kernel_options(*, ratio: int, q: KernelFrame, cmp_k: KernelFrame):
    """The kernel options, read off this geometry's own frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_k`` its reach into the gathered keys, so
        # the selection is scored at this rank's chunk against the blocks it may read.  Every
        # operator this selector can pick takes the pair -- the BF16 indexer and SLIKG -- and at
        # ``cp_size = 1`` each equals the corresponding boundary difference, so passing them is a
        # no-op there rather than a second code path.
        seqused_q=q.seqused,
        seqused_k=cmp_k.seqused,
        cmp_residual_k=cmp_k.residual,
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_MASK_MODE,
        cmp_ratio=ratio,
    )


def build_indexer_metadata(
    *,
    ratio: int,
    q: KernelFrame,
    cmp_k: KernelFrame,
    geometry: AicpuGeometry,
) -> AscLightningIndexerMetadata:
    """The selector's two AI_CPU metadata tensors for this geometry, forward and backward.

    ``max_seqlen_q``/``max_seqlen_k`` are pinned to ``None`` because both the BF16
    ``lightning_indexer_metadata`` and SLIKG's take them as optional ints; passing them makes the
    call independent of whatever default the operator ships with.
    """
    options = kernel_options(ratio=ratio, q=q, cmp_k=cmp_k)
    geometry_kwargs = dict(
        num_heads_q=geometry.index_num_heads,
        num_heads_k=1,
        head_dim=geometry.index_head_dim,
        topk=geometry.index_topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
    )
    return AscLightningIndexerMetadata(
        fwd=torch.ops.cann_ops_transformer.lightning_indexer_metadata(**geometry_kwargs, **options),
        bwd=torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            **geometry_kwargs,
            **options,
        ),
    )


class _LightningIndexerTND(torch.autograd.Function):
    """The BF16 LightningIndexer forward fused with the SLIKG backward.

    The kernels' contract: ``q``/``k`` are BF16 with a single key head and
    ``index_head_dim`` 128, ``w`` is float32 ``[T, Hi]``, and indices are
    document-local with ``-1`` padding. SLIKG requires ``topk`` to be 512 or
    an integer multiple of 1024.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_metadata,
        num_global_queries,
    ):
        # These arrive with the batch axis already squeezed off by the caller: an
        # autograd.Function's backward must return one gradient per input *in the shape it
        # was handed*, so the kernels' ``[T, H, D]`` layout has to be established outside
        # this node rather than inside it.
        options = kernel_options(ratio=ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k)
        # The precomputed metadata is read instead of re-running the AI_CPU operator per layer.
        li_metadata = attention_metadata.aicpu_metadata.lightning_indexer.fwd
        topk_indices, _ = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            return_value=1,
            **options,
        )

        # The kernel's own order is unspecified, so the selection is sorted into position
        # order here -- inside the Function, before anything is saved.  That placement is
        # the whole point: the teacher arrives on ``topk_scores``' *gradient*, positioned
        # per slot by whatever selection SMLA was handed, and SLIKG consumes it next to
        # ``topk_indices``.  Sorting after ``apply`` returns would leave the saved indices
        # in kernel order while the teacher is in sorted order, and SLIKG would pair each
        # slot's mass with a different key.  Sorting here keeps the two in step by
        # construction, and the kernel layout stays in one place.
        #
        # Descending gives both properties with one sort: the valid entries come out in
        # descending position order and ``-1``, being the smallest index, lands behind
        # every one of them.  (The reference sorts ascending instead; neither the kernel
        # schema nor the operator docs fix a direction.)
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        # The carrier for the teacher edge, fabricated rather than taken from the kernel's
        # second output.  The gradient this Function's backward receives on it *is* the
        # teacher -- ``attn_softmax_l1_norm`` there feeds SLIKG -- so it has to be an output
        # of this Function that the engine differentiates through, and returning it from
        # here is what supplies that ``grad_fn``.  A tensor built by the caller after
        # ``apply`` returns cannot have it and leaves the indexer untrained.
        #
        # Nothing reads its value: SMLAG replaces it wholesale and the attention's core
        # accepts and ignores it.  Shape, dtype and the edge are the whole contract, which
        # is why the kernel's own scores are dropped rather than permuted across to here.
        # The shape is the selection's own, ``[T, 1, K]`` -- SLIKG requires the teacher to
        # match ``sparse_indices`` exactly.
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices, num_global_queries)
        ctx.topk, ctx.ratio = topk, ratio
        # The metadata is a dataclass, so ``save_for_backward`` cannot take it; it is a
        # context attribute instead.  That is not extra retention: it holds the same
        # boundary tensors the graph already keeps for the backward, and the node -- and
        # with it this reference -- is released when that backward has run.
        ctx.attention_metadata = attention_metadata
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        # The indices are integer selections: never tracked, always None.  The
        # scores' gradient is the teacher the SLIKG kernel expects, because SLIKG
        # applies ``dI = Z * Y - p`` itself.  SMLAG's backward is what puts the raw
        # ``p`` on this edge; nothing else may, and a signed value here would be read
        # as the teacher and negate the indexer's gradient.
        del grad_indices
        if attn_softmax_l1_norm is None:
            # No teacher reached this edge, so no operand has a gradient either.  The count
            # is the same one the labelled tail below enumerates: one entry per forward
            # input, in the same order.
            return (None,) * 7
        idx_q, idx_k, idx_w, topk_indices, num_global_queries = ctx.saved_tensors
        if num_global_queries is None:
            raise RuntimeError("Set Selector.num_global_queries before training the indexer")
        # SMLAG injects raw teacher weights outside the main loss's normalization.
        # Scaling p scales SLIKG's Z * softmax(I) - p by the same factor.
        attn_softmax_l1_norm = attn_softmax_l1_norm / num_global_queries
        options = kernel_options(ratio=ctx.ratio, q=ctx.attention_metadata.q, cmp_k=ctx.attention_metadata.cmp_k)
        # Rides the same entry as the forward, so the backward skips its AI_CPU call too.
        slig_metadata = ctx.attention_metadata.aicpu_metadata.lightning_indexer.bwd
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **options,
        )
        # ``idx_w`` arrives float32 (the kernel contract); the cast back to the
        # projection dtype happens in the eager graph outside the Function.
        #
        # One gradient per forward input: q, k, w, topk, ratio, masks, query count.
        return (
            dq,
            dk,
            dw,
            None,  # topk
            None,  # ratio
            None,  # attention_metadata
            None,  # num_global_queries
        )


class AscSelector(Selector):
    """The fused BF16 LI/SLIKG selector for the V4.1 indexer layers.

    It emits the kernel's own document-local ``topk_indices``.  The fused attention
    consumes that same coordinate system, so neither side translates: the selection
    crosses the model boundary exactly as the kernels selected it, and only ``-1`` marks an
    unused slot.  The reference selector emits that numbering too, so it is the model's
    contract rather than this pair's.

    It is the consumer half of the teacher pair: ``sparse_attn.asc`` emits the teacher on
    ``topk_scores``' gradient and this backward is what reads it, so the two overrides are
    requested together or not at all -- see
    :func:`torchtitan_npu.models.deepseek_v4_1.model._check_indexer_teacher_pair`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Selector.Config):
        pass

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_metadata,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        topk = self.index_topk
        topk_indices, topk_scores = _LightningIndexerTND.apply(
            # The batch axis is the packed row, so it is squeezed rather than indexed: every
            # kernel in this path addresses one row.
            idx_q_BLHiDi.squeeze(0),
            idx_k_BNDi.squeeze(0).unsqueeze(1),
            weights_BLHi.squeeze(0).float(),
            topk,
            self.compress_ratio,
            attention_metadata,
            self.num_global_queries,
        )
        # Two layout translations and nothing else: ``_LightningIndexerTND`` already
        # emitted the selection sorted, in the kernels' ``[T, 1, K]`` layout, and the model
        # contract is ``[1, L, K]``.  Sorting at this level instead would desynchronise the
        # saved indices from the teacher (see the Function).
        topk_indices_BLK = topk_indices.reshape(1, -1, topk)
        topk_scores_BLK = topk_scores.reshape(1, -1, topk)
        return topk_indices_BLK, topk_scores_BLK, None
