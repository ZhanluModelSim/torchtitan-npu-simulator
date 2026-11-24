# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 selector's BF16 port: InnerIndexer fused with SLIKG.

The forward runs the CANN ``lightning_indexer`` selection and returns the document-local
indices and their matching FP32 logits.  The gradient of the logits carries the teacher
that SMLAG's backward produces
(``sparse_flash_mla_grad``'s ``cmp_softmax_l1_norm``).  This Function's backward consumes
that edge with ``sparse_lightning_indexer_kl_loss_grad`` and emits ``dq/dk/dw``.

The metadata is the entry the fused attention already built per geometry (``AscV4Metadata``):
the selector's two records (``lightning_indexer`` and SLIKG) live beside the attention's,
because the teacher edge crosses both kernels' backwards.

This module owns the whole AscendC selector -- the autograd bridge, the input preparation,
the kernel-scored selection and its backward -- so its kernel pair is stated once,
here.  The PyPTO port is a separate implementation of the same contract in
``lightning_indexer.pypto``; the two do not share code, so each can follow its runtime's
calling conventions without a common abstraction between them.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata,
    lightning_indexer_options,
)


class _LightningIndexerTND(torch.autograd.Function):
    """InnerIndexer forward fused with its SLIKG backward, over the CANN kernel pair.

    The kernels address the packed TND row: ``idx_q`` is ``[T, Hi, Di]``, ``idx_k`` is
    ``[Tk, 1, Di]`` and ``idx_w`` is ``[T, Hi]`` float32.  Indices are document-local with
    ``-1`` padding, and SLIKG requires ``topk`` to be 512 or an integer multiple of 1024.
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
    ):
        # The batch axis is squeezed outside this node: an autograd.Function's backward must
        # return one gradient per input in the shape it was handed, so the kernels' ``[T, ...]``
        # layout is established by the module wrapper.
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("the fused selector needs AscV4Metadata")
        asc = attention_metadata.aicpu_metadata
        li_metadata = asc.lightning_indexer.fwd
        if li_metadata is None:
            raise ValueError("the fused selector needs the entry's forward record")
        # The selector's frame is the entry's: on arch22 the pool this rank's reach covers, which
        # is what the keys below are compacted onto, and what its records were built from.
        assert attention_metadata.cmp_k is not None, "the selector needs the entry's pool"
        options = lightning_indexer_options(ratio=ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k)
        topk_indices, topk_scores = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            return_value=1,
            **options,
        )
        # Preserve SMLA's accumulation order and align the logits with the saved SLIKG slots.
        topk_indices, order = topk_indices.sort(dim=-1, descending=True)
        topk_scores = topk_scores.gather(-1, order)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices)
        ctx.ratio = ratio
        # The metadata is a dataclass, so ``save_for_backward`` cannot take it; it is a context
        # attribute instead.  It holds the same boundary tensors the graph already keeps for
        # the backward and is released when that backward has run.
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(topk_indices)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        # The indices are integer selections: never tracked, always None.  The scores' gradient
        # is the teacher SMLAG emitted on this edge; SLIKG applies ``dI = Z * Y - p`` itself.
        del grad_indices
        idx_q, idx_k, idx_w, topk_indices = ctx.saved_tensors
        metadata = ctx.attention_metadata
        slig_metadata = metadata.aicpu_metadata.lightning_indexer.bwd
        if slig_metadata is None:
            raise ValueError("the fused selector needs the entry's backward record")
        # The emitter already applied ``softmax_scale`` to the raw teacher.  One denominator
        # remains: the **row's** query-token count, the same quantity the CE loss is normalized
        # by.  It must be the whole row, not this rank's slab: under context parallelism every
        # rank owns a slab of the row and the parameter gradients are summed across the CP axis
        # by the sharded reduce-scatter, so a local denominator would scale every indexer
        # parameter's gradient by the CP degree (measured: exactly 4.000x for CP=4 while the
        # forward loss stayed aligned).
        query_rows = metadata.swa_k.cu_seqlens[-1].to(torch.float32)
        attn_softmax_l1_norm = attn_softmax_l1_norm / query_rows
        assert metadata.cmp_k is not None, "the selector needs the entry's pool"
        options = lightning_indexer_options(ratio=ctx.ratio, q=metadata.q, cmp_k=metadata.cmp_k)
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **options,
        )
        return dq.to(idx_q.dtype), dk.to(idx_k.dtype), dw.to(idx_w.dtype), None, None, None


class AscInnerIndexer(InnerIndexer):
    """The fused BF16 LI/SLIKG selector for the V4 indexer layers.

    It emits the kernel's own document-local ``topk_indices`` in ``[B, L, K]`` layout.  The
    fused attention consumes the same coordinate system, so neither side translates; only
    ``-1`` marks an unused slot.  ``topk_scores`` is the teacher carrier described above.
    """

    requires_teacher = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        # Single source, as for the fused core: the selector reads the records the fused
        # attention's builder attached, so any other entry is refused here.
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("AscInnerIndexer requires the entry the fused attention built.")
        topk_indices, topk_scores = _LightningIndexerTND.apply(
            idx_q,
            # The keys enter the Function already compacted (arch22): the gather is
            # differentiable, so the teacher's gradient scatters back onto the gathered row and
            # the wrapper's ``S(1) -> R`` reduce-scatter then lands it on the compressor.
            A3KeyWorkaround.indexer_keys(idx_k, attention_metadata, a5=_IS_A5).unsqueeze(1).contiguous(),
            idx_w.float(),
            self.index_topk,
            attention_metadata.ratio,
            attention_metadata,
        )
        return topk_indices.squeeze(1), topk_scores.squeeze(1)
