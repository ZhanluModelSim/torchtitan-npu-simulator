# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 BF16 indexer using the packed LI/SLIKG kernel pair."""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata,
    lightning_indexer_options,
)


def _uses_fp32_sort(indices: torch.Tensor) -> bool:
    return indices.device.type == "npu" and indices.dtype == torch.int32


def sort_indices_and_scores(
    indices: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort top-k indices descending and align scores, keeping NPU sort on AI Core.

    CANN dispatches ``int32.sort`` to AI_CPU while ``float32.sort`` runs on AI Core.
    Compressed document-local indices (including ``-1`` padding) are exact in FP32
    up to ``2**24`` (~16.7M compressed keys).
    """
    if _uses_fp32_sort(indices):
        sorted_fp32, order = indices.float().sort(dim=-1, descending=True, stable=True)
        indices = sorted_fp32.to(indices.dtype)
    else:
        indices, order = indices.sort(dim=-1, descending=True)
    return indices, scores.gather(-1, order)


class _LightningIndexerTND(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, w, topk, attention_metadata, options):  # pyrefly: ignore [bad-override]
        indices, scores = torch.ops.cann_ops_transformer.lightning_indexer(
            q, k, w, topk, metadata=attention_metadata.aicpu_metadata.lightning_indexer.fwd, return_value=1, **options
        )
        # Keep SMLA's accumulation order. Sort before saving the indices so SLIKG
        # sees exactly the slots used to compute SMLAG's teacher.
        indices, scores = sort_indices_and_scores(indices, scores)
        ctx.save_for_backward(q, k, w, indices)
        ctx.attention_metadata = attention_metadata
        ctx.options = options
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(indices)
        return indices, scores

    @staticmethod
    def backward(ctx, grad_indices, teacher):  # pyrefly: ignore [bad-override]
        if teacher is None:
            return (None,) * 6
        q, k, w, indices = ctx.saved_tensors
        metadata = ctx.attention_metadata.aicpu_metadata.lightning_indexer.bwd
        if metadata is None:
            raise ValueError("the fused selector needs the entry's backward metadata")
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=q,
            k=k,
            w=w,
            sparse_indices=indices,
            attn_softmax_l1_norm=teacher,
            metadata=metadata,
            **ctx.options,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dw.to(w.dtype), None, None, None


class AscIndexerMixin:
    """Run the shared BF16 LightningIndexer and SLIKG on prepared TND operands."""

    index_topk: int

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata, options):
        indices, scores = _LightningIndexerTND.apply(
            idx_q, idx_k.unsqueeze(1), idx_w.float(), self.index_topk, attention_metadata, options
        )
        return indices.squeeze(1), scores.squeeze(1)


class AscInnerIndexer(AscIndexerMixin, InnerIndexer):
    """BF16 selection for V4 compressed sparse attention."""

    requires_teacher = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):  # pyrefly: ignore [bad-override]
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("AscInnerIndexer requires the entry the fused attention built.")
        if attention_metadata.aicpu_metadata.lightning_indexer.fwd is None:
            raise ValueError("the fused selector needs the entry's forward metadata")
        assert attention_metadata.cmp_k is not None, "the selector needs the entry's pool"
        options = lightning_indexer_options(
            ratio=attention_metadata.ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k
        )
        return super().forward(
            idx_q,
            A3KeyWorkaround.indexer_keys(idx_k, attention_metadata, a5=_IS_A5).contiguous(),
            idx_w,
            attention_metadata=attention_metadata,
            options=options,
        )
