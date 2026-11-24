# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's PyPTO-Pro LI forward and auxiliary KL-gradient backward.

Select ``lightning_indexer.pypto_pro`` together with ``sparse_attn.asc``. The latter
builds the packed metadata and emits the teacher on ``topk_scores``' gradient;
this selector owns the LI/LIG autograd bridge. The optional PyPTO-Pro runtime
is loaded only inside the real NPU callbacks in ``ops.pypto_pro.li_lig_wrapper``.

The kernels require BF16 query/key, FP32 weights, 64 indexer heads of width
128, ratio 4 and right-down causal TND masking. The joint LI/LIG contract
supports topk 512/1024, at most 8192 query tokens and 262144 compressed keys.
The runtime ABI and upstream source attribution are recorded in
the LIG implementation in ``ops/pypto_pro/sparse_lightning_indexer_kl_loss_grad/``
and ``ops/pypto_pro/LICENSE``.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.ops.pypto_pro.li_lig_wrapper import (
    lightning_indexer_indices,
    sparse_lightning_indexer_kl_loss_grad,
)
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata,
    CompactSelectorFrameProof,
    lightning_indexer_options,
)


def _pro_options(metadata: AscV4Metadata, ratio: int) -> dict:
    key_frame = metadata.cmp_k
    assert key_frame is not None, "the selector needs the entry's compressed frame"
    proof = metadata.compact_selector_frame_proof
    if not isinstance(proof, CompactSelectorFrameProof):
        raise ValueError(
            "PyPTO-Pro requires compact frames validated by the eager metadata builder with tracked tensor versions; "
            "manual, noncompact or inference-mode metadata must be rebuilt outside inference_mode"
        )
    if (
        proof.q_cu is not metadata.q.cu_seqlens
        or proof.q_seqused is not metadata.q.seqused
        or proof.k_cu is not key_frame.cu_seqlens
        or proof.k_seqused is not key_frame.seqused
    ):
        raise ValueError("PyPTO-Pro compact-frame proof is stale; rebuild metadata after replacing frame tensors")
    options = lightning_indexer_options(ratio=ratio, q=metadata.q, cmp_k=key_frame)
    # This Pro kernel pair reads compact boundaries, not seqused. The eager
    # builder verified that contract for these tensors. Real opaque callbacks
    # also reject in-place changes using host version counters, without a
    # device-value read in this graph.
    del options["seqused_q"], options["seqused_k"]
    options["guard_tensors"] = proof.tensors()
    options["guard_versions"] = list(proof.versions)
    return options


class _PyPTOProIndexerTND(torch.autograd.Function):
    """Keep the saved selection and the SMLAG teacher in the same slot order."""

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
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("the fused selector needs AscV4Metadata")
        li_metadata = attention_metadata.aicpu_metadata.lightning_indexer.fwd
        if li_metadata is None:
            raise ValueError("the fused selector needs the entry's forward record")
        proof = attention_metadata.compact_selector_frame_proof
        if proof is None:
            raise ValueError(
                "PyPTO-Pro requires compact frames validated by the eager metadata builder "
                "with tracked tensor versions; "
                "manual, noncompact or inference-mode metadata must be rebuilt outside inference_mode"
            )
        torch._check(idx_q.shape[0] == proof.positions_q.numel())
        torch._check(idx_k.shape[0] == proof.key_tokens)
        topk_indices = lightning_indexer_indices(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            **_pro_options(attention_metadata, ratio),
        )
        # Sort before saving: SMLAG emits the teacher in the returned slot order.
        # Descending puts -1 padding after valid document-local positions.
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices)
        ctx.ratio = ratio
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(topk_indices)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        del grad_indices
        idx_q, idx_k, idx_w, topk_indices = ctx.saved_tensors
        metadata = ctx.attention_metadata
        slig_metadata = metadata.aicpu_metadata.lightning_indexer.bwd
        if slig_metadata is None:
            raise ValueError("the fused selector needs the entry's backward record")
        # SMLAG already applies softmax_scale. Normalize by the whole row, not
        # the CP slab. Pro's Z * student - teacher formula is linear in teacher,
        # so neither teacher renormalization nor another gradient scale belongs here.
        query_rows = metadata.cu_seq_q[-1].to(torch.float32)
        teacher = attn_softmax_l1_norm / query_rows
        dq, dk, dw, _ = sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=teacher,
            metadata=slig_metadata,
            **_pro_options(metadata, ctx.ratio),
        )
        return dq.to(idx_q.dtype), dk.to(idx_k.dtype), dw.to(idx_w.dtype), None, None, None


class PyPTOProInnerIndexer(InnerIndexer):
    """Standalone selector over compile-safe, lazily loaded PyPTO-Pro kernels."""

    requires_teacher = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("PyPTOProInnerIndexer requires the entry the fused attention built.")
        topk_indices, topk_scores = _PyPTOProIndexerTND.apply(
            idx_q,
            # index_select stays outside the Function so dk scatters back to
            # the gathered whole-row keys before the CP wrapper reduce-scatters.
            A3KeyWorkaround.indexer_keys(idx_k, attention_metadata).unsqueeze(1).contiguous(),
            idx_w.float(),
            self.index_topk,
            attention_metadata.ratio,
            attention_metadata,
        )
        return topk_indices.squeeze(1), topk_scores.squeeze(1)
