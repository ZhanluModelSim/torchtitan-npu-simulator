# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 sparse attention with FP8 SWA fake quantization.

A copy of ``torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc`` with the
quantization inserted: the sliding-window KV is fake-quantized to FP8 (MXFP8,
group 32) on the way into the kernel, and the same quantized tensor is replayed in
the backward so both passes see identical operands.  Its gradient passes straight
through to the original input, i.e. the Q/DQ is an identity STE.

The kernels' AI_CPU metadata is *not* a copy step: this core reads what the attention
override precomputed on the entry, so the quantized and the unquantized core launch the very
same ``metadata`` tensors.

The main KV is *not* quantized here.  It arrives already fake-quantized: the
source Compressor does the Q/DQ once, on the layer that produces it, and every
consuming layer reuses that tensor (``QuantCompressor``).  Quantizing on
consumption instead would repeat the work on all 40 layers rather than the four
that own a compressor, and would quantize a second time on top of it.

Everything else -- the TND layout, the metadata frames, the document-local
selection and the kernel options -- is the port's behaviour, deliberately kept
line-for-line so the quantized stack and the unquantized one cannot drift.
"""

__all__ = ["QuantV41SparseAttention"]

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch

from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16
from torchao_npu.quantized_modules.deepseek_v4_1.attention_metadata import smla_kernel_options

# The SWA branch's quantization: FP8 with the group size the FP4 main-KV cache does not
# use, because the two streams carry different precision budgets.
_SWA_QUANT_GROUP_SIZE = 32
_SWA_QUANT_MODE = "mxfp8_bf16"


def _kernel_options(attention_metadata, ratio, window_size):
    """The kernel options, read off the entry's frames.

    The metadata this backend precomputes and the launches that consume it read them from one
    builder -- ``smla_kernel_options`` -- so the two cannot disagree option-for-option.
    """
    return smla_kernel_options(
        ratio=ratio,
        q=attention_metadata.q,
        swa_k=attention_metadata.swa_k,
        cmp_k=attention_metadata.cmp_k,
        window_size=window_size,
    )


class _SparseMLA(torch.autograd.Function):
    """The port's SMLAG and teacher edge with the SWA input fake-quantized.

    Identical to the override's Function except that ``swa_k`` is Q/DQ'd before the
    kernel call and the *quantized* tensor is what gets saved, so the backward replays
    the same operand the forward used.  The quantized tensor's gradient is the gradient
    the originals would have received, which is the identity STE.

    ``topk_scores`` is an autograd carrier only: forward never reads its value, while
    backward publishes SMLAG's compressed-attention marginal as its gradient for the
    LightningIndexer SLIKG backward to consume.
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
        wants_teacher,
    ):
        options = _kernel_options(attention_metadata, ratio, window_size)
        # The metadata this backend built for these frames, not a per-layer AI_CPU launch: the
        # quantized core and the unquantized one hand the kernel the same schedule.
        if not hasattr(attention_metadata, "aicpu_metadata"):
            raise ValueError(
                "the quantized sparse attention needs the AI_CPU metadata its own config builds: "
                "the entry handed to this forward carries none. Enable the quantized attention "
                "(and the quantized selector with it) instead of the unquantized override."
            )
        smla_metadata = attention_metadata.aicpu_metadata.sparse_attn.fwd
        qdq_swa_k = fake_quantize_mx_bf16(swa_k, quant_group_size=_SWA_QUANT_GROUP_SIZE, quant_mode=_SWA_QUANT_MODE)
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=qdq_swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=topk_indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=smla_metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            qdq_swa_k,
            cmp_k,
            topk_indices,
            sinks,
            output,
            lse,
        )
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        ctx.wants_teacher = wants_teacher
        # The metadata is a dataclass, so ``save_for_backward`` cannot take it; it is a
        # context attribute instead.  That is not extra retention: it holds the same
        # boundary tensors the graph already keeps for the backward, and the node -- and
        # with it this reference -- is released when that backward has run.
        ctx.attention_metadata = attention_metadata
        # The teacher consumes the LSE as a detached constant.
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse  # the LSE is a teacher signal, never a loss input
        # Nothing but the operand tensors is saved, and ``swa_k`` here is the quantized
        # one -- the backward has to differentiate the same graph the forward built.
        q, swa_k, cmp_k, topk_indices, sinks, output, lse = ctx.saved_tensors
        options = _kernel_options(ctx.attention_metadata, ctx.ratio, ctx.window_size)
        # The gradient side rides the same entry, so the backward launches nothing either.
        smla_grad_metadata = ctx.attention_metadata.aicpu_metadata.sparse_attn.bwd
        dq, dswa_k, dcmp_k, dsinks, _, cmp_softmax_l1_norm = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=topk_indices,
            sinks=sinks,
            metadata=smla_grad_metadata,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.softmax_scale,
            **options,
        )
        # One gradient per forward input, in order: (q, swa_k, cmp_k, topk_indices,
        # sinks, attention_metadata, softmax_scale, ratio, window_size, topk_scores,
        # wants_teacher). PyTorch silently ignores extra trailing entries, so spell out
        # every slot to keep the teacher on the carrier input.
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            (cmp_softmax_l1_norm.masked_fill(topk_indices < 0, 0.0) if ctx.wants_teacher else None),
            None,
        )


class QuantV41SparseAttention(torch.nn.Module):
    """Sparse computation installed at the host's ``_compute_attention`` seam.

    The host supplies window_size, softmax_scale and compress_ratio when the
    module-swap handler binds this forward method to it.
    """

    def __init__(self, *, window_size: int, softmax_scale: float, compress_ratio: int):
        super().__init__()
        self.window_size = window_size
        self.softmax_scale = softmax_scale
        self.compress_ratio = compress_ratio

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        attention_metadata,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
    ):
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together")
        wants_teacher = self.training and cmp_k is not None
        teacher_scores = None
        if wants_teacher:
            if topk_scores is None:
                raise ValueError("QuantV41SparseAttention requires topk_scores to publish the LightningIndexer teacher")
            teacher_scores = topk_scores.reshape(-1, 1, topk_scores.shape[-1])

        # The selection arrives document-local from the fused selector, which chose it
        # with the same kernel: the two share one coordinate system, and ``-1`` is how
        # both spell an unused slot.  Order is left exactly as it came -- the selector
        # already put it in position order, and the teacher's slot positions follow that
        # order.  Only the layout is translated: the model carries ``[B, L, 1, K]`` while
        # a TND kernel wants ``[T, N2, K]``.
        # The port returns the attention output alone: the teacher's LSE never leaves
        # SMLAG, which publishes the marginal on ``topk_scores``' gradient instead.
        output, _lse = _SparseMLA.apply(
            q.squeeze(0),
            swa_k.squeeze(0).unsqueeze(1),
            None if cmp_k is None else cmp_k.squeeze(0).unsqueeze(1),
            None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1]),
            attn_sink.float(),
            attention_metadata,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            teacher_scores,
            wants_teacher,
        )
        return output.reshape_as(q)
