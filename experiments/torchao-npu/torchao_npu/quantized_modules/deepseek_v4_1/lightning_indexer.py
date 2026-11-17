# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 QLI/QSLI selection with an SLIKG backward edge.

The kernels' AI_CPU schedules are not built here either: the quantized attention's own config
precomputes them per microbatch at the eager boundary and hands them over on the entry this module
receives (``attention_metadata`` is the provider).  QSLI's per-row candidate count is derived at the
same time, from the row's lengths -- never read back off QLI's output, which is what lets the
search's schedule exist before the forward.  This module launches the pair and reads that metadata,
the way the BF16 selector reads its own.
"""

__all__ = ["QuantV41LightningIndexer"]

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch
import torch_npu

from torchao_npu.quantized_modules.deepseek_v4_1.attention_metadata import (
    QUANT_MODE,
    kernel_options,
)

# MXFP4 uses one E8M0 scale per 32 elements. The kernels and _pack_mxfp4's
# [T, H, D/64, 2] scale layout require this group size.
_MX_BLOCK_SIZE = 32


def _pack_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one ``[T, H, D]`` operand into the ds41 MXFP4 ABI."""
    if x.ndim != 3:
        raise ValueError(f"QLI operands must be rank-3 TND tensors, got shape {tuple(x.shape)}")
    rows, heads, head_dim = x.shape
    if head_dim % 64 != 0:
        raise ValueError(f"QLI head_dim must be divisible by 64, got {head_dim}")
    data, scale = torch_npu.npu_dynamic_mx_quant(
        x.contiguous(),
        axis=-1,
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=_MX_BLOCK_SIZE,
        round_mode="rint",
        scale_alg=2,
        dst_type_max=0.0,
    )
    return (
        data.view(torch.uint8).reshape(rows, heads, head_dim // 2).contiguous(),
        scale.view(torch.uint8).reshape(rows, heads, head_dim // 64, 2).contiguous(),
    )


def _selector_metadata(attention_metadata):
    """The entry's quantized selector schedules.

    They are built by the attention backend that owns this stack -- the quantized sparse attention
    -- once per microbatch, before this forward.  There is no second builder here: a stack whose
    entry carries none is a wiring error, and the two kernels' metadata is not interchangeable
    (an unquantized schedule fed to the quantized kernel yields garbage indices).
    """
    indexer = getattr(attention_metadata.aicpu_metadata, "lightning_indexer", None)
    if indexer is None or getattr(indexer, "fwd", None) is None:
        raise ValueError(
            "the attention entry carries no quantized selector schedules: the quantized selector "
            "runs only with the quantized sparse attention."
        )
    return indexer


def _validate_role(*, candidate_topk_blocks: int, is_source: bool, candidate: torch.Tensor | None) -> None:
    """Refuse a layer whose role and candidate table disagree.

    A pool's source builds the table and is handed none; a searcher is handed the source's; a layer
    with no pool never sees one.  The branch below picks the kernel from ``candidate`` alone, so a
    mismatch would silently train a searcher that scores every entry -- the pool's whole point
    being that it does not.
    """
    searcher = candidate_topk_blocks > 0 and not is_source
    if (candidate is not None) == searcher:
        return
    raise ValueError(
        "the candidate table disagrees with this layer's role: "
        + (
            "a searcher requires the source's candidate table"
            if searcher
            else "this layer cannot consume a candidate table"
        )
    )


def _quantized_forward(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    topk: int,
    *,
    schedules,
    is_source: bool,
    candidate: torch.Tensor | None,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    **options,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run QLI for a producer/pool-free layer or QSLI for a searcher."""
    _validate_role(candidate_topk_blocks=candidate_topk_blocks, is_source=is_source, candidate=candidate)
    if candidate is not None and (candidate.ndim != 4 or candidate.shape[0] != 1 or candidate.shape[2] != 1):
        raise ValueError(f"the model candidate table must have shape [1, T, 1, capacity], got {tuple(candidate.shape)}")
    idx_q_mxfp4, idx_q_scale = _pack_mxfp4(idx_q)
    idx_k_mxfp4, idx_k_scale = _pack_mxfp4(idx_k)

    if candidate is None:
        # One QLI schedule serves both QLI roles: the candidate writer's width is a launch argument
        # (this layer's own config states it), not part of the metadata, so nothing is chosen here.
        topk_indices, _, candidate_block_indices, _ = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer(
            idx_q_mxfp4,
            idx_k_mxfp4,
            idx_w,
            idx_q_scale,
            idx_k_scale,
            topk,
            QUANT_MODE,
            metadata=schedules.fwd,
            return_value=False,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **options,
        )
        return topk_indices, candidate_block_indices.unsqueeze(0) if is_source else None

    # The count and the schedule are one object: the search is bound by exactly the prefix QLI
    # published, derived from the lengths rather than read back off the candidate table.  The
    # table travels as ``[B, L, 1, C]`` between layers and a kernel wants ``[T, 1, C]``, so the
    # batch axis is squeezed here rather than at the caller.
    topk_indices, _ = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer(
        idx_q_mxfp4,
        idx_k_mxfp4,
        idx_w,
        idx_q_scale,
        candidate.squeeze(0),
        schedules.sparse_length,
        topk,
        QUANT_MODE,
        candidate_block_size,
        descale_k=idx_k_scale,
        metadata=schedules.fwd_sparse,
        return_value=False,
        **options,
    )
    return topk_indices, candidate


class _QuantV41LightningIndexerTND(torch.autograd.Function):
    """QLI/QSLI forward paired with the BF16 SLIKG training backward."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_metadata,
        is_source,
        candidate,
        num_global_queries,
        candidate_topk_blocks,
        candidate_block_size,
    ):
        # The options and the schedules both come off the same entry, so a launch and the metadata
        # it is handed describe one frame set.
        topk_indices, candidate_out = _quantized_forward(
            idx_q,
            idx_k,
            idx_w,
            topk,
            schedules=_selector_metadata(attention_metadata),
            is_source=is_source,
            candidate=candidate,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **kernel_options(ratio=ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k),
        )
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)

        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices, num_global_queries)
        ctx.topk = topk
        ctx.ratio = ratio
        # A dataclass, so ``save_for_backward`` cannot take it; the node holds the boundary
        # tensors the backward's options are read from and is released with it.
        ctx.attention_metadata = attention_metadata
        return topk_indices, topk_scores, candidate_out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx,
        grad_indices,
        attn_softmax_l1_norm,
        grad_candidate,
    ):
        del grad_indices, grad_candidate
        if attn_softmax_l1_norm is None:
            return (None,) * 11

        idx_q, idx_k, idx_w, topk_indices, num_global_queries = ctx.saved_tensors
        if num_global_queries is None:
            raise RuntimeError("Set Selector.num_global_queries before training the indexer")
        # Match the BF16 path: the trainer counts queries across the full step and DP mesh.
        attn_softmax_l1_norm = attn_softmax_l1_norm / num_global_queries
        # The backward rides the same entry, so it launches no AI_CPU operator either, and the
        # SLIKG metadata is the one the BF16 selector would read: the teacher edge is BF16 in both.
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=ctx.attention_metadata.aicpu_metadata.lightning_indexer.bwd,
            **kernel_options(ratio=ctx.ratio, q=ctx.attention_metadata.q, cmp_k=ctx.attention_metadata.cmp_k),
        )
        return (
            dq,
            dk,
            dw,
            None,  # topk
            None,  # ratio
            None,  # attention_metadata
            None,  # is_source
            None,  # candidate
            None,  # num_global_queries
            None,  # candidate_topk_blocks
            None,  # candidate_block_size
        )


class QuantV41LightningIndexer(torch.nn.Module):
    """Forward behavior installed on a DeepSeek-V4.1 ``Selector`` instance."""

    # The module-swap handler binds this forward to a Selector instance. Declare the
    # host-owned fields here so static analysis does not fall back to nn.Module's
    # dynamic attribute type (Module | Tensor).
    index_topk: int
    compress_ratio: int
    candidate_topk_blocks: int
    candidate_block_size: int
    has_candidate_pool: bool
    mode: str
    num_global_queries: torch.Tensor | None

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_metadata,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if idx_q_BLHiDi.shape[0] != 1 or idx_k_BNDi.shape[0] != 1 or weights_BLHi.shape[0] != 1:
            raise ValueError("V4.1 QLI/QSLI requires one packed TND row (model batch size 1)")

        # Which schedule this layer launches, and whether it builds the pool, follow from its own
        # role -- which the model states and the entry's schedules were sized for.
        is_source = self.has_candidate_pool and self.mode == "full"
        topk = self.index_topk
        topk_indices, topk_scores, candidate = _QuantV41LightningIndexerTND.apply(
            idx_q_BLHiDi.squeeze(0),
            idx_k_BNDi.squeeze(0).unsqueeze(1),
            weights_BLHi.squeeze(0).float(),
            topk,
            self.compress_ratio,
            attention_metadata,
            is_source,
            candidates_BL1C,
            self.num_global_queries,
            # The width this layer runs with: the sentinel outside the pool, the capacity on the
            # source -- exactly what its own selector config states.
            self.candidate_topk_blocks,
            self.candidate_block_size,
        )
        return (
            topk_indices.reshape(1, -1, topk),
            topk_scores.reshape(1, -1, topk),
            candidate,
        )
