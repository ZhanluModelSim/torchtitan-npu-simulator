# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused CANN TND kernels.

This module holds the CANN metadata layer (``CANNCompressedVarlenMetadataHandler``
+ the ``*_metadata`` fill) and the fused TND kernel
(``CANNCompressedSparseInnerAttention``); the eager golden reference lives in
``golden.py``.  The ``sparse_attn`` package ``__init__`` defines the
registrations, so the override paths stay ``override.deepseek_v4.sparse_attn.*``.

The fused path only bridges the CANN TND kernels' forward/backward with a
slim ``torch.autograd.Function``.  All CANN metadata is precomputed by the
mask handler and carried in ``CompressedVarlenMetadata``, so this module
never builds or caches metadata itself.  The container-grid compressed KV is
converted to the packed TND stream by slicing the leading ``n_blocks``
slots (the identity ``storage_indices`` mapping under the B=1 contract).

The kernel inputs follow the training autocast contract: q/k/v are bf16,
``idx_w`` and the sink are fp32 — an eager-fp32 caller fails loudly at the
kernel's dtype check.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import (
    CompressedBlockLayout,
    CompressedBlockMaskHandler,
    build_kernel_layout,
)

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


# ---------------------------------------------------------------------------
# CANN metadata layer
# ---------------------------------------------------------------------------
#
# ``CANNCompressedVarlenMetadataHandler`` extends the model-dir
# ``CompressedBlockMaskHandler``: after the document-packed layout (including
# the dense compressed-key attendability mask) is built, it fills the
# precomputed CANN ``*_metadata`` kernel outputs (lightning_indexer,
# sparse_flash_mla, sparse_flash_mla_grad,
# sparse_lightning_indexer_kl_loss_grad) into a
# ``CANNCompressedVarlenMetadata`` wrapper, keeping the model-dir metadata
# NPU-free.  Those kernels take only shape information, so their results are
# layer-invariant and reused by every DSA layer and by the backward pass.


@dataclass(kw_only=True, slots=True)
class CANNBlockLayoutMetadata:
    """CANN ``*_metadata`` kernel outputs for one compression ratio (the
    key of ``cann_plans``)."""

    smla_metadata: torch.Tensor | None = None
    """``sparse_flash_mla_metadata`` output (opaque, layer-invariant)."""

    smla_grad_metadata: torch.Tensor | None = None
    """``sparse_flash_mla_grad_metadata`` output (opaque, layer-invariant)."""

    li_metadata: torch.Tensor | None = None
    """``lightning_indexer_metadata`` output; ratio-4 plans only."""

    slig_metadata: torch.Tensor | None = None
    """``sparse_lightning_indexer_kl_loss_grad_metadata`` output;
    ratio-4 plans only."""


@dataclass(kw_only=True, slots=True)
class CANNCompressedVarlenMetadata:
    """The fused path's varlen contract: the model-dir kernel contract
    plus the CANN metadata layer.

    Standalone (not a ``CompressedVarlenMetadata`` subclass) so the
    model-dir reference tier stays required on the model-dir type and the
    fused path carries no reference tensors at all.
    """

    varlen: VarlenMetadata
    """Token-stream boundaries (``cu_seq_q`` / ``cu_seq_k``)."""

    batch_size: int
    """Container batch size (``1`` for the current packed scenario)."""

    seq_len: int
    """Container sequence length (the total token count)."""

    plans: dict[int, CompressedBlockLayout]
    """The model-dir kernel contract for each ratio present in the
    model."""

    cann_plans: dict[int, CANNBlockLayoutMetadata] = field(default_factory=dict)

    @property
    def cu_seqlens_q(self) -> torch.Tensor:
        return self.varlen.cu_seq_q

    @property
    def cu_seqlens_ori_kv(self) -> torch.Tensor:
        # CP seam: under context parallel this becomes the per-segment
        # window-pack ori ranges (end-aligned oriLen), not the bare cu_seq_k.
        return self.varlen.cu_seq_k


def _fill_cann_metadata(
    varlen: VarlenMetadata,
    ratio: int,
    plan: CompressedBlockLayout,
    record: CANNBlockLayoutMetadata,
    *,
    num_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    window_size: int,
) -> None:
    """Compute the ``*_metadata`` kernel outputs stored in ``record``.

    ``num_heads`` / ``head_dim`` describe the main attention (used by the
    SMLA kernels); ``index_n_heads`` / ``index_head_dim`` describe the
    indexer projections (used by the LightningIndexer and SLIG kernels).
    The indexer score sums over the head dimension, so tensor parallel is
    fixed at 1 for the DSA path.
    """
    has_cmp_kv = ratio > 1
    smla_kwargs: dict[str, Any] = {
        "cu_seqlens_q": varlen.cu_seq_q,
        "cu_seqlens_ori_kv": varlen.cu_seq_k,
        "cu_seqlens_cmp_kv": plan.cu_seqlens_cmp_k if has_cmp_kv else None,
        "cmp_residual_kv": plan.block_remainder if has_cmp_kv else None,
        "ori_topk_length": None,
        "cmp_topk_length": None,
        "ori_topk": 0,
        "cmp_topk": index_topk if ratio == 4 else 0,
        "cmp_ratio": ratio,
        "ori_mask_mode": _ORI_MASK_MODE,
        "cmp_mask_mode": _CMP_MASK_MODE,
        "ori_win_left": window_size - 1,
        "ori_win_right": 0,
        "layout_q": _LAYOUT,
        "layout_kv": _LAYOUT,
        "has_ori_kv": True,
        "has_cmp_kv": has_cmp_kv,
    }
    record.smla_metadata = sparse_flash_mla_metadata(
        num_heads, 1, head_dim, **smla_kwargs
    )
    grad_kwargs = dict(smla_kwargs)
    grad_kwargs.pop("ori_topk_length")
    grad_kwargs.pop("cmp_topk_length")
    record.smla_grad_metadata = sparse_flash_mla_grad_metadata(
        num_heads, 1, head_dim, **grad_kwargs
    )
    if ratio != 4:
        return

    record.li_metadata = lightning_indexer_metadata(
        index_n_heads,
        1,
        index_head_dim,
        index_topk,
        cu_seqlens_q=varlen.cu_seq_q,
        cu_seqlens_k=plan.cu_seqlens_cmp_k,
        cmp_residual_k=plan.block_remainder,
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_CMP_MASK_MODE,
        cmp_ratio=4,
    )
    record.slig_metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
        index_n_heads,
        1,
        index_head_dim,
        cu_seqlens_q=varlen.cu_seq_q,
        cu_seqlens_k=plan.cu_seqlens_cmp_k,
        cmp_residual_k=plan.block_remainder,
        topk=index_topk,
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_CMP_MASK_MODE,
        cmp_ratio=4,
    )


def _mark_dynamic(metadata: CANNCompressedVarlenMetadata) -> None:
    """Mark batch-dependent leading dimensions so Dynamo does not specialize.

    The slim contract carries only the kernel tensors; the model-dir
    reference tier is never materialized here.
    """
    tensors: list[torch.Tensor] = [metadata.varlen.cu_seq_q]
    for plan in metadata.plans.values():
        for tensor in (
            plan.cu_seqlens_cmp_k,
            plan.block_remainder,
            plan.gather_indices,
        ):
            if tensor is not None and tensor.numel() > 0:
                tensors.append(tensor)
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


class CANNCompressedVarlenMetadataHandler(CompressedBlockMaskHandler):
    """Build the DSV4 varlen contract plus the CANN metadata kernels.

    ``compress_ratios`` / ``window_size`` / ``block_size`` are declared
    directly on the (registry-built) model-dir handler config; the attention
    geometry (``num_heads``, ``head_dim``, ``index_n_heads``,
    ``index_head_dim``, ``index_topk``) is passed as override kwargs by the
    factory below, keeping the model directory backend-agnostic.  Mismatched
    geometry fails loudly: the CANN metadata kernels and the fused core
    validate against the actual tensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedBlockMaskHandler.Config):
        num_heads: int
        head_dim: int
        index_n_heads: int
        index_head_dim: int
        index_topk: int

    def post_process(  # pyrefly: ignore [bad-override]
        self, masks: VarlenMetadata
    ) -> CANNCompressedVarlenMetadata:
        cfg = self.config
        assert isinstance(cfg, CANNCompressedVarlenMetadataHandler.Config)
        # The slim contract: the model-dir kernel tier only — no reference
        # tier (dense mask, static blocks, doc/pos) is materialized.
        batch_size, seq_len, plans = build_kernel_layout(masks, self.compress_ratios)
        for ratio, plan in plans.items():
            if ratio > 1 and (
                plan.gather_indices is None or plan.gather_indices.numel() == 0
            ):
                raise ValueError(
                    f"batch has no complete compression block for ratio={ratio}; "
                    "doc-packed sequences must be long enough to produce at least "
                    "one full block per sequence."
                )
        cann_plans: dict[int, CANNBlockLayoutMetadata] = {}
        for ratio, plan in plans.items():
            record = CANNBlockLayoutMetadata()
            _fill_cann_metadata(
                masks,
                ratio,
                plan,
                record,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                index_n_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=cfg.index_topk,
                window_size=cfg.window_size,
            )
            cann_plans[ratio] = record
        result = CANNCompressedVarlenMetadata(
            varlen=masks,
            batch_size=batch_size,
            seq_len=seq_len,
            plans=plans,
            cann_plans=cann_plans,
        )
        _mark_dynamic(result)
        return result


def _compute_li_loss(
    indexer_softmax: torch.Tensor,
    teacher_mass: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Recover the reference LI loss from SLIG outputs."""
    student = indexer_softmax.float().clamp_min(1e-10)
    target = teacher_mass.float().clamp_min(0)
    target_sum = target.sum(dim=-1, keepdim=True)
    valid_target = target_sum > 1e-10
    student = torch.where(valid_target, student, torch.ones_like(student))
    teacher = target / target_sum.clamp_min(1e-10)
    log_teacher = teacher.clamp_min(1e-10).log()
    loss = (teacher * (log_teacher - student.log())).sum(dim=-1)
    return (target_sum.squeeze(-1) * loss).mean() * softmax_scale


class _SparseFlashMLATND(torch.autograd.Function):
    """Bridge SparseFlashMLA forward and SMLAG + SLIG backward.

    All CANN metadata tensors are precomputed in ``CompressedBlockLayout``.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        cmp_sparse_indices,
        sinks,
        cu_seqlens_q,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        smla_metadata,
        smla_grad_metadata,
        slig_metadata,
        idx_q,
        idx_k,
        idx_w,
        softmax_scale,
        ratio,
        window_size,
        indexer_loss_coeff,
        indexer_loss_accumulator,
    ):
        has_compressed = ratio > 1
        result, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla.default(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            cmp_sparse_indices=(
                cmp_sparse_indices if cmp_sparse_indices is not None else None
            ),
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_q,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            sinks=sinks,
            metadata=smla_metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=_CMP_MASK_MODE,
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            smla_grad_metadata,
            slig_metadata,
            idx_q,
            idx_k,
            idx_w,
            result,
            softmax_lse,
        )
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.window_size = window_size
        ctx.indexer_loss_coeff = indexer_loss_coeff
        ctx.indexer_loss_accumulator = indexer_loss_accumulator
        return result

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        (
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            smla_grad_metadata,
            slig_metadata,
            idx_q,
            idx_k,
            idx_w,
            result,
            softmax_lse,
        ) = ctx.saved_tensors
        has_compressed = ctx.ratio > 1
        has_sparse_indices = cmp_sparse_indices is not None

        (
            dquery,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = torch.ops.cann_ops_transformer.sparse_flash_mla_grad.default(
            q,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(cmp_sparse_indices if has_sparse_indices else None),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_q,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=smla_grad_metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=_CMP_MASK_MODE,
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
        )
        if not has_compressed:
            dcmp_kv = None

        didx_q = didx_k = didx_w = None
        if ctx.ratio == 4 and ctx.indexer_loss_coeff != 0:
            if any(x is None for x in (idx_q, idx_k, idx_w, slig_metadata)):
                raise RuntimeError(
                    "ratio-4 cann requires LI tensors and slig_metadata in backward."
                )
            (
                didx_q,
                didx_k,
                didx_w,
                indexer_softmax,
            ) = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad.default(
                q=idx_q,
                k=idx_k,
                w=idx_w.float(),
                sparse_indices=cmp_sparse_indices,
                attn_softmax_l1_norm=cmp_softmax_l1,
                cmp_residual_k=cmp_residual_kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_cmp_kv,
                metadata=slig_metadata,
                layout_q=_LAYOUT,
                layout_k=_LAYOUT,
                mask_mode=_CMP_MASK_MODE,
                cmp_ratio=4,
            )
            if ctx.indexer_loss_accumulator is not None:
                li_loss = _compute_li_loss(
                    indexer_softmax,
                    cmp_softmax_l1,
                    ctx.softmax_scale,
                )
                ctx.indexer_loss_accumulator.add_(li_loss.detach())
            query_rows = cmp_softmax_l1.sum(dim=-1).numel()
            grad_scale = ctx.indexer_loss_coeff * ctx.softmax_scale / float(query_rows)
            didx_q = (didx_q * grad_scale).to(idx_q.dtype)
            didx_k = (didx_k * grad_scale).to(idx_k.dtype)
            didx_w = (didx_w * grad_scale).to(idx_w.dtype)

        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            None,
            None,
            didx_q,
            didx_k,
            didx_w,
            None,  # softmax_scale
            None,  # ratio
            None,  # window_size
            None,  # indexer_loss_coeff
            None,  # indexer_loss_accumulator
        )


class CANNCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Run LI and SMLA/SMLAG/SLIG in a local TND layout.

    The model-facing tensors use the container grid; the compressed KV and the
    indexer keys are converted to the packed TND stream by slicing the
    leading ``n_blocks`` slots (the identity ``storage_indices`` mapping
    under the B=1 contract) before the CANN kernels run.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        indexer_loss_coeff: float = 0.0
        """Scales the LightningIndexer KL-loss gradient produced by SLIG."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.indexer_loss_coeff = config.indexer_loss_coeff
        self.register_buffer(
            "_indexer_loss_acc",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attn_sink=None,
        *,
        attention_masks=None,
    ):
        if not isinstance(attention_masks, CANNCompressedVarlenMetadata):
            raise TypeError(
                "cann requires CANNCompressedVarlenMetadata attention masks."
            )
        if attn_sink is None:
            raise ValueError("CANNCompressedSparseInnerAttention requires attn_sink")
        metadata = attention_masks
        plan = metadata.plans.get(self.compress_ratio)
        if plan is None:
            raise ValueError(
                f"No CompressedBlockLayout for ratio={self.compress_ratio}."
            )
        if self.compress_ratio <= 1:
            if cmp_k is not None and cmp_k.numel() != 0:
                raise ValueError("ratio-1 cann must not receive compressed KV.")
        elif cmp_k is None or cmp_k.ndim != 3:
            raise ValueError(
                "ratio>1 cann requires compressed KV in the container "
                "layout [B, S//ratio, D]."
            )
        # q / swa_k shape consistency and the kernel input layouts are
        # validated by the aclnn interface itself.
        npu = metadata.cann_plans[self.compress_ratio]
        batch_size, seqlen, _, _ = q.shape

        q = q.flatten(0, 1)
        swa_k = swa_k.flatten(0, 1).unsqueeze(1).contiguous()
        cmp_k = (
            None
            if cmp_k is None
            else cmp_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]
            .unsqueeze(1)
            .contiguous()
        )

        cmp_sparse_indices = None
        if self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError("ratio-4 cann requires all LI projection tensors.")
            idx_q = idx_q.flatten(0, 1)
            idx_k = (
                idx_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]
                .unsqueeze(1)
                .contiguous()
            )
            idx_w = idx_w.flatten(0, 1)
            cmp_sparse_indices, _ = (
                torch.ops.cann_ops_transformer.lightning_indexer.default(
                    idx_q,
                    idx_k,
                    idx_w.float(),
                    self.index_topk,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=plan.cu_seqlens_cmp_k,
                    cmp_residual_k=plan.block_remainder,
                    metadata=npu.li_metadata,
                    layout_q=_LAYOUT,
                    layout_k=_LAYOUT,
                    mask_mode=_CMP_MASK_MODE,
                    cmp_ratio=4,
                    return_value=1,
                )
            )

        indexer_loss_coeff = float(self.indexer_loss_coeff) if self.training else 0.0
        indexer_loss_accumulator = self._indexer_loss_acc

        output = _SparseFlashMLATND.apply(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            attn_sink.float(),
            metadata.cu_seqlens_q,
            plan.cu_seqlens_cmp_k,
            plan.block_remainder,
            npu.smla_metadata,
            npu.smla_grad_metadata,
            npu.slig_metadata,
            idx_q,
            idx_k,
            idx_w,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            indexer_loss_coeff,
            indexer_loss_accumulator,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])
