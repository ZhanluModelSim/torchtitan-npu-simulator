# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused CANN TND kernels.

This module holds the CANN metadata extension (``CANNMetadataExtension``
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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import (
    CompressedBlockLayout,
    CompressedVarlenMetadata,
)

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


@dataclass(frozen=True, slots=True)
class _SparseAttentionHooks:
    """Operator implementation bundle used by the fused attention bridge."""

    lightning_indexer: Callable[..., Any]
    sparse_flash_mla: Callable[..., Any]
    sparse_flash_mla_grad: Callable[..., Any]
    sparse_lightning_indexer_kl_loss_grad: Callable[..., Any]


_CANN_SPARSEATTN_HOOK = _SparseAttentionHooks(
    lightning_indexer=torch.ops.cann_ops_transformer.lightning_indexer,
    sparse_flash_mla=torch.ops.cann_ops_transformer.sparse_flash_mla,
    sparse_flash_mla_grad=torch.ops.cann_ops_transformer.sparse_flash_mla_grad,
    sparse_lightning_indexer_kl_loss_grad=torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad,
)


# ---------------------------------------------------------------------------
# CANN metadata layer
# ---------------------------------------------------------------------------
#
# ``CANNMetadataExtension`` post-processes the model-built metadata (the
# model's ``build_attention_masks``): after the document-packed layout (including
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
class CANNCompressedVarlenMetadata(CompressedVarlenMetadata):
    """The fused path's varlen contract: the common kernel contract plus
    the CANN metadata layer.

    A ``CompressedVarlenMetadata`` subclass carrying no reference tier (the
    fused path never materializes it) and the ``cann_plans`` the fused
    core consumes.
    """

    cann_plans: dict[int, CANNBlockLayoutMetadata] = field(default_factory=dict)


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
    cu_seqlens_ori_kv: torch.Tensor | None = None,
) -> None:
    """Compute the ``*_metadata`` kernel outputs stored in ``record``.

    ``num_heads`` / ``head_dim`` describe the main attention (used by the
    SMLA kernels); ``index_n_heads`` / ``index_head_dim`` describe the
    indexer projections (used by the LightningIndexer and SLIG kernels).
    The indexer score sums over the head dimension, so tensor parallel is
    fixed at 1 for the DSA path.  Under context parallel
    ``cu_seqlens_ori_kv`` is the per-segment window-pack cumsum; otherwise
    the bare ``cu_seq_k`` (identical to ``cu_seq_q`` without CP).
    """
    has_cmp_kv = ratio > 1
    ori_cu = cu_seqlens_ori_kv if cu_seqlens_ori_kv is not None else varlen.cu_seq_k
    smla_kwargs = {
        "cu_seqlens_q": varlen.cu_seq_q,
        "cu_seqlens_ori_kv": ori_cu,
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
    record.smla_metadata = sparse_flash_mla_metadata(num_heads, 1, head_dim, **smla_kwargs)
    grad_kwargs: dict[str, Any] = dict(smla_kwargs)
    grad_kwargs.pop("ori_topk_length")
    grad_kwargs.pop("cmp_topk_length")
    record.smla_grad_metadata = sparse_flash_mla_grad_metadata(num_heads, 1, head_dim, **grad_kwargs)
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
    reference tier is never materialized here.  Under context parallel the
    per-ratio block plans (``metadata.plans[ratio]``) and the window plan
    (``metadata.window``) carry the dispatcher machinery too.
    """
    tensors: list[torch.Tensor] = [metadata.varlen.cu_seq_q]
    for plan in metadata.plans.values():
        for name in (
            "cu_seqlens_cmp_k",
            "block_remainder",
            "gather_indices",
            "block_positions",
            "first_indices",
            "compressed_rows",
            "cmp_k_global_gather_indices",
        ):
            tensor = getattr(plan, name, None)
            if tensor is not None and tensor.numel() > 0:
                tensors.append(tensor)
        if plan.exchange is not None:
            tensors += [plan.exchange.send_indices, plan.exchange.recv_offsets]
    if metadata.window is not None:
        window = metadata.window
        tensors += [
            window.exchange.send_indices,
            window.exchange.recv_offsets,
            window.gather_indices,
            window.cu_seqlens_ori_kv,
        ]
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


class CANNMetadataExtension(MetadataExtension):
    """The CANN metadata extension: fills the vendor kernel tensors onto the
    model-built metadata (the model dir stays backend-agnostic).

    Receives the model-dir metadata — ``CompressedVarlenMetadata`` (non-CP)
    or the CP-shaped variant (CP, built by the model's
    ``build_attention_masks``) — and returns the slim
    ``CANNCompressedVarlenMetadata`` carrying the ``cann_plans``.  The
    attention geometry (``num_heads``, ``head_dim``, ``index_n_heads``,
    ``index_head_dim``, ``index_topk``) is passed as override kwargs by the
    factory below, keeping the model directory backend-agnostic.  Mismatched
    geometry fails loudly: the CANN metadata kernels and the fused core
    validate against the actual tensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MetadataExtension.Config):
        num_heads: int
        head_dim: int
        index_n_heads: int
        index_head_dim: int
        index_topk: int

    def __call__(self, metadata) -> CANNCompressedVarlenMetadata:
        cfg = cast("CANNMetadataExtension.Config", self.config)
        plans = metadata.plans
        for ratio, p in plans.items():
            if ratio > 1 and p.gather_indices.numel() == 0:
                raise ValueError(
                    f"batch has no complete compression block for ratio={ratio}; "
                    "doc-packed sequences must be long enough to produce at least "
                    "one full block per sequence."
                )
        cann_plans: dict[int, CANNBlockLayoutMetadata] = {}
        window = metadata.window
        ori_cu = window.cu_seqlens_ori_kv if window is not None else None
        for ratio, p in plans.items():
            record = CANNBlockLayoutMetadata()
            _fill_cann_metadata(
                metadata.varlen,
                ratio,
                p,
                record,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                index_n_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=cfg.index_topk,
                window_size=cfg.window_size,
                cu_seqlens_ori_kv=ori_cu,
            )
            cann_plans[ratio] = record
        # ``plans[ratio]`` carries the full per-ratio plan: part 1 (the
        # unified compressor/kernel contract) and part 2 (the dispatcher
        # fields — the block borrow exchange routing + assembly,
        # ``compressed_rows``/``out_width``, ``cmp_k_global_gather_indices``); the
        # window plan rides on the metadata (``window``).
        result = CANNCompressedVarlenMetadata(
            varlen=metadata.varlen,
            plans=plans,
            window=window,
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

    The ``cann_ops_transformer`` kernels carry no native autograd
    registration (unlike dsv3.2's older ``torch_npu`` ops), so the manual
    Function runs SMLA forward, SMLAG backward, and the SLIG indexer-loss
    gradient (which consumes SMLAG's ``cmp_softmax_l1``) in one bridge.
    All CANN metadata tensors are precomputed in
    ``CompressedBlockLayout``.
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
        cu_seqlens_ori_kv,
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
        hooks,
    ):
        # Non-CP self-attention uses the same cumsum for Q and original KV.
        # Resolve the alias inside the Function instead of passing one Tensor
        # object twice to autograd.Function.apply, which Dynamo rejects.
        if cu_seqlens_ori_kv is None:
            cu_seqlens_ori_kv = cu_seqlens_q
        has_compressed = ratio > 1
        result, softmax_lse = hooks.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            cmp_sparse_indices=(cmp_sparse_indices if cmp_sparse_indices is not None else None),
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
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
            cu_seqlens_ori_kv,
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
        ctx.hooks = hooks
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
            cu_seqlens_ori_kv,
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
        ) = ctx.hooks.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(cmp_sparse_indices if has_sparse_indices else None),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
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
            didx_q, didx_k, didx_w = _SparseFlashMLATND._indexer_loss_grad(
                ctx,
                cmp_softmax_l1,
                idx_q,
                idx_k,
                idx_w,
                cmp_sparse_indices,
                cmp_residual_kv,
                cu_seqlens_q,
                cu_seqlens_cmp_kv,
                slig_metadata,
            )

        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_ori_kv
            None,  # cu_seqlens_cmp_kv
            None,  # cmp_residual_kv
            None,  # smla_metadata
            None,  # smla_grad_metadata
            None,  # slig_metadata
            didx_q,
            didx_k,
            didx_w,
            None,  # softmax_scale
            None,  # ratio
            None,  # window_size
            None,  # indexer_loss_coeff
            None,  # indexer_loss_accumulator
            None,  # hooks
        )

    @staticmethod
    def _indexer_loss_grad(
        ctx,
        cmp_softmax_l1,
        idx_q,
        idx_k,
        idx_w,
        cmp_sparse_indices,
        cmp_residual_kv,
        cu_seqlens_q,
        cu_seqlens_cmp_kv,
        slig_metadata,
    ):
        """The LightningIndexer KL-loss gradient (SLIG), mirroring dsv3.2's
        ``CANNSparseIndexerLoss`` role: runs in the backward (it consumes
        SMLAG's ``cmp_softmax_l1``), scales the indexer grads, and
        accumulates the detached LI loss for logging."""
        if any(x is None for x in (idx_q, idx_k, idx_w, slig_metadata)):
            raise RuntimeError("ratio-4 cann requires LI tensors and slig_metadata in backward.")
        (
            didx_q,
            didx_k,
            didx_w,
            indexer_softmax,
        ) = ctx.hooks.sparse_lightning_indexer_kl_loss_grad(
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
        return didx_q, didx_k, didx_w


class CANNCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Run LI and SMLA/SMLAG/SLIG in a local TND layout.

    The flow mirrors ``CANNSparseInnerAttention`` (dsv3.2): the core
    contains the **kgather** — the compressed-level gather assembling the
    per-segment packed TND streams from the all-gathered padded containers
    (``cmp_k_global_gather_indices``; the plain identity slice without context
    parallel) — and then calls the CANN ops directly (LightningIndexer,
    then SparseFlashMLA).  The difference from dsv3.2 is the op set: the
    ``cann_ops_transformer`` SMLA/SMLAG/SLIG kernels carry no native
    autograd registration (unlike the older ``torch_npu``
    ``npu_sparse_flash_attention``), so the fused core needs the manual
    ``_SparseFlashMLATND`` bridge — SMLA forward, SMLAG backward, and the
    indexer KL-loss gradient (SLIG, the dsv3.2 ``CANNSparseIndexerLoss``
    role) in the same backward because it consumes SMLAG's
    ``cmp_softmax_l1``.
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
        self.hooks = _CANN_SPARSEATTN_HOOK

    @staticmethod
    def _assemble_tnd(container, plan) -> torch.Tensor:
        """The kgather: the gathered container -> the per-segment packed
        TND stream (``[T_cmp, 1, D]``)."""
        asm = plan.cmp_k_global_gather_indices
        flat = container.flatten(0, 1)
        return (flat[asm] if asm is not None else flat[: plan.cu_seqlens_cmp_k[-1]]).unsqueeze(1).contiguous()

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attn_sink: torch.Tensor | None = None,
        *,
        attention_masks=None,
    ):
        hooks = self.hooks
        if not isinstance(attention_masks, CANNCompressedVarlenMetadata):
            raise TypeError("cann requires CANNCompressedVarlenMetadata attention masks.")
        if attn_sink is None:
            raise ValueError("CANNCompressedSparseInnerAttention requires attn_sink")
        metadata = attention_masks
        plan = metadata.plans.get(self.compress_ratio)
        if plan is None:
            raise ValueError(f"No CompressedBlockLayout for ratio={self.compress_ratio}.")
        if self.compress_ratio <= 1:
            if cmp_k is not None and cmp_k.numel() != 0:
                raise ValueError("ratio-1 cann must not receive compressed KV.")
        elif cmp_k is None or cmp_k.ndim != 3:
            raise ValueError("ratio>1 cann requires compressed KV in the container layout [B, S//ratio, D].")
        # q / swa_k shape consistency and the kernel input layouts are
        # validated by the aclnn interface itself.
        npu = metadata.cann_plans[self.compress_ratio]
        batch_size, seqlen, _, _ = q.shape

        q = q.flatten(0, 1)
        swa_k = swa_k.flatten(0, 1).unsqueeze(1).contiguous()
        # The ori cumsum: the kernel's ``cu_seqlens_ori_kv`` — under context
        # parallel the window plan's packed cumsum, otherwise the bare
        # ``cu_seq_k`` (identical to ``cu_seq_q`` without CP).
        cu_seqlens_ori_kv = metadata.window.cu_seqlens_ori_kv if metadata.window is not None else None
        if cmp_k is not None:
            # The container is the ShardingConfig all-gather's output (under
            # CP the padded per-rank containers concatenated); the kgather
            # assembles the per-segment packed stream.
            cmp_k = self._assemble_tnd(cmp_k, plan)

        cmp_sparse_indices = None
        if self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError("ratio-4 cann requires all LI projection tensors.")
            idx_q = idx_q.flatten(0, 1)
            idx_k = self._assemble_tnd(idx_k, plan)
            idx_w = idx_w.flatten(0, 1)
            cmp_sparse_indices, _ = hooks.lightning_indexer(
                idx_q,
                idx_k,
                idx_w.float(),
                self.index_topk,
                cu_seqlens_q=metadata.varlen.cu_seq_q,
                cu_seqlens_k=plan.cu_seqlens_cmp_k,
                cmp_residual_k=plan.block_remainder,
                metadata=npu.li_metadata,
                layout_q=_LAYOUT,
                layout_k=_LAYOUT,
                mask_mode=_CMP_MASK_MODE,
                cmp_ratio=4,
                return_value=1,
            )

        indexer_loss_coeff = float(self.indexer_loss_coeff) if self.training else 0.0
        indexer_loss_accumulator = self._indexer_loss_acc

        output = _SparseFlashMLATND.apply(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            attn_sink.float(),
            metadata.varlen.cu_seq_q,
            cu_seqlens_ori_kv,
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
            hooks,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])
