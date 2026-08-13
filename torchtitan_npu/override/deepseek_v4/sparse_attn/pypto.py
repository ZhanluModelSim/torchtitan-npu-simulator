# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in DeepSeek-V4 sparse-attention bridge for the PyPTO LI/LIG kernels.

The metadata contract and the SMLA/SMLAG kernels are reused from ``cann.py``.
Only the LightningIndexer forward and its KL-loss backward are dispatched to
PyPTO.  Keeping the autograd bridge here leaves the default CANN path and
``cann.py`` unchanged while providing explicit replacement points for the
remaining two kernels in a future PyPTO integration.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch

from .cann import (
    _CMP_MASK_MODE,
    _LAYOUT,
    _ORI_MASK_MODE,
    CANNCompressedSparseInnerAttention,
    CANNCompressedVarlenMetadata,
    _compute_li_loss,
)

_LI_MODULE = "torchtitan_npu.ops.pypto.lightning_indexer.lightning_indexer"
_LIG_MODULE = (
    "torchtitan_npu.ops.pypto.sparse_lightning_indexer_kl_loss_grad."
    "sparse_lightning_indexer_kl_loss_grad"
)


@cache
def _load_pypto_op(module_name: str, op_name: str, device_index: int) -> Callable:
    os.environ.setdefault("TILE_FWK_DEVICE_ID", str(device_index))
    return getattr(importlib.import_module(module_name), op_name)


def _device_index(tensor: torch.Tensor) -> int:
    return 0 if tensor.device.index is None else tensor.device.index


def _pypto_lightning_indexer(q: torch.Tensor, *args, **kwargs):
    op = _load_pypto_op(_LI_MODULE, "lightning_indexer", _device_index(q))
    return op(q, *args, **kwargs)


def _pypto_sparse_lightning_indexer_kl_loss_grad(*, q: torch.Tensor, **kwargs):
    op = _load_pypto_op(
        _LIG_MODULE,
        "sparse_lightning_indexer_kl_loss_grad",
        _device_index(q),
    )
    return op(q=q, **kwargs)


# Keep all four dispatch points local.  SMLA/SMLAG intentionally call the
# existing CANN kernels today; replacing these two wrappers is sufficient when
# their PyPTO implementations become available.
def _sparse_flash_mla(*args, **kwargs):
    return torch.ops.cann_ops_transformer.sparse_flash_mla(*args, **kwargs)


def _sparse_flash_mla_grad(*args, **kwargs):
    # CANN SMLAG has no deterministic implementation. Keep the global
    # deterministic setting for the rest of the graph and exempt this call
    # only, restoring the exact mode before returning to the caller.
    deterministic_mode = torch.get_deterministic_debug_mode()
    if deterministic_mode:
        torch.set_deterministic_debug_mode(0)
    try:
        return torch.ops.cann_ops_transformer.sparse_flash_mla_grad(*args, **kwargs)
    finally:
        if deterministic_mode:
            torch.set_deterministic_debug_mode(deterministic_mode)


def _lightning_indexer(*args, **kwargs):
    return _pypto_lightning_indexer(*args, **kwargs)


def _sparse_lightning_indexer_kl_loss_grad(*, q: torch.Tensor, **kwargs):
    return _pypto_sparse_lightning_indexer_kl_loss_grad(q=q, **kwargs)


class _PyPTOSparseFlashMLATND(torch.autograd.Function):
    """Run CANN SMLA/SMLAG and replace only the LI/LIG pair with PyPTO."""

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
        result, softmax_lse = _sparse_flash_mla(
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
        ) = _sparse_flash_mla_grad(
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
                    "ratio-4 PyPTO requires LI tensors and slig_metadata in backward."
                )
            (
                didx_q,
                didx_k,
                didx_w,
                indexer_softmax,
            ) = _sparse_lightning_indexer_kl_loss_grad(
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


class PyPTOCompressedSparseInnerAttention(CANNCompressedSparseInnerAttention):
    """Use PyPTO LI/LIG with CANN metadata and SMLA/SMLAG."""

    @dataclass(kw_only=True, slots=True)
    class Config(CANNCompressedSparseInnerAttention.Config):
        pass

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
                "pypto requires CANNCompressedVarlenMetadata attention masks."
            )
        if attn_sink is None:
            raise ValueError("PyPTOCompressedSparseInnerAttention requires attn_sink")
        metadata = attention_masks
        plan = metadata.plans.get(self.compress_ratio)
        if plan is None:
            raise ValueError(
                f"No CompressedBlockLayout for ratio={self.compress_ratio}."
            )
        if self.compress_ratio <= 1:
            if cmp_k is not None and cmp_k.numel() != 0:
                raise ValueError("ratio-1 PyPTO must not receive compressed KV.")
        elif cmp_k is None or cmp_k.ndim != 3:
            raise ValueError(
                "ratio>1 PyPTO requires compressed KV in the container "
                "layout [B, S//ratio, D]."
            )

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
                raise ValueError("ratio-4 PyPTO requires all LI projection tensors.")
            idx_q = idx_q.flatten(0, 1)
            idx_k = (
                idx_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]
                .unsqueeze(1)
                .contiguous()
            )
            idx_w = idx_w.flatten(0, 1)
            cmp_sparse_indices, _ = _lightning_indexer(
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

        indexer_loss_coeff = float(self.indexer_loss_coeff) if self.training else 0.0
        output = _PyPTOSparseFlashMLATND.apply(
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
            self._indexer_loss_acc,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])
