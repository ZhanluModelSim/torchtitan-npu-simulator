# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.common import dsa_indexer_loss
from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLossLoggingHelper
from torchtitan_npu.models.deepseek_v4.model import (
    Compressor,
    InnerAttention,
    LiCompute,
    LiLoss,
    SparseAttention,
    enable_smla_varlen_attention_dispatch,
)
from torchtitan_npu.models.deepseek_v4.tnd import (
    DeepSeekV4SMLAAttentionMasks,
    smla_global_tnd_post_dataloading_process,
)
from torchtitan_npu.models.deepseek_v4.tnd import (
    max_seqlen_from_cu_seqlens as _max_seqlen_from_cu_seqlens,
)
from torchtitan_npu.ops.aclnn.builder import build_op
from torchtitan_npu.tools.device import get_npu_device_type

logger = logging.getLogger(__name__)

TORCH_MAX_INT = 9223372036854775807

# Will be compiled lazily, only when converter hits.
_li_op, _kl_op, _sas_op = None, None, None
_smla_ops_module: Any | None = None


def _smla_ops() -> Any:
    global _smla_ops_module
    if _smla_ops_module is None:
        try:
            module: Any = importlib.import_module("cann_ops_transformer")
        except ImportError as exc:
            raise RuntimeError("DeepSeekV4 A5 SMLA fusion requires the cann_ops_transformer package.") from exc
        _smla_ops_module = module.ops
    return _smla_ops_module


class SMLAMetadataCache:
    def __init__(self, model_args: Any) -> None:
        self.model_args = model_args
        self._attention_mask_cache_id: int | None = None
        self._op_metadata: dict[tuple[str, int, tuple[Any, ...]], torch.Tensor | None] = {}

    def get_or_create(
        self,
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        cmp_ratio: int,
        name: str,
        builder: Callable[..., torch.Tensor | None],
        *builder_args: Any,
    ) -> torch.Tensor | None:
        cache_id = attention_masks.cache_id if attention_masks.cache_id >= 0 else id(attention_masks)
        if self._attention_mask_cache_id != cache_id:
            self._attention_mask_cache_id = cache_id
            self._op_metadata.clear()
        key = (name, cmp_ratio, builder_args)
        if key not in self._op_metadata:
            self._op_metadata[key] = builder(attention_masks, self.model_args, cmp_ratio, *builder_args)
        return self._op_metadata[key]


def _patch_post_dataloading_process_for_smla_global_tnd() -> None:
    def make_wrapper(original):
        @wraps(original)
        def wrapper(self, *args, **kwargs):
            model_args = getattr(self, "model_config", None)
            use_global_tnd = bool(
                model_args is not None
                and getattr(model_args, "use_smla", False)
                and getattr(model_args, "use_global_tnd", False)
            )
            if not use_global_tnd:
                return original(self, *args, **kwargs)

            input_dict = args[0] if args else kwargs.get("input_dict")
            labels = args[1] if len(args) > 1 else kwargs.get("labels")
            if labels is None:
                raise TypeError("DeepSeek-V4 global TND post_dataloading_process requires labels.")
            return smla_global_tnd_post_dataloading_process(input_dict, labels, model_args)

        return wrapper

    for module_name in ("torchtitan.trainer", "torchtitan.train"):
        try:
            titan_module = importlib.import_module(module_name)
        except ImportError:
            continue

        trainer_cls = getattr(titan_module, "Trainer", None)
        if trainer_cls is None or getattr(trainer_cls, "npu_smla_global_tnd_postprocess_patched", False):
            continue

        original = trainer_cls.post_dataloading_process
        trainer_cls.post_dataloading_process = make_wrapper(original)
        trainer_cls.npu_smla_global_tnd_postprocess_patched = True
        logger.info("[NpuSMLAConverter] Registered DeepSeekV4 SMLA global TND post-dataloading hook.")


def _enable_native_smla_attention_mask_building() -> None:
    enable_smla_varlen_attention_dispatch()
    _patch_post_dataloading_process_for_smla_global_tnd()


def _none_grads(count: int) -> tuple[None, ...]:
    return (None,) * count


def _wrap_module(wrapper_cls, parent, **attrs):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.__dict__.update(parent.__dict__)
    wrapper.__dict__.update(attrs)
    return wrapper


def _require_op_metadata(
    cache: SMLAMetadataCache,
    attention_masks: DeepSeekV4SMLAAttentionMasks | None,
    cmp_ratio: int,
    name: str,
    builder: Callable[..., torch.Tensor | None],
    *builder_args: Any,
) -> torch.Tensor:
    metadata = (
        None
        if attention_masks is None
        else cache.get_or_create(attention_masks, cmp_ratio, name, builder, *builder_args)
    )
    if metadata is None:
        raise RuntimeError(
            f"Missing DeepSeek-V4 SMLA metadata {name!r}. "
            "The SMLA attention_masks must be prepared before model execution."
        )
    return metadata


def _require_attention_masks(
    attention_masks: DeepSeekV4SMLAAttentionMasks | None,
) -> DeepSeekV4SMLAAttentionMasks:
    if attention_masks is None:
        raise RuntimeError("DeepSeek-V4 SMLA attention_masks are required for this kernel.")
    return attention_masks


def _sparse_attention_metadata_kwargs(
    attention_masks: DeepSeekV4SMLAAttentionMasks,
    model_args: Any,
    cmp_ratio: int,
    num_heads_q: int,
) -> dict[str, Any]:
    cu_seqlens_cmp_kv = attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio)
    return {
        "batch_size": attention_masks.batch_size,
        "max_seqlen_q": _max_seqlen_from_cu_seqlens(attention_masks.cu_seqlens_q),
        "max_seqlen_ori_kv": _max_seqlen_from_cu_seqlens(attention_masks.cu_seqlens_ori_kv),
        "max_seqlen_cmp_kv": 0 if cu_seqlens_cmp_kv is None else _max_seqlen_from_cu_seqlens(cu_seqlens_cmp_kv),
        "num_heads_q": num_heads_q,
        "num_heads_kv": 1,
        "head_dim": model_args.head_dim,
        "cmp_topk": model_args.index_topk if cmp_ratio == 4 else 0,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "TND",
        "has_ori_kv": True,
        "has_cmp_kv": cmp_ratio > 1,
    }


class SparseAttnSharedKV(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    ):
        ori_kv_stride = ori_kv.stride(0) if ori_kv is not None else 0
        cmp_kv_stride = cmp_kv.stride(0) if cmp_kv is not None else 0
        # pyrefly: ignore [missing-attribute]
        metadata = _sas_op.npu_sparse_attn_sharedkv_metadata(
            # pyrefly: ignore [missing-attribute]
            cu_seq_lens_q if cu_seq_lens_q is not None else torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            num_heads_q,
            num_heads_kv,
            head_dim,
            batch_size,
            max_seq_len_q,
            max_seq_len_kv,
            0,  # oriTopk
            topk,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            ori_kv is not None,  # hasOriKv
            cmp_kv is not None,  # hasCmpKv
        )
        # pyrefly: ignore [missing-attribute]
        result, softmax_lse = _sas_op.npu_sparse_attn_sharedkv(
            query,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            None,  # oriBlockTable
            None,  # cmpBlockTable
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            None,  # sequsedQ
            None,  # sequsedKv
            sinks,
            metadata,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_kv_stride,
            cmp_kv_stride,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            True,  # returnSoftmaxLse
        )
        ctx.save_for_backward(
            query,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            sinks,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        return result

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        (
            query,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            sinks,
        ) = ctx.saved_tensors
        (
            query_grad,
            ori_kv_grad,
            cmp_kv_grad,
            sinks_grad,
            # pyrefly: ignore [missing-attribute]
        ) = _sas_op.npu_sparse_attn_sharedkv_grad(
            query,
            ori_kv,
            cmp_kv,
            grad_output,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            None,  # cuSeqlensQ
            None,  # cuSeqlensOriKv
            None,  # cuSeqlensCmpKv
            sinks,
            ctx.softmax_scale,
            ctx.cmp_ratio,
            ctx.ori_mask_mode,
            ctx.cmp_mask_mode,
            ctx.ori_win_left,
            ctx.ori_win_right,
            ctx.layout_q,
        )
        return (
            query_grad,
            ori_kv_grad,
            cmp_kv_grad,
            *_none_grads(5),
            sinks_grad,
            *_none_grads(15),
        )


def npu_sparse_attn_shared_kv(
    query,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    sinks,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
):
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # not support TND
    ori_sparse_indices = None  # ori kv use band mode
    batch_size, max_seq_len_q, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(1)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = "BSND"
    query = query.contiguous()  # [S, B, N, D] --> [B, S, N, D]
    ori_kv = ori_kv.unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_sparse_indices = None if cmp_ratio != 4 else cmp_sparse_indices.unsqueeze(2).contiguous()

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    )
    return output.contiguous()


def _compute_li_loss(
    softmax_out: torch.Tensor,
    cmp_softmax_l1: torch.Tensor,
    loss_scale: float,
) -> torch.Tensor:
    student = softmax_out.float().clamp_min(1e-10)
    target = cmp_softmax_l1.float().clamp_min(0)
    target_sum = target.sum(dim=-1, keepdim=True)
    valid_target = target_sum > 1e-10
    # Fully masked rows have no teacher mass; keep logits finite and let target_sum zero them out.
    student = torch.where(valid_target, student, torch.ones_like(student))
    teacher = target / target_sum.clamp_min(1e-10)
    log_teacher = teacher.clamp_min(1e-10).log()
    loss = (teacher * (log_teacher - student.log())).sum(dim=-1)
    loss = (target_sum.squeeze(-1) * loss).mean()
    return loss * loss_scale


class SparseFlashMLA(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        if kwargs:
            raise TypeError("SparseFlashMLA.forward does not accept keyword arguments.")
        ctx, *forward_args = args
        (
            query,
            ori_kv,
            cmp_kv,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            indexer_q,
            indexer_k,
            weights,
            attention_masks,
            metadata_cache,
            layer_number,
            num_layers,
        ) = forward_args
        num_heads_q = query.shape[-2]
        metadata = SparseFlashMLA.sparse_attn_metadata(metadata_cache, attention_masks, cmp_ratio, num_heads_q)
        result, softmax_lse = _smla_ops().sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_ori_kv=cu_seq_lens_ori_kv,
            cu_seqlens_cmp_kv=cu_seq_lens_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            result,
            softmax_lse,
            query,
            ori_kv,
            cmp_kv,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            indexer_q,
            indexer_k,
            weights,
        )
        ctx.attention_masks = attention_masks
        ctx.metadata_cache = metadata_cache
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        ctx.layer_number = layer_number
        ctx.num_layers = num_layers
        ctx.num_heads_q = num_heads_q
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        (
            fa_out,
            softmax_lse,
            query,
            ori_kv,
            cmp_kv,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            indexer_q,
            indexer_k,
            weights,
        ) = ctx.saved_tensors
        fag_metadata = SparseFlashMLA.sparse_flash_mla_grad_metadata(
            ctx.metadata_cache, ctx.attention_masks, ctx.cmp_ratio, ctx.num_heads_q
        )
        (
            dq,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = _smla_ops().sparse_flash_mla_grad(
            query,
            grad_output.contiguous(),
            fa_out,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_ori_kv=cu_seq_lens_ori_kv,
            cu_seqlens_cmp_kv=cu_seq_lens_cmp_kv,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=fag_metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout_q=ctx.layout_q,
            layout_kv=ctx.layout_kv,
        )
        if cmp_kv is None:
            dcmp_kv = None

        dindexer_q = dindexer_k = dw = None
        if ctx.cmp_ratio == 4:
            if indexer_q is None or indexer_k is None or weights is None:
                raise RuntimeError("DeepSeek-V4 SMLA LI tensors are required for LI grad.")
            lig_metadata = SparseFlashMLA.lightning_indexer_klloss_grad_metadata(
                ctx.metadata_cache, ctx.attention_masks, ctx.cmp_ratio, indexer_q.shape[-2]
            )
            (
                dindexer_q,
                dindexer_k,
                dw,
                softmax_out,
            ) = _smla_ops().sparse_lightning_indexer_kl_loss_grad(
                q=indexer_q,
                k=indexer_k.unsqueeze(1),
                w=weights.to(torch.float32),
                sparse_indices=cmp_sparse_indices,
                attn_softmax_l1_norm=cmp_softmax_l1,
                cmp_residual_k=cmp_residual_kv,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_cmp_kv,
                metadata=lig_metadata,
                layout_q=ctx.layout_q,
                layout_k=ctx.layout_kv,
                mask_mode=ctx.cmp_mask_mode,
                cmp_ratio=ctx.cmp_ratio,
            )
            if logger.isEnabledFor(logging.DEBUG):
                li_loss = _compute_li_loss(softmax_out, cmp_softmax_l1, ctx.softmax_scale)
                if ctx.layer_number is not None:
                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(li_loss, ctx.layer_number, ctx.num_layers)
            token_scale = 1.0 / float(cmp_softmax_l1.sum(dim=-1).numel())
            loss_backward_scale = dsa_indexer_loss.LOSS_SCALE.to(device=dindexer_q.device, dtype=torch.float32)
            grad_scale = loss_backward_scale * (ctx.softmax_scale * token_scale)
            dindexer_q = dindexer_q * grad_scale.to(dindexer_q.dtype)
            dindexer_k = dindexer_k.squeeze(1) * grad_scale.to(dindexer_k.dtype)
            dw = dw * grad_scale.to(dw.dtype)

        return (
            dq,
            dori_kv,
            dcmp_kv,
            *_none_grads(5),
            dsinks,
            *_none_grads(8),
            dindexer_q,
            dindexer_k,
            dw,
            *_none_grads(4),
        )

    @staticmethod
    def sparse_attn_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "sparse_attn",
            SparseFlashMLA._create_sparse_attn_metadata,
            num_heads_q,
        )

    @staticmethod
    def sparse_flash_mla_grad_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "sparse_flash_mla_grad",
            SparseFlashMLA._create_sparse_flash_mla_grad_metadata,
            num_heads_q,
        )

    @staticmethod
    def lightning_indexer_klloss_grad_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "lightning_indexer_klloss_grad",
            SparseFlashMLA._create_lightning_indexer_klloss_grad_metadata,
            num_heads_q,
        )

    @staticmethod
    def _create_sparse_attn_metadata(
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        model_args: Any,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor:
        return _smla_ops().sparse_flash_mla_metadata(
            cu_seqlens_q=attention_masks.cu_seqlens_q,
            cu_seqlens_ori_kv=attention_masks.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio),
            cmp_residual_kv=attention_masks.cmp_residual_k.get(cmp_ratio),
            ori_topk_length=None,
            cmp_topk_length=None,
            **_sparse_attention_metadata_kwargs(attention_masks, model_args, cmp_ratio, num_heads_q),
        )

    @staticmethod
    def _create_sparse_flash_mla_grad_metadata(
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        model_args: Any,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor:
        return _smla_ops().sparse_flash_mla_grad_metadata(
            cu_seqlens_q=attention_masks.cu_seqlens_q,
            cu_seqlens_ori_kv=attention_masks.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio),
            cmp_residual_kv=attention_masks.cmp_residual_k.get(cmp_ratio),
            ori_topk_length=None,
            cmp_topk_length=None,
            ori_topk=0,
            **_sparse_attention_metadata_kwargs(attention_masks, model_args, cmp_ratio, num_heads_q),
        )

    @staticmethod
    def _create_lightning_indexer_klloss_grad_metadata(
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        model_args: Any,
        cmp_ratio: int,
        num_heads_q: int,
    ) -> torch.Tensor | None:
        if cmp_ratio != 4:
            return None

        cu_seqlens_k = attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio)
        return _smla_ops().sparse_lightning_indexer_kl_loss_grad_metadata(
            num_heads_q,
            1,
            model_args.index_head_dim,
            cu_seqlens_q=attention_masks.cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=attention_masks.cmp_residual_k.get(cmp_ratio),
            batch_size=attention_masks.batch_size,
            max_seqlen_q=_max_seqlen_from_cu_seqlens(attention_masks.cu_seqlens_q),
            max_seqlen_k=0 if cu_seqlens_k is None else _max_seqlen_from_cu_seqlens(cu_seqlens_k),
            topk=model_args.index_topk,
            layout_q="TND",
            layout_k="TND",
            mask_mode=3,
            cmp_ratio=cmp_ratio,
        )


class LightningIndexer(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        if kwargs:
            raise TypeError("LightningIndexer.forward does not accept keyword arguments.")
        (
            _,
            query,
            key,
            weights,
            sparse_count,
            sparse_mode,
            cmp_ratio,
            attention_masks,
            metadata_cache,
            layout_query,
            layout_key,
        ) = args
        metadata = LightningIndexer.op_metadata(metadata_cache, attention_masks, cmp_ratio)
        return _smla_ops().lightning_indexer(
            query,
            key,
            weights.float(),
            sparse_count,
            cu_seqlens_q=attention_masks.cu_seqlens_q,
            cu_seqlens_k=attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio),
            cmp_residual_k=attention_masks.cmp_residual_k.get(cmp_ratio),
            metadata=metadata,
            layout_q=layout_query,
            layout_k=layout_key,
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
            return_value=1,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return _none_grads(10)

    @staticmethod
    def op_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "lightning_indexer",
            LightningIndexer._create_metadata,
        )

    @staticmethod
    def _create_metadata(
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        model_args: Any,
        cmp_ratio: int,
    ) -> torch.Tensor | None:
        if cmp_ratio != 4:
            return None

        cu_seqlens_k = attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio)
        return _smla_ops().lightning_indexer_metadata(
            model_args.index_n_heads,
            1,
            model_args.index_head_dim,
            model_args.index_topk,
            cu_seqlens_q=attention_masks.cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=attention_masks.cmp_residual_k.get(cmp_ratio),
            batch_size=attention_masks.batch_size,
            max_seqlen_q=_max_seqlen_from_cu_seqlens(attention_masks.cu_seqlens_q),
            max_seqlen_k=0 if cu_seqlens_k is None else _max_seqlen_from_cu_seqlens(cu_seqlens_k),
            layout_q="TND",
            layout_k="TND",
            mask_mode=3,
            cmp_ratio=cmp_ratio,
        )


def npu_lightning_indexer(
    q_indexer,
    k_indexer,
    weights,
    sparse_count: int,
    sparse_mode: int,
    cmp_ratio: int,
    attention_masks: DeepSeekV4SMLAAttentionMasks,
    metadata_cache: SMLAMetadataCache,
    layout_query: str = "TND",
    layout_key: str = "TND",
):
    return LightningIndexer.apply(
        q_indexer,
        k_indexer,
        weights,
        sparse_count,
        sparse_mode,
        cmp_ratio,
        attention_masks,
        metadata_cache,
        layout_query,
        layout_key,
    )


def npu_sparse_flash_mla(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    indexer_q: torch.Tensor | None = None,
    indexer_k: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    attention_masks: DeepSeekV4SMLAAttentionMasks | None = None,
    metadata_cache: SMLAMetadataCache | None = None,
    layer_number: int | None = None,
    num_layers: int = 0,
):
    if query.dim() != 3:
        raise RuntimeError(f"DeepSeek-V4 SMLA query must be TND [T, N, D], got shape {tuple(query.shape)}.")
    if ori_kv.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA ori_kv must be TND [T, D], got shape {tuple(ori_kv.shape)}.")
    if cmp_kv is not None and cmp_kv.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA cmp_kv must be TND [T, D], got shape {tuple(cmp_kv.shape)}.")
    if indexer_q is not None and indexer_q.dim() != 3:
        raise RuntimeError(f"DeepSeek-V4 SMLA indexer_q must be TND [T, N, D], got shape {tuple(indexer_q.shape)}.")
    if indexer_k is not None and indexer_k.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA indexer_k must be TND [T, D], got shape {tuple(indexer_k.shape)}.")
    if weights is not None and weights.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA weights must be TND [T, N], got shape {tuple(weights.shape)}.")

    attention_masks = _require_attention_masks(attention_masks)
    if metadata_cache is None:
        raise RuntimeError("DeepSeek-V4 SMLA metadata cache is not bound.")

    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.to(torch.int32)
        if cmp_ratio == 4:
            if cmp_sparse_indices.dim() == 2:
                cmp_sparse_indices = cmp_sparse_indices.unsqueeze(1)
            elif cmp_sparse_indices.dim() != 3 or cmp_sparse_indices.shape[1] != 1:
                raise RuntimeError(
                    "DeepSeek-V4 SMLA cmp_sparse_indices must be TND [T, 1, K] or [T, K], "
                    f"got shape {tuple(cmp_sparse_indices.shape)}."
                )
        cmp_sparse_indices = cmp_sparse_indices.contiguous()

    output = SparseFlashMLA.apply(
        query.contiguous(),
        ori_kv.unsqueeze(1).contiguous(),
        None if cmp_kv is None else cmp_kv.unsqueeze(1).contiguous(),
        attention_masks.cu_seqlens_q,
        attention_masks.cu_seqlens_ori_kv,
        attention_masks.cu_seqlens_cmp_kv.get(cmp_ratio),
        cmp_sparse_indices,
        attention_masks.cmp_residual_k.get(cmp_ratio),
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        "TND",
        "TND",
        None if indexer_q is None else indexer_q.contiguous(),
        None if indexer_k is None else indexer_k.contiguous(),
        None if weights is None else weights.contiguous(),
        attention_masks,
        metadata_cache,
        layer_number,
        num_layers,
    )
    return output.contiguous()


def sdpa_to_li_adapter_smla(
    self,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    _seqlen: int,
    _offset: int,
    attention_masks: DeepSeekV4SMLAAttentionMasks | None,
    metadata_cache: SMLAMetadataCache,
):
    if q_indexer.dim() != 3:
        raise RuntimeError(f"DeepSeek-V4 SMLA q_indexer must be TND [T, N, D], got shape {tuple(q_indexer.shape)}.")
    if k_indexer.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA k_indexer must be TND [T, D], got shape {tuple(k_indexer.shape)}.")
    if weights.dim() != 2:
        raise RuntimeError(f"DeepSeek-V4 SMLA weights must be TND [T, N], got shape {tuple(weights.shape)}.")

    attention_masks = _require_attention_masks(attention_masks)
    return npu_lightning_indexer(
        q_indexer.to(torch.bfloat16).contiguous(),
        k_indexer.to(torch.bfloat16).unsqueeze(1).contiguous(),
        weights.to(torch.bfloat16).contiguous(),
        sparse_count=self.index_topk,
        sparse_mode=3,
        cmp_ratio=self.ratio,
        attention_masks=attention_masks,
        metadata_cache=metadata_cache,
    )


class NpuLiComputeSMLA(LiCompute):
    def forward(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        seqlen: int,
        offset: int,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None = None,
    ):
        metadata_cache = cast("SMLAMetadataCache", self._smla_metadata_cache)
        return sdpa_to_li_adapter_smla(
            self,
            q_indexer,
            k_indexer,
            weights,
            seqlen,
            offset,
            attention_masks,
            metadata_cache,
        )


class NpuInnerAttentionSMLA(InnerAttention):
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_compress: torch.Tensor | None,
        q_indexer: torch.Tensor | None,
        k_indexer: torch.Tensor | None,
        weights: torch.Tensor | None,
        seqlen: int,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None = None,
    ):
        metadata_cache = cast("SMLAMetadataCache", self._smla_metadata_cache)
        offset = 0 if self.use_smla else kv.size(1)
        compress_topk_idxs = index_score = None
        layer_number = None
        num_layers = 0
        has_li = self.compress_ratio > 1 and hasattr(self, "li_compute") and q_indexer is not None
        if has_li:
            compress_topk_idxs, index_score = self.li_compute(
                q_indexer,
                k_indexer,
                weights,
                seqlen,
                offset,
                attention_masks,
            )
            li_loss = getattr(self, "li_loss", None)
            layer_number = getattr(li_loss, "layer_id", None)
            num_layers = getattr(li_loss, "n_layers", 0) if layer_number is not None else 0

        if compress_topk_idxs is not None and compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)

        output = npu_sparse_flash_mla(
            query=q,
            ori_kv=kv,
            cmp_kv=kv_compress,
            cmp_sparse_indices=compress_topk_idxs,
            sinks=self.attn_sink.float(),
            softmax_scale=self.sparse_attn.softmax_scale,
            cmp_ratio=self.sparse_attn.compress_ratio,
            indexer_q=q_indexer,
            indexer_k=k_indexer,
            weights=weights,
            attention_masks=attention_masks,
            metadata_cache=metadata_cache,
            layer_number=layer_number,
            num_layers=num_layers,
        )
        return output, compress_topk_idxs, index_score


class NpuSparseAttention(SparseAttention):
    def __init__(self, parent: SparseAttention) -> None:
        # Shallow copy of parent's __dict__ is intentional here:
        # - SparseAttention attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on SparseAttention.__init__ parameters (layer_id, window_size, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If SparseAttention had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attn_sink: torch.Tensor,
        kv_compress: torch.Tensor | None = None,
        compress_topk_idxs: torch.Tensor | None = None,
    ):
        if compress_topk_idxs is not None and compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)

        return npu_sparse_attn_shared_kv(
            query=query_states,
            ori_kv=kv_states,
            cmp_kv=kv_compress,
            cmp_sparse_indices=compress_topk_idxs if self.compress_ratio == 4 else None,
            sinks=attn_sink.float(),
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
        )


class NpuLiCompute(LiCompute):
    def __init__(self, parent: LiCompute) -> None:
        # Shallow copy of parent's __dict__ is intentional here:
        # - LiCompute attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on LiCompute.__init__ parameters (ratio, index_topk)
        # - Parent instance already has all attributes properly initialized
        # Note: If LiCompute had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        seqlen: int,
        offset: int,
    ):
        q_indexer = q_indexer.to(torch.bfloat16)
        k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
        weights = weights.to(torch.bfloat16)

        # pyrefly: ignore [missing-attribute]
        compress_topk_idxs, index_score = _li_op.npu_lightning_indexer(
            q_indexer,
            k_indexer,
            weights,
            None,  # actual_seq_q
            None,  # actual_seq_k
            None,  # block_table
            "BSND",  # layout_q
            "BSND",  # layout_k
            self.index_topk,
            3,  # sparse_mode
            TORCH_MAX_INT,  # pre_tokens
            TORCH_MAX_INT,  # next_tokens
            self.ratio,
            True,  # return_values
        )

        compress_topk_idxs: torch.Tensor = compress_topk_idxs.squeeze(2)
        index_score = index_score.squeeze(2)
        compress_topk_idxs = torch.where(compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs + offset)

        return compress_topk_idxs, index_score


class SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        scale_value,
        cmp_ratio,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        layer_number=None,
        num_layers=0,
    ):
        ctx.save_for_backward(query, key, query_index, key_index, weights, sparse_indices)
        ctx.scale_value = scale_value
        ctx.cmp_ratio = cmp_ratio
        ctx.layer_number = layer_number
        ctx.num_layers = num_layers
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens

        # Return dummy loss during fwd, real operation will be postponed
        # to bwd, to avoid redundant computation of the loss function in
        # case where activation checkpointing is enabled.
        return torch.zeros(1, dtype=torch.float32, device=query.device)[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        query, key, query_index, key_index, weights, sparse_indices = ctx.saved_tensors

        (
            d_query_index,
            d_key_index,
            d_weights,
            loss,
            # pyrefly: ignore [missing-attribute]
        ) = _kl_op.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            None,  # softmax_max
            None,  # softmax_sum
            None,  # query_rope
            None,  # key_rope
            None,  # optional query lengths
            None,  # optional key lengths
            ctx.layout,
            ctx.sparse_mode,
            ctx.pre_tokens,
            ctx.next_tokens,
            ctx.cmp_ratio,
            ctx.scale_value,
            False,  # deterministic
        )

        bsz, slen, *_ = query.shape
        token_scale = 1 / (bsz * slen)
        loss_scale = ctx.scale_value
        grad_scale = grad * token_scale * loss_scale

        d_query_index = d_query_index * grad_scale
        d_key_index = d_key_index * grad_scale
        d_weights = d_weights * grad_scale
        loss = loss * token_scale * loss_scale

        if ctx.layer_number is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss[0], ctx.layer_number, ctx.num_layers)
        return (
            *_none_grads(2),
            d_query_index,
            d_key_index,
            d_weights,
            *_none_grads(9),
        )


# Wrapper for autograd.Function to support default/keyword argument
def npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    *,
    scale_value,
    cmp_ratio,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    layer_number=None,
    num_layers=0,
):
    return SparseLightningIndexerGradKLLossWrapper.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        scale_value,
        cmp_ratio,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        layer_number,
        num_layers,
    )


class NpuLiLoss(LiLoss):
    def __init__(self, parent: LiLoss):
        # Shallow copy of parent's __dict__ is intentional here:
        # - LiLoss attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on LiLoss.__init__ parameters (n_heads, softmax_scale, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If LiLoss had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        q,
        kv,
        kv_compress,
        attn_sink,
        q_indexer,
        k_indexer,
        weights,
        sparse_indices,
        indexer_score,
        attention_masks,
        offset,
    ):
        if sparse_indices.dtype != torch.int32:
            sparse_indices = sparse_indices.to(torch.int32)

        return npu_sparse_lightning_indexer_grad_kl_loss(
            q,
            kv_compress.unsqueeze(2),
            q_indexer,
            k_indexer.unsqueeze(2),
            weights,
            sparse_indices.unsqueeze(2),
            scale_value=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
            layer_number=self.layer_id,
            num_layers=self.n_layers,
        )


class NpuSMLAConverter(ModelCustomConverter):
    @staticmethod
    def convert_smla_kernel(model: nn.Module):
        use_global_tnd = bool(getattr(model.model_args, "use_global_tnd", False))
        if not use_global_tnd:
            raise RuntimeError(
                "DeepSeekV4 A5 npu_smla requires global TND. Enable both "
                "npu_mhc_pre and npu_mhc_post together with npu_smla."
            )

        _smla_ops()

        _enable_native_smla_attention_mask_building()
        metadata_cache = SMLAMetadataCache(model.model_args)

        modules = list(model.named_modules())
        for _, module in modules:
            if isinstance(module, Compressor):
                module.use_tnd_metadata = True

        for name, module in modules:
            if isinstance(module, LiCompute):
                replace_module_with_name(
                    model,
                    name,
                    _wrap_module(
                        NpuLiComputeSMLA,
                        module,
                        _smla_metadata_cache=metadata_cache,
                    ),
                )
                logger.info("[NpuSMLAConverter] [LiCompute SMLA forward] Applied.")

        for name, module in modules:
            if isinstance(module, InnerAttention):
                replace_module_with_name(
                    model,
                    name,
                    _wrap_module(
                        NpuInnerAttentionSMLA,
                        module,
                        _smla_metadata_cache=metadata_cache,
                    ),
                )
                logger.info("[NpuSMLAConverter] [InnerAttention SMLA forward] Applied.")

    def convert(self, model: nn.Module):
        global _li_op, _kl_op, _sas_op

        use_smla_kernel = get_npu_device_type() == "A5"
        if use_smla_kernel:
            self.convert_smla_kernel(model)
            return

        for name, module in list(model.named_modules()):
            if isinstance(module, SparseAttention):
                _sas_op = build_op("sparse_attn_sharedkv", ["sparse_attn_sharedkv/binding.cpp"])
                replace_module_with_name(model, name, NpuSparseAttention(module))
                logger.info("[NpuSMLAConverter] [SparseAttention forward] Applied.")

            if isinstance(module, LiCompute):
                _li_op = build_op("lightning_indexer", ["lightning_indexer/binding.cpp"])
                replace_module_with_name(model, name, NpuLiCompute(module))
                logger.info("[NpuSMLAConverter] [LiCompute forward] Applied.")

            if isinstance(module, LiLoss):
                _kl_op = build_op(
                    "sparse_lightning_indexer_grad_kl_loss",
                    ["sparse_lightning_indexer_grad_kl_loss/binding.cpp"],
                )
                replace_module_with_name(model, name, NpuLiLoss(module))
                logger.info("[NpuSMLAConverter] [LiLoss forward] Applied.")


@register_model_converter("npu_smla")
class NpuSMLAModelConfig(ModelCustomConfig):
    model_converter = NpuSMLAConverter
