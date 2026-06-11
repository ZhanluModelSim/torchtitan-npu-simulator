# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import logging
from collections.abc import Callable
from typing import Any, cast, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torchtitan.models.common import VarlenAttention

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.distributed.context_parallel import register_cp_mask_handler
from torchtitan_npu.models.common import dsa_indexer_loss
from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLossLoggingHelper
from torchtitan_npu.models.deepseek_v4.model import (
    DeepSeekV4Model,
    InnerAttention,
    LiCompute,
    LiLoss,
    SparseAttention,
)
from torchtitan_npu.ops.aclnn.builder import build_op
from torchtitan_npu.tools.device import get_npu_device_type

logger = logging.getLogger(__name__)

TORCH_MAX_INT = 9223372036854775807

# Will be compiled lazily, only when converter hits.
_li_op, _kl_op, _sas_op = None, None, None


class DeepSeekV4SMLAAttentionMasks(NamedTuple):
    actual_seq_qlen: torch.Tensor
    actual_seq_klen: torch.Tensor
    cmp_residual_kv: dict[int, torch.Tensor]
    batch_size: int
    seq_len: int
    # Per-rank q/k lengths under CP (q == local_s, k == (cp_rank + 1) * local_s);
    # both equal ``seq_len`` without CP. See ``_smla_cp_mask_handler``.
    q_seq_len: int = -1
    kv_seq_len: int = -1


class _SMLAAttentionConfig(NamedTuple):
    inner_attention: Any
    mask_type: str = "causal"


class _SMLALayerConfig(NamedTuple):
    attention: _SMLAAttentionConfig


class _SMLALayerView:
    def __init__(self, model_args: Any) -> None:
        self._model_args = model_args
        self._inner_attention = VarlenAttention.Config()

    def __len__(self) -> int:
        return self._model_args.n_layers + self._model_args.num_mtp_modules

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(len(self))[index]]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _SMLALayerConfig(
            attention=_SMLAAttentionConfig(inner_attention=self._inner_attention)
        )


def build_smla_attention_masks(
    positions: torch.Tensor,
    model_args: Any,
) -> DeepSeekV4SMLAAttentionMasks:
    batch_size, total_seq_len = positions.shape
    seq_len = total_seq_len - model_args.num_mtp_modules
    device = positions.device
    cmp_ratios = model_args.compress_ratios + (model_args.mtp_layer_compress_ratio,)
    residual_cmp_ratio_set = set()
    for ratio in cmp_ratios:
        if ratio > 1:
            residual_cmp_ratio_set.add(ratio)
    residual_cmp_ratios = tuple(sorted(residual_cmp_ratio_set))
    actual_seq_qlen = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    actual_seq_klen = actual_seq_qlen.clone()
    return DeepSeekV4SMLAAttentionMasks(
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        cmp_residual_kv={
            ratio: torch.full(
                (batch_size,), seq_len % ratio, dtype=torch.int32, device=device
            )
            for ratio in residual_cmp_ratios
        },
        batch_size=batch_size,
        seq_len=seq_len,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
    )


def _cp_smla_seq_lengths(seq_len: int, cp_degree: int, cp_rank: int) -> tuple[int, int]:
    """Per-rank (query, key) sequence lengths under DeepSeek-V4 Context Parallel.

    Mirrors ``CompressorAttentionCP._post_hook``: rank ``r`` has ``local_s =
    seq_len // cp_degree`` query tokens and causally attends ``(r + 1) *
    local_s`` keys. ``cp_degree == 1`` is the off-CP identity.
    """
    if cp_degree <= 1:
        return seq_len, seq_len
    if seq_len % cp_degree != 0:
        raise ValueError(
            f"DeepSeek-V4 SMLA+CP requires seq_len ({seq_len}) divisible by "
            f"cp_degree ({cp_degree})."
        )
    local_s = seq_len // cp_degree
    return local_s, (cp_rank + 1) * local_s


def _smla_cp_mask_handler(attention_masks, cp_mesh):
    """CP mask handler for DeepSeek-V4 SMLA varlen metadata.

    SMLA masks are built before CP sharding, so they are sized to the full
    ``seq_len``. Resize them to this rank's per-rank shapes and return them (so
    ``cp_shard`` skips sequence-sharding); return ``None`` for other mask types.
    """
    if not isinstance(attention_masks, DeepSeekV4SMLAAttentionMasks):
        return None
    masks = attention_masks
    cp_degree = cp_mesh.size()
    if cp_degree <= 1:
        return masks
    q_len, kv_len = _cp_smla_seq_lengths(
        masks.seq_len, cp_degree, cp_mesh.get_local_rank()
    )
    device = masks.actual_seq_qlen.device
    batch_size = masks.batch_size
    return masks._replace(
        actual_seq_qlen=torch.full(
            (batch_size,), q_len, dtype=torch.int32, device=device
        ),
        actual_seq_klen=torch.full(
            (batch_size,), kv_len, dtype=torch.int32, device=device
        ),
        cmp_residual_kv={
            ratio: torch.full(
                (batch_size,), kv_len % ratio, dtype=torch.int32, device=device
            )
            for ratio in masks.cmp_residual_kv
        },
        q_seq_len=q_len,
        kv_seq_len=kv_len,
    )


register_cp_mask_handler(_smla_cp_mask_handler)


def _smla_get_attention_masks(
    self,
    *args,
    positions: torch.Tensor | None = None,
    input_batch: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    extra_inputs: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> DeepSeekV4SMLAAttentionMasks:
    del tokenizer, kwargs
    if args:
        if len(args) == 1 and positions is None and input_batch is None:
            positions = args[0]
        else:
            if input_batch is None:
                input_batch = args[0]
            if len(args) >= 3 and extra_inputs is None:
                extra_inputs = args[2]
    if positions is None and extra_inputs is not None:
        positions = extra_inputs.get("positions")
    if positions is None:
        if input_batch is None:
            raise ValueError(
                "DeepSeek-V4 SMLA attention masks require positions or input_batch."
            )
        positions = torch.arange(
            input_batch.shape[1],
            dtype=torch.int32,
            device=input_batch.device,
        ).repeat(input_batch.shape[0], 1)
    return build_smla_attention_masks(positions, self.model_args)


def _smla_layers(self):
    if getattr(self, "use_smla", False):
        return _SMLALayerView(self)
    return range(self.n_layers + self.num_mtp_modules)


def _smla_attn_type(self) -> str:
    return "varlen" if getattr(self, "use_smla", False) else "sdpa"


class SMLAMetadataBuildContext(NamedTuple):
    actual_seq_qlen: torch.Tensor
    actual_seq_klen: torch.Tensor
    batch_size: int
    seq_len: int
    model_args: Any
    # Per-rank query/key lengths under CP (== seq_len when CP is off).
    q_seq_len: int = -1
    kv_seq_len: int = -1


class SMLAMetadataCache:
    def __init__(self, model_args: Any) -> None:
        self.model_args = model_args
        self._attention_masks: DeepSeekV4SMLAAttentionMasks | None = None
        self._op_metadata: dict[tuple[str, int], torch.Tensor | None] = {}

    def get_or_create(
        self,
        attention_masks: DeepSeekV4SMLAAttentionMasks,
        cmp_ratio: int,
        name: str,
        builder: Callable[[SMLAMetadataBuildContext, int], torch.Tensor | None],
    ) -> torch.Tensor | None:
        if self._attention_masks is not attention_masks:
            self._attention_masks = attention_masks
            self._op_metadata.clear()
        key = (name, cmp_ratio)
        if key not in self._op_metadata:
            context = SMLAMetadataBuildContext(
                actual_seq_qlen=attention_masks.actual_seq_qlen,
                actual_seq_klen=attention_masks.actual_seq_klen,
                batch_size=attention_masks.batch_size,
                seq_len=attention_masks.seq_len,
                model_args=self.model_args,
                q_seq_len=attention_masks.q_seq_len,
                kv_seq_len=attention_masks.kv_seq_len,
            )
            self._op_metadata[key] = builder(context, cmp_ratio)
        return self._op_metadata[key]


def _include_deepseek_v4_in_native_attention_mask_dispatch() -> None:
    decoder_classes = []
    try:
        decoder_module = importlib.import_module("torchtitan.models.common.decoder")
        decoder = getattr(decoder_module, "Decoder", None)
        if decoder is not None:
            decoder_classes.append(decoder)
    except ImportError:
        pass

    for module_name in ("torchtitan.train", "torchtitan.trainer"):
        try:
            titan_module = importlib.import_module(module_name)
        except ImportError:
            continue

        trainer = getattr(titan_module, "Trainer", None)
        post_dataloading_process = getattr(trainer, "post_dataloading_process", None)
        if post_dataloading_process is None:
            continue

        globals_ = post_dataloading_process.__globals__
        decoder = globals_.get("Decoder")
        if decoder is not None:
            decoder_classes.append(decoder)

    for decoder in decoder_classes:
        decoder_config = getattr(decoder, "Config", None)
        if decoder_config is None:
            continue

        configs = (
            decoder_config if isinstance(decoder_config, tuple) else (decoder_config,)
        )
        if DeepSeekV4Model.Config in configs:
            continue

        decoder.Config = (*configs, DeepSeekV4Model.Config)


def _enable_native_smla_attention_mask_building() -> None:
    # Let torchtitan's native post_dataloading_process build SMLA masks by
    # presenting DeepSeek-V4 as a varlen-attention decoder config. This keeps the
    # input metadata path in the model/converter instead of patching trainer code.
    DeepSeekV4Model.get_attention_masks = _smla_get_attention_masks
    config_cls = cast(Any, DeepSeekV4Model.Config)
    if not getattr(config_cls, "npu_smla_attn_type_dispatch", False):
        config_cls.attn_type = property(_smla_attn_type)
        config_cls.npu_smla_attn_type_dispatch = True
    if not getattr(config_cls, "npu_smla_layers_dispatch", False):
        config_cls.layers = property(_smla_layers)
        config_cls.npu_smla_layers_dispatch = True
    _include_deepseek_v4_in_native_attention_mask_dispatch()


def _none_grads(count: int) -> tuple[None, ...]:
    return (None,) * count


def _add_offset_to_valid_sparse_indices(
    sparse_indices: torch.Tensor, offset: int
) -> torch.Tensor:
    if offset == 0:
        return sparse_indices
    sentinel_mask: torch.Tensor = sparse_indices.eq(-1)
    shifted_indices: torch.Tensor = sparse_indices + offset
    return torch.where(
        sentinel_mask,
        sparse_indices,
        shifted_indices,
    )


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
    builder: Callable[[SMLAMetadataBuildContext, int], torch.Tensor | None],
) -> torch.Tensor:
    metadata = _get_op_metadata(cache, attention_masks, cmp_ratio, name, builder)
    if metadata is None:
        raise RuntimeError(
            f"Missing DeepSeek-V4 SMLA metadata {name!r}. "
            "The SMLA attention_masks must be prepared before model execution."
        )
    return metadata


def _get_op_metadata(
    cache: SMLAMetadataCache,
    attention_masks: DeepSeekV4SMLAAttentionMasks | None,
    cmp_ratio: int,
    name: str,
    builder: Callable[[SMLAMetadataBuildContext, int], torch.Tensor | None],
) -> torch.Tensor | None:
    if attention_masks is None:
        return None
    return cache.get_or_create(attention_masks, cmp_ratio, name, builder)


def _require_attention_masks(
    attention_masks: DeepSeekV4SMLAAttentionMasks | None,
) -> DeepSeekV4SMLAAttentionMasks:
    if attention_masks is None:
        raise RuntimeError(
            "DeepSeek-V4 SMLA attention_masks are required for this kernel."
        )
    return attention_masks


def _sparse_attention_metadata_kwargs(
    context: SMLAMetadataBuildContext,
    cmp_ratio: int,
) -> dict[str, Any]:
    model_args = context.model_args
    return {
        "batch_size": context.batch_size,
        "max_seqlen_q": context.q_seq_len,
        "max_seqlen_kv": context.kv_seq_len,
        "num_heads_q": model_args.n_heads,
        "num_heads_kv": 1,
        "head_dim": model_args.head_dim,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "BSND",
        "layout_kv": "BSND",
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
    cmp_kv = (
        cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()
    )  # [S, B, D] --> [B, S, 1, D]
    if cmp_ratio != 4:
        cmp_sparse_indices = None
    else:
        cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()

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
    teacher = F.normalize(cmp_softmax_l1.float().clamp_min(0), p=1, dim=-1)
    teacher = teacher.clamp_min(1e-10)
    return (
        F.kl_div(student.log(), teacher, reduction="none").sum(dim=-1).mean()
        * loss_scale
    )


class LightningIndexerConfig(NamedTuple):
    sparse_count: int
    sparse_mode: int
    cmp_ratio: int
    attention_masks: DeepSeekV4SMLAAttentionMasks
    metadata_cache: SMLAMetadataCache
    layout_query: str = "BSND"
    layout_key: str = "BSND"


class SparseAttentionSMLAInputs(NamedTuple):
    query: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: torch.Tensor | None
    cmp_sparse_indices: torch.Tensor | None
    sinks: torch.Tensor
    softmax_scale: float
    cmp_ratio: int
    ori_mask_mode: int = 4
    cmp_mask_mode: int = 3
    ori_win_left: int = 127
    ori_win_right: int = 0
    indexer_q: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    attention_masks: DeepSeekV4SMLAAttentionMasks | None = None
    metadata_cache: SMLAMetadataCache | None = None
    layer_number: int | None = None
    num_layers: int = 0


class LIAdapterSMLAInputs(NamedTuple):
    q_indexer: torch.Tensor
    k_indexer: torch.Tensor
    weights: torch.Tensor
    seqlen: int
    offset: int
    attention_masks: DeepSeekV4SMLAAttentionMasks | None = None
    metadata_cache: SMLAMetadataCache | None = None


class SparseAttentionConverterSMLAInputs(NamedTuple):
    query_states: torch.Tensor
    kv_states: torch.Tensor
    attn_sink: torch.Tensor
    kv_compress: torch.Tensor | None = None
    compress_topk_idxs: torch.Tensor | None = None
    q_indexer: torch.Tensor | None = None
    k_indexer: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    attention_masks: DeepSeekV4SMLAAttentionMasks | None = None
    metadata_cache: SMLAMetadataCache | None = None
    layer_number: int | None = None
    num_layers: int = 0


class InnerAttentionSMLAForwardInputs(NamedTuple):
    q: torch.Tensor
    kv: torch.Tensor
    kv_compress: torch.Tensor | None
    q_indexer: torch.Tensor | None
    k_indexer: torch.Tensor | None
    weights: torch.Tensor | None
    seqlen: int
    attention_masks: DeepSeekV4SMLAAttentionMasks | None = None


def _bind_named_call(input_cls, field_names: tuple[str, ...], args, kwargs):
    if len(args) > len(field_names):
        raise TypeError(f"expected at most {len(field_names)} arguments")
    values = dict(zip(field_names, args))
    duplicates = set(values).intersection(kwargs)
    if duplicates:
        name = next(iter(duplicates))
        raise TypeError(f"got multiple values for argument '{name}'")
    values.update(kwargs)
    return input_cls(**values)


class SparseFlashMLAForwardInputs(NamedTuple):
    query: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: torch.Tensor | None
    cu_seq_lens_q: torch.Tensor | None
    cu_seq_lens_ori_kv: torch.Tensor | None
    cu_seq_lens_cmp_kv: torch.Tensor | None
    cmp_sparse_indices: torch.Tensor | None
    cmp_residual_kv: torch.Tensor | None
    sinks: torch.Tensor
    softmax_scale: float
    cmp_ratio: int
    ori_mask_mode: int
    cmp_mask_mode: int
    ori_win_left: int
    ori_win_right: int
    layout_q: str
    layout_kv: str
    indexer_q: torch.Tensor | None
    indexer_k: torch.Tensor | None
    weights: torch.Tensor | None
    attention_masks: DeepSeekV4SMLAAttentionMasks
    metadata_cache: SMLAMetadataCache
    layer_number: int | None = None
    num_layers: int = 0


def _run_sparse_flash_mla_forward(inputs: SparseFlashMLAForwardInputs):
    metadata = SparseFlashMLA.sparse_attn_metadata(
        inputs.metadata_cache, inputs.attention_masks, inputs.cmp_ratio
    )
    return torch_npu.npu_sparse_attn_sharedkv(
        q=inputs.query,
        ori_kv=inputs.ori_kv,
        cmp_kv=inputs.cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=inputs.cmp_sparse_indices,
        cu_seqlens_q=inputs.cu_seq_lens_q,
        cu_seqlens_ori_kv=inputs.cu_seq_lens_ori_kv,
        cu_seqlens_cmp_kv=inputs.cu_seq_lens_cmp_kv,
        sinks=inputs.sinks,
        metadata=metadata,
        softmax_scale=inputs.softmax_scale,
        cmp_ratio=inputs.cmp_ratio,
        ori_mask_mode=inputs.ori_mask_mode,
        cmp_mask_mode=inputs.cmp_mask_mode,
        ori_win_left=inputs.ori_win_left,
        ori_win_right=inputs.ori_win_right,
        layout_q=inputs.layout_q,
        layout_kv=inputs.layout_kv,
        return_softmax_lse=True,
    )


def _save_sparse_flash_mla_context(
    ctx,
    inputs: SparseFlashMLAForwardInputs,
    result: torch.Tensor,
    softmax_lse: torch.Tensor,
) -> None:
    ctx.save_for_backward(
        result,
        softmax_lse,
        inputs.query,
        inputs.ori_kv,
        inputs.cmp_kv,
        inputs.cu_seq_lens_q,
        inputs.cu_seq_lens_ori_kv,
        inputs.cu_seq_lens_cmp_kv,
        inputs.cmp_sparse_indices,
        inputs.cmp_residual_kv,
        inputs.sinks,
        inputs.indexer_q,
        inputs.indexer_k,
        inputs.weights,
    )
    ctx.attention_masks = inputs.attention_masks
    ctx.metadata_cache = inputs.metadata_cache
    ctx.softmax_scale = inputs.softmax_scale
    ctx.cmp_ratio = inputs.cmp_ratio
    ctx.ori_mask_mode = inputs.ori_mask_mode
    ctx.cmp_mask_mode = inputs.cmp_mask_mode
    ctx.ori_win_left = inputs.ori_win_left
    ctx.ori_win_right = inputs.ori_win_right
    ctx.layout_q = inputs.layout_q
    ctx.layout_kv = inputs.layout_kv
    ctx.layer_number = inputs.layer_number
    ctx.num_layers = inputs.num_layers


class SparseFlashMLASavedTensors(NamedTuple):
    fa_out: torch.Tensor
    softmax_lse: torch.Tensor
    query: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: torch.Tensor | None
    cu_seq_lens_q: torch.Tensor | None
    cu_seq_lens_ori_kv: torch.Tensor | None
    cu_seq_lens_cmp_kv: torch.Tensor | None
    cmp_sparse_indices: torch.Tensor | None
    cmp_residual_kv: torch.Tensor | None
    sinks: torch.Tensor
    indexer_q: torch.Tensor | None
    indexer_k: torch.Tensor | None
    weights: torch.Tensor | None


class SparseFlashMLAGradOutputs(NamedTuple):
    dq: torch.Tensor
    dori_kv: torch.Tensor
    dcmp_kv: torch.Tensor | None
    dsinks: torch.Tensor
    cmp_softmax_l1: torch.Tensor


def _run_sparse_flash_mla_grad(
    ctx,
    saved: SparseFlashMLASavedTensors,
    grad_output: torch.Tensor,
) -> SparseFlashMLAGradOutputs:
    fag_metadata = SparseFlashMLA.sparse_flash_mla_grad_metadata(
        ctx.metadata_cache, ctx.attention_masks, ctx.cmp_ratio
    )
    (
        dq,
        dori_kv,
        dcmp_kv,
        dsinks,
        _,
        cmp_softmax_l1,
    ) = torch_npu.npu_sparse_flash_mla_grad(
        q=saved.query,
        ori_kv=saved.ori_kv,
        cmp_kv=saved.cmp_kv,
        d_out=grad_output,
        attn_out=saved.fa_out,
        softmax_lse=saved.softmax_lse,
        cmp_sparse_indices=saved.cmp_sparse_indices,
        cmp_residual_kv=saved.cmp_residual_kv,
        cu_seqlens_q=saved.cu_seq_lens_q,
        cu_seqlens_ori_kv=saved.cu_seq_lens_ori_kv,
        cu_seqlens_cmp_kv=saved.cu_seq_lens_cmp_kv,
        sinks=saved.sinks,
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
    if saved.cmp_kv is None:
        dcmp_kv = None
    return SparseFlashMLAGradOutputs(dq, dori_kv, dcmp_kv, dsinks, cmp_softmax_l1)


def _scale_lightning_indexer_klloss_grads(
    ctx,
    cmp_softmax_l1: torch.Tensor,
    dindexer_q: torch.Tensor,
    dindexer_k: torch.Tensor,
    dw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_scale = 1.0 / float(cmp_softmax_l1.sum(dim=-1).numel())
    loss_backward_scale = dsa_indexer_loss.LOSS_SCALE.to(
        device=dindexer_q.device, dtype=torch.float32
    )
    grad_scale = loss_backward_scale * (ctx.softmax_scale * token_scale)
    return (
        dindexer_q * grad_scale.to(dindexer_q.dtype),
        dindexer_k.squeeze(2) * grad_scale.to(dindexer_k.dtype),
        dw * grad_scale.to(dw.dtype),
    )


def _require_lightning_indexer_tensor(
    tensor: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError(f"DeepSeek-V4 SMLA {name} is required for LI grad.")
    return tensor


def _run_lightning_indexer_klloss_grad(
    ctx,
    saved: SparseFlashMLASavedTensors,
    cmp_softmax_l1: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if ctx.cmp_ratio != 4:
        return None, None, None

    lig_metadata = SparseFlashMLA.lightning_indexer_klloss_grad_metadata(
        ctx.metadata_cache, ctx.attention_masks, ctx.cmp_ratio
    )
    indexer_k = _require_lightning_indexer_tensor(saved.indexer_k, "indexer_k")
    weights = _require_lightning_indexer_tensor(saved.weights, "weights")
    (
        dindexer_q,
        dindexer_k,
        dw,
        softmax_out,
    ) = torch_npu.npu_sparse_lightning_indexer_klloss_grad(
        q=saved.indexer_q,
        k=indexer_k.unsqueeze(2),
        w=weights.to(torch.float32),
        sparse_indices=saved.cmp_sparse_indices,
        attn_softmax_l1_norm=cmp_softmax_l1,
        cmp_residual_k=saved.cmp_residual_kv,
        metadata=lig_metadata,
        layout_q=ctx.layout_q,
        layout_k=ctx.layout_kv,
        mask_mode=ctx.cmp_mask_mode,
        cmp_ratio=ctx.cmp_ratio,
    )
    if logger.isEnabledFor(logging.DEBUG):
        li_loss = _compute_li_loss(softmax_out, cmp_softmax_l1, ctx.softmax_scale)
        if ctx.layer_number is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                li_loss, ctx.layer_number, ctx.num_layers
            )
    return _scale_lightning_indexer_klloss_grads(
        ctx, cmp_softmax_l1, dindexer_q, dindexer_k, dw
    )


def _sparse_flash_mla_backward_grads(
    grad_outputs: SparseFlashMLAGradOutputs,
    dindexer_q: torch.Tensor | None,
    dindexer_k: torch.Tensor | None,
    dw: torch.Tensor | None,
) -> tuple:
    return (
        (grad_outputs.dq, grad_outputs.dori_kv, grad_outputs.dcmp_kv)
        + _none_grads(5)
        + (grad_outputs.dsinks,)
        + _none_grads(8)
        + (dindexer_q, dindexer_k, dw)
        + _none_grads(4)
    )


class SparseFlashMLA(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        if kwargs:
            raise TypeError("SparseFlashMLA.forward does not accept keyword arguments.")
        ctx, *forward_args = args
        inputs = SparseFlashMLAForwardInputs(*forward_args)
        result, softmax_lse = _run_sparse_flash_mla_forward(inputs)
        _save_sparse_flash_mla_context(ctx, inputs, result, softmax_lse)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        saved = SparseFlashMLASavedTensors(*ctx.saved_tensors)
        sparse_grads = _run_sparse_flash_mla_grad(ctx, saved, grad_output)
        dindexer_q, dindexer_k, dw = _run_lightning_indexer_klloss_grad(
            ctx, saved, sparse_grads.cmp_softmax_l1
        )
        return _sparse_flash_mla_backward_grads(
            sparse_grads, dindexer_q, dindexer_k, dw
        )

    @staticmethod
    def sparse_attn_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "sparse_attn",
            SparseFlashMLA._create_sparse_attn_metadata,
        )

    @staticmethod
    def sparse_flash_mla_grad_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "sparse_flash_mla_grad",
            SparseFlashMLA._create_sparse_flash_mla_grad_metadata,
        )

    @staticmethod
    def lightning_indexer_klloss_grad_metadata(
        cache: SMLAMetadataCache,
        attention_masks: DeepSeekV4SMLAAttentionMasks | None,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return _require_op_metadata(
            cache,
            attention_masks,
            cmp_ratio,
            "lightning_indexer_klloss_grad",
            SparseFlashMLA._create_lightning_indexer_klloss_grad_metadata,
        )

    @staticmethod
    def _create_sparse_attn_metadata(
        context: SMLAMetadataBuildContext,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return torch_npu.npu_sparse_attn_sharedkv_metadata(
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=context.actual_seq_qlen,
            **_sparse_attention_metadata_kwargs(context, cmp_ratio),
        )

    @staticmethod
    def _create_sparse_flash_mla_grad_metadata(
        context: SMLAMetadataBuildContext,
        cmp_ratio: int,
    ) -> torch.Tensor:
        return torch_npu.npu_sparse_flash_mla_grad_metadata(
            seqused_q=context.actual_seq_qlen,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            **_sparse_attention_metadata_kwargs(context, cmp_ratio),
        )

    @staticmethod
    def _create_lightning_indexer_klloss_grad_metadata(
        context: SMLAMetadataBuildContext,
        cmp_ratio: int,
    ) -> torch.Tensor | None:
        if cmp_ratio != 4:
            return None

        model_args = context.model_args
        return torch_npu.npu_sparse_lightning_indexer_klloss_grad_metadata(
            batch_size=context.batch_size,
            max_seqlen_q=context.q_seq_len,
            max_seqlen_k=context.kv_seq_len,
            num_heads_q=model_args.n_heads,
            num_heads_k=1,
            head_dim=model_args.head_dim,
            cmp_ratio=cmp_ratio,
            topk=model_args.index_topk,
            layout_q="BSND",
            layout_k="BSND",
        )


class LightningIndexer(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        if kwargs:
            raise TypeError(
                "LightningIndexer.forward does not accept keyword arguments."
            )
        _, query, key, weights, config = args
        metadata = LightningIndexer.op_metadata(
            config.metadata_cache,
            config.attention_masks,
            config.cmp_ratio,
        )
        return torch_npu.npu_lightning_indexer(
            query,
            key,
            weights,
            metadata=metadata,
            layout_query=config.layout_query,
            layout_key=config.layout_key,
            sparse_count=config.sparse_count,
            sparse_mode=config.sparse_mode,
            cmp_ratio=config.cmp_ratio,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return _none_grads(4)

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
        context: SMLAMetadataBuildContext,
        cmp_ratio: int,
    ) -> torch.Tensor | None:
        if cmp_ratio != 4:
            return None

        model_args = context.model_args
        return torch_npu.npu_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=context.actual_seq_qlen,
            actual_seq_lengths_key=context.actual_seq_klen,
            num_heads_q=model_args.index_n_heads,
            num_heads_k=1,
            head_dim=model_args.index_head_dim,
            query_quant_mode=2,
            key_quant_mode=2,
            cmp_ratio=cmp_ratio,
        )


def npu_lightning_indexer(q_indexer, k_indexer, weights, config):
    return LightningIndexer.apply(
        q_indexer,
        k_indexer,
        weights,
        config,
    )


def _format_cmp_sparse_indices(
    cmp_ratio: int, sparse_indices: torch.Tensor | None
) -> torch.Tensor | None:
    if sparse_indices is None:
        return None
    if cmp_ratio != 4:
        return sparse_indices.to(torch.int32)
    return sparse_indices.unsqueeze(2)


def _build_sparse_flash_mla_inputs(
    inputs: SparseAttentionSMLAInputs,
) -> SparseFlashMLAForwardInputs:
    attention_masks = _require_attention_masks(inputs.attention_masks)
    if inputs.metadata_cache is None:
        raise RuntimeError("DeepSeek-V4 SMLA metadata cache is not bound.")
    cmp_sparse_indices = _format_cmp_sparse_indices(
        inputs.cmp_ratio, inputs.cmp_sparse_indices
    )
    cmp_sparse_indices = (
        None if cmp_sparse_indices is None else cmp_sparse_indices.contiguous()
    )
    return SparseFlashMLAForwardInputs(
        query=inputs.query.contiguous(),
        ori_kv=inputs.ori_kv.unsqueeze(2).contiguous(),
        cmp_kv=None
        if inputs.cmp_kv is None
        else inputs.cmp_kv.unsqueeze(2).contiguous(),
        cu_seq_lens_q=None,
        cu_seq_lens_ori_kv=None,
        cu_seq_lens_cmp_kv=None,
        cmp_sparse_indices=cmp_sparse_indices,
        cmp_residual_kv=attention_masks.cmp_residual_kv.get(inputs.cmp_ratio),
        sinks=inputs.sinks,
        softmax_scale=inputs.softmax_scale,
        cmp_ratio=inputs.cmp_ratio,
        ori_mask_mode=inputs.ori_mask_mode,
        cmp_mask_mode=inputs.cmp_mask_mode,
        ori_win_left=inputs.ori_win_left,
        ori_win_right=inputs.ori_win_right,
        layout_q="BSND",
        layout_kv="BSND",
        indexer_q=inputs.indexer_q,
        indexer_k=inputs.indexer_k,
        weights=inputs.weights,
        attention_masks=attention_masks,
        metadata_cache=inputs.metadata_cache,
        layer_number=inputs.layer_number,
        num_layers=inputs.num_layers,
    )


def npu_sparse_flash_mla(inputs: SparseAttentionSMLAInputs):
    forward_inputs = _build_sparse_flash_mla_inputs(inputs)
    return SparseFlashMLA.apply(*forward_inputs).contiguous()


def sdpa_to_li_adapter_smla(self, inputs: LIAdapterSMLAInputs):
    q_indexer, k_indexer, weights, _, offset, attention_masks, metadata_cache = inputs
    attention_masks = _require_attention_masks(attention_masks)
    if metadata_cache is None:
        raise RuntimeError("DeepSeek-V4 SMLA metadata cache is not bound.")
    q_indexer = q_indexer.to(torch.bfloat16)
    k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
    weights = weights.to(torch.bfloat16)

    compress_topk_idxs, index_score = npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        LightningIndexerConfig(
            sparse_count=self.index_topk,
            sparse_mode=3,
            cmp_ratio=self.ratio,
            attention_masks=attention_masks,
            metadata_cache=metadata_cache,
        ),
    )

    compress_topk_idxs = compress_topk_idxs.squeeze(2)
    compress_topk_idxs = _add_offset_to_valid_sparse_indices(compress_topk_idxs, offset)

    return compress_topk_idxs, index_score


def _run_sparse_attention_converter_smla(
    sparse_attn,
    inputs: SparseAttentionConverterSMLAInputs,
):
    (
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_idxs,
        q_indexer,
        k_indexer,
        weights,
        attention_masks,
        metadata_cache,
        layer_number,
        num_layers,
    ) = inputs
    if compress_topk_idxs is not None and compress_topk_idxs.dtype != torch.int32:
        compress_topk_idxs = compress_topk_idxs.to(torch.int32)

    return npu_sparse_flash_mla(
        SparseAttentionSMLAInputs(
            query=query_states,
            ori_kv=kv_states,
            cmp_kv=kv_compress,
            cmp_sparse_indices=compress_topk_idxs,
            sinks=attn_sink.float(),
            softmax_scale=sparse_attn.softmax_scale,
            cmp_ratio=sparse_attn.compress_ratio,
            indexer_q=q_indexer,
            indexer_k=k_indexer,
            weights=weights,
            attention_masks=attention_masks,
            metadata_cache=metadata_cache,
            layer_number=layer_number,
            num_layers=num_layers,
        )
    )


def _run_inner_attention_li_compute_smla(
    inner_attn,
    inputs: InnerAttentionSMLAForwardInputs,
):
    offset = 0 if inner_attn.use_smla else inputs.kv.size(1)
    has_li = (
        inner_attn.compress_ratio > 1
        and hasattr(inner_attn, "li_compute")
        and inputs.q_indexer is not None
    )
    if not has_li:
        return None, None, None, 0

    compress_topk_idxs, index_score = inner_attn.li_compute(
        inputs.q_indexer,
        inputs.k_indexer,
        inputs.weights,
        inputs.seqlen,
        offset,
        inputs.attention_masks,
    )
    li_loss = getattr(inner_attn, "li_loss", None)
    layer_number = getattr(li_loss, "layer_id", None)
    num_layers = getattr(li_loss, "n_layers", 0) if layer_number is not None else 0
    return compress_topk_idxs, index_score, layer_number, num_layers


def _run_inner_attention_smla(
    inner_attn,
    inputs: InnerAttentionSMLAForwardInputs,
    metadata_cache: SMLAMetadataCache,
):
    (
        compress_topk_idxs,
        index_score,
        layer_number,
        num_layers,
    ) = _run_inner_attention_li_compute_smla(inner_attn, inputs)
    output = _run_sparse_attention_converter_smla(
        inner_attn.sparse_attn,
        SparseAttentionConverterSMLAInputs(
            query_states=inputs.q,
            kv_states=inputs.kv,
            attn_sink=inner_attn.attn_sink,
            kv_compress=inputs.kv_compress,
            compress_topk_idxs=compress_topk_idxs,
            q_indexer=inputs.q_indexer,
            k_indexer=inputs.k_indexer,
            weights=inputs.weights,
            attention_masks=inputs.attention_masks,
            metadata_cache=metadata_cache,
            layer_number=layer_number,
            num_layers=num_layers,
        ),
    )
    return output, compress_topk_idxs, index_score


class NpuLiComputeSMLA(LiCompute):
    def forward(self, *args, **kwargs):
        inputs = _bind_named_call(
            LIAdapterSMLAInputs,
            (
                "q_indexer",
                "k_indexer",
                "weights",
                "seqlen",
                "offset",
                "attention_masks",
            ),
            args,
            kwargs,
        )
        metadata_cache = cast(SMLAMetadataCache, self._smla_metadata_cache)
        return sdpa_to_li_adapter_smla(
            self,
            LIAdapterSMLAInputs(
                q_indexer=inputs.q_indexer,
                k_indexer=inputs.k_indexer,
                weights=inputs.weights,
                seqlen=inputs.seqlen,
                offset=inputs.offset,
                attention_masks=inputs.attention_masks,
                metadata_cache=metadata_cache,
            ),
        )


class NpuInnerAttentionSMLA(InnerAttention):
    def forward(self, *args, **kwargs):
        inputs = _bind_named_call(
            InnerAttentionSMLAForwardInputs,
            (
                "q",
                "kv",
                "kv_compress",
                "q_indexer",
                "k_indexer",
                "weights",
                "seqlen",
                "attention_masks",
            ),
            args,
            kwargs,
        )
        metadata_cache = cast(SMLAMetadataCache, self._smla_metadata_cache)
        return _run_inner_attention_smla(self, inputs, metadata_cache)


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
        if compress_topk_idxs is not None:
            if compress_topk_idxs.dtype != torch.int32:
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

        compress_topk_idxs = compress_topk_idxs.squeeze(2)
        index_score = index_score.squeeze(2)
        compress_topk_idxs = _add_offset_to_valid_sparse_indices(
            compress_topk_idxs, offset
        )

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
        ctx.save_for_backward(
            query, key, query_index, key_index, weights, sparse_indices
        )
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
            None,  # actual_seq_qlen
            None,  # actual_seq_klen
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
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss[0], ctx.layer_number, ctx.num_layers
            )
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
        k,
        q_indexer,
        k_indexer,
        weights,
        sparse_indices,
        indexer_score,
        offset,
    ):
        if sparse_indices.dtype != torch.int32:
            sparse_indices = sparse_indices.to(torch.int32)

        return npu_sparse_lightning_indexer_grad_kl_loss(
            q,
            k.unsqueeze(2),
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
        importlib.import_module("custom_ops")

        _enable_native_smla_attention_mask_building()
        metadata_cache = SMLAMetadataCache(model.model_args)

        modules = list(model.named_modules())
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
                _sas_op = build_op(
                    "sparse_attn_sharedkv", ["sparse_attn_sharedkv/binding.cpp"]
                )
                replace_module_with_name(model, name, NpuSparseAttention(module))
                logger.info("[NpuSMLAConverter] [SparseAttention forward] Applied.")

            if isinstance(module, LiCompute):
                _li_op = build_op(
                    "lightning_indexer", ["lightning_indexer/binding.cpp"]
                )
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
