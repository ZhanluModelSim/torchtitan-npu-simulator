# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy NPU geometry layout for older experiments and checkpoints.

Maintained V4 recipes keep upstream attention and use compressor/inner-attention
overrides directly. These converters remain only for explicit legacy callers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torchtitan.config import derive
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v4.attention import DeepSeekV4Attention

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .attention import NPUAttention, NPUCompressedSparseInnerAttention, UnitScaleRMSNorm
from .compressor import LightningIndexer, NPUCompressor, NPUIndexer
from .model import DeepSeekV4Model
from .sharding import set_deepseek_v4_attention_sharding
from .state_dict_adapter import DeepSeekV4StateDictAdapter

if TYPE_CHECKING:
    from torchtitan.models.deepseek_v4.compressor import Compressor, Indexer


class NPUBackendDeepSeekV4Model(DeepSeekV4Model):
    """Upstream V4 model/block math with the NPU attention checkpoint layout."""

    state_dict_adapter_cls = DeepSeekV4StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        pass


def _split_rope(config, *, head_dim: int, rope_head_dim: int):
    # The package's split-aware RoPE config is installed before model loading.
    return replace(config, split=head_dim - rope_head_dim)


def _to_npu_compressor(config: Compressor.Config) -> NPUCompressor.Config:
    return derive(
        config,
        NPUCompressor.Config,
        rope=_split_rope(config.rope, head_dim=config.head_dim, rope_head_dim=config.rope_head_dim),
        carry_exchange=CarryExchange.Config(ratio=config.compress_ratio),
    )


def _to_npu_indexer(config: Indexer.Config) -> NPUIndexer.Config:
    return derive(
        config,
        NPUIndexer.Config,
        rope=_split_rope(config.rope, head_dim=config.index_head_dim, rope_head_dim=config.rope_head_dim),
        compressor=_to_npu_compressor(config.compressor),
    )


def to_npu_attention_config(
    config: DeepSeekV4Attention.Config,
    *,
    enable_sp: bool = False,
) -> NPUAttention.Config:
    """Translate native attention storage/metadata without mutating its config.

    The backend owns document geometry, carry exchange and the selector node
    consumed by the existing NPU component overrides. Its sink is a bare
    ``[heads]`` parameter; the upstream reference keeps ``[heads, 1]`` in a
    Linear module. Grouped output projection weights have the same flattened
    storage in both implementations.
    """
    if isinstance(config, NPUAttention.Config):
        return deepcopy(config)
    if not isinstance(config, DeepSeekV4Attention.Config):
        raise TypeError(f"Expected a native DeepSeekV4Attention.Config, got {type(config).__name__}")
    if type(config.wo_a) is not Linear.Config or type(config.attn_sink) is not Linear.Config:
        raise ValueError("Select the NPU attention backend before converting output projection or sink linears")
    if config.wo_a.bias or config.wo_a.num_linears != 1 or config.attn_sink.bias:
        raise ValueError("The NPU attention backend requires the native bias-free grouped projection and sink layout")
    if config.inner_attention.aux_loss is not None:
        raise ValueError(
            "The NPU attention backend owns its indexer objective; construct the native config "
            "with indexer_loss_coeff=0 before selecting the backend."
        )
    config = deepcopy(config)
    # Upstream denotes window-only attention by ratio 0; NPU geometry uses 1.
    compress_ratio = max(1, config.compress_ratio)
    compressor = config.compressor if config.compress_ratio == 4 else config.compressor_128
    if config.compress_ratio > 1 and compressor is None:
        raise ValueError(f"Compression ratio {config.compress_ratio} requires its compressor config")
    if config.wo_a.out_features % config.n_groups:
        raise ValueError("The grouped output projection width must be divisible by n_groups")
    sink_init = (config.attn_sink.param_init or {}).get("weight")
    param_init = dict(config.param_init or {})
    if sink_init is not None:
        param_init["attn_sink"] = sink_init
    result = NPUAttention.Config(
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        rope_head_dim=config.rope_head_dim,
        q_lora_rank=config.q_lora_rank,
        n_groups=config.n_groups,
        compress_ratio=compress_ratio,
        norm_eps=config.norm_eps,
        rope=_split_rope(config.rope, head_dim=config.head_dim, rope_head_dim=config.rope_head_dim),
        wq_a=config.wq_a,
        q_norm=config.q_norm,
        q_head_norm=UnitScaleRMSNorm.Config(normalized_shape=config.head_dim, eps=config.norm_eps),
        wq_b=config.wq_b,
        wkv=config.wkv,
        kv_norm=config.kv_norm,
        wo_a=BatchedLinear.Config(
            n_heads=config.n_groups,
            in_features=config.wo_a.in_features,
            out_features=config.wo_a.out_features // config.n_groups,
            param_init=config.wo_a.param_init,
        ),
        wo_b=config.wo_b,
        compressor=_to_npu_compressor(compressor) if compressor is not None else None,
        indexer=_to_npu_indexer(config.indexer) if config.indexer is not None else None,
        lightning_indexer=(
            LightningIndexer.Config(index_topk=config.inner_attention.index_topk)
            if config.indexer is not None
            else None
        ),
        inner_attention=NPUCompressedSparseInnerAttention.Config(
            window_size=config.inner_attention.window_size,
            compress_ratio=compress_ratio,
            softmax_scale=config.inner_attention.softmax_scale,
            index_topk=config.inner_attention.index_topk,
        ),
        param_init=param_init or None,
    )
    set_deepseek_v4_attention_sharding(result, enable_sp=enable_sp)
    return result


def with_npu_attention_backend(
    config: DeepSeekV4Model.Config,
    *,
    enable_sp: bool = False,
) -> NPUBackendDeepSeekV4Model.Config:
    """Select the NPU attention and state layout before applying child overrides."""
    config = deepcopy(config)
    for layer in (*config.layers, *(config.mtp_layers or ())):
        attention = to_npu_attention_config(layer.attention, enable_sp=enable_sp)
        layer.attention = attention  # pyrefly: ignore [bad-assignment]
    # The backend state adapter owns a concrete list of configured MTP stages.
    return derive(config, NPUBackendDeepSeekV4Model.Config, mtp_layers=config.mtp_layers or [])
