# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import torch.nn as nn
from torchtitan.config.transform import ModelConfigConverter, validate_converter_compatibility
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v4.flavors import (
    _build_mtp_layers as _upstream_build_mtp_layers,
)
from torchtitan.models.deepseek_v4.flavors import (
    _build_v4_layers as _upstream_build_v4_layers,
)
from torchtitan.models.deepseek_v4.sharding import hc_head_sharding_config

from torchtitan_npu.models.common.activation import LimitedSwiGLU

from .attention import DeepSeekV4Attention as Attention
from .attention import UnitScaleRMSNorm
from .mhc import HcHead, HcPost, HcPre
from .model import (
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
)
from .mtp import (
    DeepSeekV4MTPDecoder,
    DeepSeekV4MTPTransformerBlock,
    MTPLoss,
)

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.deepseek_v4.mtp import MTPBlock

__all__ = [
    "Attention",
    "DeepSeekV4MTPDecoder",
    "DeepSeekV4MTPTransformerBlock",
    "DeepSeekV4Model",
    "DeepSeekV4TransformerBlock",
    "HcPost",
    "HcPre",
    "MTPLoss",
    "deepseek_v4_configs",
    "model_registry",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_APE_INIT = {"ape": partial(nn.init.trunc_normal_, std=0.02)}
_SINK_INIT = {"attn_sink": partial(nn.init.trunc_normal_, std=0.02)}
_HC_PARAM_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


# SwiGLU gate/up clamp shared by all DSV4 flavors; 10.0 is the transformers
# config default (``swiglu_limit: float = 10.0``) and the inference repo's
# ``config.json`` value. Zero disables the clamp.
_SWIGLU_LIMIT = 10.0


def _build_v4_layers(**kwargs):
    """Construct upstream V4 blocks with the NPU flavor dimensions."""
    kwargs.pop("moe_comm_backend", None)
    kwargs.pop("non_blocking_capacity_factor", None)
    layers = _upstream_build_v4_layers(**kwargs, enable_sp=False, enable_ep=False)
    # The reference recipes have no indexer distillation objective.
    from torchtitan.config import derive

    for layer in layers:
        layer.attention = derive(
            layer.attention,
            Attention.Config,
            q_head_norm=UnitScaleRMSNorm.Config(
                normalized_shape=layer.attention.head_dim,
                eps=layer.attention.norm_eps,
            ),
        )
        layer.attention.inner_attention.aux_loss = None
        feed_forward = cast("MoE.Config", layer.feed_forward)
        experts = feed_forward.routed_experts
        experts.activation_fn = LimitedSwiGLU.Config(swiglu_limit=_SWIGLU_LIMIT)
        experts.token_dispatcher = dataclasses.replace(experts.token_dispatcher, absorb_router_scores=True)
        if feed_forward.shared_experts is not None:
            feed_forward.shared_experts.activation_fn = LimitedSwiGLU.Config(swiglu_limit=_SWIGLU_LIMIT)
    return layers


def _build_mtp_layers(
    inner_cfg,
    *,
    dim,
    n_main_layers,
    num_mtp_layers,
    rope,
    hc_mult=4,
    norm_eps=1e-6,
    hc_eps=1e-6,
) -> "list[MTPBlock.Config]":
    layers = _upstream_build_mtp_layers(
        inner_cfg,
        dim=dim,
        n_main_layers=n_main_layers,
        num_mtp_layers=num_mtp_layers,
        rope=rope,
        hc_mult=hc_mult,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )
    from torchtitan.config import derive

    return [derive(layer, DeepSeekV4MTPTransformerBlock.Config) for layer in layers]


def _make_v4_config(
    *,
    dim: int,
    n_layers: int,
    vocab_size: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratios: tuple[int, ...],
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    max_seq_len: int,
    compress_rope_theta: float,
    original_seq_len: int,
    rope_theta: float = 10000.0,
    rope_factor: float = 4.0,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    num_mtp_layers: int = 0,
) -> DeepSeekV4Model.Config:
    """Build a DSV4 model config from the flavor constants."""

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=max_seq_len,
        theta=rope_theta,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=max_seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=rope_factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    # The per-microbatch metadata is one entry per geometry, and the MTP depths'
    # dense (ratio 1) layers share the ratio-1 entry the main layers read.

    layers = _build_v4_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratios=compress_ratios,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=rope,
        rope_compress=rope_compress,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )

    mtp_layers = []
    if num_mtp_layers > 0:
        mtp_layers = _build_mtp_layers(
            layers[-1],
            dim=dim,
            n_main_layers=n_layers,
            num_mtp_layers=num_mtp_layers,
            rope=rope,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
    return DeepSeekV4Model.Config(
        max_context_length=max_seq_len,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=False),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=norm_eps,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=False),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
        hc_head=HcHead.Config(
            hc_mult=hc_mult,
            dim=dim,
            norm_eps=norm_eps,
            eps=hc_eps,
            param_init=_HC_PARAM_INIT,
            sharding_config=hc_head_sharding_config(enable_sp=False),
        ),
        mtp_layers=mtp_layers,
        n_mtp_layers=num_mtp_layers,
    )


def _debugmodel(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    num_mtp_layers: int = 0,
    *,
    seq_len: int = 16384,
) -> DeepSeekV4Model.Config:
    return _make_v4_config(
        dim=256,
        n_layers=4,
        vocab_size=129280,
        n_heads=16,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        n_groups=2,
        compress_ratios=(1, 1, 4, 128),
        window_size=128,
        norm_eps=1e-6,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=512,
        moe_inter_dim=256,
        num_experts=4,
        num_shared_experts=1,
        top_k=3,
        n_hash_layers=2,
        route_norm=False,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        hc_mult=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        max_seq_len=seq_len,
        compress_rope_theta=40000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=4.0,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_layers=num_mtp_layers,
    )


def _deepseek_v4_flash(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    seq_len: int = 4096,
    num_experts: int = 256,
    num_mtp_layers: int = 0,
) -> DeepSeekV4Model.Config:
    return _make_v4_config(
        dim=4096,
        n_layers=43,
        vocab_size=129280,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        n_groups=8,
        compress_ratios=(1, 1) + (4, 128) * 20 + (4,),
        window_size=128,
        norm_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        moe_inter_dim=2048,
        num_experts=num_experts,
        num_shared_experts=1,
        top_k=6,
        n_hash_layers=3,
        route_norm=True,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        hc_mult=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        max_seq_len=seq_len,
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16.0,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_layers=num_mtp_layers,
    )


def _deepseek_v4_pro(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    seq_len: int = 4096,
    num_experts: int = 384,
    num_mtp_layers: int = 0,
) -> DeepSeekV4Model.Config:
    return _make_v4_config(
        dim=7168,
        n_layers=61,
        vocab_size=129280,
        n_heads=128,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1536,
        o_lora_rank=1024,
        n_groups=16,
        compress_ratios=(128,) + (128, 4) * 30,
        window_size=128,
        norm_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        moe_inter_dim=3072,
        num_experts=num_experts,
        num_shared_experts=1,
        top_k=6,
        n_hash_layers=3,
        route_norm=True,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        hc_mult=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        max_seq_len=seq_len,
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16.0,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_layers=num_mtp_layers,
    )


deepseek_v4_configs = {
    "debugmodel": (_debugmodel, 16384),
    "deepseek_v4_flash": (_deepseek_v4_flash, 4096),
    "deepseek_v4_flash_43layers_16experts": (
        partial(_deepseek_v4_flash, num_experts=16),
        4096,
    ),
    "deepseek_v4_pro": (_deepseek_v4_pro, 4096),
    "deepseek_v4_pro_61layers_32experts": (
        partial(_deepseek_v4_pro, num_experts=32),
        4096,
    ),
    "deepseek_v4_pro_61layers_12experts": (
        partial(_deepseek_v4_pro, num_experts=12),
        4096,
    ),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism=None,
) -> DeepSeekV4Model.Config:
    if flavor not in deepseek_v4_configs:
        raise ValueError(f"Unknown deepseek_v4 flavor: {flavor}. Available: {list(deepseek_v4_configs.keys())}")
    get_config, max_context_len = deepseek_v4_configs[flavor]
    context_len = max_context_len if seq_len is None else seq_len
    if not 0 < context_len <= max_context_len:
        raise ValueError(f"Requested seq_len {context_len} must be between 1 and {max_context_len} for flavor {flavor}")
    config = get_config(
        seq_len=context_len,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_layers=num_mtp_layers,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    # Configure layouts before construction and disable the upstream in-place WGRAD path.
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common.linear import Linear

    from .sharding import set_upstream_deepseek_v4_sharding_config

    parallelism = parallelism or ParallelismConfig()
    set_upstream_deepseek_v4_sharding_config(
        config,
        enable_sp=parallelism.enable_sequence_parallel,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    for _, linear, _, _ in config.traverse(Linear.Config):
        linear.inplace_wgrad_accum = False
    return config
