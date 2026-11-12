# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
from collections.abc import Callable
from functools import partial
from typing import cast

import torch.nn as nn
from torchtitan.config.transform import ModelConfigConverter, validate_converter_compatibility
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, RoPE, SqrtSoftplus
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    fused_grouped_gate_up_param_init,
    make_ffn_config,
    make_routed_experts_config,
)
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router
from torchtitan.models.deepseek_v4.mtp import MTPBlock

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .attention import Attention, CompressedSparseInnerAttention, UnitScaleRMSNorm
from .compressor import Compressor, Indexer, LightningIndexer
from .mhc import HcHead, HcPost, HcPre
from .model import (
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
)
from .mtp import MTPLoss
from .state_dict_adapter import DeepSeekV4StateDictAdapter

__all__ = [
    "DeepSeekV4Model",
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


def _make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    norm_eps: float,
    coff: int,
    rope: RoPE.Config,
) -> Compressor.Config:
    return Compressor.Config(
        # RoPE receives the full-width kv and owns the [nope | rope] split.
        rope=dataclasses.replace(rope, split=head_dim - rope_head_dim),
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        carry_exchange=CarryExchange.Config(ratio=compress_ratio),
        param_init=_APE_INIT,
        wkv=Linear.Config(
            in_features=dim,
            out_features=coff * head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wgate=Linear.Config(
            in_features=dim,
            out_features=coff * head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=head_dim,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    num_index_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    compress_ratio: int,
    norm_eps: float,
    rope: RoPE.Config,
) -> Indexer.Config:
    coff = 2  # overlap always True for indexer
    return Indexer.Config(
        rope=dataclasses.replace(rope, split=index_head_dim - rope_head_dim),
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=num_index_heads * index_head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        weights_proj=Linear.Config(
            in_features=dim,
            out_features=num_index_heads,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        compressor=_make_compressor_config(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            coff=coff,
            rope=rope,
        ),
    )


def _make_v4_attn_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope: RoPE.Config,
) -> Attention.Config:
    hd = head_dim
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    compressor_cfg = None
    indexer_cfg = None
    lightning_indexer_cfg = None

    if compress_ratio == 4:
        coff = 2  # 1 + overlap (overlap=True when compress_ratio==4)
        compressor_cfg = _make_compressor_config(
            dim=dim,
            head_dim=hd,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            coff=coff,
            rope=rope,
        )
        indexer_cfg = _make_indexer_config(
            dim=dim,
            num_index_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            rope=rope,
        )
        lightning_indexer_cfg = LightningIndexer.Config(index_topk=index_topk)
    elif compress_ratio > 1:
        coff = 1  # no overlap
        compressor_cfg = _make_compressor_config(
            dim=dim,
            head_dim=hd,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            coff=coff,
            rope=rope,
        )
    inner_attention_cfg = CompressedSparseInnerAttention.Config(
        window_size=window_size,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        index_topk=index_topk,
    )
    return Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        n_groups=n_groups,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        lightning_indexer=lightning_indexer_cfg,
        inner_attention=inner_attention_cfg,
        # Attention sites are head_dim wide; the base rope may be reused by
        # the indexer (index_head_dim wide), so split is pinned per site.
        rope=dataclasses.replace(rope, split=head_dim - rope_head_dim),
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        q_head_norm=UnitScaleRMSNorm.Config(
            normalized_shape=hd,
            eps=norm_eps,
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=BatchedLinear.Config(
            n_heads=n_groups,
            in_features=per_group_in,
            out_features=o_lora_rank,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=per_group_out,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        # ``attn_sink`` is a bare ``[n_heads]`` fp32 parameter on the
        # attention module, initialized via the config-side param_init.
        param_init=_SINK_INIT,
        compressor=compressor_cfg,
        indexer=indexer_cfg,
    )


# SwiGLU gate/up clamp shared by all DSV4 flavors; 10.0 is the transformers
# config default (``swiglu_limit: float = 10.0``) and the inference repo's
# ``config.json`` value. Zero disables the clamp.
_SWIGLU_LIMIT = 10.0


def _make_v4_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
):
    return MoE.Config(
        num_experts=num_experts,
        router=DeepSeekV4Router.Config(
            num_experts=num_experts,
            gate=RouterGateLinear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                param_init=_depth_init(layer_id),
            ),
            top_k=top_k,
            score_func=SqrtSoftplus.Config(),  # pyrefly: ignore [bad-argument-type]
            route_scale=route_scale,
            route_norm=route_norm,
            layer_id=layer_id,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
        ),
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=moe_inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=moe_inter_dim * num_shared_experts,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
                swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
            )
            if num_shared_experts > 0
            else None
        ),
        load_balance_coeff=load_balance_coeff,
    )


def _build_v4_layers(
    *,
    n_layers: int,
    dim: int,
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
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    rope: RoPE.Config,
    rope_compress: RoPE.Config,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> list[DeepSeekV4TransformerBlock.Config]:
    if len(compress_ratios) != n_layers:
        raise ValueError(f"compress_ratios ({len(compress_ratios)} entries) must cover n_layers ({n_layers}).")
    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id]

        attn_cfg = _make_v4_attn_config(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            compress_ratio=cr,
            window_size=window_size,
            norm_eps=norm_eps,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope=rope_compress if cr > 1 else rope,
        )

        moe_cfg = _make_v4_moe_config(
            layer_id=layer_id,
            dim=dim,
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
        )

        layers.append(
            DeepSeekV4TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                moe=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )
    return layers


def _build_mtp_layers(
    inner_cfg: DeepSeekV4TransformerBlock.Config,
    *,
    dim: int,
    n_main_layers: int,
    num_mtp_layers: int,
    rope: RoPE.Config,
) -> list[MTPBlock.Config]:
    """Build independent, non-compressed DSV4 MTP depths."""
    mtp_layers = []
    for depth in range(num_mtp_layers):
        layer_id = n_main_layers + depth

        attention = cast("Attention.Config", copy.deepcopy(inner_cfg.attention))
        attention.compress_ratio = 1
        attention.compressor = None
        attention.indexer = None
        # MTP attention is dense; disable the prebuilt selector branch too.
        attention.lightning_indexer = None
        # Swap the plain rope in (the main layer may carry the compress-theta
        # variant) while keeping the per-site split pin: rope receives the
        # full-width head_dim tensor and rotates only its trailing channels.
        attention.rope = copy.deepcopy(dataclasses.replace(rope, split=attention.head_dim - attention.rope_head_dim))
        cast("CompressedSparseInnerAttention.Config", attention.inner_attention).compress_ratio = 1

        moe = copy.deepcopy(inner_cfg.moe)
        moe.router.gate.param_init = _depth_init(layer_id)
        moe.router.layer_id = layer_id  # pyrefly: ignore [missing-attribute]
        moe.router.n_hash_layers = 0  # pyrefly: ignore [missing-attribute]
        moe.routed_experts.w13.param_init = fused_grouped_gate_up_param_init(_depth_experts_init(layer_id))
        moe.routed_experts.w2.param_init = {"weight": _depth_experts_init(layer_id)["w2_EDF"]}
        if moe.shared_experts is not None:
            depth_init = _depth_init(layer_id)
            moe.shared_experts.w2.param_init = depth_init
            moe.shared_experts.w13.param_init = fused_gate_up_param_init(_LINEAR_INIT, depth_init)

        hc_attn_pre = copy.deepcopy(inner_cfg.hc_attn_pre)
        projection = _build_mtp_projection(dim)
        # Upstream builds the NPU configs via Config.build().
        mtp_layers.append(
            MTPBlock.Config(
                attention=attention,  # pyrefly: ignore [bad-argument-type]
                attention_norm=copy.deepcopy(inner_cfg.attention_norm),
                ffn_norm=copy.deepcopy(inner_cfg.ffn_norm),
                moe=moe,  # pyrefly: ignore [bad-argument-type]
                enorm=copy.deepcopy(inner_cfg.attention_norm),
                hnorm=copy.deepcopy(inner_cfg.attention_norm),
                e_proj=copy.deepcopy(projection),
                h_proj=copy.deepcopy(projection),
                mtp_norm=copy.deepcopy(inner_cfg.attention_norm),
                hc_attn_pre=hc_attn_pre,  # pyrefly: ignore [bad-argument-type]
                hc_ffn_pre=copy.deepcopy(inner_cfg.hc_ffn_pre),  # pyrefly: ignore [bad-argument-type]
                hc_post=copy.deepcopy(inner_cfg.hc_post),  # pyrefly: ignore [bad-argument-type]
                hc_head=HcHead.Config(  # pyrefly: ignore [bad-argument-type]
                    hc_mult=hc_attn_pre.hc_mult,
                    dim=dim,
                    norm_eps=hc_attn_pre.norm_eps,
                    eps=hc_attn_pre.eps,
                    param_init=_HC_PARAM_INIT,
                ),
            )
        )
    return mtp_layers


def _build_mtp_projection(dim: int) -> Linear.Config:
    return Linear.Config(
        in_features=dim,
        out_features=dim,
        bias=False,
        param_init=_LINEAR_INIT,
    )


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

    if num_mtp_layers < 0:
        raise ValueError(f"num_mtp_layers must be non-negative, got {num_mtp_layers}.")

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
        )
    return DeepSeekV4Model.Config(
        max_context_length=max_seq_len,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
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
        ),
        mtp_layers=mtp_layers,
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
    return config


DeepSeekV4Model.state_dict_adapter_cls = DeepSeekV4StateDictAdapter
