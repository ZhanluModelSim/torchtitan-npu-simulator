# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 text decoder configurations for TorchTitan v0.3.0."""

from dataclasses import fields
from functools import partial
from typing import Any, cast

from torch import nn
from torchtitan.models.common.config_utils import make_ffn_config, make_routed_experts_config
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm

from .attention import KimiDeltaAttention, KimiGatedMLA, RMSNormGated, ShortConvolution
from .feed_forward import KimiGroupedExperts, KimiLatentMoE, KimiMLP
from .model import KimiAttentionResidual, KimiK3Model, KimiK3TransformerBlock, KimiResidualScoreNorm

_WEIGHT_INIT = partial(nn.init.normal_, mean=0.0, std=0.02)
_LINEAR_INIT = {"weight": _WEIGHT_INIT}


def _linear_config(in_features, out_features):
    return Linear.Config(in_features=in_features, out_features=out_features, bias=False, param_init=_LINEAR_INIT)


def _attn_res_config(dim, eps):
    return KimiAttentionResidual.Config(
        norm=KimiResidualScoreNorm.Config(dim=dim, eps=eps), proj=_linear_config(dim, 1)
    )


def _short_conv_config(dim, kernel_size):
    return ShortConvolution.Config(
        conv=Conv1d.Config(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=kernel_size - 1,
            bias=False,
            param_init=_LINEAR_INIT,
        )
    )


def _mlp_config(dim, hidden_dim, beta, linear_beta):
    native = make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w1_param_init=_LINEAR_INIT,
        w2w3_param_init=_LINEAR_INIT,
    )
    return KimiMLP.Config(
        **{f.name: getattr(native, f.name) for f in fields(native)}, beta=beta, linear_beta=linear_beta
    )


def _moe_config(
    *,
    hidden_size,
    num_experts,
    top_k,
    moe_intermediate_size,
    num_shared_experts,
    routed_expert_hidden_size,
    latent_moe_use_norm,
    score_func,
    num_expert_groups,
    topk_group,
    routed_scaling_factor,
    renormalize,
    situ_beta,
    situ_linear_beta,
    norm_eps,
):
    latent_dim = routed_expert_hidden_size or hidden_size
    routed = make_routed_experts_config(
        dim=latent_dim,
        hidden_dim=moe_intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        param_init=dict.fromkeys(("w1_EFD", "w2_EDF", "w3_EFD"), _WEIGHT_INIT),
        comm_backend="standard",
    )
    if hasattr(routed.token_dispatcher, "absorb_router_scores"):
        cast("Any", routed.token_dispatcher).absorb_router_scores = False
    routed.inner_experts = KimiGroupedExperts.Config(
        **{f.name: getattr(routed.inner_experts, f.name) for f in fields(routed.inner_experts)},
        beta=situ_beta,
        linear_beta=situ_linear_beta,
    )
    return KimiLatentMoE.Config(
        num_experts=num_experts,
        routed_experts=routed,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=_linear_config(hidden_size, num_experts),
            top_k=top_k,
            score_func=score_func,
            num_expert_groups=num_expert_groups if num_expert_groups > 1 else None,
            num_limited_groups=topk_group if num_expert_groups > 1 else None,
            route_norm=renormalize and top_k > 1,
            route_scale=routed_scaling_factor,
        ),
        shared_experts=_mlp_config(hidden_size, moe_intermediate_size * num_shared_experts, situ_beta, situ_linear_beta)
        if num_shared_experts
        else None,
        down_proj=_linear_config(hidden_size, latent_dim) if routed_expert_hidden_size is not None else None,
        up_proj=_linear_config(latent_dim, hidden_size) if routed_expert_hidden_size is not None else None,
        norm=RMSNorm.Config(normalized_shape=latent_dim, eps=norm_eps)
        if routed_expert_hidden_size is not None and latent_moe_use_norm
        else None,
    )


# KDA layer indices from the official config, converted from 1-based to 0-based.
_FULL_KDA_LAYERS = [
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    28,
    29,
    30,
    32,
    33,
    34,
    36,
    37,
    38,
    40,
    41,
    42,
    44,
    45,
    46,
    48,
    49,
    50,
    52,
    53,
    54,
    56,
    57,
    58,
    60,
    61,
    62,
    64,
    65,
    66,
    68,
    69,
    70,
    72,
    73,
    74,
    76,
    77,
    78,
    80,
    81,
    82,
    84,
    85,
    86,
    88,
    89,
    90,
]


def _build_kimi_k3_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kda_head_dim: int,
    dense_hidden_dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: str = "sigmoid",
    num_expert_groups: int = 1,
    topk_group: int = 1,
    routed_expert_hidden_size: int | None = None,
    latent_moe_use_norm: bool = True,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    kda_layers: list[int] | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    norm_eps: float = 1e-5,
    conv_kernel_size: int = 4,
    gate_lower_bound: float | None = -5.0,
    use_full_rank_gate: bool = True,
    attn_res_block_size: int | None = 12,
) -> list[KimiK3TransformerBlock.Config]:
    """Build per-layer TransformerBlock configs with hybrid KDA/MLA attention."""
    if kda_layers is None:
        # Official pattern uses 1-based layer numbers: three KDA, then one MLA.
        kda_layers = [i for i in range(n_layers) if (i + 1) % 4 != 0]

    layers = []
    for layer_id in range(n_layers):
        # Attention config: KDA or Gated MLA
        if layer_id in kda_layers:
            projection_dim = n_heads * kda_head_dim
            attn_cfg = KimiDeltaAttention.Config(
                q_proj=_linear_config(dim, projection_dim),
                k_proj=_linear_config(dim, projection_dim),
                v_proj=_linear_config(dim, projection_dim),
                q_conv1d=_short_conv_config(projection_dim, conv_kernel_size),
                k_conv1d=_short_conv_config(projection_dim, conv_kernel_size),
                v_conv1d=_short_conv_config(projection_dim, conv_kernel_size),
                f_a_proj=_linear_config(dim, kda_head_dim),
                f_b_proj=_linear_config(kda_head_dim, projection_dim),
                b_proj=_linear_config(dim, n_heads),
                o_norm=RMSNormGated.Config(norm=RMSNorm.Config(normalized_shape=kda_head_dim, eps=norm_eps)),
                o_proj=_linear_config(projection_dim, dim),
                g_proj=_linear_config(dim, projection_dim) if use_full_rank_gate else None,
                g_a_proj=_linear_config(dim, kda_head_dim) if not use_full_rank_gate else None,
                g_b_proj=_linear_config(kda_head_dim, projection_dim) if not use_full_rank_gate else None,
                dim=dim,
                num_heads=n_heads,
                head_dim=kda_head_dim,
                conv_kernel_size=conv_kernel_size,
                gate_lower_bound=gate_lower_bound,
                use_full_rank_gate=use_full_rank_gate,
                norm_eps=norm_eps,
            )
        else:
            attn_cfg = KimiGatedMLA.Config(
                q_a_proj=_linear_config(dim, q_lora_rank),
                q_a_layernorm=RMSNorm.Config(normalized_shape=q_lora_rank, eps=1e-6),
                q_b_proj=_linear_config(q_lora_rank, n_heads * (qk_nope_head_dim + qk_rope_head_dim)),
                kv_a_proj_with_mqa=_linear_config(dim, kv_lora_rank + qk_rope_head_dim),
                kv_a_layernorm=RMSNorm.Config(normalized_shape=kv_lora_rank, eps=1e-6),
                kv_b_proj=_linear_config(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)),
                g_proj=_linear_config(dim, n_heads * v_head_dim),
                o_proj=_linear_config(n_heads * v_head_dim, dim),
                dim=dim,
                n_heads=n_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                norm_eps=1e-6,
            )

        # FFN: dense for first n_dense_layers, MoE for the rest
        is_dense = layer_id < n_dense_layers
        if is_dense:
            ffn = _mlp_config(dim, dense_hidden_dim, situ_beta, situ_linear_beta)
            moe_cfg = None
        else:
            ffn = None
            moe_cfg = _moe_config(
                hidden_size=dim,
                num_experts=num_experts,
                top_k=router_top_k,
                moe_intermediate_size=moe_inter_dim,
                num_shared_experts=num_shared_experts,
                routed_expert_hidden_size=routed_expert_hidden_size,
                latent_moe_use_norm=latent_moe_use_norm,
                score_func=router_score_func,
                num_expert_groups=num_expert_groups,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                renormalize=renormalize,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                norm_eps=norm_eps,
            )

        layers.append(
            KimiK3TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps),
                self_attention_res=_attn_res_config(dim, norm_eps) if attn_res_block_size is not None else None,
                mlp_res=_attn_res_config(dim, norm_eps) if attn_res_block_size is not None else None,
                feed_forward=ffn,
                moe=moe_cfg,
                norm_eps=norm_eps,
                dim=dim,
                layer_id=layer_id,
                attn_res_block_size=attn_res_block_size,
            )
        )
    return layers


def _make_kimi_k3_model_config(
    *,
    param_init: dict | None = None,
    vocab_size: int = 163840,
    dim: int = 7168,
    n_layers: int = 93,
    n_dense_layers: int = 1,
    n_heads: int = 96,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    kda_head_dim: int = 128,
    dense_hidden_dim: int = 33792,
    moe_inter_dim: int = 3072,
    num_experts: int = 896,
    num_shared_experts: int = 2,
    router_top_k: int = 16,
    router_score_func: str = "sigmoid",
    num_expert_groups: int = 1,
    topk_group: int = 1,
    routed_expert_hidden_size: int | None = 3584,
    latent_moe_use_norm: bool = True,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    kda_layers: list[int] | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    norm_eps: float = 1e-5,
    conv_kernel_size: int = 4,
    gate_lower_bound: float | None = -5.0,
    use_full_rank_gate: bool = True,
    attn_res_block_size: int | None = 12,
) -> KimiK3Model.Config:
    layers = _build_kimi_k3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kda_head_dim=kda_head_dim,
        dense_hidden_dim=dense_hidden_dim,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=router_top_k,
        router_score_func=router_score_func,
        num_expert_groups=num_expert_groups,
        topk_group=topk_group,
        routed_expert_hidden_size=routed_expert_hidden_size,
        latent_moe_use_norm=latent_moe_use_norm,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        kda_layers=kda_layers,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        norm_eps=norm_eps,
        conv_kernel_size=conv_kernel_size,
        gate_lower_bound=gate_lower_bound,
        use_full_rank_gate=use_full_rank_gate,
        attn_res_block_size=attn_res_block_size,
    )
    return KimiK3Model.Config(
        param_init=param_init,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_LINEAR_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps),
        lm_head=_linear_config(dim, vocab_size),
        output_attn_res=_attn_res_config(dim, norm_eps) if attn_res_block_size is not None else None,
        vocab_size=vocab_size,
        dim=dim,
        layers=layers,
        norm_eps=norm_eps,
        attn_res_block_size=attn_res_block_size,
    )


def _debug_model() -> KimiK3Model.Config:
    """Minimal debug config: 4 layers (3 KDA + 1 MLA), 8 experts, small dims."""
    return _make_kimi_k3_model_config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=32,
        dense_hidden_dim=512,
        moe_inter_dim=128,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        routed_expert_hidden_size=128,
        kda_layers=[0, 1, 2],
    )


def _16layer_reduced_model() -> KimiK3Model.Config:
    """Reduced 16-layer model matching MindSpeed-MM's A3 single-node config."""
    return _make_kimi_k3_model_config(
        vocab_size=163840,
        dim=7168,
        n_layers=16,
        n_dense_layers=1,
        n_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dense_hidden_dim=33792,
        moe_inter_dim=3072,
        num_experts=32,
        num_shared_experts=2,
        router_top_k=16,
        routed_expert_hidden_size=3584,
    )


def _full_model() -> KimiK3Model.Config:
    """Full Kimi K3 2.8T model: 93 layers, 896 experts, top-16."""
    return _make_kimi_k3_model_config(
        vocab_size=163840,
        dim=7168,
        n_layers=93,
        n_dense_layers=1,
        n_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dense_hidden_dim=33792,
        moe_inter_dim=3072,
        num_experts=896,
        num_shared_experts=2,
        router_top_k=16,
        routed_expert_hidden_size=3584,
        kda_layers=_FULL_KDA_LAYERS,
    )


kimi_k3_configs = {
    "debug": _debug_model,
    "16layer_reduced": _16layer_reduced_model,
    "full": _full_model,
}
