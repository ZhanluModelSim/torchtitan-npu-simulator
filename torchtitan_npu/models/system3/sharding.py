# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-level sharding wiring for the system3 decoder.

Tensor parallelism is rejected for this family, so every dense placement is
the replicated form: parameters are ``Replicate`` on TP and activations stay
``Replicate``/``I`` between modules. The annotations exist so the
``spmd_types`` FSDP path sees DTensor states on the full SPMD mesh (FSDP2
rejects plain tensors when ``dp_mesh_dims`` is provided) and so the MoE
expert weights carry their EP layout through the shared sparse placement.

The latent routed-expert wrapper is model-specific (the gate/up pair rides
the dispatcher as one concatenated feature dim), so the generic
``set_moe_sharding_config`` cannot be used directly; the routed-experts
wrapper reuses the shared local-map helper with this family's expert weight
layout. The shared expert is computed fully on every rank (replicated), a
documented modeling choice: the reference applies the shared expert to all
tokens outside the routed dispatch.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    set_decoder_sharding_config,
)
from torchtitan.models.common.moe_sharding import (
    _tokens_per_expert_placement,
    dense_sequence_parallel_placement,
    expert_param_placement_dense,
    expert_param_placement_sparse,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from .model import System3Model

# Routed-expert layout for the latent grouped experts (E, F, D) weights:
# gate/up colwise, down rowwise — mirrors the common MoE convention. With TP
# rejected these placements are inert on the TP axis; EP overrides them with
# the shared sparse (Shard(0) per expert) placement.
_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w_gate_EFD": spmd.S(1),
    "w_up_EFD": spmd.S(1),
    "w_down_EDF": spmd.S(2),
}


def _replicated_param_sharding(*parameters: str) -> ShardingConfig:
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in parameters})


def _latent_routed_experts_sharding_configs(*, enable_ep: bool) -> tuple[ShardingConfig, ShardingConfig]:
    """Wrapper local-map + inner expert weight state for LatentRoutedExperts.

    Same recipe as the shared ``RoutedExperts`` helper, adapted to this
    wrapper's signature: the dispatcher carries the concatenated (gate, up)
    latent pair, so both halves take the x-like activation placement and the
    expert output comes back in the latent width.
    """
    if enable_ep:
        pre_dispatch_layout = dense_sequence_parallel_placement()
        state_shardings: dict[str, object] = {name: expert_param_placement_sparse() for name in _EXPERTS_PARAM_LAYOUT}
        dispatch_input_layout = dense_sequence_parallel_placement()
        dispatch_input_grad_layout = dense_sequence_parallel_placement()
    else:
        pre_dispatch_layout = dense_activation_placement(tp=spmd.R)
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in _EXPERTS_PARAM_LAYOUT.items()
        }
        dispatch_input_layout = dense_activation_placement(tp=spmd.R)
        dispatch_input_grad_layout = dense_activation_placement(tp=spmd.P)

    tokens_per_expert_layout = _tokens_per_expert_placement(enable_ep=enable_ep)
    dispatch_output_layout = dense_activation_placement(tp=spmd.P)

    arg_names = ("gate_l_TL", "up_l_TL", "topk_scores_TK", "topk_expert_ids_TK")
    wrapper_config = ShardingConfig(
        in_src_shardings={
            "gate_l_TL": pre_dispatch_layout,
            "up_l_TL": pre_dispatch_layout,
            "topk_scores_TK": dispatch_input_layout,
            "topk_expert_ids_TK": dispatch_input_layout,
            "num_local_tokens_per_expert_E": tokens_per_expert_layout,
        },
        in_dst_shardings={
            "gate_l_TL": dispatch_input_layout,
            "up_l_TL": dispatch_input_layout,
            "topk_scores_TK": dispatch_input_layout,
            "topk_expert_ids_TK": dispatch_input_layout,
            "num_local_tokens_per_expert_E": tokens_per_expert_layout,
        },
        out_src_shardings=dispatch_output_layout,
        out_dst_shardings=dispatch_output_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(
                dispatch_input_grad_layout,
                dispatch_input_grad_layout,
                dispatch_input_grad_layout,
                dispatch_input_grad_layout,
                # num_local_tokens_per_expert_E is routing metadata, but it is
                # still a DTensor input to local_map and must have placements.
                tokens_per_expert_layout,
            )
        ),
    )
    return wrapper_config, ShardingConfig(state_shardings=state_shardings)


def _set_moe_layer_sharding(moe_cfg, *, enable_ep: bool) -> None:
    moe_cfg.gate_down.sharding_config = _replicated_param_sharding("weight")
    moe_cfg.up_down.sharding_config = _replicated_param_sharding("weight")
    moe_cfg.latent_up.sharding_config = _replicated_param_sharding("weight")

    (
        routed_experts_config,
        inner_experts_config,
    ) = _latent_routed_experts_sharding_configs(enable_ep=enable_ep)
    moe_cfg.routed_experts.sharding_config = routed_experts_config
    moe_cfg.routed_experts.inner_experts.sharding_config = inner_experts_config

    # The shared expert is applied to every token outside the routed dispatch
    # (reference semantics); it stays replicated under EP.
    for shared in moe_cfg.shared_experts:
        shared.gate_inter.sharding_config = _replicated_param_sharding("weight")
        shared.up_inter.sharding_config = _replicated_param_sharding("weight")
        shared.inter_to_latent.sharding_config = _replicated_param_sharding("weight")


def _set_engram_sharding(engram_cfg) -> None:
    for name in ("key_proj", "value_proj", "query_proj"):
        getattr(engram_cfg, name).sharding_config = _replicated_param_sharding("weight")
    for norm_name in ("q_norm", "k_norm", "input_norm", "output_norm"):
        norm = getattr(engram_cfg, norm_name)
        norm.sharding_config = _replicated_param_sharding("weight")
    # Hash tables carry default replicated placements on their own Config
    # (table weight + per-head offsets buffer).


def _set_attention_sharding(attention_cfg) -> None:
    from .model import AlternatingAttention, GDN2Attention

    if isinstance(attention_cfg, GDN2Attention.Config):
        gdn2 = attention_cfg
        for linear_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "alpha_gate",
            "beta_gate",
            "da_proj",
        ):
            getattr(gdn2, linear_name).sharding_config = _replicated_param_sharding("weight")
        gdn2.da_norm.sharding_config = _replicated_param_sharding("weight")
        return
    assert isinstance(attention_cfg, AlternatingAttention.Config)
    assert attention_cfg.q_proj is not None and attention_cfg.o_proj is not None
    attention_cfg.q_proj.sharding_config = _replicated_param_sharding("weight")
    attention_cfg.o_proj.sharding_config = _replicated_param_sharding("weight")
    if attention_cfg.k_proj is not None:
        attention_cfg.k_proj.sharding_config = _replicated_param_sharding("weight")
    if attention_cfg.v_proj is not None:
        attention_cfg.v_proj.sharding_config = _replicated_param_sharding("weight")
    if attention_cfg.indexer is not None:
        attention_cfg.indexer.q_proj.sharding_config = _replicated_param_sharding("weight")
        attention_cfg.indexer.k_proj.sharding_config = _replicated_param_sharding("weight")
        attention_cfg.indexer.q_norm.sharding_config = _replicated_param_sharding("weight")
        attention_cfg.indexer.k_norm.sharding_config = _replicated_param_sharding("weight")
    # The rotary embedding owns no parameters; the sparse core owns none.


def _set_mhc_sharding(hc_cfg) -> None:
    hc_cfg.expand.sharding_config = _replicated_param_sharding("weight")
    hc_cfg.contract.sharding_config = _replicated_param_sharding("weight")
    hc_cfg.pre_norm.sharding_config = _replicated_param_sharding("weight")
    hc_cfg.post_norm.sharding_config = _replicated_param_sharding("weight")
    # mix_weights carries a default replicated placement on its own Config.


def _set_encoder_block_sharding(block_cfg) -> None:
    block_cfg.norm1.sharding_config = _replicated_param_sharding("weight")
    block_cfg.norm2.sharding_config = _replicated_param_sharding("weight")
    block_cfg.mlp_up.sharding_config = _replicated_param_sharding("weight")
    block_cfg.mlp_down.sharding_config = _replicated_param_sharding("weight")
    block_cfg.attn.qkv.sharding_config = _replicated_param_sharding("weight")
    block_cfg.attn.o.sharding_config = _replicated_param_sharding("weight")


def _set_modality_encoder_sharding(modality_cfg) -> None:
    if modality_cfg.image_encoder is not None:
        image = modality_cfg.image_encoder
        image.patch_embed.sharding_config = _replicated_param_sharding("weight")
        for block in image.blocks:
            _set_encoder_block_sharding(block)
        image.norm.sharding_config = _replicated_param_sharding("weight")
    if modality_cfg.audio_encoder is not None:
        audio = modality_cfg.audio_encoder
        audio.stem_conv1.sharding_config = _replicated_param_sharding("weight")
        audio.stem_conv2.sharding_config = _replicated_param_sharding("weight")
        for block in audio.blocks:
            _set_encoder_block_sharding(block)
        audio.norm.sharding_config = _replicated_param_sharding("weight")
    if modality_cfg.point3d_encoder is not None:
        point3d = modality_cfg.point3d_encoder
        for linear_name in ("mlp_in", "mlp_hidden", "mlp_out"):
            getattr(point3d, linear_name).sharding_config = _replicated_param_sharding("weight")
        for block in point3d.blocks:
            _set_encoder_block_sharding(block)
        point3d.norm.sharding_config = _replicated_param_sharding("weight")
    if modality_cfg.haptic_encoder is not None:
        haptic = modality_cfg.haptic_encoder
        haptic.stem_conv.sharding_config = _replicated_param_sharding("weight")
        for block in haptic.blocks:
            _set_encoder_block_sharding(block)
        haptic.norm.sharding_config = _replicated_param_sharding("weight")
    for proj in modality_cfg.modality_projs.values():
        proj.sharding_config = _replicated_param_sharding("weight")
    modality_cfg.modality_type_embedding.sharding_config = _replicated_param_sharding("weight")


def set_system3_sharding_config(config: "System3Model.Config", *, enable_ep: bool) -> None:
    """Fill ``sharding_config`` on every system3 sub-config.

    Root states (embeddings, final norm, lm_head) come from the shared
    decoder helper; per-layer states are replicated dense placements; MoE
    routed experts reuse the common sparse/dense expert placements with this
    family's weight layout.
    """
    set_decoder_sharding_config(config, enable_sp=False)
    for layer_cfg in config.layers:
        for norm_name in (
            "attn_pre_norm",
            "attn_post_norm",
            "moe_pre_norm",
            "moe_post_norm",
        ):
            getattr(layer_cfg, norm_name).sharding_config = _replicated_param_sharding("weight")
        _set_mhc_sharding(layer_cfg.hc_pre_attn)
        _set_mhc_sharding(layer_cfg.hc_pre_ffn)
        _set_attention_sharding(layer_cfg.attention)
        if layer_cfg.moe is not None:
            _set_moe_layer_sharding(layer_cfg.moe, enable_ep=enable_ep)
        if layer_cfg.engram is not None:
            _set_engram_sharding(layer_cfg.engram)
    if config.modality_encoder is not None:
        _set_modality_encoder_sharding(config.modality_encoder)
