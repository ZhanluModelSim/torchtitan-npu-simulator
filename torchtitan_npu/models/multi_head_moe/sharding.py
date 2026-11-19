# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-level sharding wiring for the SLA2 + Multi-Head-MoE decoder.

Tensor parallelism is rejected for this family, so every placement is the
dense replicated form: parameters are ``Replicate`` on TP and activations
stay ``Replicate``/``I`` between modules. The annotations exist so the
``spmd_types`` FSDP path sees DTensor states on the full SPMD mesh (FSDP2
rejects plain tensors when ``dp_mesh_dims`` is provided) and so the MoE
expert weights carry their EP layout through the shared sparse placement.

The MoE wrapper and per-head gate are model-specific: the gate router is a
plain replicated parameter (expert sharding applies to the grouped experts
only), so the generic ``set_moe_sharding_config`` cannot be used directly;
the routed-experts and shared-experts sub-configs reuse the common helpers.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.moe_sharding import (
    _routed_experts_sharding_configs,
    _shared_experts_sharding_configs,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from .model import MultiHeadMoEModel

# Routed-expert layout for the shared GroupedExperts (E, F, D) weights:
# w1/w3 colwise, w2 rowwise — identical to the common MoE convention.
_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def _replicated_param_sharding(*parameters: str) -> ShardingConfig:
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in parameters})


def set_multi_head_moe_sharding_config(config: "MultiHeadMoEModel.Config", *, enable_ep: bool) -> None:
    """Fill ``sharding_config`` on every multi_head_moe sub-config.

    Root states (embeddings, final norm, lm_head) come from the shared
    decoder helper; per-layer states are replicated dense placements; MoE
    routed experts reuse the common sparse/dense expert placements.
    """
    set_decoder_sharding_config(config, enable_sp=False)
    for layer_cfg in config.layers:
        attention = layer_cfg.attention
        for linear in (attention.to_q, attention.to_k, attention.to_v, attention.to_out):
            linear.sharding_config = _replicated_param_sharding("weight")
        attention.norm_q.sharding_config = _replicated_param_sharding("weight")
        attention.norm_k.sharding_config = _replicated_param_sharding("weight")
        attention.inner.router.proj_q.sharding_config = _replicated_param_sharding("weight")
        attention.inner.router.proj_k.sharding_config = _replicated_param_sharding("weight")
        # InnerBlockSLA2.alpha and MultiHeadGate.router carry default
        # replicated sharding configs on their own Config classes.

        layer_cfg.attention_norm.sharding_config = _replicated_param_sharding("weight")
        layer_cfg.ffn_norm.sharding_config = _replicated_param_sharding("weight")

        if layer_cfg.feed_forward is not None:
            set_dense_ffn_sharding(
                layer_cfg.feed_forward,
                attn_x_layout=dense_activation_placement(tp=spmd.I),
                enable_sp=False,
            )
        if layer_cfg.moe is not None:
            _set_moe_layer_sharding(layer_cfg.moe, enable_ep=enable_ep)


def _set_moe_layer_sharding(moe_cfg, *, enable_ep: bool) -> None:
    moe_cfg.proj_in.sharding_config = _replicated_param_sharding("weight")
    moe_cfg.proj_out.sharding_config = _replicated_param_sharding("weight")

    (
        shared_config,
        w1_config,
        w2_config,
        w3_config,
    ) = _shared_experts_sharding_configs(enable_ep=enable_ep, enable_sp=False)
    shared = moe_cfg.shared_experts
    shared.sharding_config = shared_config
    shared.w1.sharding_config = w1_config
    shared.w2.sharding_config = w2_config
    shared.w3.sharding_config = w3_config

    routed_experts_config, inner_experts_config = _routed_experts_sharding_configs(
        enable_ep=enable_ep,
        enable_sp=False,
        expert_param_layout=_EXPERTS_PARAM_LAYOUT,
    )
    moe_cfg.routed_experts.sharding_config = routed_experts_config
    moe_cfg.routed_experts.inner_experts.sharding_config = inner_experts_config
