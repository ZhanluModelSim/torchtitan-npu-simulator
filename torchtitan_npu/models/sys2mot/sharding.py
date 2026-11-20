# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding annotations for the sys2mot VLA.

TP/CP are rejected at the config level, so every state is placed replicated
on the TP axis and left to FSDP; with EP the routed-expert grouped weights
take the canonical sparse-family placement (expert dim on the EP mesh).
Child ``Linear``/``Embedding``/``Conv2d`` configs carry no default
annotations, so this walk annotates them explicitly; modules with plain
parameters (Router, GroupedExperts, HyperMix, SparseAttention/ActionExpert
``oa``/``sink``, Engram gates, ViT delimiters, ActionMoT positions, RMSNorm)
self-annotate in their ``Config.__post_init__``.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import dense_param_placement, set_decoder_sharding_config
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from .model import Sys2MotModel


def _replicate(config, *parameters: str) -> None:
    if config is not None and config.sharding_config is None:
        # Linear/Conv2d states: the weight plus the bias when the projection
        # carries one (reference vision stem/align and action I/O projections).
        states = list(parameters)
        if "weight" in states and getattr(config, "bias", False):
            states.append("bias")
        config.sharding_config = ShardingConfig(
            state_shardings={name: dense_param_placement(tp=spmd.R) for name in states}
        )


def set_sys2mot_sharding_config(config: "Sys2MotModel.Config", *, enable_ep: bool = False) -> None:
    """Annotate every state replicated on TP, left to FSDP.

    With EP enabled the routed-expert grouped weights take the canonical
    sparse-family placement instead (DP_REPLICATE/EFSDP replicated, EP
    shard on the expert dim), matching the shared MoE decoder recipe.
    """
    set_decoder_sharding_config(config, enable_sp=False)
    for layer_cfg in config.layers:
        attention = layer_cfg.attention
        _replicate(attention.qa, "weight")
        _replicate(attention.qb, "weight")
        _replicate(attention.kv, "weight")
        _replicate(attention.ob, "weight")
        if attention.compressor is not None:
            _replicate(attention.compressor.kv, "weight")
            _replicate(attention.compressor.gate, "weight")
        if attention.indexer is not None:
            _replicate(attention.indexer.q, "weight")
            _replicate(attention.indexer.weights, "weight")
            _replicate(attention.indexer.key, "weight")
        moe = layer_cfg.moe
        _replicate(moe.shared.gate, "weight")
        _replicate(moe.shared.up, "weight")
        _replicate(moe.shared.down, "weight")
        if enable_ep:
            experts = moe.routed_experts.inner_experts
            experts.sharding_config = ShardingConfig(
                state_shardings={
                    name: expert_param_placement_sparse() for name in ("w_gate_EFD", "w_up_EFD", "w_down_EDF")
                }
            )
        if layer_cfg.engram is not None:
            _replicate(layer_cfg.engram.table, "weight")
            _replicate(layer_cfg.engram.project, "weight")
        _replicate(layer_cfg.attn_mix.fn, "weight")
        _replicate(layer_cfg.ffn_mix.fn, "weight")
    vision = config.vision
    _replicate(vision.patch, "weight")
    _replicate(vision.align1, "weight")
    _replicate(vision.align2, "weight")
    for block in vision.blocks:
        _replicate(block.qkv, "weight")
        _replicate(block.out, "weight")
        _replicate(block.ffn.gate, "weight")
        _replicate(block.ffn.up, "weight")
        _replicate(block.ffn.down, "weight")
    action = config.action
    _replicate(action.state_in, "weight")
    _replicate(action.action_in, "weight")
    _replicate(action.time_in, "weight")
    _replicate(action.time_out, "weight")
    _replicate(action.out, "weight")
    for expert in action.blocks:
        _replicate(expert.qa, "weight")
        _replicate(expert.qb, "weight")
        _replicate(expert.kv, "weight")
        _replicate(expert.ob, "weight")
        _replicate(expert.ffn.gate, "weight")
        _replicate(expert.ffn.up, "weight")
        _replicate(expert.ffn.down, "weight")
        _replicate(expert.modulation, "weight", "bias")
