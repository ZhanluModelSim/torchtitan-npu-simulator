# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native component sharding for Kimi's attention and latent MoE."""

from dataclasses import replace

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .attention import KimiDeltaAttention
from .model import KimiResidualScoreNorm


def set_kimi_sharding_config(config, *, enable_sp: bool, enable_ep: bool):
    replicated = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})
    for _, child, _, _ in config.traverse(Module.Config, recurse=True):
        if isinstance(child, (Linear.Config, RMSNorm.Config, Conv1d.Config, KimiResidualScoreNorm.Config)):
            child.sharding_config = replicated
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    activation = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I)
    for layer in config.layers:
        layer.attention_norm.sharding_config = norm_config(enable_sp=enable_sp)
        layer.ffn_norm.sharding_config = norm_config(enable_sp=enable_sp)
        attention = layer.attention
        attention.sharding_config = ShardingConfig(
            in_src_shardings={"hidden_states": activation},
            in_dst_shardings={"hidden_states": dense_activation_placement(tp=spmd.R)},
        )
        if isinstance(attention, KimiDeltaAttention.Config):
            attention.sharding_config.state_shardings = {
                "A_log": dense_param_placement(tp=spmd.S(0)),
                "dt_bias": dense_param_placement(tp=spmd.S(0)),
            }
            colwise = ("q_proj", "k_proj", "v_proj", "f_b_proj", "b_proj")
            colwise += ("g_proj",) if attention.use_full_rank_gate else ("g_b_proj",)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                getattr(attention, name).conv.sharding_config = ShardingConfig(
                    state_shardings={"weight": dense_param_placement(tp=spmd.S(0))}
                )
        else:
            colwise = ("q_b_proj", "kv_b_proj", "g_proj")
        for name in colwise:
            column = colwise_config()
            column.out_src_shardings = dense_activation_placement(tp=spmd.S(-1), cp=spmd.R)
            getattr(attention, name).sharding_config = column
        attention.o_proj.sharding_config = rowwise_config(output_sp=enable_sp)
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(layer.feed_forward, attn_x_layout=activation, enable_sp=enable_sp)
        if layer.moe is not None:
            moe = layer.moe
            set_moe_sharding_config(
                moe,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout={"w1_EFD": spmd.S(1), "w2_EDF": spmd.S(2), "w3_EFD": spmd.S(1)},
            )
            # A latent RMSNorm cannot consume a TP partial sum. Complete the
            # native expert reduction before applying norm and up projection.
            completed = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I)
            moe.routed_experts.sharding_config = replace(
                moe.routed_experts.sharding_config, out_dst_shardings=completed
            )
            moe.sharding_config = replace(moe.sharding_config, out_src_shardings=completed)
            if not enable_sp and moe.shared_experts is not None:
                moe.shared_experts.w2.sharding_config = rowwise_config()
            for projection in (moe.down_proj, moe.up_proj):
                if projection is not None:
                    projection.sharding_config = replicated
            if moe.norm is not None:
                moe.norm.sharding_config = norm_config(enable_sp=enable_sp)
