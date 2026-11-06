# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    pre_lm_head_norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    token_id_placement,
)
from torchtitan.protocols.sharding import ShardingConfig

from torchtitan_npu.models.common.sharding import set_moe_sharding_config

_dense_param_rep = dense_param_placement(tp=spmd.R)
_act_shard0_tp_rep = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
_attn_sink_placement = dense_param_placement(tp=spmd.S(0))
DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP
if TYPE_CHECKING:
    from torchtitan_npu.models.deepseek_v4.model import (
        DeepSeekV4Model,
        DeepSeekV4TransformerBlock,
    )

_replicate_weight = ShardingConfig(
    state_shardings={"weight": _dense_param_rep},
)


def dense_token_ids_sequence_parallel_placement() -> SpmdType:
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP)),
    )


def set_compressed_sparse_attention_sharding(attention_cfg) -> None:
    """The CP boundary of the sparse core and its selector (the V4.1 pattern).

    ``replicated_activation`` is TP-replicated and CP-sharded at the producer side; ``row`` is
    the gathered whole row the consumer addresses.  The core's ``ShardingConfig`` turns the
    former into the latter for both KV streams, and the selector gathers the index keys the
    same way; both are the ``S(0) -> R`` conversion V4.1 declares at its own call sites.
    """
    q = attention_activation_placement()
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    row = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    sink = _attn_sink_placement

    placements = {
        "q": q,
        "swa_k": replicated_activation,
        "cmp_k": replicated_activation,
        "topk_indices": replicated_activation,
        "topk_scores": replicated_activation,
        "attn_sink": sink,
    }
    # The core consumes the gathered KV streams, the local query chunks and the sink, while
    # the selection arrives from the selector already in the model's document-local numbering.
    consumed = {**placements, "swa_k": row, "cmp_k": row}
    attention_cfg.inner_attention.sharding_config = ShardingConfig(
        in_src_shardings=placements,
        in_dst_shardings=consumed,
        out_src_shardings=q,
        out_dst_shardings=q,
    )

    if attention_cfg.lightning_indexer is not None:
        # The selector scores the whole row's index keys against this rank's chunks: the query
        # side stays replicated, the keys get the same ``S(0) -> R`` gather as the KV streams.
        attention_cfg.lightning_indexer.sharding_config = ShardingConfig(
            in_src_shardings={
                "idx_q": replicated_activation,
                "idx_k": replicated_activation,
                "idx_w": replicated_activation,
            },
            in_dst_shardings={
                "idx_q": replicated_activation,
                "idx_k": row,
                "idx_w": replicated_activation,
            },
        )


def set_deepseek_v4_attention_sharding(attention_cfg, *, enable_sp):
    at = attention_cfg
    attn_x_layout = (
        dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
        },
        # ``attn_sink`` is a bare ``[n_heads]`` parameter used as a head-wise
        # vector, so TP shards its head dimension.
        state_shardings={"attn_sink": _attn_sink_placement},
    )

    # The core and the selector are the CP boundary for the KV/index streams.
    set_compressed_sparse_attention_sharding(at)

    # Attention submodule configs are explicit fields, so their parameter
    # layouts can be assigned before construction.
    at.wq_a.sharding_config = _replicate_weight
    at.q_norm.sharding_config = _replicate_weight
    at.wq_b.sharding_config = colwise_config(input_layout=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)))
    q_head_layout = attention_activation_placement()
    at.q_head_norm.sharding_config = ShardingConfig(
        state_shardings={"weight": _dense_param_rep},
        in_src_shardings={"x": q_head_layout},
        in_dst_shardings={"x": q_head_layout},
        out_src_shardings=q_head_layout,
        out_dst_shardings=q_head_layout,
        local_spmd=True,
    )
    at.wkv.sharding_config = _replicate_weight
    at.kv_norm.sharding_config = _replicate_weight
    # ``wo_a`` stores per-group matrices flattened across group and output
    # dimensions, so TP shards its flattened row dimension (dim 0).
    at.wo_a.sharding_config = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.S(0))})
    at.wo_b.sharding_config = rowwise_config(
        output_layout=(
            dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
    )
    at.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )

    if at.compressor is not None:
        set_compressor_sharding(at.compressor)
    if at.indexer is not None:
        set_indexer_sharding(at.indexer)


def set_compressor_sharding(compressor_cfg):
    # Compressor projections, normalization, and APE are explicit config
    # fields, allowing each parameter layout to be assigned before build.
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = _replicate_weight
    # ``ape`` is a plain parameter on the Compressor module itself.  The output is the local
    # compressed chunk (CP-sharded); state it explicitly so the wrapper's ``S(0) -> R`` gather
    # has a source placement to convert from rather than inferring one.
    local_chunk = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    compressor_cfg.sharding_config = ShardingConfig(
        state_shardings={"ape": _dense_param_rep},
        in_src_shardings={"x": local_chunk},
        in_dst_shardings={"x": local_chunk},
        out_src_shardings=local_chunk,
        out_dst_shardings=local_chunk,
    )


def set_indexer_sharding(indexer_cfg):
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    indexer_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
        in_dst_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
    )
    indexer_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    indexer_cfg.wq_b.sharding_config = ShardingConfig(
        state_shardings={"weight": _dense_param_rep},
    )
    indexer_cfg.weights_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": _dense_param_rep},
    )
    set_compressor_sharding(indexer_cfg.compressor)


def set_deepseek_v4_layer_sharding(
    layer_cfg: "DeepSeekV4TransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    hc_pre_rep = ShardingConfig(
        state_shardings={
            "hc_fn": _dense_param_rep,
            "hc_base": _dense_param_rep,
            "hc_scale": _dense_param_rep,
        },
    )
    layer_cfg.hc_attn_pre.sharding_config = hc_pre_rep
    layer_cfg.hc_ffn_pre.sharding_config = hc_pre_rep

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_layout = (
        dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )

    set_deepseek_v4_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)

    # MoE FFN (every DSV4 layer is a MoE layer).
    set_moe_sharding_config(
        layer_cfg.moe,
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )
    moe_sharding = layer_cfg.moe.sharding_config
    assert moe_sharding is not None
    if moe_sharding.in_dst_shardings is None:
        moe_sharding.in_dst_shardings = dict(moe_sharding.in_src_shardings or {})

    input_ids_src_placement = token_id_placement()
    # Hash routing owns the token-ID redistribution to the score shard.
    if moe_sharding.in_src_shardings is None:
        moe_sharding.in_src_shardings = {}
    moe_sharding.in_src_shardings["input_ids_T"] = input_ids_src_placement
    if layer_cfg.moe.router.layer_id < layer_cfg.moe.router.n_hash_layers:  # pyrefly: ignore [missing-attribute]
        router_sharding = layer_cfg.moe.router.sharding_config or ShardingConfig()
        router_sharding.state_shardings["tid2eid"] = _dense_param_rep
        layer_cfg.moe.router.sharding_config = router_sharding


def set_deepseek_v4_sharding_config(
    config: "DeepSeekV4Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_decoder_sharding_config(config, enable_sp=enable_sp)

    config.hc_head.sharding_config = ShardingConfig(
        state_shardings={
            "hc_fn": _dense_param_rep,
            "hc_base": _dense_param_rep,
            "hc_scale": _dense_param_rep,
        },
    )

    for layer_cfg in config.layers:
        set_deepseek_v4_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)

    if len(config.mtp_layers) > 0:
        _set_deepseek_v4_mtp_sharding(
            config,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )


def _set_deepseek_v4_mtp_sharding(
    config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    activation = (
        dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    norm = norm_config(enable_sp=enable_sp)

    def projection_sharding() -> ShardingConfig:
        return ShardingConfig(
            state_shardings={"weight": _dense_param_rep},
            in_src_shardings={"input": activation},
            out_src_shardings=activation,
        )

    for mtp_layer_cfg in config.mtp_layers:
        set_deepseek_v4_layer_sharding(
            mtp_layer_cfg,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )

        block_sharding = mtp_layer_cfg.sharding_config or ShardingConfig()
        if enable_sp:
            if block_sharding.in_src_shardings is None:
                block_sharding.in_src_shardings = {}
            if block_sharding.in_dst_shardings is None:
                block_sharding.in_dst_shardings = {}
            block_sharding.in_src_shardings["mtp_input_valid_mask"] = token_id_placement()
            block_sharding.in_dst_shardings["mtp_input_valid_mask"] = dense_token_ids_sequence_parallel_placement()
        mtp_layer_cfg.sharding_config = block_sharding

        mtp_layer_cfg.hc_head.sharding_config = ShardingConfig(
            state_shardings={
                "hc_fn": _dense_param_rep,
                "hc_base": _dense_param_rep,
                "hc_scale": _dense_param_rep,
            },
        )

        mtp_layer_cfg.enorm.sharding_config = norm
        mtp_layer_cfg.hnorm.sharding_config = norm
        mtp_layer_cfg.mtp_norm.sharding_config = pre_lm_head_norm_config(enable_sp=enable_sp)
        mtp_layer_cfg.e_proj.sharding_config = projection_sharding()
        mtp_layer_cfg.h_proj.sharding_config = projection_sharding()
