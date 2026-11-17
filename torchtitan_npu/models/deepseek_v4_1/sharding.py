# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The complete V4.1 parameter/activation placement policy.

Derived from the V4 policy; the V4.1 differences: no LightningIndexer
wrapper, no compressor ``ape`` (V4.1 builds no APE parameter), no MTP
depths, and the V4.1-only placements (vision markers, the VL router
bias).
"""

import spmd_types as spmd
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
)
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse, set_moe_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

DP, CP, TP = MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP

from .indexer import REUSE

_dense_param_rep = dense_param_placement(tp=spmd.R)
_attn_sink_placement = dense_param_placement(tp=spmd.S(0))

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
    "w13_lora_a": spmd.R,
    "w1_lora_b": spmd.S(1),
    "w3_lora_b": spmd.S(1),
    "w2_lora_a": spmd.S(2),
    "w2_lora_b": spmd.R,
}

_replicate_weight = ShardingConfig(
    state_shardings={"weight": _dense_param_rep},
)


def _engram_hidden_placement(*, enable_sp: bool) -> SpmdLayout:
    if not enable_sp:
        return dense_activation_placement(tp=spmd.I)
    return SpmdLayout(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=(DP, (CP, TP), None, None),
    )


def set_engram_sharding_config(engram_cfg, *, enable_ep: bool, enable_sp: bool) -> None:
    """Declare the dense mHC boundary and the sparse table storage family."""
    dense_param_rep = dense_param_placement(tp=spmd.R)
    table_state = {
        "weight": expert_param_placement_sparse() if enable_ep else dense_param_rep,
        "token_id_map": dense_param_rep,
        "head_vocab_sizes": dense_param_rep,
        "offsets": dense_param_rep,
        "hash_multipliers": dense_param_rep,
        "compressed_pad_id": dense_param_rep,
    }
    engram_cfg.table.sharding_config = ShardingConfig(state_shardings=table_state)

    engram_cfg.gate.sharding_config = ShardingConfig(
        state_shardings={
            "wkv": dense_param_rep,
            "q_weight": dense_param_rep,
            "k_weight": dense_param_rep,
        }
    )
    hidden = _engram_hidden_placement(enable_sp=enable_sp)
    token_src = dense_activation_placement(tp=spmd.R)
    token_dst = dense_token_ids_sequence_parallel_placement() if enable_sp else token_src
    engram_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_states_BLMD": hidden,
            "input_ids_BL": token_src,
            "positions_BL": token_src,
            "image_mask": token_src,
            "token_history": token_src,
        },
        in_dst_shardings={
            "hidden_states_BLMD": hidden,
            "input_ids_BL": token_dst,
            "positions_BL": token_dst,
            "image_mask": token_dst,
            "token_history": token_src,
        },
        out_src_shardings=hidden,
        out_dst_shardings=hidden,
        local_map=LocalMapConfig(
            in_grad_placements=(hidden, None, None, None, None),
        ),
    )


def dense_token_ids_sequence_parallel_placement():
    DP, CP, TP = MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP
    return SpmdLayout(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=(DP, (CP, TP)),
    )


def set_inner_attention_sharding(inner_cfg) -> None:
    """Placement for Attention Gym's ``gather_attn`` call.

    Under context parallelism the KV streams are the **whole row** -- gathered declaratively at
    this boundary, the V4 compressed-container pattern -- while ``q``, ``topk_indices`` and the
    sink keep the placements they have at every degree: the model does the query side's slicing
    and movement itself, so a producer needs no ``S(1)`` source declaration.
    """
    replicated_activation = dense_activation_placement(tp=spmd.R)
    row = dense_activation_placement(tp=spmd.R, cp=spmd.R)

    placements = {
        "q": dense_activation_placement(tp=spmd.S(2)),
        "swa_k": replicated_activation,
        "cmp_k": replicated_activation,
        "topk_indices": replicated_activation,
        "attn_sink": _attn_sink_placement,
    }
    # ``swa_k`` and ``cmp_k`` are the gathered row on the consumer side; their frames' ``seqused``
    # is what limits a rank's view, so the gather is the whole of the CP conversion here.
    consumed = {**placements, "swa_k": row, "cmp_k": row}
    inner_cfg.sharding_config = ShardingConfig(
        in_src_shardings=placements,
        in_dst_shardings=consumed,
        out_src_shardings=placements["q"],
        out_dst_shardings=placements["q"],
    )


def set_compressor_sharding(compressor_cfg):
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    if compressor_cfg.wgate is not None:
        compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = _replicate_weight


def set_indexer_sharding(indexer_cfg):
    replicated_activation = dense_activation_placement(tp=spmd.R)
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
    # The selector scores the **whole row's** index keys against this rank's chunks, so the keys
    # get the same ``S(1) -> R`` conversion the sparse core declares for its KV streams; the
    # query side and the candidate pool keep the placements they have at every degree.
    indexer_cfg.selector.sharding_config = ShardingConfig(
        in_src_shardings={
            "idx_q_BLHiDi": replicated_activation,
            "idx_k_BNDi": replicated_activation,
            "weights_BLHi": replicated_activation,
        },
        in_dst_shardings={
            "idx_q_BLHiDi": replicated_activation,
            "idx_k_BNDi": dense_activation_placement(tp=spmd.R, cp=spmd.R),
            "weights_BLHi": replicated_activation,
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
    if indexer_cfg.wk is not None:
        indexer_cfg.wk.sharding_config = _replicate_weight
    if indexer_cfg.k_norm is not None:
        indexer_cfg.k_norm.sharding_config = _replicate_weight


def set_v41_attention_sharding(attention_cfg, *, enable_sp: bool):
    at = attention_cfg
    attn_x_layout = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I)

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R),
        },
        state_shardings={"attn_sink": _attn_sink_placement},
    )

    set_inner_attention_sharding(at.inner_attention)

    at.wq_a.sharding_config = _replicate_weight
    at.q_norm.sharding_config = _replicate_weight
    at.wq_b.sharding_config = colwise_config()
    at.wkv.sharding_config = _replicate_weight
    at.kv_norm.sharding_config = _replicate_weight
    at.wo_a.sharding_config = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.S(0))})
    at.wo_b.sharding_config = rowwise_config(output_sp=enable_sp)
    at.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )

    if at.compressor.is_source:
        set_compressor_sharding(at.compressor)
    if at.indexer.mode is not REUSE:
        set_indexer_sharding(at.indexer)


def set_v41_layer_sharding(layer_cfg, *, enable_sp: bool, enable_ep: bool) -> None:
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

    set_v41_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)

    set_moe_sharding_config(
        layer_cfg.moe,
        enable_ep=enable_ep,
        enable_sp=enable_sp,
        expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
    )
    # Both router extensions are placement contracts as well as parameters: the hash
    # table reads ``input_ids`` and the vision bias reads ``image_mask``, so each is
    # declared only where the stack has it.  No V4.1 layer hashes, so its MoE receives no
    # ``input_ids``; only the multimodal flavor receives an ``image_mask``.
    if getattr(layer_cfg.moe.router, "hash", False):
        input_ids_src_placement = dense_activation_placement(tp=spmd.R)
        input_ids_dst_placement = (
            dense_token_ids_sequence_parallel_placement() if enable_ep else dense_activation_placement(tp=spmd.R)
        )
        layer_cfg.moe.sharding_config.in_src_shardings[  # pyrefly: ignore [missing-attribute]
            "input_ids"
        ] = input_ids_src_placement
        layer_cfg.moe.sharding_config.in_dst_shardings[  # pyrefly: ignore [missing-attribute]
            "input_ids"
        ] = input_ids_dst_placement
    if getattr(layer_cfg.moe.router, "vision_enabled", False):
        layer_cfg.moe.router.sharding_config = ShardingConfig(state_shardings={"bias_vl": _dense_param_rep})
        image_mask_placement = dense_activation_placement(tp=spmd.R)
        layer_cfg.moe.sharding_config.in_src_shardings[  # pyrefly: ignore [missing-attribute]
            "image_mask"
        ] = image_mask_placement
        layer_cfg.moe.sharding_config.in_dst_shardings[  # pyrefly: ignore [missing-attribute]
            "image_mask"
        ] = image_mask_placement


def set_deepseek_v4_1_sharding_config(
    config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Assign the complete V4.1 placement policy."""
    from .model import DeepSeekV41MultimodalModel

    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer in config.layers:
        if layer.engram is not None:
            set_engram_sharding_config(layer.engram, enable_ep=enable_ep, enable_sp=enable_sp)

    # The markers exist only on the multimodal stack, so their placement follows the
    # config's class rather than a ``None`` field.
    if isinstance(config, DeepSeekV41MultimodalModel.Config):
        config.image_marker_embeddings.sharding_config = ShardingConfig(
            state_shardings=dict.fromkeys(
                ("image_start", "image_newline", "image_end"),
                _dense_param_rep,
            )
        )

    for layer_cfg in config.layers:
        set_v41_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
