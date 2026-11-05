# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native component sharding for GLM-5.3-Flash (spmd_types backend).

v1 declares everything replicated: the capture matrix is single + FSDP2,
where FSDP2 owns the data-parallel sharding and the spmd_types residency
only requires every parameter/buffer to be a DTensor on the full SPMD mesh.
TP/EP placements for the custom KDA/DSA/MoE modules land with the
parallelism matrix phase.
"""

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    set_decoder_sharding_config,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, LayerNorm, RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .attention import DSAAttention, DSAIndexer
from .feed_forward import GlmMoE, TopkRouter
from .hyper import HyperConnection
from .kda import KDALinearAttention


def _replicated_for(config) -> ShardingConfig:
    """Replicated placement for every own parameter of a protocol module."""
    names = ["weight"]
    if getattr(config, "bias", False) or (isinstance(config, LayerNorm.Config) and config.elementwise_affine):
        # nn.LayerNorm always owns a bias and carries no bias field.
        names.append("bias")
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in names})


def set_glm5_next_sharding_config(config, *, enable_sp: bool = False, enable_ep: bool = False) -> None:
    """Declare placements for every state in the GLM config tree.

    Called from ``Glm5NextModel.Config.update_from_config``; the parallelize
    step then walks the tree and distributes each parameter/buffer.
    """
    del enable_ep  # EP sharding lands with the parallelism matrix phase

    # Decoder root: tok_embeddings (Shard(0) vocab), norm, lm_head.
    set_decoder_sharding_config(config, enable_sp=enable_sp)

    # Protocol children nested in the custom module configs.
    for _, child, _, _ in config.traverse(Module.Config, recurse=True):
        is_protocol_dense = isinstance(
            child, (Linear.Config, RMSNorm.Config, LayerNorm.Config, Conv1d.Config, Embedding.Config)
        )
        if is_protocol_dense and child.sharding_config is None:
            child.sharding_config = _replicated_for(child)

    # Custom modules with own parameters/buffers.
    replicated = dense_param_placement(tp=spmd.R)
    for _, child, _, _ in config.traverse(Module.Config, recurse=True):
        if isinstance(child, HyperConnection.Config) and child.sharding_config is None:
            child.sharding_config = ShardingConfig(
                state_shardings={"fn": replicated, "base": replicated, "scale": replicated}
            )

    for layer in config.layers:
        attention = layer.attention
        if isinstance(attention, KDALinearAttention.Config):
            attention.forget_gate.sharding_config = ShardingConfig(
                state_shardings={"A_log": replicated, "dt_bias": replicated}
            )
            attention.o_norm.sharding_config = ShardingConfig(state_shardings={"weight": replicated})
        else:
            assert isinstance(attention, DSAAttention.Config)
            indexer = attention.indexer
            assert isinstance(indexer, DSAIndexer.Config)
            indexer.sharding_config = ShardingConfig(
                state_shardings={
                    "index_kpool_compress_ape": replicated,
                    "index_kpool_compress_gate": replicated,
                }
            )
        if layer.moe is not None:
            moe = layer.moe
            assert isinstance(moe, GlmMoE.Config)
            assert isinstance(moe.router, TopkRouter.Config)
            moe.router.sharding_config = ShardingConfig(state_shardings={"e_score_correction_bias": replicated})
            moe.routed_experts.inner_experts.sharding_config = ShardingConfig(
                state_shardings={
                    "w1_EFD": replicated,
                    "w2_EDF": replicated,
                    "w3_EFD": replicated,
                }
            )
