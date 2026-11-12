# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debug recipes with the native typed V4 context-parallel backend."""

from torchtitan.config.transform import ContextParallelTransform

from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


def _debugmodel_cp(*, num_mtp_layers: int):
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscV4ContextParallelSparseAttention

    config = deepseek_v4_debugmodel(num_mtp_layers=num_mtp_layers)
    transform = ContextParallelTransform(inner_attention=AscV4ContextParallelSparseAttention)
    config.model = transform.transform(config.model)
    for _, attention, _, _ in config.model.traverse(AscV4ContextParallelSparseAttention.Config):
        # Match the geometry of the fused debug recipe, including its MTP blocks.
        attention.num_heads = 16
        attention.index_num_heads = 8
    return config


def deepseek_v4_debugmodel_cp():
    """Debug model for the CP2 + EP2 smoke case."""
    return _debugmodel_cp(num_mtp_layers=0)


def deepseek_v4_debugmodel_mtp():
    """Debug model with one MTP depth for the CP2 smoke case."""
    return _debugmodel_cp(num_mtp_layers=1)
