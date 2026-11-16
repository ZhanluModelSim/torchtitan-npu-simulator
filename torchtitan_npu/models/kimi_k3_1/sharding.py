# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 indexers stay replicated on TP, independently of the MLA heads."""

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.sharding import ShardingConfig

from .dsa import KimiIndexerLayerNorm


def set_indexer_sharding_config(config) -> None:
    for _, child, _, _ in config.traverse(KimiIndexerLayerNorm.Config, recurse=True):
        child.sharding_config = ShardingConfig(
            state_shardings={name: dense_param_placement(tp=spmd.R) for name in ("weight", "bias")}
        )
