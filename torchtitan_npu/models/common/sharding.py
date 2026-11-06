# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MoE storage declarations missing from the pinned upstream baseline."""

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe_sharding import set_moe_sharding_config as _set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig


def set_moe_sharding_config(moe_cfg, *, enable_ep: bool, enable_sp: bool) -> None:
    _set_moe_sharding_config(moe_cfg, enable_ep=enable_ep, enable_sp=enable_sp)
    if not enable_ep:
        # ebdfc8 leaves these weights as plain tensors. FSDP's dp_mesh_dims
        # path requires DTensor storage even without expert parallelism.
        for projection in (moe_cfg.routed_experts.w13, moe_cfg.routed_experts.w2):
            projection.sharding_config = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})
