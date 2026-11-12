# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Decoder and MoE layouts for NPU model configurations."""

from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_expert_linear_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
    shared_expert_rowwise_sharding_config,
    shared_experts_output_placement,
    shared_experts_sharding_config,
)


def set_decoder_sharding_config(config, *, enable_sp: bool) -> None:
    config.tok_embeddings.sharding_config = token_embedding_sharding_config(enable_sp=enable_sp)
    config.norm.sharding_config = norm_sharding_config(enable_sp=enable_sp)
    config.lm_head.sharding_config = lm_head_sharding_config()


def set_moe_sharding_config(moe_cfg, *, enable_ep: bool, enable_sp: bool) -> None:
    moe_cfg.sharding_config = moe_sharding_config(enable_sp=enable_sp)
    moe_cfg.router.sharding_config = router_sharding_config()
    moe_cfg.router.gate.sharding_config = router_gate_sharding_config()
    experts = moe_cfg.routed_experts
    experts.sharding_config = routed_experts_sharding_config()
    for projection in (experts.w13, experts.w2):
        projection.sharding_config = routed_expert_linear_sharding_config(enable_ep=enable_ep)
    if moe_cfg.shared_experts is not None:
        shared = moe_cfg.shared_experts
        shared.sharding_config = shared_experts_sharding_config(enable_sp=enable_sp)
        shared.w2.sharding_config = shared_expert_rowwise_sharding_config(
            output_layout=shared_experts_output_placement(enable_sp=enable_sp)
        )
