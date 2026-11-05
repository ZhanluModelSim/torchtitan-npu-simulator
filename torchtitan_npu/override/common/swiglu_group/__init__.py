# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent SwigluGroup overrides (registry-facing module)."""

from torchtitan.config import derive, override
import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.sharding import ShardingConfig

from .ascendc import (
    AscFeedForward,
    AscGroupedExperts,
    swiglu_group_activation,
)

__all__ = [
    "AscFeedForward",
    "AscGroupedExperts",
    "asc",
    "asc_shared_experts",
    "swiglu_group_activation",
]


@override(
    target=GroupedExperts.Config,
    fqns=["*.moe.routed_experts.inner_experts"],
    # Not exact: subclasses (GlmExperts.Config) must derive too; the fqn
    # selector already scopes this to routed experts.
    description="Use cann_ops_nn.swiglu_group for routed experts.",
)
def asc(cfg: GroupedExperts.Config) -> AscGroupedExperts.Config:
    # The vendor ops load lazily at forward time: derivation also runs on the
    # meta capture path, where the ops come from the simulator's fake bridge.
    if isinstance(cfg, AscGroupedExperts.Config):
        return cfg
    return derive(cfg, AscGroupedExperts.Config)


@override(
    target=FeedForward.Config,
    fqns=["*.moe.shared_experts", "*.feed_forward"],
    # Not exact: the shared expert / dense FFN may be a FeedForward subclass
    # (GlmMLP.Config); dense swiglu runs the same fused kernel on device.
    description="Use cann_ops_nn.swiglu_group for shared and dense feed-forwards.",
)
def asc_shared_experts(cfg: FeedForward.Config) -> AscFeedForward.Config:
    derived = derive(cfg, AscFeedForward.Config)
    return _stamp_clamp_buffers(derived)


def _stamp_clamp_buffers(config: AscFeedForward.Config) -> AscFeedForward.Config:
    """Declare the clamped-swiglu bound buffers (replicated constants) on the
    derived config, merging into any sharding the tree already carries."""
    placement = dense_param_placement(tp=spmd.R)
    if config.sharding_config is None:
        config.sharding_config = ShardingConfig(state_shardings={})
    state_shardings = config.sharding_config.state_shardings
    state_shardings["_min_bound"] = placement
    state_shardings["_max_bound"] = placement
    return config
