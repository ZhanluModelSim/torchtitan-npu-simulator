# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent SwigluGroup overrides (registry-facing module)."""

from torchtitan.config import derive, override
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts

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
    return derive(cfg, AscFeedForward.Config)
