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
    _ensure_cann_ops_loaded,
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
    exact=True,
    description="Use cann_ops_nn.swiglu_group for routed experts.",
)
def asc(cfg: GroupedExperts.Config) -> AscGroupedExperts.Config:
    _ensure_cann_ops_loaded()
    if isinstance(cfg, AscGroupedExperts.Config):
        return cfg
    return derive(cfg, AscGroupedExperts.Config)


@override(
    target=FeedForward.Config,
    fqns=["*.moe.shared_experts"],
    exact=True,
    description="Use cann_ops_nn.swiglu_group for shared experts.",
)
def asc_shared_experts(cfg: FeedForward.Config) -> AscFeedForward.Config:
    _ensure_cann_ops_loaded()
    return derive(cfg, AscFeedForward.Config)
