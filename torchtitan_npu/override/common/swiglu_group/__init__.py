# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent SwigluGroup overrides (registry-facing module)."""

__all__ = ["AscSwiGLU", "asc", "asc_shared_experts", "swiglu_group_activation"]

from torchtitan.config import override
from torchtitan.models.common.activation import SwiGLU

from .ascendc import AscSwiGLU, _ensure_cann_ops_loaded, swiglu_group_activation


def _activation_config(cfg: SwiGLU.Config) -> AscSwiGLU.Config:
    return AscSwiGLU.Config(swiglu_limit=getattr(cfg, "swiglu_limit", 0.0))


@override(
    target=SwiGLU.Config,
    fqns=["*.feed_forward.routed_experts.activation_fn"],
    description="Use cann_ops_nn.swiglu_group for routed experts.",
)
def asc(cfg: SwiGLU.Config) -> AscSwiGLU.Config:
    _ensure_cann_ops_loaded()
    return _activation_config(cfg)


@override(
    target=SwiGLU.Config,
    fqns=["*.feed_forward.shared_experts.activation_fn"],
    description="Use cann_ops_nn.swiglu_group for shared experts.",
)
def asc_shared_experts(cfg: SwiGLU.Config) -> AscSwiGLU.Config:
    _ensure_cann_ops_loaded()
    return _activation_config(cfg)
