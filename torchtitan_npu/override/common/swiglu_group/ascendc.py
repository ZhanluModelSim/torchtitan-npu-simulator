# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common FFN implementation backed by CANN ``SwigluGroup``."""

import importlib
from dataclasses import dataclass

import torch

# Keep the PyTorch dispatcher entry used by TorchAO weight wrappers.
from torchtitan.models.common.activation import BinaryActivationFn


def _effective_swiglu_limit(module) -> float | None:
    swiglu_limit = getattr(module, "swiglu_limit", None)
    if swiglu_limit is None or swiglu_limit <= 0:
        return None
    return float(swiglu_limit)


def swiglu_group_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the documented ``cann_ops_nn.swiglu_group`` op."""
    weight = None if routed_scores is None else routed_scores.to(dtype=torch.float32).contiguous()
    clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
    return torch.ops.cann_ops_nn.swiglu_group.default(
        h.contiguous(),
        weight=weight,
        group_index=None,
        clamp_limit=clamp_limit,
    )


class AscSwiGLU(BinaryActivationFn):
    """CANN activation for first-class fused gate/up projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        swiglu_limit: float = 0.0

    def __init__(self, config):
        self.swiglu_limit = config.swiglu_limit

    def _activation_fn(self, gate, up):
        raise NotImplementedError("CANN implements the fused activation")

    def __call__(self, gate_up, **kwargs):
        # Routed score multiplication belongs to RoutedExperts, after this op.
        return swiglu_group_activation(
            gate_up.flatten(-2),
            _effective_swiglu_limit(self),
        )


def _ensure_cann_ops_loaded() -> None:
    importlib.import_module("cann_ops_nn.ops")
