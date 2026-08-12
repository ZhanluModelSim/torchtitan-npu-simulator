# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable NPU MoE building blocks shared by model converters."""

__all__ = ["NpuSharedExperts", "SharedExpertActivationFn", "native_shared_expert_activation"]

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torchtitan.models.common.feed_forward import FeedForward

SharedExpertActivationFn = Callable[
    [torch.Tensor, float | None],
    torch.Tensor,
]


def native_shared_expert_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Apply the native small-op shared-expert SwiGLU decomposition."""
    gate, up = h.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


class NpuSharedExperts(FeedForward):
    """Shared-expert FeedForward with an injectable activation implementation."""

    @classmethod
    def convert(
        cls,
        parent: FeedForward,
        *,
        activation_fn: SharedExpertActivationFn = native_shared_expert_activation,
    ) -> "NpuSharedExperts":
        if not isinstance(parent, cls):
            try:
                # In-place conversion preserves parameters, hooks, and FQNs.
                parent.__class__ = cls  # pyrefly: ignore [bad-assignment]
            except TypeError as exc:
                raise RuntimeError("Cannot convert shared_experts to NpuSharedExperts in place") from exc
        converted = cast("NpuSharedExperts", parent)
        converted.set_expert_activation(activation_fn)
        return converted

    def set_expert_activation(self, activation_fn: SharedExpertActivationFn) -> None:
        self._expert_activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
        activation_fn = getattr(self, "_expert_activation_fn", native_shared_expert_activation)
        hidden = activation_fn(
            packed,
            getattr(self, "swiglu_limit", None),
        )
        return self.w2(hidden)
