# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only autograd bridge for BF16 grouped experts on meta tensors."""

from __future__ import annotations

from typing import Callable

import torch


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    """Create a shape-only gradient without adding a fabricated L0 op."""
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


class _SimGroupedExperts(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        run_forward,
        w13,
        w2,
        x,
        num_tokens_per_expert,
        swiglu_limit,
        routed_scores,
    ):
        ctx.save_for_backward(w13, w2, x)
        ctx.routed_scores = routed_scores
        return run_forward(
            w13,
            w2,
            None,
            x,
            num_tokens_per_expert,
            swiglu_limit,
            routed_scores,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        w13, w2, x = ctx.saved_tensors
        routed_scores = ctx.routed_scores

        # Meta simulation only needs shape and lifetime information. Returning
        # shape-correct tensors makes expert parameters participate in the same
        # public optimizer path as parameters backed by ordinary autograd ops.
        dw13 = _uncaptured_empty_like(w13)
        dw2 = _uncaptured_empty_like(w2)
        dx = _uncaptured_empty_like(x)
        drouted_scores = (
            _uncaptured_empty_like(routed_scores)
            if isinstance(routed_scores, torch.Tensor)
            else None
        )
        return None, dw13, dw2, dx, None, None, drouted_scores


def run_meta_grouped_experts(
    run_forward: Callable,
    w13: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None,
    routed_scores: torch.Tensor | None,
) -> torch.Tensor:
    """Run grouped experts with a simulator-owned shape-only backward."""
    return _SimGroupedExperts.apply(
        run_forward,
        w13,
        w2,
        x,
        num_tokens_per_expert,
        swiglu_limit,
        routed_scores,
    )
