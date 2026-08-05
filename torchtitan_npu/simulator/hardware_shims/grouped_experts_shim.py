# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only autograd bridge for BF16 grouped experts on meta tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


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


def _uncaptured_empty(
    shape: tuple[int, ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None:
        return torch.empty(
            shape,
            dtype=reference.dtype,
            device=reference.device,
        )
    with capture.suspend_recording():
        return torch.empty(
            shape,
            dtype=reference.dtype,
            device=reference.device,
        )


def _current_module_path() -> str:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None or capture.module_path_tracker is None:
        return ""
    return capture.module_path_tracker.current_path()


def _record_op(
    raw_op_type: str,
    inputs: list[torch.Tensor],
    output: torch.Tensor,
    module_path: str,
) -> None:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type,
            inputs=inputs,
            outputs=[output],
            module_path=module_path,
        )


def _record_grouped_mm(
    inputs: list[torch.Tensor],
    output: torch.Tensor,
    module_path: str,
) -> None:
    _record_op(
        "aten._grouped_mm.default",
        inputs,
        output,
        module_path,
    )


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
        pre_activation = _uncaptured_empty(
            (x.shape[0], w13.shape[-2]),
            x,
        )
        activated_hidden = _uncaptured_empty(
            (x.shape[0], w2.shape[-1]),
            x,
        )
        scaled_hidden = _uncaptured_empty_like(activated_hidden)
        ctx.save_for_backward(
            w13,
            w2,
            x,
            num_tokens_per_expert,
            pre_activation,
            activated_hidden,
            scaled_hidden,
        )
        ctx.routed_scores = routed_scores
        ctx.module_path = _current_module_path()
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
        (
            w13,
            w2,
            x,
            num_tokens_per_expert,
            pre_activation,
            activated_hidden,
            scaled_hidden,
        ) = ctx.saved_tensors
        routed_scores = ctx.routed_scores

        dw13 = _uncaptured_empty_like(w13)
        dw2 = _uncaptured_empty_like(w2)
        dx = _uncaptured_empty_like(x)
        grad_scaled_hidden = _uncaptured_empty_like(scaled_hidden)

        # Each forward GMM contributes one activation-gradient GMM and one
        # weight-gradient GMM. These are shape-only events, but their count,
        # tensor metadata, dependencies, and FLOPs follow the production
        # grouped-expert backward.
        _record_grouped_mm(
            [grad_output, w2, num_tokens_per_expert],
            grad_scaled_hidden,
            ctx.module_path,
        )
        _record_grouped_mm(
            [scaled_hidden, grad_output, num_tokens_per_expert],
            dw2,
            ctx.module_path,
        )

        if isinstance(routed_scores, torch.Tensor):
            grad_activated_hidden = _uncaptured_empty_like(activated_hidden)
            drouted_scores = _uncaptured_empty_like(routed_scores)
            _record_op(
                "aten.mul.Tensor",
                [grad_scaled_hidden, routed_scores],
                grad_activated_hidden,
                ctx.module_path,
            )
            _record_op(
                "aten.mul.Tensor",
                [grad_scaled_hidden, activated_hidden],
                drouted_scores,
                ctx.module_path,
            )
        else:
            grad_activated_hidden = grad_scaled_hidden
            drouted_scores = None

        grad_pre_activation = _uncaptured_empty_like(pre_activation)
        _record_op(
            "npu.npu_swiglu_backward.default",
            [grad_activated_hidden, pre_activation],
            grad_pre_activation,
            ctx.module_path,
        )

        _record_grouped_mm(
            [grad_pre_activation, w13, num_tokens_per_expert],
            dx,
            ctx.module_path,
        )
        _record_grouped_mm(
            [x, grad_pre_activation, num_tokens_per_expert],
            dw13,
            ctx.module_path,
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
