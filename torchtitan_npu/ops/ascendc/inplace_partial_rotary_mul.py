# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4 partial-RoPE: functional autograd.Function over the CANN inplace kernel."""

from __future__ import annotations

import cann_ops_transformer.ops.inplace_partial_rotary_mul  # noqa: F401  (registers the torch.ops below)
import torch

__all__ = ["inplace_partial_rotary_mul"]


class _PartialRotaryMulFn(torch.autograd.Function):
    """Rotate a fresh clone and return it; the caller's tensor is never touched.

    ``r1``/``r2`` are saved as passed to the forward, so inverse sites (whose
    callers already negate the sine) need no special casing — the conjugate
    of the negated rotation is their exact transpose.
    """

    @staticmethod
    def forward(ctx, x, r1, r2, rotary_mode, partial_slice):  # pyrefly: ignore [bad-override]
        ctx.save_for_backward(r1, r2)
        ctx.rotary_mode = rotary_mode
        ctx.partial_slice = partial_slice
        output = x.clone(memory_format=torch.contiguous_format)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            output,
            r1,
            r2,
            rotary_mode=rotary_mode,
            partial_slice=partial_slice,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        if grad_output is None:
            return None, None, None, None, None
        r1, r2 = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul_backward(
            grad_input,
            r1,
            r2,
            rotary_mode=ctx.rotary_mode,
            partial_slice=ctx.partial_slice,
        )
        return grad_input, None, None, None, None


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: list[int] | None = None,
) -> torch.Tensor:
    """Return a partially-rotated clone of ``x`` (CANN inplace kernel)."""
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    return _PartialRotaryMulFn.apply(x, r1, r2, rotary_mode, partial_slice)
