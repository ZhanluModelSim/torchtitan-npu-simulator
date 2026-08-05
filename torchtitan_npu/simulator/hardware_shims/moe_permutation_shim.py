# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autograd bridge for NPU MoE token permutation on meta tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

if TYPE_CHECKING:
    from collections.abc import Callable


class _SimMoeTokenPermute(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        run_forward,
        tokens,
        indices,
        num_out_tokens,
        padded_mode,
    ):
        permuted_tokens, sorted_indices = run_forward(
            tokens,
            indices,
            num_out_tokens=num_out_tokens,
            padded_mode=padded_mode,
        )
        ctx.save_for_backward(tokens, indices, sorted_indices)
        ctx.padded_mode = padded_mode
        ctx.mark_non_differentiable(sorted_indices)
        return permuted_tokens, sorted_indices

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_permuted_tokens, _grad_sorted_indices):
        tokens, indices, sorted_indices = ctx.saved_tensors
        grad_tokens = None
        if ctx.needs_input_grad[1] and grad_permuted_tokens is not None:
            grad_tokens = torch_npu.npu_moe_token_permute_grad(
                tokens,
                grad_permuted_tokens,
                indices,
                sorted_indices,
                padded_mode=ctx.padded_mode,
            )
        return None, grad_tokens, None, None, None


def run_meta_moe_token_permute(
    run_forward: Callable,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int | None = None,
    padded_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _SimMoeTokenPermute.apply(
        run_forward,
        tokens,
        indices,
        num_out_tokens,
        padded_mode,
    )
