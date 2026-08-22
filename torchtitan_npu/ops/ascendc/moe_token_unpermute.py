# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom op for MoE token unpermutation."""

__all__ = ["npu_moe_token_unpermute"]

import torch
import torch_npu


@torch.library.custom_op(
    "torchtitan_npu::npu_moe_token_unpermute",
    mutates_args=(),
)
def npu_moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore token order, optionally applying routing probabilities."""

    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens,
        sorted_indices,
        probs,
    )


@npu_moe_token_unpermute.register_fake
def _npu_moe_token_unpermute_fake(permuted_tokens, sorted_indices, probs=None):
    del sorted_indices
    if probs is None:
        return torch.empty_like(permuted_tokens)

    output_shape = (*probs.shape[:-1], *permuted_tokens.shape[1:])
    return torch.empty(
        output_shape,
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )


def _npu_moe_token_unpermute_setup_context(ctx, inputs, output):
    permuted_tokens, sorted_indices, probs = inputs
    ctx.save_for_backward(permuted_tokens, sorted_indices)
    ctx.probs = probs


def _npu_moe_token_unpermute_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None

    permuted_tokens, sorted_indices = ctx.saved_tensors
    grad_tokens, grad_probs = torch_npu.npu_moe_token_unpermute_grad(
        permuted_tokens,
        grad_output,
        sorted_indices,
        probs=ctx.probs,
    )
    # Some torch_npu versions return a placeholder grad for omitted probs.
    return grad_tokens, None, grad_probs if ctx.probs is not None else None


npu_moe_token_unpermute.register_autograd(
    _npu_moe_token_unpermute_backward,
    setup_context=_npu_moe_token_unpermute_setup_context,
)
