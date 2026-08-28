# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom op for MoE token unpermutation.

Backward (both eager and compiled):
- probs present: native ``npu_moe_token_unpermute_grad`` (safe on zero-row
  inputs).
- probs=None (EP paths): exact ``scatter_add`` inverse — the native grad
  crashes on zero-row inputs (error 561002), and neither ``torch.cond`` nor
  a ``numel()==0`` guard survives the compile chain.
"""

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
    return permuted_tokens.new_empty(output_shape)


def _npu_moe_token_unpermute_setup_context(ctx, inputs, output):
    permuted_tokens, sorted_indices, probs = inputs
    ctx.save_for_backward(permuted_tokens, sorted_indices)
    ctx.probs = probs


def _npu_moe_token_unpermute_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None

    permuted_tokens, sorted_indices = ctx.saved_tensors
    probs = ctx.probs

    if probs is not None:
        # With probs present, CANN's MoeTokenUnpermuteGrad handles zero-row
        # inputs (verified: an empty (0, K) probs tensor passes). Keep the
        # native kernel for the weighted combine path.
        grad_tokens, grad_probs = torch_npu.npu_moe_token_unpermute_grad(
            permuted_tokens,
            grad_output,
            sorted_indices,
            probs=probs,
        )
        return grad_tokens, None, grad_probs

    # probs=None: a plain (unweighted) scatter; the backward is its exact
    # inverse gather-scatter. The native grad crashes on zero-row inputs
    # (error 561002), and a rank can legitimately receive zero tokens in EP.
    grad_tokens = torch.zeros_like(permuted_tokens)
    grad_tokens.scatter_add_(
        0,
        sorted_indices.unsqueeze(-1).expand(-1, permuted_tokens.size(1)),
        grad_output,
    )
    return grad_tokens, None, None


npu_moe_token_unpermute.register_autograd(
    _npu_moe_token_unpermute_backward,
    setup_context=_npu_moe_token_unpermute_setup_context,
)
