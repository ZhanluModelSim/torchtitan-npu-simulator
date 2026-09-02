# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom op for MoE token unpermutation.

Backward strategy, chosen by ``probs``:

- ``probs`` requires grad: native ``npu_moe_token_unpermute_grad`` with the
  saved ``permuted_tokens``, which is needed for ``grad_probs``.
- ``probs`` frozen (pre-W2 absorption passes ``ones_like``): the same kernel
  with a zero placeholder — ``grad_tokens`` does not read the forward values,
  so the GMM2/W2 output need not stay alive.
- ``probs is None`` (unweighted EP paths): the same native kernel under
  ``torch.cond`` — for non-empty inputs it is bitwise-equal to the scatter
  inverse and does not read forward values; CANN tiling rejects zero-row
  inputs (error 561002), so ranks with zero routed tokens take the zero
  branch. ``torch.cond`` keeps that runtime branch alive under the graph
  trainer's aot_fx_trace chain.
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
    ctx.probs = probs
    # permuted_tokens is only needed for grad_probs.
    if probs is not None and probs.requires_grad:
        ctx.save_for_backward(permuted_tokens, sorted_indices)
    else:
        ctx.save_for_backward(sorted_indices)


def _npu_moe_token_unpermute_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None

    probs = ctx.probs
    if probs is None:
        (sorted_indices,) = ctx.saved_tensors

        def native_branch():
            # grad_tokens does not read forward values in the unweighted path,
            # so grad_output doubles as the shape-only permuted_tokens argument.
            grad_tokens, _ = torch_npu.npu_moe_token_unpermute_grad(
                grad_output,
                grad_output,
                sorted_indices,
                probs=None,
            )
            return grad_tokens

        def zero_branch():
            # CANN tiling rejects num_out_tokens == 0 (error 561002): a rank
            # can legitimately receive zero routed tokens in EP.
            return torch.zeros_like(grad_output)

        grad_tokens = torch.cond(grad_output.numel() == 0, zero_branch, native_branch)
        return grad_tokens, None, None

    if probs.requires_grad:
        permuted_tokens, sorted_indices = ctx.saved_tensors
    else:
        # probs is (T, K); permuted_tokens is (T*K, D).
        permuted_tokens = grad_output.new_zeros((probs.numel(), grad_output.size(1)))
        (sorted_indices,) = ctx.saved_tensors

    grad_tokens, grad_probs = torch_npu.npu_moe_token_unpermute_grad(
        permuted_tokens,
        grad_output,
        sorted_indices,
        probs=probs,
    )
    return grad_tokens, None, grad_probs


npu_moe_token_unpermute.register_autograd(
    _npu_moe_token_unpermute_backward,
    setup_context=_npu_moe_token_unpermute_setup_context,
)
