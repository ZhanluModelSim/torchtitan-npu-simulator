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
- ``probs is None`` (unweighted EP paths): an opaque backward op keeps the
  zero-row guard at runtime through compile and graph-trainer capture.
  Non-empty inputs use the native kernel; empty inputs skip its unsupported
  zero-row tiling path.
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


@torch.library.custom_op("torchtitan_npu::npu_moe_token_unpermute_grad_unweighted", mutates_args=())
def _npu_moe_token_unpermute_grad_unweighted(grad_output: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    # CANN tiling rejects zero rows. Keep this guard inside the opaque op:
    # a cond subgraph can emit out-of-scope symbols for dynamic EP sizes.
    if grad_output.numel() == 0:
        return torch.empty_like(grad_output)
    # The unweighted gradient does not read forward token values.
    grad_tokens, _ = torch_npu.npu_moe_token_unpermute_grad(grad_output, grad_output, sorted_indices, probs=None)
    return grad_tokens


@_npu_moe_token_unpermute_grad_unweighted.register_fake
def _npu_moe_token_unpermute_grad_unweighted_fake(grad_output, sorted_indices):
    return torch.empty_like(grad_output)


def _npu_moe_token_unpermute_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None

    probs = ctx.probs
    if probs is None:
        (sorted_indices,) = ctx.saved_tensors

        grad_tokens = _npu_moe_token_unpermute_grad_unweighted(grad_output, sorted_indices)
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
