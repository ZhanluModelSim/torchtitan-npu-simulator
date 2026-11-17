# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native token permutation with a symbolic-size-safe backward."""

import torch
import torch_npu


@torch.library.custom_op("torchtitan_npu::npu_moe_token_permute", mutates_args=())
def npu_moe_token_permute(tokens: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_moe_token_permute(tokens, indices)


@npu_moe_token_permute.register_fake
def _permute_fake(tokens, indices):
    return tokens.new_empty((indices.numel(), tokens.shape[1])), indices.new_empty(indices.numel(), dtype=torch.int32)


@torch.library.custom_op("torchtitan_npu::npu_moe_token_permute_backward", mutates_args=())
def _permute_backward_op(
    grad: torch.Tensor, sorted_indices: torch.Tensor, num_tokens: int, dtype: torch.dtype, topk: int
) -> torch.Tensor:
    # The vendor schema takes int here. Keep SymInt in our graph boundary and
    # invoke the same kernel only once runtime dimensions have concrete values.
    return torch_npu.npu_moe_token_permute_grad_v2(grad, sorted_indices, num_tokens, dtype, topk)


@_permute_backward_op.register_fake
def _permute_backward_fake(grad, sorted_indices, num_tokens, dtype, topk):
    return grad.new_empty((num_tokens, grad.shape[1]), dtype=dtype)


def _setup_context(ctx, inputs, output):
    tokens, indices = inputs
    ctx.num_tokens = tokens.shape[0]
    ctx.dtype = tokens.dtype
    ctx.topk = indices.shape[1] if indices.ndim == 2 else 1
    ctx.save_for_backward(output[1])
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, grad, grad_indices):
    (sorted_indices,) = ctx.saved_tensors
    return _permute_backward_op(grad, sorted_indices, ctx.num_tokens, ctx.dtype, ctx.topk), None


npu_moe_token_permute.register_autograd(_backward, setup_context=_setup_context)
