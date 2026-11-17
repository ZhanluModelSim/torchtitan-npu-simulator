# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom op for EP MoE re-routing."""

__all__ = ["npu_moe_re_routing"]

import torch
import torch_npu

from .moe_token_unpermute import npu_moe_token_unpermute


@torch.library.custom_op(
    "torchtitan_npu::npu_moe_re_routing",
    mutates_args=(),
)
def npu_moe_re_routing(
    routed_tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reroute rank-major EP output to expert-major layout.

    ``per_token_scales`` (optional) follows the same rank-major -> expert-major
    permutation as ``routed_tokens`` and is returned as ``permuted_scales``.
    The third output is the inverse permutation used by the matching unpermute
    kernel in combine and in backward.
    """

    (
        permuted_tokens,
        permuted_scales,
        token_order_indices,
        num_global_tokens_per_local_expert,
    ) = torch_npu.npu_moe_re_routing(
        routed_tokens,
        expert_token_num_per_rank,
        per_token_scales=per_token_scales,
        expert_token_num_type=1,
        idx_type=0,
    )
    if num_global_tokens_per_local_expert.dtype != torch.int64:
        num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(torch.int64)

    # The native kernel returns the forward gather order. Its inverse is used
    # by the unpermute kernel in combine and in backward.
    restore_indices = torch.argsort(token_order_indices.to(torch.float32)).to(token_order_indices.dtype)
    return permuted_tokens, permuted_scales, restore_indices, num_global_tokens_per_local_expert


@npu_moe_re_routing.register_fake
def _npu_moe_re_routing_fake(
    routed_tokens,
    expert_token_num_per_rank,
    per_token_scales=None,
):
    num_tokens = routed_tokens.shape[0]
    if per_token_scales is None:
        permuted_scales = routed_tokens.new_empty(0)
    else:
        permuted_scales = per_token_scales.new_empty((num_tokens,))
    return (
        torch.empty_like(routed_tokens),
        permuted_scales,
        routed_tokens.new_empty((num_tokens,), dtype=torch.int32),
        expert_token_num_per_rank.new_empty(
            (expert_token_num_per_rank.shape[1],),
            dtype=torch.int64,
        ),
    )


def _npu_moe_re_routing_setup_context(ctx, inputs, output):
    _routed_tokens, _expert_token_num_per_rank, per_token_scales = inputs
    _permuted_tokens, permuted_scales, restore_indices, num_global_tokens_per_local_expert = output
    ctx.save_for_backward(restore_indices)
    ctx.mark_non_differentiable(restore_indices, num_global_tokens_per_local_expert)
    ctx.has_scales = per_token_scales is not None
    if per_token_scales is None:
        ctx.mark_non_differentiable(permuted_scales)


def _npu_moe_re_routing_backward(
    ctx,
    grad_permuted_tokens,
    grad_permuted_scales,
    _grad_restore_indices,
    _grad_num_global_tokens_per_local_expert,
):
    if grad_permuted_tokens is None and grad_permuted_scales is None:
        return None, None, None

    (restore_indices,) = ctx.saved_tensors
    grad_routed_tokens = None
    if grad_permuted_tokens is not None:
        grad_routed_tokens = npu_moe_token_unpermute(
            grad_permuted_tokens,
            restore_indices,
            None,
        )

    grad_per_token_scales = None
    if ctx.has_scales and grad_permuted_scales is not None:
        grad_per_token_scales = npu_moe_token_unpermute(
            grad_permuted_scales.reshape(-1, 1),
            restore_indices,
            None,
        ).reshape(-1)
    return grad_routed_tokens, None, grad_per_token_scales


npu_moe_re_routing.register_autograd(
    _npu_moe_re_routing_backward,
    setup_context=_npu_moe_re_routing_setup_context,
)
