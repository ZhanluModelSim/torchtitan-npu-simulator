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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reroute rank-major EP output to expert-major layout."""

    (
        permuted_tokens,
        _permuted_scales,
        token_order_indices,
        num_global_tokens_per_local_expert,
    ) = torch_npu.npu_moe_re_routing(
        routed_tokens,
        expert_token_num_per_rank,
        expert_token_num_type=1,
        idx_type=0,
    )
    if num_global_tokens_per_local_expert.dtype != torch.int64:
        num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(torch.int64)

    # The native kernel returns the forward gather order. Its inverse is used
    # by the unpermute kernel in combine and in backward.
    restore_indices = torch.argsort(token_order_indices.to(torch.float32)).to(token_order_indices.dtype)
    return permuted_tokens, restore_indices, num_global_tokens_per_local_expert


@npu_moe_re_routing.register_fake
def _npu_moe_re_routing_fake(routed_tokens, expert_token_num_per_rank):
    num_tokens = routed_tokens.shape[0]
    return (
        torch.empty_like(routed_tokens),
        torch.empty((num_tokens,), dtype=torch.int32, device=routed_tokens.device),
        torch.empty(
            (expert_token_num_per_rank.shape[1],),
            dtype=torch.int64,
            device=expert_token_num_per_rank.device,
        ),
    )


def _npu_moe_re_routing_setup_context(ctx, inputs, output):
    _routed_tokens, _expert_token_num_per_rank = inputs
    _permuted_tokens, restore_indices, num_global_tokens_per_local_expert = output
    ctx.save_for_backward(restore_indices)
    ctx.mark_non_differentiable(restore_indices, num_global_tokens_per_local_expert)


def _npu_moe_re_routing_backward(
    ctx,
    grad_permuted_tokens,
    _grad_restore_indices,
    _grad_num_global_tokens_per_local_expert,
):
    if grad_permuted_tokens is None:
        return None, None

    (restore_indices,) = ctx.saved_tensors
    grad_routed_tokens = npu_moe_token_unpermute(
        grad_permuted_tokens,
        restore_indices,
        None,
    )
    return grad_routed_tokens, None


npu_moe_re_routing.register_autograd(
    _npu_moe_re_routing_backward,
    setup_context=_npu_moe_re_routing_setup_context,
)
