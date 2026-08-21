# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""NPU MoE permutation, unpermutation, and rerouting APIs."""

import torch
import torch_npu


class NPUMoeTokenUnpermute(torch.autograd.Function):
    """functional npu_moe_token_unpermute"""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        if not permuted_tokens.numel():
            return permuted_tokens

        output, _, _, _ = torch_npu._npu_moe_token_unpermute_with_routing_map(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=None,
            routing_map=None,
            drop_and_pad=False,
        )
        ctx.restore_shape = restore_shape
        ctx.sorted_indices = sorted_indices

        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, unpermuted_tokens_grad):
        if not unpermuted_tokens_grad.numel():
            return unpermuted_tokens_grad, None, None
        if ctx.needs_input_grad[0]:
            sorted_indices = ctx.sorted_indices
            act_grad, _ = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                unpermuted_tokens_grad,
                sorted_indices,
                sorted_indices,
                routing_map=None,
                permuted_tokens=None,
                probs=None,
                drop_and_pad=False,
                restore_shape=ctx.restore_shape,
            )
            return act_grad, None, None

        return None, None, None


class NPUMoeReRouting(torch.autograd.Function):
    """Reroute EP all-to-all output with the NPU MoE rerouting kernel.

    This replaces the standard eager ExpertParallel local reroute sequence
    `repeat_interleave + npu_moe_token_permute` after EP all-to-all. The
    wrapper keeps the baseline ExpertParallel dispatcher in eager mode while
    using the Ascend NPU MoE rerouting kernel and a matching unpermute-based
    backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        routed_tokens: torch.Tensor,
        expert_token_num_per_rank: torch.Tensor,
        per_token_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
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

        # Integer argsort on NPU can fall back to AICPU; token ids are exact in
        # fp32 for the reroute sizes used here, so keep this construction on NPU.
        sort_input = (
            token_order_indices.to(torch.float32) if token_order_indices.numel() < (1 << 24) else token_order_indices
        )
        restore_indices = torch.argsort(sort_input).to(token_order_indices.dtype)
        ctx.save_for_backward(restore_indices)

        ctx.mark_non_differentiable(restore_indices, num_global_tokens_per_local_expert)

        if per_token_scales is None:
            permuted_scales = None

        return (
            permuted_tokens,
            permuted_scales,
            restore_indices,
            num_global_tokens_per_local_expert,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx,
        permuted_tokens_grad,
        permuted_scales_grad,
        _permuted_indices_grad,
        _num_global_tokens_per_local_expert_grad,
    ):
        (restore_indices,) = ctx.saved_tensors

        routed_tokens_grad = None
        if ctx.needs_input_grad[0] and permuted_tokens_grad is not None:
            # This custom backward returns a first-order gradient. The raw
            # NPU op has no Autograd dispatch registration, so never build a
            # second-order graph through it.
            with torch.no_grad():
                routed_tokens_grad = torch_npu.npu_moe_token_unpermute(
                    permuted_tokens_grad,
                    restore_indices,
                    None,
                )

        per_token_scales_grad = None
        if ctx.needs_input_grad[2] and permuted_scales_grad is not None:
            with torch.no_grad():
                per_token_scales_grad = torch_npu.npu_moe_token_unpermute(
                    permuted_scales_grad,
                    restore_indices,
                    None,
                )

        return routed_tokens_grad, None, per_token_scales_grad
