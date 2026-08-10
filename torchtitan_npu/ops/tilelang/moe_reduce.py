# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registered TileLang-Ascend MoE combine/reduce operator."""

from __future__ import annotations

import torch

from torchtitan_npu.ops.tilelang.runtime import (
    get_cached_backward_kernel,
    get_cached_forward_kernel,
    require_raw_storage,
)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _normalize_forward_inputs(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"TileLang MoE reduce expects a 2-D input, got shape {tuple(x.shape)}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"TileLang MoE reduce does not support input dtype {x.dtype}")
    if token_topk_to_pos.ndim != 2:
        raise ValueError(
            f"TileLang MoE reduce expects a 2-D token_topk_to_pos tensor, got shape {tuple(token_topk_to_pos.shape)}"
        )
    if token_topk_to_pos.device != x.device:
        raise ValueError("TileLang MoE reduce input and routing metadata must be on the same device")

    # The custom-op implementation is the only owner of the TileLang ABI.
    # These calls are metadata-only for already conforming production inputs.
    return x.contiguous(), token_topk_to_pos.to(dtype=torch.int32).contiguous()


def _normalize_backward_inputs(
    token_topk_to_pos: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_topk_to_pos.ndim != 2:
        raise ValueError(
            f"TileLang MoE reduce backward expects 2-D routing metadata, got shape {tuple(token_topk_to_pos.shape)}"
        )
    if grad_output.ndim != 2:
        raise ValueError(
            f"TileLang MoE reduce backward expects a 2-D grad_output, got shape {tuple(grad_output.shape)}"
        )
    if token_topk_to_pos.shape[0] != grad_output.shape[0]:
        raise ValueError("TileLang MoE reduce backward routing rows must match grad_output rows")
    if token_topk_to_pos.device != grad_output.device:
        raise ValueError("TileLang MoE reduce backward tensors must be on the same device")
    if grad_output.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"TileLang MoE reduce backward does not support dtype {grad_output.dtype}")

    return token_topk_to_pos.to(dtype=torch.int32).contiguous(), grad_output.contiguous()


def _empty_kernel_argument(size: int | tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    return torch.empty(size, dtype=torch.float32, device=reference.device)


@torch.library.custom_op(
    "torchtitan_npu::npu_moe_reduce_fused_tilelang_backward",
    mutates_args=(),
)
def moe_reduce_fused_tilelang_backward_op(
    token_topk_to_pos: torch.Tensor,
    grad_output: torch.Tensor,
    num_expanded_tokens: int,
) -> torch.Tensor:
    token_topk_to_pos, grad_output = _normalize_backward_inputs(token_topk_to_pos, grad_output)
    num_tokens, num_topk = token_topk_to_pos.shape
    hidden = int(grad_output.shape[1])
    grad_x = grad_output.new_empty((num_expanded_tokens, hidden))

    if num_tokens == 0:
        return grad_x.zero_()

    backward_kernel = get_cached_backward_kernel(hidden, int(num_topk), grad_output.dtype)
    topk_weights = _empty_kernel_argument(num_tokens * num_topk, grad_output)
    sf = _empty_kernel_argument(1, grad_output)
    x_sf = _empty_kernel_argument(num_expanded_tokens, grad_output)
    dtopk_weights = _empty_kernel_argument(num_tokens * num_topk, grad_output)
    dx_sf = _empty_kernel_argument(num_expanded_tokens, grad_output)
    dsf = _empty_kernel_argument(1, grad_output)

    if grad_output.device.type == "npu":
        require_raw_storage(token_topk_to_pos, "token_topk_to_pos")
        require_raw_storage(grad_output, "grad_output")
        require_raw_storage(grad_x, "grad_x")

    # The compile-time unweighted/unscaled kernel never reads its x argument.
    # Reuse the output buffer as that ABI placeholder so backward neither saves
    # the forward activation nor allocates another [T*K, H] temporary tensor.
    backward_kernel(
        grad_x,
        topk_weights,
        token_topk_to_pos.view(-1),
        sf,
        x_sf,
        grad_output,
        grad_x,
        dtopk_weights,
        dx_sf,
        dsf,
    )
    return grad_x


@moe_reduce_fused_tilelang_backward_op.register_fake
def _moe_reduce_fused_tilelang_backward_fake(
    token_topk_to_pos: torch.Tensor,
    grad_output: torch.Tensor,
    num_expanded_tokens: int,
) -> torch.Tensor:
    return grad_output.new_empty((num_expanded_tokens, grad_output.shape[1]))


@torch.library.custom_op(
    "torchtitan_npu::npu_moe_reduce_fused_tilelang",
    mutates_args=(),
)
def moe_reduce_fused_tilelang_op(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    x, token_topk_to_pos = _normalize_forward_inputs(x, token_topk_to_pos)
    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    output = x.new_empty((num_tokens, hidden))

    if num_tokens == 0:
        return output

    forward_kernel = get_cached_forward_kernel(int(hidden), int(num_topk), x.dtype)
    topk_weights = _empty_kernel_argument(num_tokens * num_topk, x)
    sf = _empty_kernel_argument(1, x)
    x_sf = _empty_kernel_argument(num_expanded_tokens, x)

    if x.device.type == "npu":
        require_raw_storage(x, "x")
        require_raw_storage(token_topk_to_pos, "token_topk_to_pos")
        require_raw_storage(output, "output")

    forward_kernel(
        x,
        topk_weights,
        token_topk_to_pos.view(-1),
        sf,
        x_sf,
        output,
    )
    return output


@moe_reduce_fused_tilelang_op.register_fake
def _moe_reduce_fused_tilelang_fake(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((token_topk_to_pos.shape[0], x.shape[1]))


def _moe_reduce_fused_tilelang_setup_context(ctx, inputs, output) -> None:
    x, token_topk_to_pos = inputs
    ctx.save_for_backward(token_topk_to_pos)
    ctx.num_expanded_tokens = x.shape[0]


def _moe_reduce_fused_tilelang_backward(ctx, grad_output: torch.Tensor):
    (token_topk_to_pos,) = ctx.saved_tensors
    grad_x = moe_reduce_fused_tilelang_backward_op(
        token_topk_to_pos,
        grad_output,
        ctx.num_expanded_tokens,
    )
    return grad_x, None


moe_reduce_fused_tilelang_op.register_autograd(
    _moe_reduce_fused_tilelang_backward,
    setup_context=_moe_reduce_fused_tilelang_setup_context,
)


def tilelang_moe_reduce_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    """Run the registered zero-copy TileLang MoE reduce forward/backward."""

    return moe_reduce_fused_tilelang_op(x, token_topk_to_pos)
