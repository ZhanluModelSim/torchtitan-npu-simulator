# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A5 TileKernels head-compute-mix custom op for AOT/Inductor tracing."""

from importlib import import_module

import torch

_A5_TILE_TOKEN_CHUNK = 512


def _tilelang_load_tile_kernels_op():
    try:
        tile_kernels_ops = import_module("tile_kernels.modeling.mhc.ops")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "A5 TileLang HcHead requires the tile_kernels package and a loaded CANN environment"
        ) from exc
    return tile_kernels_ops.mhc_head_compute_mix


def _tilelang_run_tile_kernels(
    input_mix: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if input_mix.ndim != 3 or not input_mix.is_contiguous():
        raise ValueError(
            "A5 TileLang HcHead mix input must be contiguous [batch, tokens, hc_mult], "
            f"got shape={tuple(input_mix.shape)}, stride={input_mix.stride()}"
        )

    original_shape = input_mix.shape
    flat_mix = input_mix.reshape(1, -1, input_mix.shape[-1]).contiguous()
    tile_kernels_op = _tilelang_load_tile_kernels_op()
    chunks = flat_mix.split(_A5_TILE_TOKEN_CHUNK, dim=1)
    output = torch.cat(
        [tile_kernels_op(chunk, hc_scale, hc_base, eps) for chunk in chunks],
        dim=1,
    )
    return output.reshape(original_shape)


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_head_compute_mix_a5",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_head_compute_mix_a5(
    input_mix: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the A5 TileKernels mix op behind a compile-safe custom-op schema."""
    return _tilelang_run_tile_kernels(input_mix, hc_scale, hc_base, eps)


@tilelang_mhc_head_compute_mix_a5.register_fake
def _tilelang_mhc_head_compute_mix_a5_fake(input_mix, hc_scale, hc_base, eps):
    del hc_scale, hc_base, eps
    return torch.empty_like(input_mix)


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_head_compute_mix_a5_bwd",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_head_compute_mix_a5_bwd(
    grad_output: torch.Tensor,
    input_mix: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the analytic sigmoid derivatives behind the external op."""
    pre_sigmoid = input_mix * hc_scale + hc_base
    sigmoid = torch.sigmoid(pre_sigmoid)
    derivative = grad_output * sigmoid * (1 - sigmoid)
    input_mix_grad = derivative * hc_scale
    hc_scale_grad = (derivative * input_mix).sum().reshape_as(hc_scale)
    hc_base_grad = derivative.sum(dim=(0, 1))
    return input_mix_grad, hc_scale_grad, hc_base_grad.reshape_as(hc_base)


@tilelang_mhc_head_compute_mix_a5_bwd.register_fake
def _tilelang_mhc_head_compute_mix_a5_bwd_fake(grad_output, input_mix, hc_scale, hc_base, eps):
    del grad_output, eps
    return (
        torch.empty_like(input_mix),
        torch.empty_like(hc_scale),
        torch.empty_like(hc_base),
    )


def _tilelang_mhc_head_compute_mix_a5_setup_context(ctx, inputs, output):
    input_mix, hc_scale, hc_base, eps = inputs
    del output
    ctx.save_for_backward(input_mix, hc_scale, hc_base)
    ctx.eps = eps


def _tilelang_mhc_head_compute_mix_a5_backward(ctx, grad_output):
    input_mix, hc_scale, hc_base = ctx.saved_tensors
    return (*tilelang_mhc_head_compute_mix_a5_bwd(grad_output, input_mix, hc_scale, hc_base, ctx.eps), None)


torch.library.register_autograd(
    tilelang_mhc_head_compute_mix_a5,
    _tilelang_mhc_head_compute_mix_a5_backward,
    setup_context=_tilelang_mhc_head_compute_mix_a5_setup_context,
)
