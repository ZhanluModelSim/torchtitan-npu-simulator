# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom-op boundary for the optional TileKernels HcPost op."""

import importlib

import torch


def _load_tile_kernels():
    try:
        # Keep this optional dependency out of the static import graph.  The
        # package is supplied by the target CANN/TileLang environment and is
        # intentionally absent from the repository lint environment.
        tile_kernels = importlib.import_module("tile_kernels.mhc.post_kernel")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileLang HcPost requires the installed tile_kernels package and a loaded CANN environment"
        ) from error
    return tile_kernels.mhc_post_fwd, tile_kernels.mhc_post_bwd


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_post_fwd",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Run the TileKernels forward kernel behind a compile-safe schema."""
    mhc_post_fwd, _ = _load_tile_kernels()
    return mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix)


@tilelang_mhc_post_fwd.register_fake
def _tilelang_mhc_post_fwd_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual)


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_post_bwd",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the TileKernels backward kernel behind a compile-safe schema."""
    _, mhc_post_bwd = _load_tile_kernels()
    return mhc_post_bwd(x, residual, post_layer_mix, comb_res_mix, grad_output)


@tilelang_mhc_post_bwd.register_fake
def _tilelang_mhc_post_bwd_fake(x, residual, post_layer_mix, comb_res_mix, grad_output):
    return (
        torch.empty_like(x),
        torch.empty_like(residual),
        torch.empty_like(post_layer_mix),
        torch.empty_like(comb_res_mix),
    )


def _tilelang_mhc_post_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _tilelang_mhc_post_backward(ctx, grad_output):
    return (*tilelang_mhc_post_bwd(*ctx.saved_tensors, grad_output),)


tilelang_mhc_post_fwd.register_autograd(
    _tilelang_mhc_post_backward,
    setup_context=_tilelang_mhc_post_setup_context,
)


def tilelang_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Return the TileLang HcPost result for eager and compiled callers."""
    return tilelang_mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix)
