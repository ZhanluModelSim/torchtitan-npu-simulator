# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monkey patch for torchao.prototype.moe_training.mxfp8_grouped_mm

This module patches _to_mxfp8_then_scaled_grouped_mm to support NPU by:
1. Adding NpuMXFP8GroupedMM autograd function for NPU-specific grouped MM
2. Patching _to_mxfp8_then_scaled_grouped_mm to use NPU path when available

Note: In torchao v0.17.0, _to_mxfp8_then_scaled_grouped_mm was moved from
scaled_grouped_mm.py to mxfp8_grouped_mm.py.
"""

import torch
import torch_npu
from einops import rearrange


@torch._dynamo.allow_in_graph
class NpuMXFP8GroupedMM(torch.autograd.Function):
    """
    NPU-specific MXFP8 grouped GEMM autograd function.

    Uses torch_npu operations for efficient MX-format grouped matrix multiplication
    on NPU hardware.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1)
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
        )

        return torch_npu.npu_grouped_matmul(
            [x_mxfp8],
            [weight_mxfp8],
            bias=None,
            scale=[weight_scale],
            per_token_scale=[x_scale],
            group_list=group_list.to(torch.int64),  # npu_grouped_matmul requires group_list to have dtype int64
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        grad_mxfp8, grad_scale = torch_npu.npu_dynamic_mx_quant(
            grad, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_mxfp8],
            [rearrange(weight_mxfp8, "n h f -> n f h")],
            bias=None,
            scale=[rearrange(weight_scale, "n h f g -> n f h g")],
            per_token_scale=[grad_scale],
            group_list=group_list.to(torch.int64),  # npu_grouped_matmul requires group_list to have dtype int64
            group_type=0,
            output_dtype=grad.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]
        x_mxfp8, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x,
            group_list.to(torch.int32),  # npu_grouped_dynamic_mx_quant requires group_list to have dtype int32
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            blocksize=32,
        )
        grad_mxfp8, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad,
            group_list.to(torch.int32),  # npu_grouped_dynamic_mx_quant requires group_list to have dtype int32
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            blocksize=32,
        )
        grad_weight = torch_npu.npu_grouped_matmul(
            [x_mxfp8.t()],
            [grad_mxfp8],
            bias=None,
            scale=[grad_scale],
            per_token_scale=[rearrange(x_scale, "n h f -> h n f")],
            group_list=group_list.to(torch.int64),  # npu_grouped_matmul requires group_list to have dtype int64
            group_type=2,
            output_dtype=x.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]
        return grad_input, grad_weight, None


def _patched_to_mxfp8_then_scaled_grouped_mm(
    A,
    B_t,
    offs=None,
    block_size=None,
    out_dtype=torch.bfloat16,
    kernel_preference=None,
    wgrad_with_hp=False,
    scale_calculation_mode=None,
    pad_token_groups_for_grouped_mm=False,
):
    """
    Patched version of _to_mxfp8_then_scaled_grouped_mm that uses NPU path when available.

    Accepts extra keyword arguments (block_size, out_dtype, kernel_preference,
    wgrad_with_hp, scale_calculation_mode, pad_token_groups_for_grouped_mm)
    for compatibility with MXFP8TrainingWeightWrapperTensor's
    __torch_function__ call site, which passes these through from config.
    """

    return NpuMXFP8GroupedMM.apply(A, B_t, offs)


def apply_patches():
    """
    Apply patches to torchao.prototype.moe_training.mxfp8_grouped_mm.

    Patches _to_mxfp8_then_scaled_grouped_mm to use NPU-specific implementation
    when NPU is available.

    Note: In torchao v0.17.0, this function was moved from scaled_grouped_mm.py
    to mxfp8_grouped_mm.py.
    """
    import torchao.prototype.moe_training as moe_training_pkg
    import torchao.prototype.moe_training.mxfp8_grouped_mm as target_module

    # Store original function for potential rollback
    target_module._original_to_mxfp8_then_scaled_grouped_mm = target_module._to_mxfp8_then_scaled_grouped_mm

    # Apply the patch to the mxfp8_grouped_mm module
    target_module._to_mxfp8_then_scaled_grouped_mm = _patched_to_mxfp8_then_scaled_grouped_mm

    # Also patch the reference in the parent package (__init__.py),
    # because _quantize_then_scaled_grouped_mm in utils.py imports
    # via `from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm`,
    # which resolves to the __init__.py namespace, not the mxfp8_grouped_mm module directly.
    moe_training_pkg._to_mxfp8_then_scaled_grouped_mm = _patched_to_mxfp8_then_scaled_grouped_mm


# Apply patches when this module is imported (skip if torchao is not installed)
try:
    apply_patches()
except ModuleNotFoundError:
    from torchtitan.tools.logging import logger

    logger.warning(
        "torchao is not installed, and mxfp8_grouped_mm NPU patch is skipped. "
        "MXFP8 MoE training features will not be available."
    )
