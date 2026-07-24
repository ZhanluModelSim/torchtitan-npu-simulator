# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monkey patch for torchao.prototype.mx_formats.mx_linear

This module patches mx_mm autograd function to support NPU by:
1. Adding NpuMXFP8MM autograd function for NPU-specific MX matmul
2. Patching _to_mxfp8_then_scaled_mm to use NPU path when available

AC Optimization:
    Uses dual-axis quant + ctx/stack bridge to avoid re-quantization in backward
    when activation checkpoint is enabled. Falls back to re-quantization when
    torch.compile is enabled (to avoid Dynamo HOP side-effect errors).
"""

import torch
import torch_npu

from torchtitan_npu.patches.torchao_npu.activation_checkpoint_state import (
    is_in_recomputation,
    pop_mx_quant_data,
    push_mx_quant_data,
)
from torchtitan_npu.patches.torchao_npu.mx_ac_config import (
    ac_enabled_context,
    get_ac_mode,
    should_save_bwd_quant_for_mx,
)


def view_as_n_dim(input_tensor, dim=2):
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1 :])
    return input_tensor


@torch._dynamo.allow_in_graph
class NpuMXFP8MM(torch.autograd.Function):
    """
    NPU-specific MX matmul autograd function.

    Uses torch_npu operations for efficient MX-format matrix multiplication
    on NPU hardware.

    AC Optimization:
        - When AC disabled: dual-axis quant in forward, ctx stores bwd data.
        - When AC enabled + compile disabled: stack bridge passes bwd data
          from recomputation forward to backward.
        - When AC enabled + compile enabled: falls back to re-quantization
          in backward (Dynamo-safe).
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight):
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        ctx.save_for_backward(x, weight)
        ctx.x_dtype = x.dtype

        save_bwd = should_save_bwd_quant_for_mx()
        in_recomp = is_in_recomputation()

        # Always use the same forward quantization path to ensure AC recomputation
        # produces identical output. The dual-axis quant may produce slightly
        # different forward results than single-axis quant on some NPU firmware.
        if save_bwd:
            x_mxfp8, x_scale, x_mxfp8_bwd, x_scale_bwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                view_as_n_dim(x), dst_type=torch.float8_e4m3fn, scale_alg=1
            )
            weight_mxfp8, weight_scale, weight_mxfp8_bwd, weight_scale_bwd = (
                torch_npu.npu_dynamic_mx_quant_with_dual_axis(weight, dst_type=torch.float8_e4m3fn, scale_alg=1)
            )

            if in_recomp:
                push_mx_quant_data(
                    {
                        "x_mxfp8_bwd": x_mxfp8_bwd,
                        "x_scale_bwd": x_scale_bwd,
                        "weight_mxfp8_bwd": weight_mxfp8_bwd,
                        "weight_scale_bwd": weight_scale_bwd,
                    }
                )
            else:
                # Normal forward (AC disabled): save directly on ctx.
                ctx.x_mxfp8_bwd = x_mxfp8_bwd
                ctx.x_scale_bwd = x_scale_bwd
                ctx.weight_mxfp8_bwd = weight_mxfp8_bwd
                ctx.weight_scale_bwd = weight_scale_bwd

        else:
            # AC enabled + normal forward: use single-axis quant for forward,
            # save original tensors for re-quantization in backward.
            x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(
                view_as_n_dim(x), axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
            )
            weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
                weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
            )

        output = torch_npu.npu_quant_matmul(
            x_mxfp8,
            weight_mxfp8.t(),
            weight_scale.transpose(0, 1),
            pertoken_scale=x_scale,
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(x.shape) != 2:
            output = output.reshape(*x.shape[:-1], *output.shape[1:])
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grads):
        x_dtype = ctx.x_dtype

        grads_mxfp8_di, grads_scale_di, grads_mxfp8_dw, grads_scale_dw = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            view_as_n_dim(grads), dst_type=torch.float8_e4m3fn, scale_alg=1
        )

        # Unpack saved tensors first to trigger recompute before popping stack.
        # This ensures the stack data corresponds to the current backward pass,
        # matching the Grouped-MM backward pattern.
        x, weight = ctx.saved_tensors

        stack_data = pop_mx_quant_data()
        if stack_data is not None:
            x_mxfp8_bwd = stack_data["x_mxfp8_bwd"]
            x_scale_bwd = stack_data["x_scale_bwd"]
            weight_mxfp8_bwd = stack_data["weight_mxfp8_bwd"]
            weight_scale_bwd = stack_data["weight_scale_bwd"]

        elif hasattr(ctx, "x_mxfp8_bwd"):
            x_mxfp8_bwd = ctx.x_mxfp8_bwd
            x_scale_bwd = ctx.x_scale_bwd
            weight_mxfp8_bwd = ctx.weight_mxfp8_bwd
            weight_scale_bwd = ctx.weight_scale_bwd

        else:
            weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
                weight, axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
            )

            dx = torch_npu.npu_quant_matmul(
                grads_mxfp8_di,
                weight_mxfp8,
                weight_scale,
                pertoken_scale=grads_scale_di,
                output_dtype=grads.dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
            if len(grads.shape) != 2:
                dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

            x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(
                view_as_n_dim(x), axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
            )

            dw = torch_npu.npu_quant_matmul(
                grads_mxfp8_dw.t(),
                x_mxfp8,
                x_scale,
                pertoken_scale=grads_scale_dw.transpose(0, 1),
                output_dtype=x_dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
            return dx, dw

        # Fast path: use pre-saved / stack quant data
        dx = torch_npu.npu_quant_matmul(
            grads_mxfp8_di,
            weight_mxfp8_bwd,
            weight_scale_bwd,
            pertoken_scale=grads_scale_di,
            output_dtype=grads.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        dw = torch_npu.npu_quant_matmul(
            grads_mxfp8_dw.t(),
            x_mxfp8_bwd,
            x_scale_bwd,
            pertoken_scale=grads_scale_dw.transpose(0, 1),
            output_dtype=x_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )

        return dx, dw


def _patched_to_mxfp8_then_scaled_mm(
    input_hp,
    weight_hp,
    kernel_preference=None,
    scale_calculation_mode=None,
    wgrad_with_hp=False,
) -> torch.Tensor:
    """
    Patched version of _to_mxfp8_then_scaled_mm that uses NPU path when available.

    Accepts extra keyword arguments (kernel_preference, scale_calculation_mode,
    wgrad_with_hp) for compatibility with MXFP8TrainingWeightWrapperTensor's
    __torch_function__ call site, which passes these through from config.

    AC state is read from the global get_ac_mode() (set in entry.py),
    consistent with GMM's approach.
    """
    ac_enabled = get_ac_mode() != "none"
    with ac_enabled_context(ac_enabled):
        return NpuMXFP8MM.apply(input_hp, weight_hp)


def apply_patches():
    """
    Apply patches to torchao.prototype.mx_formats.mx_linear.

    Patches _to_mxfp8_then_scaled_mm to use NPU-specific implementation when NPU is available.
    """
    import torchao.prototype.mx_formats.mx_linear as target_module

    # Store original function for potential rollback
    target_module._original_to_mxfp8_then_scaled_mm = target_module._to_mxfp8_then_scaled_mm

    # Apply the patch
    target_module._to_mxfp8_then_scaled_mm = _patched_to_mxfp8_then_scaled_mm


# Apply patches when this module is imported (skip if torchao is not installed)
try:
    apply_patches()
except ModuleNotFoundError:
    from torchtitan.tools.logging import logger

    logger.warning(
        "torchao is not installed, and mx_linear NPU patch is skipped. MXFP8 training features will not be available."
    )
