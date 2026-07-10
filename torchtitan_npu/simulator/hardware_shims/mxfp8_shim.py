# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-safe shadow implementations of NpuMXFP8MM and NpuMXFP8GroupedMM.

These shims are used when ``_is_meta_simulation=True`` to:
1. Record the real NPU op names (``npu_dynamic_mx_quant``, ``npu_quant_matmul``,
   ``npu_grouped_matmul``) via ``record_synthetic_op`` so they appear in the
   captured L0 graph.
2. Use standard ``torch.matmul`` for shape inference on meta tensors (no
   real quantization or NPU kernel needed).

See ``meta_env._patch_mxfp8_for_meta`` for installation.
"""

from __future__ import annotations

import torch

from .dispatch_capture import get_active_capture


def _record_quant_and_matmul(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    is_grouped: bool = False,
) -> torch.Tensor:
    """Record npu_dynamic_mx_quant + npu_quant_matmul ops, return matmul result."""
    cap = get_active_capture()

    # Simulate quantization: shape stays the same, dtype changes to fp8
    # On meta device we just need the shape, so empty_like is sufficient
    x_quant = torch.empty_like(input_tensor)
    w_quant = torch.empty_like(weight)

    if cap is not None:
        cap.record_synthetic_op(
            "npu.npu_dynamic_mx_quant.default", [input_tensor], [x_quant]
        )
        cap.record_synthetic_op(
            "npu.npu_dynamic_mx_quant.default", [weight], [w_quant]
        )

    # Simulate matmul: standard matmul for shape inference
    if is_grouped:
        # Grouped matmul: input is 2D, weight is 3D (n_experts, in_features, out_features)
        # Use standard matmul per-group for shape inference
        out = torch.matmul(input_tensor.unsqueeze(0), weight.transpose(1, 2)).squeeze(0)
    else:
        out = torch.matmul(input_tensor, weight.t())

    if cap is not None:
        op_name = "npu.npu_grouped_matmul.default" if is_grouped else "npu.npu_quant_matmul.default"
        cap.record_synthetic_op(op_name, [x_quant, w_quant], [out])

    return out


class SimMXFP8MM(torch.autograd.Function):
    """Meta-safe shadow of NpuMXFP8MM for linear layers.

    Records ``npu_dynamic_mx_quant`` + ``npu_quant_matmul`` in forward,
    and the same pattern for dx/dw in backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight):
        out = _record_quant_and_matmul(x, weight, is_grouped=False)
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grads):
        x, weight = ctx.saved_tensors
        cap = get_active_capture()

        # dx = grads @ weight (with quant)
        dx = _record_quant_and_matmul(grads, weight.t(), is_grouped=False)

        # dw = grads.T @ x (with quant)
        dw = _record_quant_and_matmul(grads.t(), x, is_grouped=False)

        return dx, dw


class SimMXFP8GroupedMM(torch.autograd.Function):
    """Meta-safe shadow of NpuMXFP8GroupedMM for MoE expert layers.

    Records ``npu_dynamic_mx_quant`` + ``npu_grouped_matmul`` in forward,
    and the same pattern for dx/dw in backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, A, B_t, offs):
        # A: (total_tokens, in_features) — all tokens across all experts
        # B_t: (n_experts, out_features, in_features) — transposed weights
        # offs: offsets for splitting A into per-expert chunks
        out = _record_quant_and_matmul(A, B_t, is_grouped=True)
        ctx.save_for_backward(A, B_t, offs)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grads):
        A, B_t, offs = ctx.saved_tensors
        # dx: per-expert matmul with weight
        dx = _record_quant_and_matmul(grads, B_t.transpose(1, 2), is_grouped=True)
        # dw: per-expert weight gradient
        dw = _record_quant_and_matmul(grads.t(), A, is_grouped=True)
        return dx, dw, None
