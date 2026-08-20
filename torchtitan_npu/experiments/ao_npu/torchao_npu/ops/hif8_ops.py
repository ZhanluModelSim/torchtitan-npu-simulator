# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-tensor HiF8 (``torch_npu.hifloat8``) low-precision matmul ops for NPU.
"""

import torch
import torch_npu

from ..quantization.quant_configs import HiF8QuantizeConfig


def _ensure_bf16_or_fp16(t: torch.Tensor) -> torch.Tensor:
    """``npu_dynamic_quant``'s pertensor path only accepts bf16/fp16 input."""
    if t.dtype not in (torch.float16, torch.bfloat16):
        return t.to(torch.bfloat16)
    return t


# =========================================================================
# Real per-tensor HiF8 quantized matmul
# =========================================================================


class _HiF8QuantMM(torch.autograd.Function):
    """Per-tensor HiF8 quantized matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: HiF8QuantizeConfig,
        config_B: HiF8QuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        A_flat = _ensure_bf16_or_fp16(A.reshape(-1, A.shape[-1]))
        B = _ensure_bf16_or_fp16(B)

        A_q, A_s = torch_npu.npu_dynamic_quant(A_flat, dst_type=config_A.elem_dtype, quant_mode="pertensor")
        B_q, B_s = torch_npu.npu_dynamic_quant(B, dst_type=config_B.elem_dtype, quant_mode="pertensor")

        Y = torch_npu.npu_quant_matmul(
            A_q,
            B_q,
            B_s,
            pertoken_scale=A_s,
            output_dtype=A_flat.dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_B.elem_dtype,
        )

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q, A_s, B_q, B_s)
        ctx.A_dtype = A_flat.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, dY: torch.Tensor):
        A_q, A_s, B_q, B_s = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B

        dY_flat = _ensure_bf16_or_fp16(dY.reshape(-1, dY.shape[-1]))
        dY_q, dY_s = torch_npu.npu_dynamic_quant(dY_flat, dst_type=config_A.elem_dtype, quant_mode="pertensor")

        # --- dgrad  dA = dY @ B^T  (contract over N) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q,
            B_q.t(),
            B_s,
            pertoken_scale=dY_s,
            output_dtype=A_dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_B.elem_dtype,
        )

        # --- wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q.t(),
            dY_q,
            dY_s,
            pertoken_scale=A_s,
            output_dtype=A_dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_A.elem_dtype,
        )

        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_hif8_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: HiF8QuantizeConfig,
    config_B: HiF8QuantizeConfig,
) -> torch.Tensor:
    """Per-tensor HiF8 quantized matrix multiply: ``A @ B``.

    ``A`` can be n-dimensional (n >= 2); leading dimensions are flattened
    before quantization and restored in the output. ``B`` must be 2D.

    Args:
        A: Shape ``(..., M, K)`` (n >= 2).
        B: Shape ``(K, N)``.
        config_A: Config for A's (and dY's) quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(..., M, N)``, dtype matching ``A``.
    """
    return _HiF8QuantMM.apply(A, B, config_A, config_B)


class _HiF8QuantGroupedMM(torch.autograd.Function):
    """Per-tensor HiF8 quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        group_list: torch.Tensor,
        config_A: HiF8QuantizeConfig,
        config_B: HiF8QuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        group_list = group_list.to(torch.int64)
        num_experts = B.shape[0]

        A_c = _ensure_bf16_or_fp16(A)
        B_c = _ensure_bf16_or_fp16(B)

        A_q, A_s = torch_npu.npu_dynamic_quant(A_c, dst_type=config_A.elem_dtype, quant_mode="pertensor")
        B_q, B_s = torch_npu.npu_dynamic_quant(B_c, dst_type=config_B.elem_dtype, quant_mode="pertensor")

        Y = torch_npu.npu_grouped_matmul(
            [A_q],
            [B_q],
            # group_type=0: scale (weight-role) broadcasts over experts.
            scale=[B_s.reshape(1).expand(num_experts).contiguous()],
            # group_type=0: per_token_scale (activation-role) broadcasts over tokens.
            per_token_scale=[A_s.reshape(1).expand(A.shape[0]).contiguous()],
            group_list=group_list,
            group_type=0,
            bias=None,
            split_item=3,
            output_dtype=A_c.dtype,
            group_list_type=0,
            x_dtype=config_A.elem_dtype,
            weight_dtype=config_B.elem_dtype,
        )[0]

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q, A_s, B_c, group_list)
        ctx.A_dtype = A_c.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, dY: torch.Tensor):
        A_q, A_s, B, group_list = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        num_experts = B.shape[0]

        # --- dgrad  dA = dY @ B^T  (group_type=0, contract over N) ---
        dY_c = _ensure_bf16_or_fp16(dY)
        B_t = B.transpose(-1, -2)
        dY_q, dY_s = torch_npu.npu_dynamic_quant(dY_c, dst_type=config_A.elem_dtype, quant_mode="pertensor")
        B_t_q, B_t_s = torch_npu.npu_dynamic_quant(B_t, dst_type=config_B.elem_dtype, quant_mode="pertensor")
        dA = torch_npu.npu_grouped_matmul(
            [dY_q],
            [B_t_q],
            bias=None,
            scale=[B_t_s.reshape(1).expand(num_experts).contiguous()],
            per_token_scale=[dY_s.reshape(1).expand(dY.shape[0]).contiguous()],
            group_list=group_list,
            group_type=0,
            split_item=3,
            output_dtype=A_dtype,
            group_list_type=0,
            x_dtype=config_A.elem_dtype,
            weight_dtype=config_B.elem_dtype,
        )[0]

        # --- dweight  dB = A^T @ dY  (group_type=2, split-K: BOTH scales broadcast
        # over num_experts, not over token count
        dB = torch_npu.npu_grouped_matmul(
            [A_q.t()],
            [dY_q],
            scale=[dY_s.reshape(1).expand(num_experts).contiguous()],
            per_token_scale=[A_s.reshape(1).expand(num_experts).contiguous()],
            group_list=group_list,
            group_type=2,
            bias=None,
            split_item=3,
            output_dtype=A_dtype,
            group_list_type=0,
            x_dtype=config_A.elem_dtype,
            weight_dtype=config_A.elem_dtype,
        )[0]

        return dA, dB, None, None, None


def to_hif8_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: HiF8QuantizeConfig,
    config_B: HiF8QuantizeConfig,
) -> torch.Tensor:
    """Per-tensor HiF8 quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``). Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``. ``E`` is read from ``B.shape[0]``.

    Args:
        A: Shape ``(M, K)``.
        B: Shape ``(E, K, N)``.
        group_list: Cumulative row offsets per group, shape ``(E,)``.
        config_A: Config for A's (and dY's) quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _HiF8QuantGroupedMM.apply(A, B, group_list, config_A, config_B)
