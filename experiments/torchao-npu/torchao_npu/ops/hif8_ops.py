# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-tensor HiF8 low-precision matmul operators for NPU.

Current/constant tensor scaling (CTS) only: every quantize call recomputes
its scale fresh via ``torch_npu.npu_dynamic_quant``. There is no persistent
per-parameter scale state and no delayed/windowed scaling.
"""

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig
from torchao_npu.quantization.quant_primitives.hif8 import quantize_hifloat8


def _ensure_bf16_or_fp16(t: torch.Tensor) -> torch.Tensor:
    """``npu_dynamic_quant``'s per-tensor path accepts bf16/fp16 inputs."""
    if t.dtype not in (torch.float16, torch.bfloat16):
        return t.to(torch.bfloat16)
    return t


def _per_expert_scale(scale: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Per-tensor ``scale`` as one ``npu_grouped_matmul`` scale-list entry."""
    return scale.reshape(1).expand(num_experts).contiguous()


def _per_expert_scale_product(a: torch.Tensor, b: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``a * b`` (both per-tensor) as one ``npu_grouped_matmul`` scale entry.

    ``(a * b).expand(E).contiguous()`` costs a multiply plus a copy; the
    broadcasting multiply below writes the E-long contiguous result in a
    single kernel and is bit-identical (every lane is the same scalar
    product of the same two fp32 values).
    """
    return a.reshape(1).expand(num_experts) * b.reshape(1)


class _HiF8QuantMM(torch.autograd.Function):
    """Per-tensor HiF8 matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``."""

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

        A_q, A_s = quantize_hifloat8(A_flat, config_A)
        B_q, B_s = quantize_hifloat8(B, config_B)

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

        dY_q, dY_s = quantize_hifloat8(dY.reshape(-1, dY.shape[-1]), config_A)

        dA = torch_npu.npu_quant_matmul(
            dY_q,
            B_q.t(),
            B_s,
            pertoken_scale=dY_s,
            output_dtype=A_dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_B.elem_dtype,
        )

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
    """Run per-tensor HiF8 matrix multiplication ``A @ B``.

    ``A`` may have leading dimensions; they are flattened during the
    operation and restored in the result. ``B`` must be two-dimensional.
    """
    return _HiF8QuantMM.apply(A, B, config_A, config_B)


class _HiF8QuantGroupedMM(torch.autograd.Function):
    """Per-tensor HiF8 grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``."""

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
        B_dtype = B.dtype

        A_q, A_s = quantize_hifloat8(A_c, config_A)
        B_q, B_s = quantize_hifloat8(B, config_B)

        # scale slot: [E], the per-tensor weight scale broadcast per expert,
        # times the activation's per-tensor scale.
        fwd_scale = _per_expert_scale_product(A_s, B_s, num_experts)

        Y = torch_npu.npu_grouped_matmul(
            [A_q],
            [B_q],
            scale=[fwd_scale],
            per_token_scale=None,
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

        ctx.B_dtype = B_dtype
        ctx.save_for_backward(A_q, A_s, B_q, B_s, group_list)
        ctx.A_dtype = A_c.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, dY: torch.Tensor):
        A_q, A_s, B_q, B_s, group_list = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        num_experts = B_q.shape[0]

        dY_q, dY_s = quantize_hifloat8(dY, config_A)

        # B's per-tensor scale is unchanged by transpose (a single global
        # scalar), so B_q's transpose is reused directly instead of
        # re-quantizing B_t.
        B_t_q = B_q.transpose(-1, -2)
        B_t_s = B_s

        dgrad_scale = _per_expert_scale_product(dY_s, B_t_s, num_experts)

        dA = torch_npu.npu_grouped_matmul(
            [dY_q],
            [B_t_q],
            bias=None,
            scale=[dgrad_scale],
            per_token_scale=None,
            group_list=group_list,
            group_type=0,
            split_item=3,
            output_dtype=A_dtype,
            group_list_type=0,
            x_dtype=config_A.elem_dtype,
            weight_dtype=config_B.elem_dtype,
        )[0]

        wgrad_scale = _per_expert_scale(dY_s, num_experts)
        wgrad_pertoken = _per_expert_scale(A_s, num_experts)

        dB = torch_npu.npu_grouped_matmul(
            [A_q.t()],
            [dY_q],
            scale=[wgrad_scale],
            per_token_scale=[wgrad_pertoken],
            group_list=group_list,
            group_type=2,
            bias=None,
            split_item=3,
            output_dtype=A_dtype,
            group_list_type=0,
            x_dtype=config_A.elem_dtype,
            weight_dtype=config_A.elem_dtype,
        )[0]

        if dB.dtype != ctx.B_dtype:
            dB = dB.to(ctx.B_dtype)
        return dA, dB, None, None, None


def to_hif8_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: HiF8QuantizeConfig,
    config_B: HiF8QuantizeConfig,
) -> torch.Tensor:
    """Run per-tensor HiF8 grouped matmul ``A @ B``.

    ``group_list`` contains cumulative row offsets for the expert groups.
    """
    return _HiF8QuantGroupedMM.apply(A, B, group_list, config_A, config_B)


class _HiF8QuantBMM(torch.autograd.Function):
    """Per-tensor HiF8 batched matmul: ``A[b,M,K] @ B[b,K,N] = Y[b,M,N]``.

    Both operands are 3D and share the leading batch dimension. Forward
    per-tensor quantizes both operands and runs the low-precision batched
    matmul contracting over K. Backward reuses the forward-quantized
    ``A_q/B_q`` transposed and only quantizes ``dY`` fresh -- the same scheme
    as :class:`_HiF8QuantMM`.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: HiF8QuantizeConfig,
        config_B: HiF8QuantizeConfig,
    ):
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[0] == B.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B[0]={B.shape[0]}"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        A_c = _ensure_bf16_or_fp16(A)

        A_q, A_s = quantize_hifloat8(A_c, config_A)
        B_q, B_s = quantize_hifloat8(B, config_B)

        Y = torch_npu.npu_quant_matmul(
            A_q,
            B_q,
            B_s,
            pertoken_scale=A_s,
            output_dtype=A_c.dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_B.elem_dtype,
        )

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q, A_s, B_q, B_s)
        ctx.A_dtype = A_c.dtype
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
        assert dY.ndim == 3, f"dY must be 3D, got {dY.ndim}D"

        dY_q, dY_s = quantize_hifloat8(dY, config_A)

        # dgrad  dA = dY @ B^T  (contract over N)
        dA = torch_npu.npu_quant_matmul(
            dY_q,
            B_q.transpose(-1, -2),
            B_s,
            pertoken_scale=dY_s,
            output_dtype=A_dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_B.elem_dtype,
        )

        # wgrad  dB = A^T @ dY  (contract over M)
        dB = torch_npu.npu_quant_matmul(
            A_q.transpose(-1, -2),
            dY_q,
            dY_s,
            pertoken_scale=A_s,
            output_dtype=A_dtype,
            x1_dtype=config_A.elem_dtype,
            x2_dtype=config_A.elem_dtype,
        )

        return dA, dB, None, None


def to_hif8_then_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: HiF8QuantizeConfig,
    config_B: HiF8QuantizeConfig,
) -> torch.Tensor:
    """Run per-tensor HiF8 batched matmul ``A @ B``.

    Both operands must be 3D and share the leading batch dimension.
    Quantization configs are drawn from ``config_A`` (for A and dY) and
    ``config_B`` (for B).
    """
    return _HiF8QuantBMM.apply(A, B, config_A, config_B)
