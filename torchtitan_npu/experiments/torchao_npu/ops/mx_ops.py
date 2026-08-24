# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu

from ..quantization.quant_configs import MXQuantizeConfig
from ..quantization.quant_primitives.mx import mxfp4_dequantize

__all__ = [
    "mxfp4_fake_quantize",
    "mxfp8_dequantize",
    "to_mx_then_bmm",
    "to_mx_then_grouped_mm",
    "to_mx_then_mm",
]


def _to_npu_dtype_override(dtype: torch.dtype) -> int | None:
    """Return NPU dtype int for ``x1_dtype``/``x2_dtype``, or ``None`` to omit.

    ``npu_quant_matmul``'s ``x1_dtype``/``x2_dtype`` parameters must only be
    passed for dtypes where the tensor's storage dtype differs from its logical
    dtype.  ``float4_e2m1fn_x2`` is stored as ``uint8`` (two FP4 values per
    byte), so the kernel needs the explicit hint ``torch_npu.float4_e2m1fn_x2``.
    Standard FP8 types (``float8_e4m3fn``, ``float8_e5m2``) are natively
    represented by the tensor's dtype and **must not** be passed — doing so
    causes a ``RuntimeError`` under ``torch.compile``'s fake-tensor tracing.
    """
    if dtype == torch.float4_e2m1fn_x2:
        return torch_npu.float4_e2m1fn_x2
    return None


class MXFP4FakeQuantize(torch.autograd.Function):
    """MX FP4 fake quantize via autograd.Function.

    Forward:  quantize to FP4 via ``npu_dynamic_mx_quant``,
              then dequantize back via ``mxfp4_dequantize``.
    Backward: straight-through estimator (gradient passes through unchanged).

    Takes an ``MXQuantizeConfig`` as a single extra argument.
    """

    @staticmethod
    def forward(ctx, hp: torch.Tensor, config: MXQuantizeConfig, axis: int = -1):  # pyrefly: ignore [bad-override]
        weight_mx, w_scale = torch_npu.npu_dynamic_mx_quant(
            hp,
            dst_type=torch_npu.float4_e2m1fn_x2,
            axis=axis,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        return mxfp4_dequantize(
            weight_mx,
            w_scale,
            axis=axis,
            block_size=config.block_size,
            output_shape=hp.shape,
            output_dtype=hp.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        return grad_output, None, None


def mxfp4_fake_quantize(
    hp: torch.Tensor,
    config: MXQuantizeConfig,
    axis: int = -1,
) -> torch.Tensor:
    """Thin wrapper around ``MXFP4FakeQuantize.apply``."""
    return MXFP4FakeQuantize.apply(hp, config, axis)


def mxfp8_dequantize(
    data: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    block_size: int,
    output_shape: torch.Size,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP8 data from ``torch_npu.npu_dynamic_mx_quant`` output.

    Unlike ``mxfp4_dequantize``, FP8 uses 1 byte per value so no LUT-based
    nibble unpacking is needed. The scale format (E8M0, paired) is the same.

    Args:
        data: FP8 tensor (output y from npu_dynamic_mx_quant).
              Shape matches the original input shape.
        scale: uint8 E8M0 tensor (output mxscale_out).
               ndim = data.ndim + 1, with a trailing 2 dim packing scale pairs.
        axis: Quantization axis used in npu_dynamic_mx_quant.
        block_size: Block size used in npu_dynamic_mx_quant.
        output_dtype: Target dtype for the dequantized output.
    """
    assert output_shape[axis] % block_size == 0, f"quant dim must be divisible by block_size ({block_size})"

    orig_shape = output_shape
    pos_axis = axis if axis >= 0 else axis + data.ndim
    qdim = orig_shape[pos_axis]
    num_blocks = qdim // block_size

    # Convert FP8 → output dtype (no nibble unpacking needed)
    values = data.to(output_dtype)

    # Unpack scale (same paired E8M0 format as mxfp4_dequantize)
    s = scale.to(torch.uint8)
    s = s.movedim(-1, pos_axis + 1)
    s = s.flatten(pos_axis, pos_axis + 1)  # [..., packed_blocks*2, ...]
    s = s.narrow(pos_axis, 0, num_blocks)  # trim padding when num_blocks is odd

    # E8M0 uint8 → bf16 scale: 2^(e - 127)
    s = torch.exp2(s.to(torch.bfloat16) - 127.0)

    # Block-wise broadcast multiply
    values = values.unflatten(pos_axis, (num_blocks, block_size))
    s = s.unsqueeze(pos_axis + 1)

    result = values * s  # broadcasts over block_size
    result = result.reshape(orig_shape)

    return result.to(output_dtype)


# =========================================================================
# Real MX quantized matmul ops
# =========================================================================


class _MXQuantMM(torch.autograd.Function):
    """Per-axis MX quantized matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert A.shape[-2] % config_A.block_size == 0, (
            f"A.shape[-2]={A.shape[-2]} must be a multiple of config_A.block_size"
        )
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A with dual-axis MX quant ---
        A_q1, A_s1, A_q2, A_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            A.reshape(-1, A.shape[-1]),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: quantize B with dual-axis MX quant ---
        B_q1, B_s1, B_q2, B_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            B,
            round_mode=config_B.round_mode,
            dst_type=config_B.elem_dtype,
            scale_alg=config_B.scale_alg,
        )

        # --- Step 3: low-precision matmul, contracting over K ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q2,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_B.elem_dtype),
        )

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q1, B_s1)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q1, B_s1 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B

        # --- Step 1: quantize dY with dual-axis MX quant ---
        dY_q1, dY_s1, dY_q2, dY_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            dY.reshape(-1, dY.shape[-1]),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q1.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_B.elem_dtype),
        )

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q2.t(),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(0, 1),
            output_dtype=A_dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_A.elem_dtype),
        )

        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_mx_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized matrix multiply: ``A @ B``.

    ``A`` can be n-dimensional (n >= 2); leading dimensions are flattened
    before quantization and restored in the output.  ``B`` must be 2D.

    Quantization configs are drawn from ``config_A`` (for A and dY) and
    ``config_B`` (for B).

    Args:
        A: Shape ``(..., M, K)`` (n >= 2).
        B: Shape ``(K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(..., M, N)``, dtype matching ``A``.
    """
    return _MXQuantMM.apply(A, B, config_A, config_B)


class _MXQuantGroupedMM(torch.autograd.Function):
    """Per-axis MX quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        group_list: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A along K-dim (contracting dim for forward) ---
        A_q1, A_s1 = torch_npu.npu_dynamic_mx_quant(
            A,
            axis=-1,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            block_size=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: quantize A along M-dim with grouped quant (zero boundaries) ---
        A_q2, A_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            A,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: dual-axis quantize B (q1/s1 = N-dim, q2/s2 = K-dim) ---
        B_q1, B_s1, B_q2, B_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            B,
            round_mode=config_B.round_mode,
            dst_type=config_B.elem_dtype,
            scale_alg=config_B.scale_alg,
        )

        # --- Step 4: grouped low-precision matmul, group_type=0 (contract over K) ---
        Y = torch_npu.npu_grouped_matmul(
            [A_q1],
            [B_q2],
            scale=[B_s2],
            per_token_scale=[A_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=A.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q1, B_s1, group_list)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q1, B_s1, group_list = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        # --- Step 1: quantize dY along N-dim (for dgrad) ---
        dY_q1, dY_s1 = torch_npu.npu_dynamic_mx_quant(
            dY,
            axis=-1,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            block_size=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: quantize dY along M-dim with grouped quant (zero boundaries) ---
        dY_q2, dY_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dY,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: dgrad  dA = dY @ B^T  (group_type=0, contract over N) ---
        dA = torch_npu.npu_grouped_matmul(
            [dY_q1],
            [B_q1.transpose(-1, -2)],
            scale=[B_s1.transpose(1, 2)],
            per_token_scale=[dY_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=A_dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        # --- Step 4: wgrad  dB = A^T @ dY  (group_type=2, contract over M) ---
        dB = torch_npu.npu_grouped_matmul(
            [A_q2.t()],
            [dY_q2],
            scale=[dY_s2],
            per_token_scale=[A_s2.transpose(0, 1)],
            group_list=group_list.to(torch.int64),
            group_type=2,
            output_dtype=A_dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        return dA, dB, None, None, None


def to_mx_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``). Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``.

    Args:
        A: Shape ``(M, K)``.
        B: Shape ``(E, K, N)``.
        group_list: Cumulative row offsets per group, shape ``(E,)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _MXQuantGroupedMM.apply(A, B, group_list, config_A, config_B)


class _MXQuantBMM(torch.autograd.Function):
    """Per-axis MX quantized batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and share the leading batch dimension. The forward
    dual-axis quantizes both operands, then performs the low-precision matmul
    contracting over K. The backward reuses the forward-quantized ``A_q2/A_s2``
    and ``B_q1/B_s1`` transposed and only quantizes ``dY`` fresh.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[0] == B.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B[0]={B.shape[0]}"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"
        assert A.shape[-2] % config_A.block_size == 0, (
            f"A.shape[-2]={A.shape[-2]} must be a multiple of config_A.block_size"
        )
        assert B.shape[-2] % config_B.block_size == 0, (
            f"B.shape[-2]={B.shape[-2]} must be a multiple of config_B.block_size"
        )
        assert B.shape[-1] % config_B.block_size == 0, (
            f"B.shape[-1]={B.shape[-1]} must be a multiple of config_B.block_size"
        )

        # --- Step 1: dual-axis quantize A (left operand) ---
        A_q1, A_s1, A_q2, A_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            A,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: dual-axis quantize B (right operand) ---
        B_q1, B_s1, B_q2, B_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            B,
            round_mode=config_B.round_mode,
            dst_type=config_B.elem_dtype,
            scale_alg=config_B.scale_alg,
        )

        # --- Step 3: low-precision batched matmul, contracting over K ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q2,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_B.elem_dtype),
        )

        ctx.save_for_backward(A_q2, A_s2, B_q1, B_s1)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q1, B_s1 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B

        # --- Step 1: quantize dY with dual-axis MX quant ---
        dY_q1, dY_s1, dY_q2, dY_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            dY,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q1.transpose(-1, -2),
            B_s1.transpose(-2, -3),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_B.elem_dtype),
        )

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q2.transpose(-1, -2),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(-2, -3),
            output_dtype=A_dtype,
            group_sizes=[1, 1, config_A.block_size],
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=_to_npu_dtype_override(config_A.elem_dtype),
            x2_dtype=_to_npu_dtype_override(config_A.elem_dtype),
        )

        return dA, dB, None, None


def to_mx_then_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and must share the leading batch dimension.
    Quantization configs are drawn from ``config_A`` (for A and dY) and
    ``config_B`` (for B).

    Args:
        A: Shape ``(B, M, K)``.
        B: Shape ``(B, K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(B, M, N)``, dtype matching ``A``.
    """
    return _MXQuantBMM.apply(A, B, config_A, config_B)
