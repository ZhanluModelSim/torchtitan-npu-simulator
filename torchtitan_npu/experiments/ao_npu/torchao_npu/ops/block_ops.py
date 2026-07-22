# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block FP8 low-precision matmul ops for NPU.

Quantization scheme (non-grouped ``to_block_fp8_then_mm``):
- A (activation):  ``npu_dynamic_mx_quant_with_dual_axis``
  - s1 = per-32-group scale along dim -1  (``A_s1``)
  - s2 = per-32-group scale along dim -2  (``A_s2``)
- B (weight):      ``npu_dynamic_block_mx_quant``
  - s1 = per-32-group scale along dim -1  (``B_s1``)
  - s2 = per-32-group scale along dim -2  (``B_s2``)

Quantization scheme (grouped ``to_block_fp8_then_grouped_mm``):
- A (activation), K-dim:    ``npu_dynamic_mx_quant(axis=-1)``
- A (activation), M-dim:    ``npu_grouped_dynamic_mx_quant``
- B (weight):               ``npu_dynamic_block_mx_quant``
- dY, N-dim:                ``npu_dynamic_mx_quant(axis=-1)``
- dY, M-dim:                ``npu_grouped_dynamic_mx_quant``
"""

import torch
import torch_npu

from ..quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from ..quantization.quant_primitives.block_fp8 import quantize_right_operand


class _BlockFP8QuantMM(torch.autograd.Function):
    """Block FP8 matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``.

    Forward:
      1. ``A_q1, A_s1, A_q2, A_s2 = npu_dynamic_mx_quant_with_dual_axis(A, **config_A)``
         -- quantize A along both axes.  A_s1 = K-dim scale ``[M, ceil(ceil(K/32)/2), 2]``.
      2. ``B_q, B_s1, B_s2 = npu_dynamic_block_mx_quant(B, **config_B)``
         -- quantize B in 32×32 blocks.  B_s2 = K-dim scale ``[ceil(ceil(K/32)/2), N, 2]``.
      3. ``Y = npu_quant_matmul(A_q1, B_q, pertoken_scale=A_s1, scale=B_s2)``
         -- low-precision matmul contracting over K.

    Backward (dY = grad_output):
      1. ``dY_q1, dY_s1, dY_q2, dY_s2 = npu_dynamic_mx_quant_with_dual_axis(dY, **config_A)``
         -- quantize dY along both axes.
      2. dgrad: ``dA = dY @ B^T`` (contract over N)
         x1 = dY_q1, x2 = B_q.t()
         pertoken_scale = dY_s1  ``[M, ceil(ceil(N/32)/2), 2]``  (N-dim of dY)
         scale          = B_s1.transpose(0,1)  ``[ceil(ceil(N/32)/2), K, 2]``  (N-dim of B)
      3. wgrad: ``dB = A^T @ dY`` (contract over M)
         x1 = A_q2.t(), x2 = dY_q2
         pertoken_scale = A_s2.transpose(0,1)  ``[K, ceil(ceil(M/32)/2), 2]``  (M-dim of A)
         scale          = dY_s2  ``[ceil(ceil(M/32)/2), N, 2]``  (M-dim of dY)
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockQuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert A.shape[-2] % config_A.block_size == 0, (
            f"A.shape[-2]={A.shape[-2]} must be a multiple of config_A.block_size"
        )
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A with dual-axis MX quant ---
        # Flatten leading dims inside the quant call (like MXfp8MM uses view_as_n_dim)
        A_q1, A_s1, A_q2, A_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            A.reshape(-1, A.shape[-1]),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )

        # --- Step 2: block FP8 quantize B (with optional mxfp4 fake-quant pre-pass) ---
        # B [K, N] → B_s1 (N-dim scale) [K, ceil(ceil(N/32)/2), 2]
        #           → B_s2 (K-dim scale) [ceil(ceil(K/32)/2), N, 2]
        B_q, B_s1, B_s2 = quantize_right_operand(B, axis=-2, config=config_B)

        # --- Step 3: low-precision matmul, contracting over K ---
        # x1 = A_q1 [M, K] → pertoken_scale = A_s1 [M, ceil(ceil(K/32)/2), 2]
        # x2 = B_q  [K, N] → scale          = B_s2 [ceil(ceil(K/32)/2), N, 2]
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, config_B.block_size],
        )

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        Y.requires_grad_(A.requires_grad or B.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s1)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s1 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        config_B = ctx.config_B

        # --- Step 1: quantize dY with dual-axis MX quant ---
        # Flatten leading dims inside the quant call (like MXfp8MM)
        dY_q1, dY_s1, dY_q2, dY_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            dY.reshape(-1, dY.shape[-1]),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N) ---
        # x1 = dY_q1 [M, N] → pertoken_scale = dY_s1 [M, ceil(ceil(N/32)/2), 2]
        # x2 = B_q.t() [N, K] → scale = B_s1.transpose(0,1) [ceil(ceil(N/32)/2), K, 2]
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, config_B.block_size],
        )

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        # x1 = A_q2.t() [K, M] → pertoken_scale = A_s2.transpose(0,1) [K, ceil(ceil(M/32)/2), 2]
        # x2 = dY_q2 [M, N]   → scale          = dY_s2 [ceil(ceil(M/32)/2), N, 2]
        # group_sizes[2] assumes config_A.block_size == config_B.block_size and uses config_B.
        dB = torch_npu.npu_quant_matmul(
            A_q2.t(),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(0, 1),
            output_dtype=A_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, config_B.block_size],
        )

        # Restore dA to original shape
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_block_fp8_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockQuantizeConfig,
) -> torch.Tensor:
    """Block FP8 matrix multiply: ``A @ B`` with mixed-format quantization.

    Quantization scheme:
    - A: ``npu_dynamic_mx_quant_with_dual_axis`` (params from ``config_A``)
    - B: ``npu_dynamic_block_mx_quant``, 32×32 blocks (params from ``config_B``)

    See ``_BlockFP8QuantMM`` for details of the three GEMMs.

    Args:
        A: Shape ``(M, K)``.
        B: Shape ``(K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _BlockFP8QuantMM.apply(A, B, config_A, config_B)


class _BlockFP8QuantGroupedMM(torch.autograd.Function):
    """Block FP8 grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``).  Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``.

    Quantization:
    - A's K-dim (contraction):  ``npu_dynamic_mx_quant(axis=-1)`` (params from ``config_A``)
    - A's M-dim (for wgrad):    ``npu_grouped_dynamic_mx_quant`` (zero-boundary padded;
                                params from ``config_A``)
    - B:                        ``npu_dynamic_block_mx_quant``, 32×32 blocks
                                (params from ``config_B``)

    Forward:
      1. ``A_q1, A_s1 = npu_dynamic_mx_quant(A, axis=-1, **config_A)``
         -- quantize A along K.  A_s1 = K-dim scale ``[M, ceil(K/64), 2]``.
      2. ``A_q2, A_s2 = npu_grouped_dynamic_mx_quant(A, group_list, **config_A)``
         -- quantize A along M with zero-boundary padding.
            A_s2 = M-dim scale ``[M//64+E, K, 2]``.
      3. ``B_q, B_s1, B_s2 = npu_dynamic_block_mx_quant(B, **config_B)``
         -- quantize B in 32×32 blocks.
            B_s2 = K-dim scale ``[E, ceil(K/64), N, 2]``.
      4. ``Y = npu_grouped_matmul([A_q1], [B_q], group_type=0,
             per_token_scale=[A_s1], scale=[B_s2])``
         -- grouped low-precision matmul contracting over K.

    Backward (dY = grad_output):
      1. ``dY_q1, dY_s1 = npu_dynamic_mx_quant(dY, axis=-1, **config_A)``
         -- quantize dY along N.  dY_s1 = N-dim scale ``[M, ceil(N/64), 2]``.
      2. ``dY_q2, dY_s2 = npu_grouped_dynamic_mx_quant(dY, group_list, **config_A)``
         -- quantize dY along M with zero-boundary padding.
            dY_s2 = M-dim scale ``[M//64+E, N, 2]``.
      3. dgrad: ``dA = dY @ B^T`` (group_type=0, contract over N)
         x = dY_q1, weight = B_q.transpose(-1,-2)
         per_token_scale = dY_s1  ``[M, ceil(N/64), 2]``
         scale           = B_s1.transpose(1,2)  ``[E, ceil(N/64), K, 2]``
      4. wgrad: ``dB = A^T @ dY`` (group_type=2, contract over M)
         x = A_q2.t(), weight = dY_q2
         per_token_scale = A_s2  ``[M//64+E, K, 2]``  (untransposed — group_type=2
                                          puts the contracting-block dim first)
         scale           = dY_s2 ``[M//64+E, N, 2]``
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        group_list: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockQuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A along K-dim (contracting dim for forward) ---
        # A [M, K] → A_s1 (K-dim scale) [M, ceil(ceil(K/32)/2), 2]
        A_q1, A_s1 = torch_npu.npu_dynamic_mx_quant(
            A,
            axis=-1,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            block_size=config_A.block_size,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )

        # --- Step 2: quantize A along M-dim with grouped quant (zero boundaries) ---
        # A [M, K] → A_s2 (M-dim scale) [M//64 + E, K, 2]
        A_q2, A_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            A,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: block FP8 quantize B (with optional mxfp4 fake-quant pre-pass) ---
        # B [E, K, N] → B_s1 (N-dim scale) [E, K, ceil(ceil(N/32)/2), 2]
        #              → B_s2 (K-dim scale) [E, ceil(ceil(K/32)/2), N, 2]
        B_q, B_s1, B_s2 = quantize_right_operand(B, axis=-2, config=config_B)

        # --- Step 4: grouped low-precision matmul, group_type=0 (contract over K) ---
        # x = A_q1 [M, K]    → per_token_scale = A_s1 [M, ceil(K/64), 2]
        # weight = B_q [E, K, N] → scale = B_s2 [E, ceil(K/64), N, 2]
        Y = torch_npu.npu_grouped_matmul(
            [A_q1],
            [B_q],
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

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s1, group_list)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        ctx.config_B = config_B
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s1, group_list = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        # --- Step 1: quantize dY along N-dim (for dgrad) ---
        # dY [M, N] → dY_s1 (N-dim scale) [M, ceil(ceil(N/32)/2), 2]
        dY_q1, dY_s1 = torch_npu.npu_dynamic_mx_quant(
            dY,
            axis=-1,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            block_size=config_A.block_size,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )

        # --- Step 2: quantize dY along M-dim with grouped quant (zero boundaries) ---
        # dY [M, N] → dY_s2 (M-dim scale) [M//64 + E, N, 2]
        dY_q2, dY_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dY,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: dgrad  dA = dY @ B^T  (group_type=0, contract over N) ---
        # x = dY_q1 [M, N]    → per_token_scale = dY_s1 [M, ceil(N/64), 2]
        # weight = B_q.t() [E, N, K] → scale = B_s1.transpose(1,2) [E, ceil(N/64), K, 2]
        dA = torch_npu.npu_grouped_matmul(
            [dY_q1],
            [B_q.transpose(-1, -2)],
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
        # x = A_q2.t() [K, M]            → per_token_scale = A_s2.transpose(0,1) [K, M//64+E, 2]
        # weight = dY_q2 [M, N]           → scale = dY_s2 [M//64+E, N, 2]
        #
        # Note: A_s2 must be transposed to match A_q2.t()'s transposition,
        # otherwise the kernel errors: "transposition of perTokenScale/x should be equal".
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


def to_block_fp8_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockQuantizeConfig,
) -> torch.Tensor:
    """Block FP8 grouped matrix multiply with mixed-format quantization.

    ``A`` is a 2D tensor ``(M, K)`` whose rows are partitioned among
    ``E`` groups via ``group_list`` (cumsum offsets, length ``E``).
    ``B`` is a 3D tensor ``(E, K, N)``.  Computes the equivalent of
    ``A @ B`` where each group's assigned rows are multiplied by its
    corresponding slice of ``B``.

    Quantization scheme:
    - A along K-dim: ``npu_dynamic_mx_quant(axis=-1)`` (params from ``config_A``)
    - A along M-dim: ``npu_grouped_dynamic_mx_quant`` (params from ``config_A``)
    - B:             ``npu_dynamic_block_mx_quant``, 32×32 blocks (params from ``config_B``)

    See ``_BlockFP8QuantGroupedMM`` for details of the three GEMMs.

    Args:
        A: Input tensor, shape ``(M, K)``.
        B: Tensor, shape ``(E, K, N)``.
        group_list: Cumulative row offsets per group, shape ``(E,)``
            (no leading zero, length equals number of groups ``E``).
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _BlockFP8QuantGroupedMM.apply(A, B, group_list, config_A, config_B)
