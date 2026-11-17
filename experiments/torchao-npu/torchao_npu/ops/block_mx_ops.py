# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block MX low-precision matmul ops for NPU.

Quantization scheme (non-grouped ``to_block_mx_then_mm``):
- A (activation):  ``npu_dynamic_mx_quant_with_dual_axis``
  - s1 = per-32-group scale along dim -1  (``A_s1``)
  - s2 = per-32-group scale along dim -2  (``A_s2``)
- B (weight):      ``npu_dynamic_block_mx_quant``
  - s1 = per-32-group scale along dim -1  (``B_s1``)
  - s2 = per-32-group scale along dim -2  (``B_s2``)

Quantization scheme (grouped ``to_block_mx_then_grouped_mm``):
- A (activation), K-dim:    ``npu_dynamic_mx_quant(axis=-1)``
- A (activation), M-dim:    ``npu_grouped_dynamic_mx_quant``
- B (weight):               ``npu_dynamic_block_mx_quant``
- dY, N-dim:                ``npu_dynamic_mx_quant(axis=-1)``
- dY, M-dim:                ``npu_grouped_dynamic_mx_quant``
"""

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize
from torchao_npu.quantization.quant_primitives.mx import mx_quantize, mx_quantize_dual_axis

__all__ = [
    "to_block_mx_then_bmm",
    "to_block_mx_then_bmm_from_prequantized",
    "to_block_mx_then_grouped_mm",
    "to_block_mx_then_grouped_mm_from_prequantized",
    "to_block_mx_then_linear_from_prequantized",
    "to_block_mx_then_mm",
    "to_block_mx_then_mm_from_prequantized",
]


class _BlockMXQuantMM(torch.autograd.Function):
    """Block MX matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``.

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
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A with dual-axis MX quant ---
        # Flatten leading dims inside the quant call (like MXfp8MM uses view_as_n_dim)
        A_q1, A_s1, A_q2, A_s2 = mx_quantize_dual_axis(A.reshape(-1, A.shape[-1]), config_A)

        # --- Step 2: block MX quantize B (with optional mxfp4 fake-quant pre-pass) ---
        # B [K, N] → B_s1 (N-dim scale) [K, ceil(ceil(N/32)/2), 2]
        #           → B_s2 (K-dim scale) [ceil(ceil(K/32)/2), N, 2]
        B_q, B_s1, B_s2 = block_mx_quantize(B, config=config_B, axis=-2)

        # --- Step 3: low-precision matmul, contracting over K ---
        # x1 = A_q1 [M, K] → pertoken_scale = A_s1 [M, ceil(ceil(K/32)/2), 2]
        # x2 = B_q  [K, N] → scale          = B_s2 [ceil(ceil(K/32)/2), N, 2]
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=config_B.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_B.npu_matmul_dtype,
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
        dY_q1, dY_s1, dY_q2, dY_s2 = mx_quantize_dual_axis(dY.reshape(-1, dY.shape[-1]), config_A)

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N) ---
        # x1 = dY_q1 [M, N] → pertoken_scale = dY_s1 [M, ceil(ceil(N/32)/2), 2]
        # x2 = B_q.t() [N, K] → scale = B_s1.transpose(0,1) [ceil(ceil(N/32)/2), K, 2]
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=config_B.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_B.npu_matmul_dtype,
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
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, config_B.block_size],
        )

        # Restore dA to original shape
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_block_mx_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX matrix multiply: ``A @ B`` with mixed-format quantization.

    Quantization scheme:
    - A: ``npu_dynamic_mx_quant_with_dual_axis`` (params from ``config_A``)
    - B: ``npu_dynamic_block_mx_quant``, 32×32 blocks (params from ``config_B``)

    See ``_BlockMXQuantMM`` for details of the three GEMMs.

    Args:
        A: Shape ``(M, K)``.
        B: Shape ``(K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _BlockMXQuantMM.apply(A, B, config_A, config_B)


class _BlockMXQuantGroupedMM(torch.autograd.Function):
    """Block MX grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

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
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: quantize A along K-dim (contracting dim for forward) ---
        # A [M, K] → A_s1 (K-dim scale) [M, ceil(ceil(K/32)/2), 2]
        A_q1, A_s1 = mx_quantize(A, -1, config_A)

        # --- Step 2: quantize A along M-dim with grouped quant (zero boundaries) ---
        # A [M, K] → A_s2 (M-dim scale) [M//64 + E, K, 2]
        A_q2, A_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            A,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.npu_elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: block MX quantize B (with optional mxfp4 fake-quant pre-pass) ---
        # B [E, K, N] → B_s1 (N-dim scale) [E, K, ceil(ceil(N/32)/2), 2]
        #              → B_s2 (K-dim scale) [E, ceil(ceil(K/32)/2), N, 2]
        B_q, B_s1, B_s2 = block_mx_quantize(B, config=config_B, axis=-2)

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
            scale_dtype=config_B.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_B.npu_matmul_dtype,
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
        config_B = ctx.config_B
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        # --- Step 1: quantize dY along N-dim (for dgrad) ---
        # dY [M, N] → dY_s1 (N-dim scale) [M, ceil(ceil(N/32)/2), 2]
        dY_q1, dY_s1 = mx_quantize(dY, -1, config_A)

        # --- Step 2: quantize dY along M-dim with grouped quant (zero boundaries) ---
        # dY [M, N] → dY_s2 (M-dim scale) [M//64 + E, N, 2]
        dY_q2, dY_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dY,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.npu_elem_dtype,
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
            scale_dtype=config_B.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_B.npu_matmul_dtype,
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
            # wgrad: the weight operand is dY_q2, quantized with config_A
            scale_dtype=config_A.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_A.npu_matmul_dtype,
            split_item=3,
        )[0]

        return dA, dB, None, None, None


def to_block_mx_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX grouped matrix multiply with mixed-format quantization.

    ``A`` is a 2D tensor ``(M, K)`` whose rows are partitioned among
    ``E`` groups via ``group_list`` (cumsum offsets, length ``E``).
    ``B`` is a 3D tensor ``(E, K, N)``.  Computes the equivalent of
    ``A @ B`` where each group's assigned rows are multiplied by its
    corresponding slice of ``B``.

    Quantization scheme:
    - A along K-dim: ``npu_dynamic_mx_quant(axis=-1)`` (params from ``config_A``)
    - A along M-dim: ``npu_grouped_dynamic_mx_quant`` (params from ``config_A``)
    - B:             ``npu_dynamic_block_mx_quant``, 32×32 blocks (params from ``config_B``)

    See ``_BlockMXQuantGroupedMM`` for details of the three GEMMs.

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
    return _BlockMXQuantGroupedMM.apply(A, B, group_list, config_A, config_B)


class _BlockMXQuantBMM(torch.autograd.Function):
    """Block MX batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and share the leading batch dimension. The forward
    dual-axis quantizes ``A`` (params from ``config_A``) and block quantizes
    ``B`` in 32×32 blocks (params from ``config_B``, via
    ``block_mx_quantize``), then performs the low-precision matmul
    contracting over K. The backward reuses the forward-quantized ``A_q2/A_s2``
    and ``B_q/B_s1`` transposed and only quantizes ``dY`` fresh.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[0] == B.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B[0]={B.shape[0]}"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        # --- Step 1: dual-axis MX quantize A (left operand) ---
        A_q1, A_s1, A_q2, A_s2 = mx_quantize_dual_axis(A, config_A)

        # --- Step 2: block MX quantize B (right operand, optional mxfp4 pre-pass) ---
        B_q, B_s1, B_s2 = block_mx_quantize(B, config=config_B, axis=-2)

        # --- Step 3: low-precision batched matmul, contracting over K ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=config_B.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_B.npu_matmul_dtype,
            group_sizes=[1, 1, config_B.block_size],
        )

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

        # --- Step 1: dual-axis MX quantize dY ---
        dY_q1, dY_s1, dY_q2, dY_s2 = mx_quantize_dual_axis(dY, config_A)

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.transpose(-1, -2),
            B_s1.transpose(-2, -3),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=config_B.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_B.npu_matmul_dtype,
            group_sizes=[1, 1, config_B.block_size],
        )

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q2.transpose(-1, -2),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(-2, -3),
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, config_B.block_size],
        )

        return dA, dB, None, None


def to_block_mx_then_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and must share the leading batch dimension.
    Quantization configs are drawn from ``config_A`` (MX, for A and dY) and
    ``config_B`` (Block MX, for B).

    Args:
        A: Shape ``(B, M, K)``.
        B: Shape ``(B, K, N)``.
        config_A: MX config for A's quantization.
        config_B: Block MX config for B's quantization.

    Returns:
        Output tensor, shape ``(B, M, N)``, dtype matching ``A``.
    """
    return _BlockMXQuantBMM.apply(A, B, config_A, config_B)


# =========================================================================
# Pre-quantized Block MX matmul ops (weight already quantized in
# ``fsdp_pre_all_gather``). These skip the on-the-fly ``block_mx_quantize``
# and consume the pre-quantized ``B_q/B_s1/B_s2`` stored on the wrapper.
#
# The wrapper is passed as the weight input so the backward returns a BF16
# gradient routed to the local shard by FSDP (the wrapper's logical dtype is
# BF16 even though ``_data`` holds FP8).
# =========================================================================


@torch._dynamo.allow_in_graph
class _BlockMXQuantMMFromPreQuantized(torch.autograd.Function):
    """Block MX matmul with a pre-quantized weight (dense ``A @ B``).

    ``wrapper._data`` is the FP8 weight ``B_q`` in ``[K, N]`` layout, with
    ``_scale_s1`` (N-dim) and ``_scale_s2`` (K-dim) scales. Forward uses
    ``B_q + B_s2``, backward uses ``B_q + B_s1``.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        wrapper: torch.Tensor,
        config_A: MXQuantizeConfig,
    ):
        B_q = wrapper._data  # pyrefly: ignore [missing-attribute]
        B_s1 = wrapper._scale_s1  # pyrefly: ignore [missing-attribute]
        B_s2 = wrapper._scale_s2  # pyrefly: ignore [missing-attribute]
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B_q.ndim == 2, f"B_q must be 2D, got {B_q.ndim}D"
        assert A.shape[-1] == B_q.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B_q[-2]={B_q.shape[-2]}"

        # --- Step 1: quantize A with dual-axis MX quant ---
        A_q1, A_s1, A_q2, A_s2 = mx_quantize_dual_axis(A.reshape(-1, A.shape[-1]), config_A)

        # --- Step 2: low-precision matmul, contracting over K (use B_s2) ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        Y.requires_grad_(A.requires_grad or wrapper.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s1)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s1 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A

        # --- Step 1: quantize dY with dual-axis MX quant ---
        dY_q1, dY_s1, dY_q2, dY_s2 = mx_quantize_dual_axis(dY.reshape(-1, dY.shape[-1]), config_A)

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N, use B_s1) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q2.t(),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(0, 1),
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        return dA, dB, None


def to_block_mx_then_mm_from_prequantized(
    A: torch.Tensor,
    wrapper: torch.Tensor,
    config_A: MXQuantizeConfig,
) -> torch.Tensor:
    """Block MX matmul with a pre-quantized weight (dense ``A @ B``).

    ``wrapper`` must be a ``BlockMXTrainingWeightWrapperTensor`` carrying
    pre-quantized FP8 data (``_data`` = ``B_q``, ``_scale_s1``, ``_scale_s2``).
    """
    return _BlockMXQuantMMFromPreQuantized.apply(A, wrapper, config_A)


@torch._dynamo.allow_in_graph
class _BlockMXQuantLinearFromPreQuantized(torch.autograd.Function):
    """Block MX linear with a pre-quantized weight (``A @ W.T + bias``).

    ``wrapper._data`` is the FP8 weight ``B_q`` in ``[N, K]`` layout (the
    ``nn.Linear`` weight), with ``_scale_s1`` (K-dim) and ``_scale_s2`` (N-dim)
    scales. Forward uses ``B_q.t() + B_s1.transpose(0,1)`` (K-dim scale),
    backward uses ``B_q + B_s2`` (N-dim scale).
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        wrapper: torch.Tensor,
        config_A: MXQuantizeConfig,
    ):
        B_q = wrapper._data  # pyrefly: ignore [missing-attribute]
        B_s1 = wrapper._scale_s1  # pyrefly: ignore [missing-attribute]
        B_s2 = wrapper._scale_s2  # pyrefly: ignore [missing-attribute]
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B_q.ndim == 2, f"B_q must be 2D, got {B_q.ndim}D"
        assert A.shape[-1] == B_q.shape[-1], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B_q[-1]={B_q.shape[-1]}"

        # --- Step 1: quantize A with dual-axis MX quant ---
        A_q1, A_s1, A_q2, A_s2 = mx_quantize_dual_axis(A.reshape(-1, A.shape[-1]), config_A)

        # --- Step 2: low-precision matmul A @ B_q.T, contract over K (use B_s1) ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        Y.requires_grad_(A.requires_grad or wrapper.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s2)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s2 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A

        # --- Step 1: quantize dY with dual-axis MX quant ---
        dY_q1, dY_s1, dY_q2, dY_s2 = mx_quantize_dual_axis(dY.reshape(-1, dY.shape[-1]), config_A)

        # --- Step 2: dgrad  dA = dY @ B_q  (contract over N, use B_s2) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q,
            B_s2,
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        # ``A_q2.t() @ dY_q2`` yields ``[K, N]``, but the pre-quantized weight
        # ``B_q`` is stored in the ``nn.Linear`` layout ``[N, K]``, so the
        # gradient must be transposed to match (``Y = A @ B_q.T``).
        dB = torch_npu.npu_quant_matmul(
            A_q2.t(),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(0, 1),
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        ).t()

        return dA, dB, None


def to_block_mx_then_linear_from_prequantized(
    A: torch.Tensor,
    wrapper: torch.Tensor,
    config_A: MXQuantizeConfig,
) -> torch.Tensor:
    """Block MX linear with a pre-quantized weight (``A @ W.T``).

    ``wrapper`` must be a ``BlockMXTrainingWeightWrapperTensor`` carrying
    pre-quantized FP8 data in the ``[N, K]`` (``nn.Linear`` weight) layout.
    """
    return _BlockMXQuantLinearFromPreQuantized.apply(A, wrapper, config_A)


@torch._dynamo.allow_in_graph
class _BlockMXQuantBMMFromPreQuantized(torch.autograd.Function):
    """Block MX batched matmul with a pre-quantized weight (``A[B,M,K] @ B[B,K,N]``).

    ``wrapper._data`` is the FP8 weight ``B_q`` in ``[B, K, N]`` layout, with
    ``_scale_s1`` (N-dim) and ``_scale_s2`` (K-dim) scales.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        wrapper: torch.Tensor,
        config_A: MXQuantizeConfig,
    ):
        B_q = wrapper._data  # pyrefly: ignore [missing-attribute]
        B_s1 = wrapper._scale_s1  # pyrefly: ignore [missing-attribute]
        B_s2 = wrapper._scale_s2  # pyrefly: ignore [missing-attribute]
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B_q.ndim == 3, f"B_q must be 3D, got {B_q.ndim}D"
        assert A.shape[0] == B_q.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B_q[0]={B_q.shape[0]}"
        assert A.shape[-1] == B_q.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B_q[-2]={B_q.shape[-2]}"

        # --- Step 1: dual-axis MX quantize A ---
        A_q1, A_s1, A_q2, A_s2 = mx_quantize_dual_axis(A, config_A)

        # --- Step 2: low-precision batched matmul, contract over K (use B_s2) ---
        Y = torch_npu.npu_quant_matmul(
            A_q1,
            B_q,
            B_s2,
            pertoken_scale=A_s1,
            output_dtype=A.dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        Y.requires_grad_(A.requires_grad or wrapper.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s1)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s1 = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A

        # --- Step 1: dual-axis MX quantize dY ---
        dY_q1, dY_s1, dY_q2, dY_s2 = mx_quantize_dual_axis(dY, config_A)

        # --- Step 2: dgrad  dA = dY @ B^T  (contract over N, use B_s1) ---
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.transpose(-1, -2),
            B_s1.transpose(-2, -3),
            pertoken_scale=dY_s1,
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        # --- Step 3: wgrad  dB = A^T @ dY  (contract over M) ---
        dB = torch_npu.npu_quant_matmul(
            A_q2.transpose(-1, -2),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(-2, -3),
            output_dtype=A_dtype,
            scale_dtype=config_A.npu_scale_dtype,
            pertoken_scale_dtype=config_A.npu_scale_dtype,
            x1_dtype=config_A.npu_matmul_dtype,
            x2_dtype=config_A.npu_matmul_dtype,
            group_sizes=[1, 1, 32],
        )

        return dA, dB, None


def to_block_mx_then_bmm_from_prequantized(
    A: torch.Tensor,
    wrapper: torch.Tensor,
    config_A: MXQuantizeConfig,
) -> torch.Tensor:
    """Block MX batched matmul with a pre-quantized weight (``A[B,M,K] @ B[B,K,N]``)."""
    return _BlockMXQuantBMMFromPreQuantized.apply(A, wrapper, config_A)


@torch._dynamo.allow_in_graph
class _BlockMXQuantGroupedMMFromPreQuantized(torch.autograd.Function):
    """Block MX grouped matmul with a pre-quantized weight (``A[M,K] @ B[E,K,N]``).

    ``wrapper._data`` is the FP8 weight ``B_q`` in ``[E, K, N]`` layout, with
    ``_scale_s1`` (N-dim) and ``_scale_s2`` (K-dim) scales.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        wrapper: torch.Tensor,
        group_list: torch.Tensor,
        config_A: MXQuantizeConfig,
    ):
        B_q = wrapper._data  # pyrefly: ignore [missing-attribute]
        B_s1 = wrapper._scale_s1  # pyrefly: ignore [missing-attribute]
        B_s2 = wrapper._scale_s2  # pyrefly: ignore [missing-attribute]
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B_q.ndim == 3, f"B_q must be 3D, got {B_q.ndim}D"
        assert A.shape[-1] == B_q.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B_q[-2]={B_q.shape[-2]}"

        # --- Step 1: quantize A along K-dim (contracting dim for forward) ---
        A_q1, A_s1 = mx_quantize(A, -1, config_A)

        # --- Step 2: quantize A along M-dim with grouped quant (zero boundaries) ---
        A_q2, A_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            A,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.npu_elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: grouped low-precision matmul, group_type=0 (contract over K) ---
        Y = torch_npu.npu_grouped_matmul(
            [A_q1],
            [B_q],
            scale=[B_s2],
            per_token_scale=[A_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=A.dtype,
            group_list_type=0,
            scale_dtype=config_A.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_A.npu_matmul_dtype,
            split_item=3,
        )[0]

        Y.requires_grad_(A.requires_grad or wrapper.requires_grad)

        ctx.save_for_backward(A_q2, A_s2, B_q, B_s1, group_list)
        ctx.A_dtype = A.dtype
        ctx.config_A = config_A
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        A_q2, A_s2, B_q, B_s1, group_list = ctx.saved_tensors
        A_dtype = ctx.A_dtype
        config_A = ctx.config_A
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        # --- Step 1: quantize dY along N-dim (for dgrad) ---
        dY_q1, dY_s1 = mx_quantize(dY, -1, config_A)

        # --- Step 2: quantize dY along M-dim with grouped quant (zero boundaries) ---
        dY_q2, dY_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dY,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.npu_elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )

        # --- Step 3: dgrad  dA = dY @ B^T  (group_type=0, contract over N) ---
        dA = torch_npu.npu_grouped_matmul(
            [dY_q1],
            [B_q.transpose(-1, -2)],
            scale=[B_s1.transpose(1, 2)],
            per_token_scale=[dY_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=A_dtype,
            group_list_type=0,
            scale_dtype=config_A.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_A.npu_matmul_dtype,
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
            scale_dtype=config_A.npu_scale_dtype,
            per_token_scale_dtype=config_A.npu_scale_dtype,
            x_dtype=config_A.npu_matmul_dtype,
            weight_dtype=config_A.npu_matmul_dtype,
            split_item=3,
        )[0]

        return dA, dB, None, None


def to_block_mx_then_grouped_mm_from_prequantized(
    A: torch.Tensor,
    wrapper: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
) -> torch.Tensor:
    """Block MX grouped matmul with a pre-quantized weight (``A[M,K] @ B[E,K,N]``)."""
    return _BlockMXQuantGroupedMMFromPreQuantized.apply(A, wrapper, group_list, config_A)
