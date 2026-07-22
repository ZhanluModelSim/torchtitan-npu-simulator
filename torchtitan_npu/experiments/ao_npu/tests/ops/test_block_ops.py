# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao.float8.float8_utils import compute_error

from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.block_ops import (
    to_block_fp8_then_grouped_mm,
    to_block_fp8_then_mm,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_primitives.block_fp8 import quantize_right_operand


def _npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, config_A, config_B",
    [
        (128, 64, 256, MXQuantizeConfig(), BlockQuantizeConfig()),
        (256, 128, 128, MXQuantizeConfig(), BlockQuantizeConfig()),
        (64, 256, 64, MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_forward_shape_and_dtype(M, K, N, config_A, config_B):
    """Output shape and dtype match input expectations.

    B is passed as a transposed column-major tensor, simulating the common
    pattern where a linear weight is stored as [N, K] and ``.T`` is applied.
    """
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    # weight stored [N, K] (out_features, in_features), use .T to get [K, N]
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16)
    B = weight.T  # [K, N] with column-major strides

    out = to_block_fp8_then_mm(A, B, config_A, config_B)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == A.dtype, f"Expected {A.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, config_A, config_B, sqnr_threshold",
    [
        (128, 64, 256, MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (256, 128, 128, MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (
            128,
            64,
            256,
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            12.0,
        ),
    ],
)
def test_sqnr_forward(M, K, N, config_A, config_B, sqnr_threshold):
    """Block FP8 forward output has acceptable SQNR vs high-precision matmul."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out_ref = A @ B
    out_fp8 = to_block_fp8_then_mm(A, B, config_A, config_B)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"Forward SQNR too low: {sqnr:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, config_A, config_B, sqnr_threshold",
    [
        (128, 64, 256, MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (256, 128, 128, MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (
            128,
            64,
            256,
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            12.0,
        ),
    ],
)
def test_sqnr_gradients(M, K, N, config_A, config_B, sqnr_threshold):
    """Block FP8 backward gradients have acceptable SQNR vs high-precision backward.

    Checks both dA (grad wrt A) and dB (grad wrt B).
    """
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16)
    B = weight.T

    # --- Reference: high-precision forward + backward ---
    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)
    out_ref = A_ref @ B_ref
    out_ref.sum().backward()

    # --- Block FP8 forward + backward ---
    A_fp8 = A.clone().detach().requires_grad_(True)
    B_fp8 = B.clone().detach().requires_grad_(True)
    out_fp8 = to_block_fp8_then_mm(A_fp8, B_fp8, config_A, config_B)
    out_fp8.sum().backward()

    # SQNR of dA (grad wrt A)
    sqnr_dA = compute_error(A_ref.grad.float(), A_fp8.grad.float()).item()
    assert sqnr_dA > sqnr_threshold, f"dA SQNR too low: {sqnr_dA:.2f} dB"

    # SQNR of dB (grad wrt B)
    sqnr_dB = compute_error(B_ref.grad.float(), B_fp8.grad.float()).item()
    assert sqnr_dB > sqnr_threshold, f"dB SQNR too low: {sqnr_dB:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, config_A, config_B",
    [
        (32, 64, 128, MXQuantizeConfig(), BlockQuantizeConfig()),
        (128, 32, 64, MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_backward_finiteness(M, K, N, config_A, config_B):
    """Gradients are finite (no NaN/Inf) and non-zero."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    B = weight.T

    out = to_block_fp8_then_mm(A, B, config_A, config_B)
    out.sum().backward()

    for name, g in [("A", A.grad), ("weight", weight.grad)]:
        assert g is not None, f"{name}.grad is None"
        g_cpu = g.float().cpu()
        assert torch.isfinite(g_cpu).all(), f"{name}.grad has non-finite values"
        assert g_cpu.norm().item() > 0, f"{name}.grad is all zeros"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_non_2d_input(config_A, config_B):
    """3D + 2D inputs produce correct output; 1D B raises."""
    A_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out = to_block_fp8_then_mm(A_3d, B, config_A, config_B)
    assert out.shape == (4, 32, 128)

    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    B_1d = torch.randn(64, device="npu", dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        to_block_fp8_then_mm(A, B_1d, config_A, config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_contracting_dim_mismatch(config_A, config_B):
    """Mismatched contracting dimensions raise an error."""
    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    # weight [64, 256] → weight.T [256, 64], A[-1]=64 != B[-2]=256
    weight = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    B = weight.T

    with pytest.raises((AssertionError, RuntimeError)):
        to_block_fp8_then_mm(A, B, config_A, config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_no_requires_grad(config_A, config_B):
    """Gradient tracking is not required on either operand."""
    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out = to_block_fp8_then_mm(A, B, config_A, config_B)
    assert out.shape == (32, 128)
    assert not out.requires_grad


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_dtype_preservation(dtype, config_A, config_B):
    """Output dtype matches A's dtype."""
    A = torch.randn(64, 128, device="npu", dtype=dtype)
    weight = torch.randn(32, 128, device="npu", dtype=dtype)
    B = weight.T
    out = to_block_fp8_then_mm(A, B, config_A, config_B)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# --- helpers for grouped mm tests ---


def _group_list_from_sizes(group_sizes: list[int], device: str = "npu") -> torch.Tensor:
    """Build a cumsum group_list for the given per-group sizes.

    The API expects ``group_list`` to have length ``E`` (number of groups),
    containing cumulative token counts per group: ``[s0, s0+s1, ..., M]``.
    Use ``group_list_type=0`` (cumsum format).
    """
    gs = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return gs.cumsum(0)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes, config_A, config_B",
    [
        (128, 64, 128, 2, [64, 64], MXQuantizeConfig(), BlockQuantizeConfig()),
        (192, 64, 64, 3, [64, 64, 64], MXQuantizeConfig(), BlockQuantizeConfig()),
        (128, 64, 64, 4, [32, 32, 32, 32], MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_forward_shape_and_dtype(M, K, N, E, group_sizes, config_A, config_B):
    """Output shape and dtype match expectations for grouped matmul."""
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    out = to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == A.dtype, f"Expected {A.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes, config_A, config_B, sqnr_threshold",
    [
        (192, 64, 128, 3, [64, 64, 64], MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (
            192,
            64,
            128,
            3,
            [64, 64, 64],
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            12.0,
        ),
    ],
)
def test_grouped_sqnr_forward(M, K, N, E, group_sizes, config_A, config_B, sqnr_threshold):
    """Grouped block FP8 forward output has acceptable SQNR."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    # Reference: group-by-group high-precision matmul
    out_ref = []
    for i in range(E):
        s = group_list[i - 1].item() if i > 0 else 0
        e = group_list[i].item()
        out_ref.append(A[s:e] @ B[i])
    out_ref = torch.cat(out_ref, dim=0)

    out_fp8 = to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"Forward SQNR too low: {sqnr:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes, config_A, config_B, sqnr_threshold",
    [
        (192, 64, 128, 3, [64, 64, 64], MXQuantizeConfig(), BlockQuantizeConfig(), 17.0),
        (
            192,
            64,
            128,
            3,
            [64, 64, 64],
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            12.0,
        ),
    ],
)
def test_grouped_sqnr_gradients(M, K, N, E, group_sizes, config_A, config_B, sqnr_threshold):
    """Grouped block FP8 backward gradients have acceptable SQNR."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    # --- Reference ---
    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)
    out_ref = []
    for i in range(E):
        s = group_list[i - 1].item() if i > 0 else 0
        e = group_list[i].item()
        out_ref.append(A_ref[s:e] @ B_ref[i])
    out_ref = torch.cat(out_ref, dim=0)
    out_ref.sum().backward()

    # --- Block FP8 ---
    A_fp8 = A.clone().detach().requires_grad_(True)
    B_fp8 = B.clone().detach().requires_grad_(True)
    out_fp8 = to_block_fp8_then_grouped_mm(A_fp8, B_fp8, group_list, config_A, config_B)
    out_fp8.sum().backward()

    sqnr_dA = compute_error(A_ref.grad.float(), A_fp8.grad.float()).item()
    assert sqnr_dA > sqnr_threshold, f"dA SQNR too low: {sqnr_dA:.2f} dB"

    sqnr_dB = compute_error(B_ref.grad.float(), B_fp8.grad.float()).item()
    assert sqnr_dB > sqnr_threshold, f"dB SQNR too low: {sqnr_dB:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes, config_A, config_B",
    [
        (128, 64, 128, 2, [64, 64], MXQuantizeConfig(), BlockQuantizeConfig()),
        (192, 64, 128, 3, [64, 64, 64], MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_backward_finiteness(M, K, N, E, group_sizes, config_A, config_B):
    """Grouped backward gradients are finite and non-zero."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16, requires_grad=True)
    group_list = _group_list_from_sizes(group_sizes)

    out = to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)
    out.sum().backward()

    for name, g in [("A", A.grad), ("B", B.grad)]:
        assert g is not None, f"{name}.grad is None"
        g_cpu = g.float().cpu()
        assert torch.isfinite(g_cpu).all(), f"{name}.grad has non-finite values"
        assert g_cpu.norm().item() > 0, f"{name}.grad is all zeros"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_non_2d_input(config_A, config_B):
    """Non-2D / non-3D inputs raise an error in grouped matmul."""
    # A 3D
    A_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    B = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_fp8_then_grouped_mm(A_3d, B, group_list, config_A, config_B)

    # B 2D (must be 3D)
    A = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    B_2d = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_fp8_then_grouped_mm(A, B_2d, group_list, config_A, config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_contracting_dim_mismatch(config_A, config_B):
    """Mismatched contracting dimensions raise an error in grouped matmul."""
    A = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    # B shape [E=2, K=128, N=64], but A[-1]=64 != B[-2]=128
    B = torch.randn(2, 128, 64, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([32, 64], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_no_requires_grad(config_A, config_B):
    """Gradient tracking not required on grouped operands."""
    A = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")

    out = to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)
    assert out.shape == (128, 128)
    assert not out.requires_grad


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "config_A, config_B",
    [
        (MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_grouped_dtype_preservation(dtype, config_A, config_B):
    """Output dtype matches A's dtype in grouped matmul."""
    A = torch.randn(128, 128, device="npu", dtype=dtype)
    B = torch.randn(2, 128, 64, device="npu", dtype=dtype)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")

    out = to_block_fp8_then_grouped_mm(A, B, group_list, config_A, config_B)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# =========================================================================
# quantize_right_operand
# =========================================================================


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "K, N, axis, config_B",
    [
        (64, 128, -2, BlockQuantizeConfig()),
        (
            64,
            128,
            -2,
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
        (128, 64, -2, BlockQuantizeConfig()),
        (
            128,
            64,
            -2,
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
    ],
)
def test_quantize_right_operand_shape_and_dtype_2d(K, N, axis, config_B):
    """2D B: returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)

    B_q, _, _ = quantize_right_operand(B, axis=axis, config=config_B)

    assert B_q.dtype == config_B.elem_dtype
    assert B_q.shape == B.shape


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "E, K, N, axis, config_B",
    [
        (2, 64, 128, -2, BlockQuantizeConfig()),
        (
            2,
            64,
            128,
            -2,
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
        (3, 128, 64, -2, BlockQuantizeConfig()),
    ],
)
def test_quantize_right_operand_shape_and_dtype_3d(E, K, N, axis, config_B):
    """3D B (grouped): returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)

    B_q, _, _ = quantize_right_operand(B, axis=axis, config=config_B)

    assert B_q.dtype == config_B.elem_dtype
    assert B_q.shape == B.shape
