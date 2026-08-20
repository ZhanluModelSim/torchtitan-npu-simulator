# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for to_hif8_then_mm / to_hif8_then_grouped_mm.

SQNR thresholds here are conservative placeholders. They are set low enough
to catch gross correctness bugs (wrong operand order, wrong scale broadcast,
transposed gradient, etc.) without being tuned to an actual measured
baseline; tighten them once real NPU numbers are available.
"""

import pytest
import torch
from torchao.float8.float8_utils import compute_error

from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.hif8_ops import (
    to_hif8_then_grouped_mm,
    to_hif8_then_mm,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import HiF8QuantizeConfig

_SQNR_THRESHOLD = 12.0


def _npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


# =========================================================================
# to_hif8_then_mm
# =========================================================================


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128, 64, 256),
        (256, 128, 128),
        (64, 256, 64),
    ],
)
def test_forward_shape_and_dtype(M, K, N, dtype):
    """Output shape and dtype match input expectations.

    B is passed as a transposed column-major tensor, simulating the common
    pattern where a linear weight is stored as [N, K] and ``.T`` is applied.
    """
    cfg = HiF8QuantizeConfig()
    A = torch.randn(M, K, device="npu", dtype=dtype)
    weight = torch.randn(N, K, device="npu", dtype=dtype)
    B = weight.T  # [K, N] with column-major strides

    out = to_hif8_then_mm(A, B, cfg, cfg)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == A.dtype, f"Expected {A.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("M, K, N", [(128, 64, 256), (256, 128, 128)])
def test_sqnr_forward(M, K, N):
    """HiF8 forward output has acceptable SQNR vs high-precision matmul."""
    torch.manual_seed(42)
    cfg = HiF8QuantizeConfig()
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out_ref = A @ B
    out_hif8 = to_hif8_then_mm(A, B, cfg, cfg)

    sqnr = compute_error(out_ref.float(), out_hif8.float()).item()
    assert sqnr > _SQNR_THRESHOLD, f"Forward SQNR too low: {sqnr:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("M, K, N", [(128, 64, 256), (256, 128, 128)])
def test_sqnr_gradients(M, K, N):
    """HiF8 backward gradients have acceptable SQNR vs high-precision backward.

    Checks both dA (grad wrt A) and dB (grad wrt B).
    """
    torch.manual_seed(42)
    cfg = HiF8QuantizeConfig()
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(N, K, device="npu", dtype=torch.bfloat16)
    B = weight.T

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)
    out_ref = A_ref @ B_ref
    out_ref.sum().backward()

    A_hif8 = A.clone().detach().requires_grad_(True)
    B_hif8 = B.clone().detach().requires_grad_(True)
    out_hif8 = to_hif8_then_mm(A_hif8, B_hif8, cfg, cfg)
    out_hif8.sum().backward()

    sqnr_dA = compute_error(A_ref.grad.float(), A_hif8.grad.float()).item()
    assert sqnr_dA > _SQNR_THRESHOLD, f"dA SQNR too low: {sqnr_dA:.2f} dB"

    sqnr_dB = compute_error(B_ref.grad.float(), B_hif8.grad.float()).item()
    assert sqnr_dB > _SQNR_THRESHOLD, f"dB SQNR too low: {sqnr_dB:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_non_2d_input():
    """3D A + 2D B produce correct output; 1D B raises."""
    cfg = HiF8QuantizeConfig()
    A_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out = to_hif8_then_mm(A_3d, B, cfg, cfg)
    assert out.shape == (4, 32, 128)

    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    B_1d = torch.randn(64, device="npu", dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_mm(A, B_1d, cfg, cfg)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_contracting_dim_mismatch():
    """Mismatched contracting dimensions raise an error."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    B = weight.T

    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_mm(A, B, cfg, cfg)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_no_requires_grad():
    """Gradient tracking is not required on either operand."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = weight.T

    out = to_hif8_then_mm(A, B, cfg, cfg)
    assert out.shape == (32, 128)
    assert not out.requires_grad


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_fp32_input_is_cast_to_bf16():
    """fp32 operands are internally cast to bf16 (npu_dynamic_quant requires bf16/fp16)."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(32, 64, device="npu", dtype=torch.float32)
    weight = torch.randn(128, 64, device="npu", dtype=torch.float32)
    B = weight.T

    out = to_hif8_then_mm(A, B, cfg, cfg)
    assert out.dtype == torch.bfloat16, f"Expected bf16 (cast from fp32), got {out.dtype}"


# =========================================================================
# to_hif8_then_grouped_mm
# =========================================================================


def _group_list_from_sizes(group_sizes: list[int], device: str = "npu") -> torch.Tensor:
    """Build a cumsum group_list for the given per-group sizes."""
    gs = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return gs.cumsum(0)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes",
    [
        (128, 64, 128, 2, [64, 64]),
        (192, 64, 64, 3, [64, 64, 64]),
        (128, 64, 64, 4, [32, 32, 32, 32]),
    ],
)
def test_grouped_forward_shape_and_dtype(M, K, N, E, group_sizes, dtype):
    """Output shape and dtype match expectations for grouped matmul."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(M, K, device="npu", dtype=dtype)
    B = torch.randn(E, K, N, device="npu", dtype=dtype)
    group_list = _group_list_from_sizes(group_sizes)

    out = to_hif8_then_grouped_mm(A, B, group_list, cfg, cfg)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == A.dtype, f"Expected {A.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("M, K, N, E, group_sizes", [(192, 64, 128, 3, [64, 64, 64])])
def test_grouped_sqnr_forward(M, K, N, E, group_sizes):
    """Grouped HiF8 forward output has acceptable SQNR."""
    torch.manual_seed(42)
    cfg = HiF8QuantizeConfig()
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

    out_hif8 = to_hif8_then_grouped_mm(A, B, group_list, cfg, cfg)

    sqnr = compute_error(out_ref.float(), out_hif8.float()).item()
    assert sqnr > _SQNR_THRESHOLD, f"Forward SQNR too low: {sqnr:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("M, K, N, E, group_sizes", [(192, 64, 128, 3, [64, 64, 64])])
def test_grouped_sqnr_gradients(M, K, N, E, group_sizes):
    """Grouped HiF8 backward gradients have acceptable SQNR."""
    torch.manual_seed(42)
    cfg = HiF8QuantizeConfig()
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16, requires_grad=True)
    group_list = _group_list_from_sizes(group_sizes)

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)
    out_ref = []
    for i in range(E):
        s = group_list[i - 1].item() if i > 0 else 0
        e = group_list[i].item()
        out_ref.append(A_ref[s:e] @ B_ref[i])
    out_ref = torch.cat(out_ref, dim=0)
    out_ref.sum().backward()

    A_hif8 = A.clone().detach().requires_grad_(True)
    B_hif8 = B.clone().detach().requires_grad_(True)
    out_hif8 = to_hif8_then_grouped_mm(A_hif8, B_hif8, group_list, cfg, cfg)
    out_hif8.sum().backward()

    sqnr_dA = compute_error(A_ref.grad.float(), A_hif8.grad.float()).item()
    assert sqnr_dA > _SQNR_THRESHOLD, f"dA SQNR too low: {sqnr_dA:.2f} dB"

    sqnr_dB = compute_error(B_ref.grad.float(), B_hif8.grad.float()).item()
    assert sqnr_dB > _SQNR_THRESHOLD, f"dB SQNR too low: {sqnr_dB:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_grouped_non_2d_3d_input():
    """Non-2D A / non-3D B raise an error in grouped matmul."""
    cfg = HiF8QuantizeConfig()
    # A 3D (must be 2D)
    A_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    B = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_grouped_mm(A_3d, B, group_list, cfg, cfg)

    # B 2D (must be 3D)
    A = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    B_2d = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_grouped_mm(A, B_2d, group_list, cfg, cfg)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_grouped_contracting_dim_mismatch():
    """Mismatched contracting dimensions raise an error in grouped matmul."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    B = torch.randn(2, 128, 64, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([32, 64], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_grouped_mm(A, B, group_list, cfg, cfg)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_grouped_no_requires_grad():
    """Gradient tracking not required on grouped operands."""
    cfg = HiF8QuantizeConfig()
    A = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    B = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")

    out = to_hif8_then_grouped_mm(A, B, group_list, cfg, cfg)
    assert out.shape == (128, 128)
    assert not out.requires_grad


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize("E", [2, 3, 5])
def test_grouped_scale_broadcast_tracks_weight_expert_count(E):
    """Scale broadcast width is derived from B.shape[0] (E), not any external state.

    Regression guard for the design decision (see ops/hif8_ops.py module
    docstring) to read the expert count directly off the weight tensor's
    shape instead of depending on an external mutable dict -- sweep several
    values of E and confirm forward/backward run and produce finite,
    correctly-shaped gradients for each.
    """
    cfg = HiF8QuantizeConfig()
    M, K, N = 64, 32, 32
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16, requires_grad=True)
    base = M // E
    group_sizes = [base] * (E - 1) + [M - base * (E - 1)]
    group_list = _group_list_from_sizes(group_sizes)

    out = to_hif8_then_grouped_mm(A, B, group_list, cfg, cfg)
    assert out.shape == (M, N)

    out.sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad.float()).all()
    assert B.grad is not None and B.grad.shape == B.shape and torch.isfinite(B.grad.float()).all()
