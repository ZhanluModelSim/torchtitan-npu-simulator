# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchao.float8.float8_utils import compute_error

from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.block_ops import (
    to_block_fp8_then_bmm,
    to_block_fp8_then_grouped_mm,
    to_block_fp8_then_mm,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.mx_ops import (
    mxfp4_fake_quantize,
    mxfp8_dequantize,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_primitives.block_fp8 import block_fp8_quantize


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
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


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
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


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
# block_fp8_quantize
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
def test_block_fp8_quantize_shape_and_dtype_2d(K, N, axis, config_B):
    """2D B: returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)

    B_q, _, _ = block_fp8_quantize(B, axis=axis, config=config_B)

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
def test_block_fp8_quantize_shape_and_dtype_3d(E, K, N, axis, config_B):
    """3D B (grouped): returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)

    B_q, _, _ = block_fp8_quantize(B, axis=axis, config=config_B)

    assert B_q.dtype == config_B.elem_dtype
    assert B_q.shape == B.shape


# =========================================================================
# to_block_fp8_then_bmm (block FP8 batched matmul)
# =========================================================================


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "B, M, K, N, config_A, config_B",
    [
        (4, 2048, 4096, 2048, MXQuantizeConfig(), BlockQuantizeConfig()),
        (4, 4096, 2048, 4096, MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_bmm_forward_shape_and_dtype(B, M, K, N, config_A, config_B):
    """Block FP8 batched matmul output shape and dtype match expectations."""
    act = torch.randn(B, M, K, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(B, N, K, device="npu", dtype=torch.bfloat16).transpose(-1, -2)  # [B, K, N]

    out = to_block_fp8_then_bmm(act, weight, config_A, config_B)

    assert out.shape == (B, M, N), f"Expected ({B}, {M}, {N}), got {out.shape}"
    assert out.dtype == act.dtype, f"Expected {act.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "B, M, K, N, config_A, config_B, sqnr_threshold",
    [
        (4, 2048, 4096, 2048, MXQuantizeConfig(), BlockQuantizeConfig(), 27.5),
        (
            4,
            2048,
            4096,
            2048,
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            17.5,
        ),
    ],
)
def test_bmm_sqnr_forward(B, M, K, N, config_A, config_B, sqnr_threshold):
    """Block FP8 batched matmul forward output has acceptable SQNR."""
    torch.manual_seed(42)
    act = torch.randn(B, M, K, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(B, N, K, device="npu", dtype=torch.bfloat16).transpose(-1, -2)  # [B, K, N]

    out_ref = torch.bmm(act, weight)
    out_fp8 = to_block_fp8_then_bmm(act, weight, config_A, config_B)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"BMM forward SQNR too low: {sqnr:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "B, M, K, N, config_A, config_B, sqnr_threshold",
    [
        (4, 2048, 4096, 2048, MXQuantizeConfig(), BlockQuantizeConfig(), 30.5),
        (
            4,
            2048,
            4096,
            2048,
            MXQuantizeConfig(),
            BlockQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
            17.5,
        ),
    ],
)
def test_bmm_sqnr_gradients(B, M, K, N, config_A, config_B, sqnr_threshold):
    """Block FP8 batched matmul backward gradients have acceptable SQNR.

    The weight is stored as ``[B, N, K]`` and transposed for the bmm (matching
    model usage); gradients are checked on the stored leaf parameter.
    """
    torch.manual_seed(42)
    act = torch.randn(B, M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(B, N, K, device="npu", dtype=torch.bfloat16, requires_grad=True)

    # --- Reference ---
    act_ref = act.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    torch.bmm(act_ref, weight_ref.transpose(-1, -2)).sum().backward()

    # --- Block FP8 ---
    act_fp8 = act.clone().detach().requires_grad_(True)
    weight_fp8 = weight.clone().detach().requires_grad_(True)
    to_block_fp8_then_bmm(act_fp8, weight_fp8.transpose(-1, -2), config_A, config_B).sum().backward()

    sqnr_dA = compute_error(act_ref.grad.float(), act_fp8.grad.float()).item()
    assert sqnr_dA > sqnr_threshold, f"dA SQNR too low: {sqnr_dA:.2f} dB"

    sqnr_dB = compute_error(weight_ref.grad.float(), weight_fp8.grad.float()).item()
    assert sqnr_dB > sqnr_threshold, f"dB SQNR too low: {sqnr_dB:.2f} dB"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "B, M, K, N, config_A, config_B",
    [
        (4, 1024, 2048, 1024, MXQuantizeConfig(), BlockQuantizeConfig()),
    ],
)
def test_bmm_backward_finiteness(B, M, K, N, config_A, config_B):
    """Block FP8 batched matmul gradients are finite and non-zero."""
    torch.manual_seed(42)
    act = torch.randn(B, M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(B, N, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight_b = weight.transpose(-1, -2)  # [B, K, N]

    to_block_fp8_then_bmm(act, weight_b, config_A, config_B).sum().backward()

    for name, g in [("act", act.grad), ("weight", weight.grad)]:
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "shape, axis",
    [
        ((64, 128), -2),
        ((64, 128), -1),
        ((2, 64, 128), -2),
        ((2, 64, 128), -1),
        ((4, 128, 256), -2),
        ((4, 128, 256), -1),
        ((1024, 2048), -2),
        ((1024, 2048), -1),
    ],
)
def test_mxfp4_fused_op_equivalence(shape, axis):
    """
    Old (mxfp4_fake_quantize + dynamic_block_mx_quant) and new
    (fused cann_ops_nn.mx_to_block_mx_quant) paths produce identical results.
    """

    torch.manual_seed(42)
    B = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    # --- Old path: two-step ---
    hp_tensor = mxfp4_fake_quantize(B, config.mxfp4_fake_quantize_config, axis=axis)
    B_q_old, B_s1_old, B_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp_tensor,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: fused op ---
    B_q_new, B_s1_new, B_s2_new = block_fp8_quantize(B, axis=axis, config=config)

    # --- Compare: dequantize block FP8 to bf16, then torch.equal ---
    # Block FP8 scale is broadcast across each 32×32 block (MXFP8 format).
    # Dequant along K-dim using B_s2 (forward scale) or N-dim using B_s1 (backward scale).
    block_size = 32
    for axis, s_old, s_new, label in [(-2, B_s2_old, B_s2_new, "s2"), (-1, B_s1_old, B_s1_new, "s1")]:
        dq_old = mxfp8_dequantize(
            B_q_old, s_old, axis=axis, block_size=block_size, output_shape=B.shape, output_dtype=B.dtype
        )
        dq_new = mxfp8_dequantize(
            B_q_new, s_new, axis=axis, block_size=block_size, output_shape=B.shape, output_dtype=B.dtype
        )
        assert torch.equal(dq_old, dq_new), f"Dequantized values differ with {label}"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, axis",
    [
        (128, 64, 256, -2),
        (64, 128, 64, -1),
    ],
)
def test_mxfp4_matmul_equivalence(M, K, N, axis):
    """Matmul output using old vs new quantization paths has acceptable SQNR."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_A = MXQuantizeConfig()

    # Quantize A once (shared)
    A_q1, A_s1, _, _ = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        A.reshape(-1, A.shape[-1]),
        round_mode=config_A.round_mode,
        dst_type=config_A.elem_dtype,
        scale_alg=config_A.scale_alg,
        dst_type_max=config_A.dst_type_max,
    )

    # --- Old path: quantize B ---
    hp = mxfp4_fake_quantize(B, config.mxfp4_fake_quantize_config, axis=axis)
    B_q_old, _, B_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    Y_old = torch_npu.npu_quant_matmul(
        A_q1,
        B_q_old,
        B_s2_old,
        pertoken_scale=A_s1,
        output_dtype=A.dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    if A.ndim != 2:
        Y_old = Y_old.reshape(*A.shape[:-1], *Y_old.shape[1:])

    # --- New path: quantize B ---
    B_q_new, _, B_s2_new = block_fp8_quantize(B, axis=axis, config=config)
    Y_new = torch_npu.npu_quant_matmul(
        A_q1,
        B_q_new,
        B_s2_new,
        pertoken_scale=A_s1,
        output_dtype=A.dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    if A.ndim != 2:
        Y_new = Y_new.reshape(*A.shape[:-1], *Y_new.shape[1:])

    assert torch.equal(Y_old, Y_new), "Matmul results differ between old and new quantization paths"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes",
    [
        (192, 64, 128, 3, [64, 64, 64]),
    ],
)
def test_mxfp4_grouped_matmul_equivalence(M, K, N, E, group_sizes):
    """Grouped matmul output using old vs new quantization paths is identical."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)
    config = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_A = MXQuantizeConfig()

    # Quantize A once (shared)
    A_q1, A_s1 = torch_npu.npu_dynamic_mx_quant(
        A,
        axis=-1,
        round_mode=config_A.round_mode,
        dst_type=config_A.elem_dtype,
        block_size=config_A.block_size,
        scale_alg=config_A.scale_alg,
        dst_type_max=config_A.dst_type_max,
    )

    # --- Old path: quantize B ---
    hp = mxfp4_fake_quantize(B, config.mxfp4_fake_quantize_config, axis=-2)
    B_q_old, _, B_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    Y_old = torch_npu.npu_grouped_matmul(
        [A_q1],
        [B_q_old],
        scale=[B_s2_old],
        per_token_scale=[A_s1],
        group_list=group_list.to(torch.int64),
        group_type=0,
        output_dtype=A.dtype,
        group_list_type=0,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=3,
    )[0]

    # --- New path: quantize B ---
    B_q_new, _, B_s2_new = block_fp8_quantize(B, axis=-2, config=config)
    Y_new = torch_npu.npu_grouped_matmul(
        [A_q1],
        [B_q_new],
        scale=[B_s2_new],
        per_token_scale=[A_s1],
        group_list=group_list.to(torch.int64),
        group_type=0,
        output_dtype=A.dtype,
        group_list_type=0,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=3,
    )[0]

    assert torch.equal(Y_old, Y_new), "Grouped matmul results differ between old and new quantization paths"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, axis",
    [
        (128, 64, 256, -2),
        (64, 128, 64, -1),
    ],
)
def test_mxfp4_matmul_backward_equivalence(M, K, N, axis):
    """Backward gradients using old vs new quantization paths are identical."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_A = MXQuantizeConfig()

    A_flat = A.reshape(-1, A.shape[-1])

    # Quantize A once (shared)
    A_q1, A_s1, A_q2, A_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        A_flat,
        round_mode=config_A.round_mode,
        dst_type=config_A.elem_dtype,
        scale_alg=config_A.scale_alg,
        dst_type_max=config_A.dst_type_max,
    )

    # --- Old path: quantize B ---
    hp = mxfp4_fake_quantize(B, config.mxfp4_fake_quantize_config, axis=axis)
    B_q_old, B_s1_old, B_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: quantize B ---
    B_q_new, B_s1_new, B_s2_new = block_fp8_quantize(B, axis=axis, config=config)

    def _backward(dY, A_q2, A_s2, B_q, B_s1):
        dY_q1, dY_s1, dY_q2, dY_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            dY.reshape(-1, dY.shape[-1]),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )
        dA = torch_npu.npu_quant_matmul(
            dY_q1,
            B_q.t(),
            B_s1.transpose(0, 1),
            pertoken_scale=dY_s1,
            output_dtype=A.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        dB = torch_npu.npu_quant_matmul(
            A_q2.t(),
            dY_q2,
            dY_s2,
            pertoken_scale=A_s2.transpose(0, 1),
            output_dtype=A.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])
        return dA, dB

    # Forward matmul + backward for both paths
    Y_old = torch_npu.npu_quant_matmul(
        A_q1,
        B_q_old,
        B_s2_old,
        pertoken_scale=A_s1,
        output_dtype=A.dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    if A.ndim != 2:
        Y_old = Y_old.reshape(*A.shape[:-1], *Y_old.shape[1:])
    dY = torch.randn_like(Y_old)
    dA_old, dB_old = _backward(dY, A_q2, A_s2, B_q_old, B_s1_old)

    Y_new = torch_npu.npu_quant_matmul(
        A_q1,
        B_q_new,
        B_s2_new,
        pertoken_scale=A_s1,
        output_dtype=A.dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    if A.ndim != 2:
        Y_new = Y_new.reshape(*A.shape[:-1], *Y_new.shape[1:])
    dA_new, dB_new = _backward(dY, A_q2, A_s2, B_q_new, B_s1_new)

    assert torch.equal(dA_old, dA_new), "dA differs between old and new quantization paths"
    assert torch.equal(dB_old, dB_new), "dB differs between old and new quantization paths"


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N, E, group_sizes",
    [
        (192, 64, 128, 3, [64, 64, 64]),
    ],
)
def test_mxfp4_grouped_matmul_backward_equivalence(M, K, N, E, group_sizes):
    """Grouped backward gradients using old vs new quantization paths are identical."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)
    config = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_A = MXQuantizeConfig()

    # Quantize A once (shared)
    A_q1, A_s1 = torch_npu.npu_dynamic_mx_quant(
        A,
        axis=-1,
        round_mode=config_A.round_mode,
        dst_type=config_A.elem_dtype,
        block_size=config_A.block_size,
        scale_alg=config_A.scale_alg,
        dst_type_max=config_A.dst_type_max,
    )
    A_q2, A_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
        A,
        group_list.to(torch.int32),
        round_mode=config_A.round_mode,
        dst_type=config_A.elem_dtype,
        blocksize=config_A.block_size,
        scale_alg=config_A.scale_alg,
    )

    # --- Old path: quantize B ---
    hp = mxfp4_fake_quantize(B, config.mxfp4_fake_quantize_config, axis=-2)
    B_q_old, B_s1_old, B_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: quantize B ---
    B_q_new, B_s1_new, B_s2_new = block_fp8_quantize(B, axis=-2, config=config)

    def _grouped_backward(dY, A_q2, A_s2, B_q, B_s1):
        dY_q1, dY_s1 = torch_npu.npu_dynamic_mx_quant(
            dY,
            axis=-1,
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            block_size=config_A.block_size,
            scale_alg=config_A.scale_alg,
            dst_type_max=config_A.dst_type_max,
        )
        dY_q2, dY_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dY,
            group_list.to(torch.int32),
            round_mode=config_A.round_mode,
            dst_type=config_A.elem_dtype,
            blocksize=config_A.block_size,
            scale_alg=config_A.scale_alg,
        )
        dA = torch_npu.npu_grouped_matmul(
            [dY_q1],
            [B_q.transpose(-1, -2)],
            scale=[B_s1.transpose(1, 2)],
            per_token_scale=[dY_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=A.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]
        dB = torch_npu.npu_grouped_matmul(
            [A_q2.t()],
            [dY_q2],
            scale=[dY_s2],
            per_token_scale=[A_s2.transpose(0, 1)],
            group_list=group_list.to(torch.int64),
            group_type=2,
            output_dtype=A.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]
        return dA, dB

    # Forward + backward for both paths
    Y_old = torch_npu.npu_grouped_matmul(
        [A_q1],
        [B_q_old],
        scale=[B_s2_old],
        per_token_scale=[A_s1],
        group_list=group_list.to(torch.int64),
        group_type=0,
        output_dtype=A.dtype,
        group_list_type=0,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=3,
    )[0]
    dY = torch.randn_like(Y_old)
    dA_old, dB_old = _grouped_backward(dY, A_q2, A_s2, B_q_old, B_s1_old)

    Y_new = torch_npu.npu_grouped_matmul(
        [A_q1],
        [B_q_new],
        scale=[B_s2_new],
        per_token_scale=[A_s1],
        group_list=group_list.to(torch.int64),
        group_type=0,
        output_dtype=A.dtype,
        group_list_type=0,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=3,
    )[0]
    dA_new, dB_new = _grouped_backward(dY, A_q2, A_s2, B_q_new, B_s1_new)

    assert torch.equal(dA_old, dA_new), "dA differs between old and new quantization paths"
    assert torch.equal(dB_old, dB_new), "dB differs between old and new quantization paths"
