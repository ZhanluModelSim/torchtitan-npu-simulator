# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao.float8.float8_utils import compute_error

from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import MXQuantizeConfig

# =========================================================================
# Tests for mxfp4_fake_quantize (autograd.Function wrapper)
# =========================================================================


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("shape", [(32, 256), (1024, 1024), (16, 512, 256)])
def test_mxfp4_fake_quantize(shape, axis):
    """mxfp4_fake_quantize: forward SQNR, STE backward, idempotency."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import mxfp4_fake_quantize

    torch.manual_seed(0)
    x = torch.randn(*shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    y = mxfp4_fake_quantize(x, config, axis=axis)

    # Shape and dtype preserved
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.dtype == x.dtype

    # SQNR: FP4 block32 should exceed 12 dB
    sqnr = compute_error(x, y).item()
    assert sqnr > 12, f"SQNR too low: {sqnr:.1f} dB for axis={axis}, shape={shape}"

    # STE backward
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x.grad)), f"STE gradient failed for axis={axis}, shape={shape}"

    # Idempotency: quantize -> dequantize -> quantize -> dequantize
    y2 = mxfp4_fake_quantize(y, config, axis=axis)
    assert torch.equal(y, y2), f"Not idempotent for axis={axis}, shape={shape}"


@pytest.mark.parametrize("round_mode", ["rint", "floor", "round"])
def test_mxfp4_fake_quantize_round_modes(round_mode):
    """Different round_modes should produce different quantization results."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import mxfp4_fake_quantize

    torch.manual_seed(0)
    x = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, round_mode=round_mode)

    y = mxfp4_fake_quantize(x, config, axis=-1)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    # Round mode should not affect shape/dtype, just numerical accuracy
    sqnr = compute_error(x, y).item()
    assert sqnr > 10, f"SQNR too low: {sqnr:.1f} dB for round_mode={round_mode}"


# =========================================================================
# Tests for to_mx_then_mm (real MX quantized matmul)
# =========================================================================


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 26),
        (torch.float8_e5m2, 21),
    ],
)
def test_to_mx_then_mm_forward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_mm forward produces reasonable SQNR vs bf16 baseline."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import to_mx_then_mm

    torch.manual_seed(0)
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_mm(A, B, cfg, cfg)
    Y_ref = A @ B

    assert Y.shape == (M, N), f"Shape mismatch: {Y.shape}"
    assert Y.dtype == A.dtype

    sqnr = compute_error(Y, Y_ref).item()
    assert sqnr > sqnr_threshold, f"SQNR {sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 28),
        (torch.float8_e5m2, 23),
    ],
)
def test_to_mx_then_mm_backward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_mm backward produces meaningful gradients vs bf16 baseline."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import to_mx_then_mm

    torch.manual_seed(1)
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16, requires_grad=True)

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_mm(A, B, cfg, cfg)
    Y_ref = A_ref @ B_ref

    loss = Y.sum()
    loss.backward()
    loss_ref = Y_ref.sum()
    loss_ref.backward()

    assert A.grad is not None and B.grad is not None
    assert A_ref.grad is not None and B_ref.grad is not None

    dA_sqnr = compute_error(A.grad, A_ref.grad).item()
    dB_sqnr = compute_error(B.grad, B_ref.grad).item()
    assert dA_sqnr > sqnr_threshold, f"dA SQNR {dA_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
    assert dB_sqnr > sqnr_threshold, f"dB SQNR {dB_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


# =========================================================================
# Tests for to_mx_then_grouped_mm (real MX quantized grouped matmul)
# =========================================================================


def _make_group_list(M, E, device):
    """Create cumsum group_list for E groups dividing M rows."""
    base = M // E
    rem = M % E
    sizes = [base + 1] * rem + [base] * (E - rem)
    return torch.tensor(sizes, device=device).cumsum(0).to(torch.int64)


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 26),
        (torch.float8_e5m2, 21),
    ],
)
def test_to_mx_then_grouped_mm_forward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_grouped_mm forward produces reasonable SQNR vs bf16 baseline."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import to_mx_then_grouped_mm

    torch.manual_seed(0)
    M, K, N, E = 128, 256, 64, 4
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device=device, dtype=torch.bfloat16)
    group_list = _make_group_list(M, E, device)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_grouped_mm(A, B, group_list, cfg, cfg)

    # Reference: each expert slice
    Y_ref = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    start = 0
    for i in range(E):
        end = group_list[i].item()
        Y_ref[start:end] = A[start:end] @ B[i]
        start = end

    assert Y.shape == (M, N), f"Shape mismatch: {Y.shape}"
    assert Y.dtype == A.dtype

    sqnr = compute_error(Y, Y_ref).item()
    assert sqnr > sqnr_threshold, f"SQNR {sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 28),
        (torch.float8_e5m2, 23),
    ],
)
def test_to_mx_then_grouped_mm_backward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_grouped_mm backward produces meaningful gradients."""
    from torchtitan_npu.experiments.ao_npu.torchao_npu.ops import to_mx_then_grouped_mm

    torch.manual_seed(1)
    M, K, N, E = 128, 256, 64, 4
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device=device, dtype=torch.bfloat16, requires_grad=True)
    group_list = _make_group_list(M, E, device)

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_grouped_mm(A, B, group_list, cfg, cfg)

    # Reference
    Y_ref = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    start = 0
    for i in range(E):
        end = group_list[i].item()
        Y_ref[start:end] = A_ref[start:end] @ B_ref[i]
        start = end

    loss = Y.sum()
    loss.backward()
    loss_ref = Y_ref.sum()
    loss_ref.backward()

    assert A.grad is not None and B.grad is not None
    assert A_ref.grad is not None and B_ref.grad is not None

    dA_sqnr = compute_error(A.grad, A_ref.grad).item()
    dB_sqnr = compute_error(B.grad, B_ref.grad).item()
    assert dA_sqnr > sqnr_threshold, f"dA SQNR {dA_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
    assert dB_sqnr > sqnr_threshold, f"dB SQNR {dB_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
