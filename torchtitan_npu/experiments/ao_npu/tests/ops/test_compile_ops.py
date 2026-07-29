# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-mode tests for NPU quantized matmul ops.

These tests verify that ``torch.compile`` works correctly for all
quantization paths (MX FP8, MX FP4, Block FP8).  The key concern is that
``npu_quant_matmul``'s ``x1_dtype``/``x2_dtype`` parameters are correctly
omitted for standard FP8 types, where the tensor's native dtype is
sufficient, but passed for FP4 where the tensor is stored as ``uint8``.

The compilation uses ``backend="inductor"`` with ``inductor_npu_ext``,
which is the standard NPU compile path used by torchtitan-npu.

First invocation can be slow due to TBE kernel compilation (~minutes);
subsequent runs with the same shape use the cached graph.
"""

# Extends the inductor backend for NPU; import order matters (before torch.compile).
import inductor_npu_ext  # noqa: F401
import pytest
import torch

from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.block_ops import (
    to_block_fp8_then_mm,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.ops.mx_ops import to_mx_then_mm
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)


def _npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


# ============================================================================
# MX quantized matmul (to_mx_then_mm)
# ============================================================================


class MXMMModel(torch.nn.Module):
    def __init__(self, config_A, config_B):
        super().__init__()
        self.config_A = config_A
        self.config_B = config_B

    def forward(self, A, B):
        return to_mx_then_mm(A, B, self.config_A, self.config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_mx_fp8_matmul_compile(M, K, N):
    """``to_mx_then_mm`` with FP8 works under torch.compile."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig()  # default: float8_e4m3fn

    model = MXMMModel(config, config).npu()
    compiled = torch.compile(model, backend="inductor", dynamic=False)

    Y = compiled(A, B)
    assert Y.shape == (M, N), f"Expected ({M}, {N}), got {Y.shape}"
    assert Y.dtype == torch.bfloat16


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "npu_dynamic_mx_quant_with_dual_axis meta registration computes "
        "y1.shape[-1] = K*2 instead of K/2 for FP4 under compile"
    ),
)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_mx_fp4_matmul_compile(M, K, N):
    """``to_mx_then_mm`` with FP4 works under torch.compile.

    FP4 tensors are stored as ``uint8`` (two FP4 values per byte), so
    ``x1_dtype``/``x2_dtype`` must still be passed.
    """
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    model = MXMMModel(config, config).npu()
    compiled = torch.compile(model, backend="inductor", dynamic=False)

    Y = compiled(A, B)
    assert Y.shape == (M, N), f"Expected ({M}, {N}), got {Y.shape}"
    assert Y.dtype == torch.bfloat16


# ============================================================================
# Block FP8 quantized matmul (to_block_fp8_then_mm)
# ============================================================================


class BlockMMModel(torch.nn.Module):
    def __init__(self, config_A, config_B):
        super().__init__()
        self.config_A = config_A
        self.config_B = config_B

    def forward(self, A, B):
        return to_block_fp8_then_mm(A, B, self.config_A, self.config_B)


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_block_fp8_without_mxfp4_compile(M, K, N):
    """``to_block_fp8_then_mm`` without MXFP4 fake-quant works under compile."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config_A = MXQuantizeConfig()
    config_B = BlockQuantizeConfig()  # no mxfp4_fake_quantize_config

    model = BlockMMModel(config_A, config_B).npu()
    compiled = torch.compile(model, backend="inductor", dynamic=False)

    Y = compiled(A, B)
    assert Y.shape == (M, N), f"Expected ({M}, {N}), got {Y.shape}"
    assert Y.dtype == torch.bfloat16


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_block_fp8_with_mxfp4_compile(M, K, N):
    """``to_block_fp8_then_mm`` with MXFP4 fake-quant works under compile."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="npu", dtype=torch.bfloat16)
    config_A = MXQuantizeConfig()
    config_B = BlockQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    model = BlockMMModel(config_A, config_B).npu()
    compiled = torch.compile(model, backend="inductor", dynamic=False)

    Y = compiled(A, B)
    assert Y.shape == (M, N), f"Expected ({M}, {N}), got {Y.shape}"
    assert Y.dtype == torch.bfloat16
