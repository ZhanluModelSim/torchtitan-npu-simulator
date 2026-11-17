# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-mode tests for NPU quantized matmul ops.

These tests verify that ``torch.compile`` works correctly for all
quantization paths (MX FP8, MX FP4, Block MX).  The key concern is that
``npu_quant_matmul``'s ``x1_dtype``/``x2_dtype`` parameters are correctly
omitted for standard FP8 types, where the tensor's native dtype is
sufficient, but passed for FP4 where the tensor is stored as ``uint8``.

The compilation uses ``backend="inductor"`` and selects the bundled AscendC
codegen with the ``npu_backend`` compile option.

First invocation can be slow due to TBE kernel compilation (~minutes);
subsequent runs with the same shape use the cached graph.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.ops.block_mx_ops import (
    to_block_mx_then_mm,
)
from torchao_npu.ops.hif8_ops import to_hif8_then_bmm, to_hif8_then_mm
from torchao_npu.ops.mx_ops import to_mx_then_mm
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)


def _compile_and_assert_output(model, lhs, rhs, *, expected_shape):
    compiled = torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        options={"npu_backend": "ascendc"},
    )
    output = compiled(lhs, rhs)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    assert output.dtype == torch.bfloat16


# ============================================================================
# MX quantized matmul (to_mx_then_mm)
# ============================================================================


class MXMMModel(torch.nn.Module):
    def __init__(self, config_a, config_b):
        super().__init__()
        self.config_a = config_a
        self.config_b = config_b

    def forward(self, a, b):
        return to_mx_then_mm(a, b, self.config_a, self.config_b)


@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_mx_fp8_matmul_compile(m, k, n):
    """``to_mx_then_mm`` with FP8 works under torch.compile."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig()  # default: float8_e4m3fn

    model = MXMMModel(config, config).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(m, n))


@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_mx_fp4_matmul_compile(m, k, n):
    """``to_mx_then_mm`` with FP4 works under torch.compile.

    FP4 tensors are stored as ``uint8`` (two FP4 values per byte), so
    ``x1_dtype``/``x2_dtype`` must still be passed.
    """
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    model = MXMMModel(config, config).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(m, n))


# ============================================================================
# Block MX quantized matmul (to_block_mx_then_mm)
# ============================================================================


class BlockMXMMModel(torch.nn.Module):
    def __init__(self, config_a, config_b):
        super().__init__()
        self.config_a = config_a
        self.config_b = config_b

    def forward(self, a, b):
        return to_block_mx_then_mm(a, b, self.config_a, self.config_b)


@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_block_mx_without_mxfp4_compile(m, k, n):
    """``to_block_mx_then_mm`` without MXFP4 fake-quant works under compile."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config_a = MXQuantizeConfig()
    config_b = BlockMXQuantizeConfig()  # no mxfp4_fake_quantize_config

    model = BlockMXMMModel(config_a, config_b).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(m, n))


@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_block_mx_with_mxfp4_compile(m, k, n):
    """``to_block_mx_then_mm`` with MXFP4 fake-quant works under compile."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config_a = MXQuantizeConfig()
    config_b = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    model = BlockMXMMModel(config_a, config_b).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(m, n))


# ============================================================================
# HiF8 quantized matmul (to_hif8_then_mm / to_hif8_then_bmm)
# ============================================================================


class HiF8MMModel(torch.nn.Module):
    def __init__(self, config_a, config_b):
        super().__init__()
        self.config_a = config_a
        self.config_b = config_b

    def forward(self, a, b):
        return to_hif8_then_mm(a, b, self.config_a, self.config_b)


@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 4096, 2048),
        (4096, 2048, 4096),
    ],
)
def test_hif8_matmul_compile(m, k, n):
    """``to_hif8_then_mm`` works under torch.compile."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config = HiF8QuantizeConfig()

    model = HiF8MMModel(config, config).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(m, n))


class HiF8BMMModel(torch.nn.Module):
    def __init__(self, config_a, config_b):
        super().__init__()
        self.config_a = config_a
        self.config_b = config_b

    def forward(self, a, b):
        return to_hif8_then_bmm(a, b, self.config_a, self.config_b)


def test_hif8_bmm_compile_with_transposed_activation():
    """``to_hif8_then_bmm`` works under ``torch.compile`` when the activation is
    a non-contiguous permutation view -- the ``BatchedLinear`` layout that
    ``quantize_hifloat8``'s transpose-avoidance path (descending-stride
    permute, no ``is_contiguous()`` guard) targets.
    """
    torch.manual_seed(42)
    T, H, K, N = 512, 8, 512, 512
    base = torch.randn(T, H, K, device="npu", dtype=torch.bfloat16)
    a = base.transpose(0, 1)  # non-contiguous view: [H, T, K]
    b = torch.randn(H, K, N, device="npu", dtype=torch.bfloat16)
    config = HiF8QuantizeConfig()

    model = HiF8BMMModel(config, config).npu()
    _compile_and_assert_output(model, a, b, expected_shape=(H, T, N))
