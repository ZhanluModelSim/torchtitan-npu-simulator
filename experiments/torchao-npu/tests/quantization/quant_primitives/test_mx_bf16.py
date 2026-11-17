# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao_npu.quantization.quant_primitives.mx_bf16 import (
    dequantize_mx_bf16,
    kv_cache_layout,
)

_FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _pack_cache(data: torch.Tensor, scales: torch.Tensor, row_bytes: int) -> torch.Tensor:
    """Build cache rows with nonzero padding from quantized data and BF16 scales."""

    data_bytes = data.contiguous().view(torch.uint8)
    scale_bytes = scales.contiguous().view(torch.uint8)
    cache = torch.full((data.shape[0], row_bytes), 0xA5, dtype=torch.uint8)
    cache[:, : data_bytes.shape[1]] = data_bytes
    cache[:, data_bytes.shape[1] : data_bytes.shape[1] + scale_bytes.shape[1]] = scale_bytes
    return cache


def _legacy_dequantize(
    cache: torch.Tensor,
    d: int,
    quant_mode: int,
    group_size: int,
    fp8_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Retain the former decoder as the before-change equivalence reference."""

    num_groups = d // group_size
    data_bytes = d if quant_mode == 2 else d // 2
    payload_bytes = data_bytes + 2 * num_groups
    raw = cache.detach().view(torch.uint8).contiguous()

    scale_bytes = raw[:, data_bytes:payload_bytes].reshape(raw.shape[0], num_groups, 2)
    scale_bits = scale_bytes[..., 0].to(torch.int32) | (scale_bytes[..., 1].to(torch.int32) << 8)
    scales = (scale_bits << 16).contiguous().view(torch.float32)

    if quant_mode == 2:
        assert fp8_dtype is not None
        values = raw[:, :data_bytes].contiguous().view(fp8_dtype).float()
    else:
        packed = raw[:, :data_bytes]
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(raw.shape[0], d).long()
        values = _FP4_VALUES[codes]

    restored = values.reshape(raw.shape[0], num_groups, group_size) * scales.unsqueeze(-1)
    return restored.reshape(raw.shape[0], d).to(torch.bfloat16)


@pytest.mark.parametrize(
    ("fp8_dtype", "native_storage"),
    [
        pytest.param(torch.float8_e4m3fn, False, id="e4m3-uint8"),
        pytest.param(torch.float8_e4m3fn, True, id="e4m3-native"),
        pytest.param(torch.float8_e5m2, False, id="e5m2-uint8"),
        pytest.param(torch.float8_e5m2, True, id="e5m2-native"),
    ],
)
def test_mxfp8_bf16_decode_matches_legacy_and_independent_expected(fp8_dtype, native_storage):
    d = 64
    group_size = 32
    _, _, _, row_bytes = kv_cache_layout(d, "mxfp8_bf16", group_size)
    row = torch.tensor([-6.0, -3.0, -1.5, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    data = torch.stack((row.repeat(4), row.flip(0).repeat(4))).to(fp8_dtype)
    scales = torch.tensor([[0.25, 2.0], [0.5, 4.0]], dtype=torch.bfloat16)
    cache_bytes = _pack_cache(data, scales, row_bytes)
    cache = cache_bytes.view(fp8_dtype) if native_storage else cache_bytes
    explicit_dtype = None if native_storage else fp8_dtype

    result = dequantize_mx_bf16(
        cache,
        d,
        "mxfp8_bf16",
        group_size,
        explicit_dtype,
    )
    legacy = _legacy_dequantize(cache, d, 2, group_size, fp8_dtype)
    expected = (
        (data.float().reshape(2, d // group_size, group_size) * scales.float().unsqueeze(-1))
        .reshape(2, d)
        .to(torch.bfloat16)
    )

    assert result.dtype == torch.bfloat16
    assert torch.equal(result, legacy)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("group_size", [16, 32], ids=["group16", "group32"])
def test_mxfp4_bf16_decode_matches_legacy_and_independent_expected(group_size):
    d = 64
    num_groups, _, _, row_bytes = kv_cache_layout(d, "mxfp4_bf16", group_size)
    codes = torch.arange(16, dtype=torch.uint8).repeat(4)
    codes = torch.stack((codes, codes.roll(5)))
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.tensor(
        [[0.25, 0.5, 1.0, 2.0], [4.0, 2.0, 0.5, 0.25]],
        dtype=torch.bfloat16,
    )[:, :num_groups]
    cache = _pack_cache(packed, scales, row_bytes)

    result = dequantize_mx_bf16(cache, d, "mxfp4_bf16", group_size)
    legacy = _legacy_dequantize(cache, d, 4, group_size)
    expected = (
        (_FP4_VALUES[codes.long()].reshape(2, num_groups, group_size) * scales.float().unsqueeze(-1))
        .reshape(2, d)
        .to(torch.bfloat16)
    )

    assert result.dtype == torch.bfloat16
    assert torch.equal(result, legacy)
    assert torch.equal(result, expected)


def test_dequantize_mx_bf16_uses_output_dtype():
    d = 32
    group_size = 32
    _, _, _, row_bytes = kv_cache_layout(d, "mxfp4_bf16", group_size)
    codes = torch.arange(16, dtype=torch.uint8).repeat(2).unsqueeze(0)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.tensor([[0.3]], dtype=torch.bfloat16)
    cache = _pack_cache(packed, scales, row_bytes)

    result = dequantize_mx_bf16(
        cache,
        d,
        "mxfp4_bf16",
        group_size,
        output_dtype=torch.float32,
    )
    expected = _FP4_VALUES[codes.long()] * scales.float().unsqueeze(-1)

    assert result.dtype == torch.float32
    assert torch.equal(result, expected.reshape(1, d))
