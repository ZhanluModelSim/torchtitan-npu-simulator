# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao_npu.ops.float8_ops import float8_rowwise_fake_quantize


def test_float8_rowwise_fake_quantize_preserves_shape_dtype_and_uses_ste():
    weight = torch.linspace(-2, 2, 64, dtype=torch.float32).reshape(2, 32).requires_grad_()
    config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow(dim=-1))

    result = float8_rowwise_fake_quantize(weight, config, PerRow(dim=-1))

    assert result.shape == weight.shape
    assert result.dtype == weight.dtype
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_float8_rowwise_fake_quantize_rejects_noncontiguous_quantized_dim():
    weight = torch.randn(4, 8).transpose(0, 1)
    config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow(dim=-1))

    with pytest.raises(AssertionError, match="contiguous"):
        float8_rowwise_fake_quantize(weight, config, PerRow(dim=-1))
