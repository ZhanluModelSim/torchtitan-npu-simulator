# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 row-wise fake-quantization ops for NPU."""

import torch
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.quant_primitives import _choose_scale_float8
from torchao.quantization.utils import get_block_size


class Float8RowwiseFakeQuantize(torch.autograd.Function):
    """Per-row FP8 fake-quantize with straight-through estimator."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        weight: torch.Tensor,
        config: Float8FakeQuantizeConfig,
        granularity: PerRow,
    ) -> torch.Tensor:
        assert weight.stride(granularity.dim) == 1, "Fake-quantized dim should be contiguous."

        original_dtype = weight.dtype
        float8_dtype = config.dtype
        weight_fp32 = weight.to(torch.float32)

        # Compute scale
        block_size = get_block_size(weight.shape, granularity)
        scale = _choose_scale_float8(
            weight_fp32,
            block_size,
            float8_dtype,
            hp_value_lb=config.hp_value_lb,
            hp_value_ub=config.hp_value_ub,
        ).detach()
        scale = torch.clamp_min(scale, torch.finfo(original_dtype).eps)

        # Quantize
        weight_scaled = weight_fp32 / scale
        max_value = torch.finfo(float8_dtype).max
        weight_clamped = weight_scaled.clamp(min=-max_value, max=max_value)
        q = weight_clamped.to(float8_dtype)

        # Dequantize
        q_fp32 = q.to(torch.float32)
        dq = (q_fp32 * scale).to(original_dtype)
        return dq

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        return grad_output, None, None


def float8_rowwise_fake_quantize(
    weight: torch.Tensor,
    config: Float8FakeQuantizeConfig,
    granularity: PerRow,
) -> torch.Tensor:
    """Thin wrapper around ``Float8RowwiseFakeQuantize.apply``."""
    return Float8RowwiseFakeQuantize.apply(weight, config, granularity)
