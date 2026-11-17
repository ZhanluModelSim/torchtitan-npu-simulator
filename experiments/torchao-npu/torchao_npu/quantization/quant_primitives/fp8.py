# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 per-token-head quantization with FP32 dequantization scales."""

__all__ = ["fp8_quantize"]

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import FP8QuantizeConfig


def fp8_quantize(
    tensor: torch.Tensor,
    config: FP8QuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each last-axis vector independently to E4M3FN.

    For TND / BSND inputs, return FP8 data with the input shape and FP32
    descales shaped TN / BSN. Reconstruction is ``data.float() *
    descale.unsqueeze(-1)``. DynamicQuant reduces the last axis, so each
    token/head gets its own scale without flattening or mixing heads.
    Scale calculation and zero-vector handling follow the fused NPU kernel.
    """
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NPU FP8 DynamicQuant requires float16 or bfloat16 input")
    if tensor.ndim < 2:
        raise ValueError("NPU FP8 DynamicQuant requires input with at least two dimensions")
    return torch_npu.npu_dynamic_quant(
        tensor,
        dst_type=config.npu_elem_dtype,
        quant_mode=config.quant_mode,
    )
