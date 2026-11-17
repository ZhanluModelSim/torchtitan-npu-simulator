# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KV cache fake quantization with BF16 group scales."""

import custom_ops  # noqa: F401  # pyrefly: ignore[missing-import]
import torch

from torchao_npu.quantization.quant_primitives.mx_bf16 import dequantize_mx_bf16, kv_cache_layout


class _MXBF16FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, quant_group_size, quant_mode):  # pyrefly: ignore [bad-override]
        if kv.dtype != torch.bfloat16 or kv.ndim < 2:
            raise ValueError("KV cache quantization requires BF16 input with shape [..., head_dim]")
        head_dim = kv.shape[-1]
        _, _, _, cache_width = kv_cache_layout(head_dim, quant_mode, quant_group_size)
        rows = kv.reshape(-1, head_dim).contiguous()
        cache_dtype = torch.float8_e4m3fn if quant_mode == "mxfp8_bf16" else torch.uint8
        cache = torch.zeros((rows.shape[0], cache_width), dtype=cache_dtype, device=kv.device)
        slots = torch.arange(rows.shape[0], dtype=torch.int32, device=kv.device)
        torch.ops.custom.kv_compress_epilog_v2.default(  # pyrefly: ignore [missing-attribute]
            cache,
            rows,
            slots,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale=False,
        )
        return dequantize_mx_bf16(
            cache,
            d=head_dim,
            quant_mode=quant_mode,
            group_size=quant_group_size,
            output_dtype=kv.dtype,
        ).reshape_as(kv)

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        return grad_output, None, None


def fake_quantize_mx_bf16(
    kv: torch.Tensor,
    *,
    quant_group_size: int = 16,
    quant_mode: str = "mxfp4_bf16",
) -> torch.Tensor:
    """Quantize/dequantize KV, preserving its shape/dtype with an identity STE."""
    if quant_mode not in ("mxfp4_bf16", "mxfp8_bf16"):
        raise ValueError(f"unsupported KV quantization mode: {quant_mode}")
    return _MXBF16FakeQuantize.apply(kv, quant_group_size, quant_mode)
