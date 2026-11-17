# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16


@pytest.mark.parametrize("shape", [(2, 3, 32), (6, 1, 32)], ids=["compressor", "smla"])
def test_kv_fake_quantize_decodes_packed_cache_and_preserves_ste(monkeypatch, shape):
    x = torch.linspace(-1, 1, 192, dtype=torch.bfloat16).reshape(shape).requires_grad_()
    seen = []

    def packed_fp4_encoder(cache, rows, slots, *, quant_group_size, quant_mode, round_scale):
        seen.append(rows.detach().clone())
        assert cache.dtype == torch.uint8
        assert cache.shape == (6, 32)
        assert torch.equal(slots, torch.arange(6, dtype=torch.int32))
        assert (quant_group_size, quant_mode, round_scale) == (16, "mxfp4_bf16", False)
        # E2M1 low-first pair [0.5, 1.0], with two distinct BF16 group scales.
        cache[:, :16] = 0x21
        cache[:, 16:20] = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16).view(torch.uint8)

    monkeypatch.setattr(torch.ops.custom.kv_compress_epilog_v2, "default", packed_fp4_encoder)

    result = fake_quantize_mx_bf16(x)

    expected = torch.tensor([0.25, 0.5] * 8 + [1.0, 2.0] * 8, dtype=x.dtype).repeat(6).reshape(shape)
    assert len(seen) == 1
    assert torch.equal(seen[0], x.detach().reshape(6, 32))
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.equal(result, expected)
    grad = torch.linspace(-0.5, 0.5, x.numel(), dtype=x.dtype).reshape_as(x)
    result.backward(grad)
    assert torch.equal(x.grad, grad)


@pytest.mark.parametrize(
    ("quant_mode", "group_size"),
    [("mxfp4_bf16", 16), ("mxfp8_bf16", 32)],
    ids=["main-kv-fp4", "swa-fp8"],
)
def test_kv_fake_quantize_real_epilog_round_trip_and_backward(quant_mode, group_size):
    x = torch.tensor([0.5, 1.0, 2.0], device="npu", dtype=torch.bfloat16)[:, None].repeat(1, 512)
    x.requires_grad_()

    result = fake_quantize_mx_bf16(x, quant_group_size=group_size, quant_mode=quant_mode)

    # FP8 round_scale=False uses amax / 448 stored as BF16; FP4 powers of two
    # are exact. Account for BF16 rounding of FP8's stored group scale.
    expected = (
        x.detach() if quant_mode == "mxfp4_bf16" else (x.detach().float() / 448).bfloat16().float().mul(448).bfloat16()
    )
    assert result.dtype == x.dtype
    assert torch.equal(result, expected)
    upstream = torch.linspace(-1, 1, x.numel(), device=x.device, dtype=x.dtype).reshape_as(x)
    result.backward(upstream)
    assert torch.equal(x.grad, upstream)
