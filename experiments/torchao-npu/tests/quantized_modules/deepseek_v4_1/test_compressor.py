# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.quantization.qat import QATStep
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import DeepSeekV41QuantCompressorConfig


class _CompressorHost(torch.nn.Module):
    """Small host with distinct latent/rotated outputs and a shared-KV branch."""

    def __init__(self, is_source=True):
        super().__init__()
        self.is_source = is_source
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))

    def forward(self, x, positions, cmp_k=None):
        if not self.is_source:
            return cmp_k, None
        latent = x * self.weight
        return latent + positions.unsqueeze(-1), latent


def test_quant_compressor_preserves_latent_parameters_and_ste(monkeypatch):
    module = _CompressorHost()
    original_weight = module.weight
    original_keys = tuple(module.state_dict())
    original_forward = module.forward
    quantize_(module, DeepSeekV41QuantCompressorConfig(), filter_fn=lambda candidate, _: candidate is module)
    # Preparing twice must not wrap an already swapped forward recursively.
    quantize_(module, DeepSeekV41QuantCompressorConfig(), filter_fn=lambda candidate, _: candidate is module)
    captured = []

    def encode(cache, rows, slots, **kwargs):
        captured.append(rows.clone())
        assert kwargs["quant_mode"] == "mxfp4_bf16"
        assert kwargs["quant_group_size"] == 16
        cache[:, :16] = 0x21
        cache[:, 16:20] = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16).view(torch.uint8)

    monkeypatch.setattr(torch.ops.custom.kv_compress_epilog_v2, "default", encode)
    x = torch.linspace(0.125, 1, 64, dtype=torch.bfloat16).reshape(1, 2, 32).requires_grad_()
    positions = torch.tensor([[0, 2]], dtype=torch.bfloat16)

    main_kv, latent = module(x, positions)

    assert module.weight is original_weight
    assert tuple(module.state_dict()) == original_keys
    assert module._torchao_npu_original_forward == original_forward
    assert len(captured) == 1
    assert torch.equal(captured[0], (x.detach() * 2 + positions.unsqueeze(-1)).reshape(2, 32))
    assert torch.equal(latent, x.detach() * 2)
    expected = torch.tensor([0.25, 0.5] * 8 + [1.0, 2.0] * 8, dtype=x.dtype).repeat(2).reshape_as(x)
    assert main_kv.dtype == x.dtype
    assert torch.equal(main_kv, expected)
    (main_kv.float().sum() + 3 * latent.float().sum()).backward()
    assert torch.equal(x.grad, torch.full_like(x, 8))
    assert torch.equal(module.weight.grad, (x.detach() * 4).sum())

    reuser = _CompressorHost(is_source=False)
    quantize_(reuser, DeepSeekV41QuantCompressorConfig(), filter_fn=lambda candidate, _: candidate is reuser)
    shared, shared_latent = reuser(x, positions, main_kv)
    assert shared is main_kv
    assert shared_latent is None
    assert len(captured) == 1


def test_quant_compressor_convert_restores_host_forward():
    module = _CompressorHost()
    quantize_(module, DeepSeekV41QuantCompressorConfig(), filter_fn=lambda candidate, _: candidate is module)
    quantize_(
        module,
        DeepSeekV41QuantCompressorConfig(step=QATStep.CONVERT),
        filter_fn=lambda candidate, _: candidate is module,
    )
    x = torch.ones((1, 1, 32), dtype=torch.bfloat16)

    main_kv, latent = module(x, torch.tensor([[3]], dtype=torch.bfloat16))

    assert torch.equal(main_kv, x * 5)
    assert torch.equal(latent, x * 2)
