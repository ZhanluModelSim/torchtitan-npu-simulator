# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan_npu.models.deepseek_v4_1 import model_registry


@pytest.mark.parametrize("layer_id", [2, 20], ids=["ratio-two", "ratio-one"])
def test_compressor_pooling_outputs_and_gradients_match_explicit_groups(layer_id):
    config = model_registry("deepseek_v4_1_debugmodel_text").layers[layer_id].attention.compressor
    with torch.random.fork_rng(devices=[]):
        compressor = config.build()
        compressor.init_states(buffer_device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(51)
    x = torch.randn(8, config.wkv.in_features, generator=generator, requires_grad=True)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    rotated, latent = compressor(x_TD=x, positions_T=positions)

    projected = F.linear(x, compressor.wkv.weight)
    ratio = config.compress_ratio
    groups = []
    for start in range(0, x.size(0), ratio):
        values = projected[start : start + ratio]
        if ratio > 1:
            gates = F.linear(x[start : start + ratio], compressor.wgate.weight).float().softmax(0)
            values = (values.float() * gates).sum(0)
        else:
            values = values[0]
        groups.append(values)
    pooled = torch.stack(groups)
    expected_latent = pooled * (pooled.square().mean(-1, keepdim=True) + config.norm.eps).rsqrt()
    expected_latent = expected_latent * compressor.norm.weight
    group_positions = torch.tensor([int(positions[start]) for start in range(0, x.size(0), ratio)])
    expected_rotated = compressor.rope(expected_latent.unsqueeze(1), positions=group_positions).squeeze(1)
    upstream = torch.randn(rotated.shape, generator=generator)
    inputs = (x, *compressor.parameters())
    actual_gradients = torch.autograd.grad(rotated, inputs, upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_rotated, inputs, upstream)

    torch.testing.assert_close(latent, expected_latent, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(rotated, expected_rotated, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
