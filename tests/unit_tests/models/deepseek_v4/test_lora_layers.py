# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.protocols.sharding import ShardingConfig

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _lora_model_config


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


def _cpu_grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
    starts = [0, *offsets_E[:-1].tolist()]
    return torch.cat(
        [
            F.linear(input_RI[start:end], weight_EOI[expert].to(input_RI.dtype))
            for expert, (start, end) in enumerate(zip(starts, offsets_E.tolist(), strict=True))
        ]
    )


@pytest.mark.parametrize("kind", ["dense", "batched", "grouped"], ids=["dense", "per-head", "routed"])
@pytest.mark.parametrize("replicated", [False, True], ids=["unsharded", "replicated"])
def test_standard_lora_projection_output_and_gradients_match_independent_formula(kind, replicated, monkeypatch):
    config = _lora_model_config()
    if kind == "dense":
        projection = config.layers[0].attention.wq_a
    elif kind == "batched":
        projection = config.layers[0].attention.wo_a
    else:
        projection = config.layers[0].moe.routed_experts.w13
        monkeypatch.setattr(GroupedLinear, "_grouped_mm", _cpu_grouped_mm)
    if replicated:
        projection.sharding_config = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})
    module = projection.build()
    if replicated:
        for adapter in (module.lora_a, module.lora_b):
            assert adapter._sharding_config.state_shardings["weight"] == dense_param_placement(tp=spmd.R)
    module.init_states(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        module.lora_b.weight.normal_(std=0.1)
    shape = (5, projection.n_heads, projection.in_features) if kind == "batched" else (5, projection.in_features)
    inputs = torch.randn(shape, requires_grad=True)
    a, b = module.lora_a.weight, module.lora_b.weight
    if kind == "grouped":
        offsets = torch.tensor([2, 5], dtype=torch.int32)
        actual = module(inputs, offsets)
        experts = torch.tensor([0, 0, 1, 1, 1])
        base = torch.einsum("ti,tsoi->tso", inputs, module.weight[experts])
        hidden = torch.einsum("ti,tri->tr", inputs, a[experts])
        expected = base + 2 * torch.einsum("tr,tsor->tso", hidden, b[experts])
    elif kind == "batched":
        actual = module(inputs)
        base = torch.einsum("thi,hoi->tho", inputs, module.weight.view(projection.n_heads, projection.out_features, -1))
        hidden = torch.einsum("thi,ri->thr", inputs, a)
        expected = base + 2 * torch.einsum(
            "thr,hor->tho", hidden, b.view(projection.n_heads, projection.out_features, -1)
        )
    else:
        actual = module(inputs)
        expected = F.linear(inputs, module.weight) + 2 * (inputs @ a.T @ b.T)
    torch.testing.assert_close(actual, expected)
    upstream = torch.linspace(-0.5, 0.7, actual.numel()).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, (inputs, a, b), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (inputs, a, b), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert module.weight.grad is None and not module.weight.requires_grad
