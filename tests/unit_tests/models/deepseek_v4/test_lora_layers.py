# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torchtitan.components.optim import AdamW
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import GroupedLinear

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _lora_model_config
from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.models.common.deepseek_lora import BatchedLinearLoRA
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _apply_lora, _make_trainer_config


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


def test_native_wo_a_grouped_output_uses_both_lora_factors():
    config = _lora_model_config().layers[0].attention
    attention = config.build().double()
    projection = attention.wo_a
    projection.init_states(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        projection.lora_b.weight.normal_(std=0.1)
    inputs = torch.randn(3, attention.n_groups, projection.in_features, dtype=torch.float64)
    scale = config.wo_a.alpha / config.wo_a.rank
    weights = projection.weight.view(attention.n_groups, attention.o_lora_rank, -1)
    adapter_b = projection.lora_b.weight.view(attention.n_groups, attention.o_lora_rank, -1)
    expected = torch.stack(
        [
            F.linear(
                inputs[:, group],
                weights[group] + scale * (adapter_b[group] @ projection.lora_a.weight),
            )
            for group in range(attention.n_groups)
        ],
        dim=1,
    )
    output = attention.project_output_groups(inputs)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    output.square().sum().backward()
    assert projection.weight.grad is None
    assert torch.count_nonzero(projection.lora_a.weight.grad) > 0
    assert torch.count_nonzero(projection.lora_b.weight.grad) > 0


@pytest.mark.parametrize(
    "model,kind",
    [
        pytest.param("dsv4", "dense", id="dsv4-dense"),
        pytest.param(
            "dsv4",
            "batched",
            marks=pytest.mark.skip(reason="Native DSV4 wo_a uses Linear; batched LoRA is covered by V4.1."),
            id="dsv4-per-group",
        ),
        pytest.param("dsv4", "grouped", id="dsv4-routed"),
        pytest.param("dsv4_1", "batched", id="v41-per-group"),
    ],
)
@pytest.mark.parametrize("replicated", [False, True], ids=["unsharded", "replicated"])
def test_standard_lora_projection_output_and_gradients_match_independent_formula(model, kind, replicated, monkeypatch):
    if model == "dsv4_1":
        lora_config = _make_trainer_config("deepseek_v4_1_debugmodel_text")
        lora_config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)])
        config = _apply_lora(lora_config).model
    else:
        config = _lora_model_config()
    if kind == "dense":
        projection = config.layers[0].attention.wq_a
    elif kind == "batched":
        projection = config.layers[0].attention.wo_a
        assert isinstance(projection, BatchedLinearLoRA.Config)
    else:
        projection = config.layers[0].feed_forward.routed_experts.w13
        monkeypatch.setattr(GroupedLinear, "_grouped_mm", _cpu_grouped_mm)
    if replicated:
        projection.sharding_config = ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})
    module = projection.build()
    if replicated:
        for adapter in (module.lora_a, module.lora_b):
            assert adapter._sharding_config.state_shardings["weight"] == dense_param_placement(tp=spmd.R)
    module.init_states(buffer_device=torch.device("cpu"))
    if kind == "batched":
        module = module.to(dtype=torch.float64)
    with torch.no_grad():
        if kind == "batched":
            module.lora_a.weight.normal_(std=0.1)
        module.lora_b.weight.normal_(std=0.1)
    shape = (5, projection.n_heads, projection.in_features) if kind == "batched" else (5, projection.in_features)
    inputs = torch.randn(shape, dtype=module.weight.dtype, requires_grad=True)
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
        weights = module.weight.view(projection.n_heads, projection.out_features, projection.in_features)
        group_b = b.view(projection.n_heads, projection.out_features, projection.rank)
        expected = torch.stack(
            [
                inputs[:, group] @ weights[group].T
                + (projection.alpha / projection.rank) * (inputs[:, group] @ a.T @ group_b[group].T)
                for group in range(projection.n_heads)
            ],
            dim=1,
        )
    else:
        actual = module(inputs)
        expected = F.linear(inputs, module.weight) + 2 * (inputs @ a.T @ b.T)
    torch.testing.assert_close(actual, expected)
    upstream = torch.linspace(-0.5, 0.7, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, (inputs, a, b), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (inputs, a, b), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert module.weight.grad is None and not module.weight.requires_grad
