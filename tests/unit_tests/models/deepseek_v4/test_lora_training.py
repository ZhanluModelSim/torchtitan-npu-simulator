# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config
from torchtitan_npu.models import deepseek_v4 as registry
from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform, is_lora_parameter


def test_registry_lora_forward_keeps_zero_adapter_base_output_and_updates_only_adapters(monkeypatch):
    monkeypatch.setitem(
        registry.deepseek_v4_configs,
        "debugmodel",
        (lambda *args, **kwargs: _build_model_config(num_layers=1), 64),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        config = registry.model_registry("debugmodel")
        base = config.build()
        base.init_states(buffer_device=torch.device("cpu"))
        model = deepseek_v4_lora_transform(rank=8).transform(copy.deepcopy(config)).build()
        model.init_states(buffer_device=torch.device("cpu"))
        model.load_state_dict(base.state_dict(), strict=False)
        tokens = torch.arange(8)
        positions = torch.arange(8)
        expected = base(tokens, positions=positions, attention_metadata=base._get_attention_metadata(positions))
        actual = model(tokens, positions=positions, attention_metadata=model._get_attention_metadata(positions))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None}
    assert trained and all(is_lora_parameter(name) for name in trained)
    assert any(p.grad.count_nonzero() for name, p in model.named_parameters() if name.endswith("lora_b.weight"))
