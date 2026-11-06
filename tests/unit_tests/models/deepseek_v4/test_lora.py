# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torchtitan.protocols.module import Module

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config, _lora_model_config
from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter


def test_standard_transform_adapts_projections_and_freezes_indexer_and_base():
    config = _lora_model_config(num_layers=4)
    configs = {name: cfg for name, cfg, _, _ in config.traverse(Module.Config, recurse=True)}
    assert all(
        hasattr(configs[f"layers.0.{suffix}"], "rank")
        for suffix in (
            "attention.wq_a",
            "attention.wo_a",
            "moe.router.gate",
            "moe.shared_experts.w13",
            "moe.routed_experts.w13",
            "moe.routed_experts.w2",
        )
    )
    assert not any(hasattr(cfg, "rank") for name, cfg in configs.items() if ".indexer." in name)
    with torch.device("meta"):
        model = config.build()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all(is_lora_parameter(name) for name in trainable)
    assert "layers.0.attention.wo_a.lora_b.weight" in trainable
    assert "lm_head.lora_b.weight" in trainable


def test_standard_lora_parameters_follow_model_construction_dtype():
    config = _lora_model_config()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = config.build()
    finally:
        torch.set_default_dtype(old_dtype)
    assert {p.dtype for name, p in model.named_parameters() if is_lora_parameter(name)} == {torch.bfloat16}


def test_transform_keeps_mtp_and_adapts_its_projection_modules():
    from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform

    config = deepseek_v4_lora_transform(rank=8).transform(_build_model_config(num_layers=1, num_mtp_layers=1))
    assert config.mtp_layers[0].e_proj.rank == 8
    assert config.mtp_layers[0].h_proj.rank == 8
    assert config.mtp_layers[0].moe.routed_experts.w13.rank == 8
