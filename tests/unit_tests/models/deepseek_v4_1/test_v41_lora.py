# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torchtitan.components.optim import AdamW
from torchtitan.models.base.module import Module

from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager
from torchtitan_npu_recipes.models.deepseek_v4_1 import (
    deepseek_v4_1_debugmodel_lora,
    deepseek_v4_1_flash_lora,
    deepseek_v4_1_flash_text_lora,
)
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41MultimodalModel
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter


@pytest.mark.parametrize(
    "factory,vision",
    [
        (deepseek_v4_1_flash_lora, True),
        (deepseek_v4_1_flash_text_lora, False),
    ],
    ids=["multimodal", "text"],
)
def test_lora_recipe_preserves_vision_and_uses_standard_adamw_and_peft(factory, vision):
    config = factory()
    assert isinstance(config.model, DeepSeekV41MultimodalModel.Config) is vision
    assert config.model.dim == 5120 and config.model.n_layers == 40
    assert isinstance(config.checkpointer, DeepSeekV41LoRACheckpointManager.Config)
    assert [type(group) for group in config.optim.optimizer.optimizers] == [AdamW.Config]
    adapters = []
    for name, cfg, _, _ in config.model.traverse(Module.Config, recurse=True):
        if hasattr(cfg, "rank"):
            adapters.append((name, cfg))
    assert adapters and all(cfg.rank == 16 for _, cfg in adapters)
    assert not any("indexer" in name or "engram" in name or "vision" in name for name, _ in adapters)


def test_lora_model_meta_build_freezes_engram_vision_and_base():
    config = deepseek_v4_1_debugmodel_lora()
    config.model.update_from_config(config=config)
    with torch.device("meta"):
        model = config.model.build()
    assert any("engram.table.weight" in name for name, _ in model.named_parameters())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(is_lora_parameter(name) for name in trainable)
    assert not any("engram" in name for name in trainable)
    adapter = DeepSeekV41StateDictAdapter(config.model, None)
    metadata = adapter.peft_adapter_config()
    assert metadata["r"] == 16 and metadata["target_modules"] == []
    assert not any("engram" in name or "indexer" in name for name in metadata["target_parameters"])
