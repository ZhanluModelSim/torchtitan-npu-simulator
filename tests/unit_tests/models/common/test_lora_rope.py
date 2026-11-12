# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest
import torch
from torchtitan.config import derive
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.models.common.deepseek_lora import deepseek_v4_lora_transform
from torchtitan_npu.override.common.rope import AscComplexRoPE, AscPartialComplexRoPE, WorkaroundComplexRoPE
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash, deepseek_v4_flash_lora_compiled
from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_flash_lora, deepseek_v4_1_flash_text_lora


class CustomComplexRoPE(ComplexRoPE):
    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.scale = torch.nn.Parameter(torch.ones(()))


@pytest.mark.parametrize(
    "factory",
    [deepseek_v4_flash_lora_compiled, deepseek_v4_1_flash_lora, deepseek_v4_1_flash_text_lora],
    ids=["v4-compiled", "v4.1-multimodal", "v4.1-text"],
)
@pytest.mark.parametrize(
    "variant,rope_cls",
    [("workaround", WorkaroundComplexRoPE), ("asc_complex", AscComplexRoPE), ("asc_partial", AscPartialComplexRoPE)],
    ids=["workaround", "asc-complex", "asc-partial"],
)
def test_lora_recipe_selects_requested_rope_override(factory, variant, rope_cls):
    config = factory()
    config.model.update_from_config(config=config)
    original = {
        fqn: dataclasses.asdict(cfg)
        for fqn, cfg, _, _ in config.model.traverse(ComplexRoPE.Config, recurse=True)
    }
    assert original
    assert all(
        type(cfg) is ComplexRoPE.Config
        for _, cfg, _, _ in config.model.traverse(ComplexRoPE.Config, recurse=True)
    )
    with torch.device("meta"):
        original_model = config.model.build()
    original_trainable = {name for name, parameter in original_model.named_parameters() if parameter.requires_grad}
    assert original_trainable
    config.override.imports = [f"torchtitan_npu.override.common.rope.{variant}"]

    apply_overrides(config.override, config)

    selected = {fqn: cfg for fqn, cfg, _, _ in config.model.traverse(ComplexRoPE.Config, recurse=True)}
    assert selected.keys() == original.keys()
    for fqn, cfg in selected.items():
        assert type(cfg) is rope_cls.Config
        assert dataclasses.asdict(cfg) == original[fqn]
    with torch.device("meta"):
        model = config.model.build()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == original_trainable
    assert all(type(model.get_submodule(fqn)) is rope_cls for fqn in selected)


@pytest.mark.parametrize("variant", ["workaround", "asc_complex", "asc_partial"])
def test_lora_rope_override_preserves_custom_subclass_and_its_frozen_parameter(variant):
    config = deepseek_v4_flash()
    fqn, cfg, parent, attr = next(config.model.traverse(ComplexRoPE.Config, recurse=True))
    assert parent is not None and isinstance(attr, str)
    setattr(parent, attr, derive(cfg, CustomComplexRoPE.Config))
    config.model = deepseek_v4_lora_transform().transform(config.model)
    config.override.imports = [f"torchtitan_npu.override.common.rope.{variant}"]

    apply_overrides(config.override, config)

    with torch.device("meta"):
        model = config.model.build()
    custom_rope = model.get_submodule(fqn)
    assert type(custom_rope) is CustomComplexRoPE
    assert not custom_rope.scale.requires_grad
