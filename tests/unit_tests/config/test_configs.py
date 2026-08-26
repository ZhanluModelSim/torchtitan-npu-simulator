# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from dataclasses import fields

from torchtitan.config import ConfigManager, TrainingConfig as UpstreamTrainingConfig
from torchtitan.trainer import Trainer

from torchtitan_npu.config import (
    ExtensionConfig,
    TrainerConfig,
    TrainingConfig as NPUTrainingConfig,
    TrainingExtensionConfig,
)
from torchtitan_npu.distributed import utils as distributed_utils


def test_from_trainer_config_wraps_training_extension():
    source = Trainer.Config(
        dump_folder="custom-output",
        training=UpstreamTrainingConfig(
            local_batch_size=3,
            seq_len=4096,
            steps=17,
        ),
    )

    adapted = TrainerConfig.from_trainer_config(source)

    assert isinstance(adapted, TrainerConfig)
    assert isinstance(adapted.extension, ExtensionConfig)
    assert isinstance(adapted.training, NPUTrainingConfig)
    assert isinstance(adapted.training.extension, TrainingExtensionConfig)
    assert adapted.training.extension.allow_hf32 is True
    for config_field in fields(Trainer.Config):
        if config_field.name == "training":
            continue
        assert getattr(adapted, config_field.name) == getattr(source, config_field.name)
    for config_field in fields(UpstreamTrainingConfig):
        assert getattr(adapted.training, config_field.name) == getattr(
            source.training, config_field.name
        )


def test_config_manager_parses_training_extension(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_test_config_registry"
    registry = types.ModuleType(module_name)

    def test_config() -> Trainer.Config:
        return Trainer.Config(
            hf_assets_path=str(tmp_path),
            training=UpstreamTrainingConfig(steps=23),
        )

    registry.test_config = test_config
    monkeypatch.setitem(sys.modules, module_name, registry)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--training.extension.no-allow-hf32",
        ]
    )

    assert isinstance(config, TrainerConfig)
    assert isinstance(config.training, NPUTrainingConfig)
    assert config.training.extension.allow_hf32 is False
    assert config.training.steps == 23


def test_trainer_config_build_applies_extension_before_parent(monkeypatch):
    events = []
    expected_result = object()

    def apply_runtime(allow_hf32):
        events.append(("apply", allow_hf32))

    def parent_build(config, **kwargs):
        events.append(("build", kwargs))
        return expected_result

    monkeypatch.setattr(distributed_utils, "set_allow_hf32", apply_runtime)
    monkeypatch.setattr(Trainer.Config, "build", parent_build)

    config = TrainerConfig(
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        )
    )
    result = config.build(example="value")

    assert result is expected_result
    assert events == [
        ("apply", False),
        ("build", {"example": "value"}),
    ]
