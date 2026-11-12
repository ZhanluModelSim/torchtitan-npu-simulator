# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NPU conversion at the upstream recipe loader boundary."""

import sys
import types

import pytest
from torchtitan.config import ConfigLoader
from torchtitan.trainer import Trainer

from torchtitan_npu.config import CompileConfig, OptimConfig, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


def test_loader_converts_upstream_recipe_before_applying_launch_options(monkeypatch, tmp_path):
    module = types.ModuleType("_npu_config_loader_recipe")
    module.recipe = lambda: Trainer.Config(model=None)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    config = ConfigLoader().load(["--module", module.__name__, "--config", "recipe", "--output-dir", str(tmp_path)])

    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.optim, OptimConfig)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    assert isinstance(config.training, TrainingConfig)
    assert config.compile is None
    assert config.dump_folder == str(tmp_path)


def test_trainer_rejects_model_compile_added_after_config_construction():
    config = TrainerEx.Config(model=None)
    config.compile = CompileConfig()

    with pytest.raises(NotImplementedError, match="NPU model compilation"):
        TrainerEx(config)


def test_trainer_rejects_ema_until_npu_state_is_adapted():
    from torchtitan.components.optim import EMA

    config = TrainerEx.Config(model=None)
    config.optim.ema = EMA.Config(decays=[0.9])

    with pytest.raises(NotImplementedError, match="NPU EMA"):
        TrainerEx(config)
