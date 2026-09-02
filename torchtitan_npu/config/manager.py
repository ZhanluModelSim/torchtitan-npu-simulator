# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the NPU config schema before TorchTitan invokes Tyro."""

from functools import wraps

from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer

from .configs import TrainerConfig

_original_load_config = ConfigManager._load_config


@wraps(_original_load_config)
def _patched_load_config(self, args: list[str]) -> tuple[object, list[str]]:
    config, filtered_args = _original_load_config(self, args)
    if isinstance(config, Trainer.Config) and not isinstance(config, GraphTrainer.Config):
        config = TrainerConfig.from_trainer_config(config)
    return config, filtered_args


def apply() -> None:
    ConfigManager._load_config = _patched_load_config  # type: ignore[method-assign]


apply()
