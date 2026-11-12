# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from torchtitan.trainer import Trainer

from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_config_cls: type[DeepSeekV4Model.Config],
) -> GraphTrainer.Config:
    """Project DSV4's NPU recipe config onto GraphTrainer's upstream schema."""
    if isinstance(base_config, TrainerEx.Config):
        base_config = Trainer.Config(
            **{config_field.name: getattr(base_config, config_field.name) for config_field in fields(Trainer.Config)}
        )
    from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config as _to_graph_trainer_config

    return _to_graph_trainer_config(base_config, model_config_cls)
