# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from torch.distributed.tensor.parallel.style import ParallelStyle

from torchtitan.distributed import ParallelDims
from torchtitan.trainer import Trainer


class ModelCustomConverter(ABC):
    def __init__(self, trainer_config: Trainer.Config, parallel_dims: ParallelDims):
        self.trainer_config = trainer_config
        self.parallel_dims = parallel_dims
        self.model_name = trainer_config.model_spec.name

    @abstractmethod
    def convert(self, model: nn.Module):
        pass


class ParallelizePlanUpdater(ABC):
    """Abstract base class for layer plan updaters"""

    @classmethod
    @abstractmethod
    def update(
        cls, parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None
    ) -> ParallelStyle | dict[str, ParallelStyle] | None:
        """Update the layer plan"""
        pass


class StateDictUpdater(ABC):
    """Abstract base class for state dict updaters"""

    @classmethod
    @abstractmethod
    def to_hf(cls, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Transform state dict to HF format"""
        pass

    @classmethod
    @abstractmethod
    def from_hf(cls, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Transform state dict from HF format"""
        pass


@dataclass
class ModelCustomConfig:
    """Model customization configuration"""

    name: str = "default"
    model_converter: type["ModelCustomConverter"] | None = None
    parallelize_plan_updater: type["ParallelizePlanUpdater"] | None = None
    state_dict_updater: type["StateDictUpdater"] | None = None
