# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch.distributed.tensor.parallel.style import ParallelStyle
from torchtitan.protocols.model_spec import ModelSpec


class ModelCustomConverter(ABC):
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        self.model_name = self.model_spec.name

    @abstractmethod
    def convert(self, model: nn.Module) -> None:
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
    state_dict_updater_predicate: Callable[[nn.Module], bool] | None = None
