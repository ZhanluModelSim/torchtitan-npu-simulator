# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass

from torchtitan.config import derive

from torchtitan_npu.config import (
    OptimizerConfig,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig


class DeepSeekV41Trainer(TrainerEx):
    """Build the V4.1 optimizer for the configured Engram tables."""

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        """Expose the model switch while DeepSeekV41Model.Config is suppressed from the CLI."""

        engram_enabled: bool = True

        def __post_init__(self) -> None:
            assert self.model is not None and isinstance(self.model, DeepSeekV41Model.Config)
            if not self.engram_enabled:
                model = copy.deepcopy(self.model)
                self.model = model
                assert isinstance(model, DeepSeekV41Model.Config)
                # Remove the HostSparse capability together with Engram.
                optimizer = derive(copy.deepcopy(self.optim.optimizer), OptimizerConfig)
                self.optim.optimizer = optimizer
                for layer in model.layers:
                    layer.engram = None
                optimizer.optimizers = [
                    group for group in optimizer.optimizers if not isinstance(group, EngramTableOptimizerConfig)
                ]
            TrainerEx.Config.__post_init__(self)
