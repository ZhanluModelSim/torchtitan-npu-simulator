# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config import derive
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_npu.config import (
    OptimizerConfig,
)
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx, NPUGraphTrainingEngine
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable
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


class DeepSeekV41GraphTrainingEngine(NPUGraphTrainingEngine):
    def _initialize_forward_backward(self) -> None:
        # Optimizer groups must claim the trainable host weights first. Graph
        # replay then differentiates gate anchors while SparseAdam continues to
        # update the same CPU parameters from the lookup's sparse gradients.
        for part in self.model_parts:
            for module in part.modules():
                if isinstance(module, HostEngramTable):
                    module.enable_graph_training()
        super()._initialize_forward_backward()


class DeepSeekV41GraphTrainer(DeepSeekV41Trainer, GraphTrainerEx):
    engine_cls = DeepSeekV41GraphTrainingEngine

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Trainer.Config):  # pyrefly: ignore [bad-override]
        _supports_graph_compile: ClassVar[bool] = True
        compile: GraphTrainerCompileConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=GraphTrainerCompileConfig
        )
        npu_auto_overlap_debug: bool = False

        def _validate_upstream_config(self) -> None:
            GraphTrainer.Config.__post_init__(self)  # pyrefly: ignore [bad-argument-type]
