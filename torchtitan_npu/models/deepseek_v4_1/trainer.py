# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from functools import cached_property

import torch
from torchtitan.config import derive
from torchtitan.distributed import utils as dist_utils

from torchtitan_npu.config import (
    OptimizerConfig,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx, TrainingEngineEx
from torchtitan_npu.models.deepseek_v4_1.indexer import Selector
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig


class DeepSeekV41TrainingEngine(TrainingEngineEx):
    """Set the LI objective denominator before model preprocessing shards inputs."""

    @cached_property
    def _li_selectors(self) -> list[Selector]:
        # Resolve after model initialization has applied overrides and wrappers.
        return [
            module
            for part in self.model_parts
            for module in part.modules()
            if isinstance(module, Selector) and type(module) is not Selector
        ]

    def forward_backward(self, *, microbatch_groups, global_loss_token_counts, global_routing_token_counts):
        if self._li_selectors:
            num_queries = sum(
                microbatch.as_input_dict()["input"].numel() for group in microbatch_groups for microbatch in group
            )
            denominator = torch.tensor(num_queries, device=self.device, dtype=torch.float32)
            if self.parallelism_context.dp_enabled:
                denominator = dist_utils.dist_sum_tensor(denominator, self.parallelism_context.get_mesh("dp"))
            for selector in self._li_selectors:
                selector.num_global_queries = denominator
        return super().forward_backward(
            microbatch_groups=microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )


class DeepSeekV41Trainer(TrainerEx):
    """V4.1 trainer with an engine-owned LI objective denominator."""

    engine_cls = DeepSeekV41TrainingEngine

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
                optimizer._optimizer_defaults = {
                    name: [group for group in groups if not isinstance(group, EngramTableOptimizerConfig)]
                    for name, groups in optimizer._optimizer_defaults.items()
                }
            TrainerEx.Config.__post_init__(self)
