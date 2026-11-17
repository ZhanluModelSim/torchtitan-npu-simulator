# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4598
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refresh cached optimizer state before TorchFT exports recovery state.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch.distributed as dist
import torch.nn as nn
import torchtitan.experiments.torchft.checkpoint
from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.components.dataloader import BaseDataLoader
    from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
    from torchtitan.experiments.torchft.manager import TorchFTManager
    from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter


def patched_init(
    self,
    config: TorchFTCheckpointManager.Config,
    *,
    dataloader: BaseDataLoader | None,
    model_parts: list[nn.Module],
    optimizers: OptimizersContainer,
    lr_schedulers: LRSchedulersContainer,
    states: dict[str, Any],
    sd_adapter: BaseStateDictAdapter | None,
    base_folder: str = "",
    ft_manager: TorchFTManager | None = None,
) -> None:
    # Initialize the base checkpoint manager (without FT)
    super(TorchFTCheckpointManager, self).__init__(
        config,
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states=states,
        sd_adapter=sd_adapter,
        base_folder=base_folder,
    )

    self.ft_manager = ft_manager.manager if ft_manager and ft_manager.enabled else None
    self.enable_ft_dataloader_checkpoints = self.ft_manager and config.enable_ft_dataloader_checkpoints

    if self.ft_manager and not self.enable_ft_dataloader_checkpoints:
        logger.warning(
            "Fault tolerance is enabled but enable_ft_dataloader_checkpoints "
            "is False. This means replicas can retrain over the same data "
            "multiple times, which can result in overfitting."
        )

    if not self.enable:
        return

    if self.ft_manager:
        optimizers.init_cache_state_dict()

        def state_dict():
            assert isinstance(optimizers, TorchFTOptimizersContainer)
            # Added to the v0.3.0 class by the optimizer backport.
            # pyrefly: ignore [missing-attribute]
            optimizers._refresh_cached_state_dict()
            ret = {}
            for k, v in self.states.items():
                if k in {MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE}:
                    ret[k] = v.state_dict()
            return ret

        def load_state_dict(state_dict):
            assert state_dict is not None
            for k, v in state_dict.items():
                self.states[k].load_state_dict(v)

        # pyrefly: ignore [missing-attribute]
        self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
        assert ft_manager is not None
        self.ft_replica_id = ft_manager.replica_id

    # FT may need staging even without async_with_pinned_mem
    if self.enable_ft_dataloader_checkpoints:
        self.enable_staging = True
        self.ft_states = {DATALOADER: dataloader}

        # FT needs gloo pg for async dataloader checkpoints
        if self.pg is None:
            self.pg = cast("dist.ProcessGroup", dist.new_group(backend="gloo"))


def apply() -> None:
    torchtitan.experiments.torchft.checkpoint.TorchFTCheckpointManager.__init__ = patched_init


apply()
