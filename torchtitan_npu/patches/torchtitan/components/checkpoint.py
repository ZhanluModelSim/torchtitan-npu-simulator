# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport EMA-aware checkpoint construction and loading from TorchTitan PR #3985.

Remove this module after the TorchTitan dependency includes the PR.
"""

from dataclasses import dataclass, field
from typing import Any

import torch.distributed.checkpoint as dcp
import torchtitan.components.checkpoint
import torchtitan.components.checkpointer
from torchtitan.components.checkpointer import CheckpointManager as OriginalCheckpointManager

from torchtitan_npu.patches.torchtitan.components.ema import EMA_OPTIMIZER, EMAOptimizersContainer


class EMACheckpointManager(OriginalCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(OriginalCheckpointManager.Config):
        ema_weights: EMAOptimizersContainer.Config = field(default_factory=EMAOptimizersContainer.Config, repr=False)

    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.ema_optimizer = config.ema_weights.build(model_parts=kwargs["model_parts"])
        if self.enable:
            self.states[EMA_OPTIMIZER] = self.ema_optimizer

    @staticmethod
    def _checkpoint_has_prefix(checkpoint_id: str, prefix: str) -> bool:
        try:
            metadata = dcp.FileSystemReader(checkpoint_id).read_metadata()
            return any(key.startswith(prefix) for key in metadata.state_dict_metadata)
        except FileNotFoundError:
            return False

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool = False,
        from_quantized: bool = False,
    ) -> Any:
        needs_ema_reseed = False
        if EMA_OPTIMIZER in state_dict:
            state_dict = state_dict.copy()
            needs_ema_reseed = self._maybe_exclude_missing_ema(state_dict, checkpoint_id)
        elif self.ema_optimizer is not None:
            needs_ema_reseed = True
        result = super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)
        if needs_ema_reseed:
            self._reseed_ema_from_model()
        return result

    def _maybe_exclude_missing_ema(self, states_to_load: dict[str, Any], checkpoint_id: str) -> bool:
        if self._checkpoint_has_prefix(checkpoint_id, f"{EMA_OPTIMIZER}."):
            return False
        del states_to_load[EMA_OPTIMIZER]
        return True

    def _reseed_ema_from_model(self) -> None:
        ema_optimizer = self.ema_optimizer
        if ema_optimizer is not None:
            ema_optimizer.load_state_dict({})


def apply() -> None:
    # torchtitan v0.3.0 split CheckpointManager into the checkpointer package;
    # the legacy components.checkpoint module is now a re-export shim. Replace
    # the name in every namespace later consumers import it from.
    torchtitan.components.checkpointer.dcp.CheckpointManager = EMACheckpointManager
    torchtitan.components.checkpointer.CheckpointManager = EMACheckpointManager
    torchtitan.components.checkpoint.CheckpointManager = EMACheckpointManager


apply()
