# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Recover interrupted TorchFT steps and discard their transient state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torchtitan.models.common.moe import MoE
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn
    from torch.distributed.checkpoint.stateful import Stateful
    from torchtitan.components.optimizer import LRSchedulersContainer

    from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx

    from .hccl import HcclRecovery


@torch.no_grad()
def reset_failed_iteration(model_parts):
    """Call only after device recovery, before starting another attempt."""
    for model in model_parts:
        if isinstance(model, FSDPModule):
            model.reset_iter_state()
        for parameter in model.parameters():
            parameter.grad = None
        for module in model.modules():
            if isinstance(module, FSDPModule):
                for group in module._get_fsdp_state()._fsdp_param_groups:
                    for fsdp_param in group.fsdp_params:
                        unsharded = getattr(fsdp_param, "_unsharded_param", None)
                        if unsharded is not None:
                            unsharded.grad = None
                        fsdp_param.unsharded_accumulated_grad = None
    reset_rejected_moe_state(model_parts)


@torch.no_grad()
def reset_rejected_moe_state(model_parts):
    for model in model_parts:
        for module in model.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert_E.zero_()


class StepRecovery:
    def __init__(
        self,
        hccl: HcclRecovery,
        *,
        manager: ManagerEx,
        model_parts: list[nn.Module],
        train_state: Stateful,
        lr_schedulers: LRSchedulersContainer,
        max_consecutive_recoveries: int,
        reset_auxiliary_state: Callable[[], None],
    ):
        self._hccl = hccl
        self._manager = manager
        self._model_parts = model_parts
        self._train_state = train_state
        self._lr_schedulers = lr_schedulers
        self._max_consecutive_recoveries = max_consecutive_recoveries
        self._reset_auxiliary_state = reset_auxiliary_state
        self._consecutive_recoveries = 0

    def run_step(self, step, data_iterator):
        try:
            result = step(data_iterator)
        except RuntimeError as error:
            if not self._hccl.recover(error):
                raise
            if self._hccl.optimizer_update_completed:
                raise RuntimeError("HCCL recovery after an optimizer commit is not supported") from error
            self._consecutive_recoveries += 1
            if self._consecutive_recoveries > self._max_consecutive_recoveries:
                raise RuntimeError("Exceeded the consecutive HCCL recovery limit") from error
            attempt_state = self._train_state.state_dict()
            reset_failed_iteration(self._model_parts)
            self._reset_auxiliary_state()
            # Interrupted backward can leave idle blocks that compete with
            # operator workspaces outside the PyTorch caching allocator.
            torch_npu.npu.empty_cache()
            self._manager.discard_interrupted_step(error)
            self._lr_schedulers.step()
            logger.warning(
                "Discarded interrupted TorchFT attempt %d after device recovery; next step reads new data",
                attempt_state["step"],
            )
            return
        if not self._manager.commit_authorized:
            reset_rejected_moe_state(self._model_parts)
            self._reset_auxiliary_state()
        if self._hccl.optimizer_update_completed:
            self._consecutive_recoveries = 0
        return result
