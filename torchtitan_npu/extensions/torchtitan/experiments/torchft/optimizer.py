# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One quorum decision and one optimizer update per training step."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer

if TYPE_CHECKING:
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx


class TorchFTOptimizersContainerEx(TorchFTOptimizersContainer):
    _transaction_manager: "ManagerEx"

    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTOptimizersContainer.Config):
        pass

    def __init__(self, config, *, model_parts, ft_manager):
        self.hccl_recovery = ft_manager.process_group.recovery
        super().__init__(config, model_parts=model_parts, ft_manager=ft_manager)
        self._transaction_manager = ft_manager.manager

    def zero_grad(self, set_to_none: bool = True) -> None:  # pyrefly: ignore [bad-override]
        self.hccl_recovery.begin_attempt()
        self._transaction_manager._begin_step()
        super().zero_grad(set_to_none=set_to_none)

    def step(  # pyrefly: ignore [bad-override]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        if closure is not None:
            raise ValueError("TorchFT optimizer does not support closures")
        if not self._transaction_manager.should_commit():
            return None
        self.hccl_recovery.apply_optimizer(self._step_optimizers)

    def _step_optimizers(self) -> None:
        # Direct dispatch preserves the upstream fix for recursive container hooks.
        for optimizer in self.optimizers:
            optimizer.step()
