# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4578
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4598
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport non-recursive TorchFT optimizer dispatch and cached-state refresh.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torchtitan.experiments.torchft.optimizer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.optimizer.utils import get_flat_optim_state_dict, init_optim_state

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn
    from torchtitan.experiments.torchft.manager import TorchFTManager
    from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer


def patched_init(
    self,
    config: TorchFTOptimizersContainer.Config,
    *,
    model_parts: list[nn.Module],
    ft_manager: TorchFTManager,
) -> None:
    OptimizersContainer.__init__(self, config, model_parts=model_parts)
    # Materialize optimizer state without going through the FT quorum path.
    for optim in self.optimizers:
        init_optim_state(optim)
    self.cache_state_dict = {}
    # Semi-sync algorithms manage quorum in their own synchronization hooks.
    self._quorum_manager = ft_manager.manager if ft_manager.use_async_quorum else None


def patched_step(self, closure: Callable[[], float] | None = None) -> float | None:
    assert closure is None, "OptimizersContainer does not support closures"
    if self._quorum_manager is not None and not self._quorum_manager.should_commit():
        return None
    # Call inner optimizers directly to avoid re-entering container hooks.
    for optimizer in self.optimizers:
        optimizer.step()
    return None


def patched_zero_grad(self, set_to_none: bool = True) -> None:
    if self._quorum_manager is not None:
        self._quorum_manager.start_quorum()
    OptimizersContainer.zero_grad(self, set_to_none=set_to_none)


def _refresh_cached_state_dict(self) -> None:
    if not self.cache_state_dict:
        return
    # Refresh scalar metadata while preserving the cache and tensor references.
    for optimizer in self.optimizers:
        self.cache_state_dict.update(get_flat_optim_state_dict(optimizer))


def apply() -> None:
    # Preserve class identity so existing Config factories and imported aliases
    # continue to construct the same container with a single public hook boundary.
    torchtitan.experiments.torchft.optimizer.TorchFTOptimizersContainer.__init__ = patched_init
    # PR #4578 replaces the legacy variadic dispatch signatures.
    # pyrefly: ignore [bad-assignment]
    torchtitan.experiments.torchft.optimizer.TorchFTOptimizersContainer.step = patched_step
    # pyrefly: ignore [bad-assignment]
    torchtitan.experiments.torchft.optimizer.TorchFTOptimizersContainer.zero_grad = patched_zero_grad
    torchtitan.experiments.torchft.optimizer.TorchFTOptimizersContainer._refresh_cached_state_dict = (
        _refresh_cached_state_dict
    )


apply()
