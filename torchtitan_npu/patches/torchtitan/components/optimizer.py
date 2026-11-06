# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Upstream-PR: https://github.com/pytorch/torchtitan/pull/4411

"""Use standard checkpoint layouts for replica-deduplicated Muon state."""

import functools
import logging
from contextlib import ExitStack
from typing import Any

import torch
import torchtitan.components.optim.optimizer

logger = logging.getLogger(__name__)

original_optimizers_state_dict = torchtitan.components.optim.optimizer.OptimizersContainer.state_dict
original_optimizers_load_state_dict = torchtitan.components.optim.optimizer.OptimizersContainer.load_state_dict


def _replica_dedup_contexts(optimizers: list[torch.optim.Optimizer], name: str) -> ExitStack:
    """Open temporary standard-layout state contexts for native DCP helpers."""
    from torchtitan_npu.patches.torchtitan.distributed.flex_shard import replica_dedup

    stack = ExitStack()
    for optimizer in optimizers:
        if not hasattr(optimizer, "_replica_dedup_all_layouts"):
            continue
        context = replica_dedup.standard_state_dict_layout if name == "save" else replica_dedup.standard_load_layout
        stack.enter_context(context(optimizer))
    return stack


@functools.wraps(original_optimizers_state_dict)
def patched_optimizers_state_dict(self: Any) -> dict[str, Any]:
    """Save deduplicated Muon state through TorchTitan's standard DCP layout."""
    with _replica_dedup_contexts(self.optimizers, "save"):
        return original_optimizers_state_dict(self)


@functools.wraps(original_optimizers_load_state_dict)
def patched_optimizers_load_state_dict(self: Any, state_dict: dict[str, Any]) -> None:
    """Install standard state templates before native DCP unflattening."""
    with _replica_dedup_contexts(self.optimizers, "load"):
        original_optimizers_load_state_dict(self, state_dict)


def apply() -> None:
    logger.info("[PATCH] OptimizersContainer save/load -> replica-dedup layout contexts (upstream #4411)")
    torchtitan.components.optim.optimizer.OptimizersContainer.state_dict = patched_optimizers_state_dict
    torchtitan.components.optim.optimizer.OptimizersContainer.load_state_dict = patched_optimizers_load_state_dict


apply()
