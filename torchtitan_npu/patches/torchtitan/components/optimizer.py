# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4474

"""Patch partial optimizer-state materialization for checkpointing.

Remove this module after the TorchTitan dependency includes the PR.
"""

import functools
import logging
from contextlib import ExitStack
from typing import Any

import torch
import torchtitan.components.checkpoint_utils
import torchtitan.components.optimizer

logger = logging.getLogger(__name__)

original_init_optim_state = torchtitan.components.checkpoint_utils.init_optim_state
original_optimizers_state_dict = torchtitan.components.optimizer.OptimizersContainer.state_dict
original_optimizers_load_state_dict = torchtitan.components.optimizer.OptimizersContainer.load_state_dict


@functools.wraps(original_init_optim_state)
def patched_init_optim_state(optim: torch.optim.Optimizer) -> None:
    """Materialize missing optimizer state while preserving existing state."""
    params = [param for param_group in optim.param_groups for param in param_group["params"]]
    params_to_initialize = [param for param in params if param.requires_grad and not optim.state.get(param)]
    if not params_to_initialize:
        return

    saved_grads = [param.grad for param in params]
    for param in params:
        param.grad = None
    for param in params_to_initialize:
        param.grad = torch.zeros_like(param)

    saved_lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            saved_lrs.append(param_group["lr"])
            param_group["lr"] = torch.tensor(0.0) if isinstance(param_group["lr"], torch.Tensor) else 0.0
    optim.step(closure=None)

    if isinstance(optim, (torch.optim.Adam, torch.optim.AdamW)):
        for param in params_to_initialize:
            state = optim.state[param]
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].zero_()

    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = saved_lrs.pop(0)
    for param, grad in zip(params, saved_grads, strict=True):
        param.grad = grad


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
    logger.info("[PATCH] checkpoint_utils.init_optim_state and optimizer.init_optim_state -> patched_init_optim_state")
    # torchtitan v0.3.0 moved init_optim_state into the components.optimizer
    # package (optimizer/utils.py) and stopped re-exporting it from the package
    # root. The sole runtime caller binds it in optimizer/optimizer.py via
    # "from .utils import init_optim_state", so patch every consumer namespace.
    torchtitan.components.optimizer.utils.init_optim_state = patched_init_optim_state
    torchtitan.components.optimizer.optimizer.init_optim_state = patched_init_optim_state
    torchtitan.components.checkpoint_utils.init_optim_state = patched_init_optim_state
    torchtitan.components.optimizer.OptimizersContainer.state_dict = patched_optimizers_state_dict
    torchtitan.components.optimizer.OptimizersContainer.load_state_dict = patched_optimizers_load_state_dict


apply()
