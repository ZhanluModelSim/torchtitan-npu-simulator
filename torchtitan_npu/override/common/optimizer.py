# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch_npu
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
from torchtitan.components.checkpoint import AsyncMode, CheckpointManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.tools.utils import GarbageCollection


def _make_swap(t: torch.Tensor) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    # A DTensor may have p.numel() > 0 globally but local.numel() == 0 on ranks
    # that own an empty shard. Swap-memory allocation rejects zero-sized tensors,
    # so preserve such local shards with a regular empty tensor instead.
    out = (
        torch.empty_like(local)
        if local.numel() == 0
        else torch_npu.empty_with_swapped_memory(
            local.size(), dtype=local.dtype, device=local.device
        )
    )
    if isinstance(t, DTensor):
        return DTensor.from_local(
            out,
            t.device_mesh,
            t.placements,
            shape=t.size(),
            stride=t.stride(),
            run_check=False,
        )
    return out


def _swap_state_init_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self.optimizers:
            opt.register_step_pre_hook(_swap_state_init_hook)


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config:
    return derive(cfg, VirtualOptimizersContainer.Config)


# This checkpoint specialization only makes synchronous native DCP saves
# compatible with live Virtual Optimizer states. It is not a standalone checkpoint
# feature: disabling copy-ahead avoids an incompatible staging path for the
# host-backed, NPU-addressable storage created above.
class VirtualCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        if async_mode != AsyncMode.DISABLED or to_hf:
            return super().dcp_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_id,
                async_mode=async_mode,
                enable_garbage_collection=enable_garbage_collection,
                to_hf=to_hf,
            )

        dcp.save(
            state_dict,
            storage_writer=dcp.FileSystemWriter(
                checkpoint_id,
                per_thread_copy_ahead=0,
            ),
        )
        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        return None


@override(
    target=CheckpointManager.Config,
    description="Make synchronous DCP saves compatible with Virtual Optimizer states",
    exact=True,
)
def checkpoint_virtual(
    cfg: CheckpointManager.Config,
) -> VirtualCheckpointManager.Config:
    return derive(cfg, VirtualCheckpointManager.Config)
