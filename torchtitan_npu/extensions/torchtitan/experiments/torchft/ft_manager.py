# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous TorchFT manager for the NPU FSDP/EP training path."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torchft
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torchtitan.experiments.torchft.manager import TorchFTManager

from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx


def _set_fsdp_all_reduce_hook(
    module: FSDPModule,
    hook: Callable[[torch.Tensor], None],
) -> None:
    state = module._get_fsdp_state()
    param_groups = state._fsdp_param_groups
    if len(param_groups) <= 1:
        module.set_all_reduce_hook(hook)
        return

    if not all(hasattr(group, "_all_reduce_hook") for group in param_groups):
        raise RuntimeError(
            "The installed PyTorch FSDP parameter-group API is incompatible with TorchFT all-reduce hooks."
        )
    for param_group in param_groups:
        param_group._all_reduce_hook = hook


def _install_all_reduce_hooks(
    model_parts: list[torch.nn.Module],
    hook: Callable[[torch.Tensor], None],
) -> int:
    num_fsdp_modules = 0

    def apply_hook(module: torch.nn.Module) -> None:
        nonlocal num_fsdp_modules
        if not isinstance(module, FSDPModule):
            return
        num_fsdp_modules += 1
        _set_fsdp_all_reduce_hook(module, hook)

    for model_part in model_parts:
        model_part.apply(apply_hook)
    return num_fsdp_modules


class FTManagerEx(TorchFTManager):
    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTManager.Config):
        enable: bool = True
        process_group: str = "hccl"
        quorum_timeout_seconds: int = 60
        """Allow time for replica-local state export, loading and disk operations."""
        device_recovery_timeout_seconds: int = 60
        max_consecutive_hccl_recoveries: int = 3
        quorum_warmup_timeout_ms: int = 90000

    @staticmethod
    def validate_config(config: Config) -> None:
        if not config.enable or config.process_group != "hccl":
            raise ValueError("NPU TorchFT requires enabled HCCL fault tolerance")
        if config.semi_sync_method is not None:
            raise ValueError("NPU TorchFT supports synchronous training only; LocalSGD and DiLoCo are unsupported")
        if config.group_size < 2:
            raise ValueError("NPU TorchFT requires at least two elastic DP replicas")
        if not 1 <= config.min_replica_size <= config.group_size:
            raise ValueError("min_replica_size must be between 1 and group_size")
        if not 0 <= config.replica_id < config.group_size:
            raise ValueError("replica_id must be between 0 and group_size - 1")

    def __init__(self, config: Config) -> None:
        self.validate_config(config)
        self.group_size = config.group_size
        self.replica_id = config.replica_id
        # Required by TorchTitan's optimizer container interface. Quorum
        # scheduling itself is fixed to synchronous mode below.
        self.use_async_quorum = False
        if config.quorum_timeout_seconds <= 0:
            raise ValueError("quorum_timeout_seconds must be positive")
        if config.device_recovery_timeout_seconds <= 0 or config.max_consecutive_hccl_recoveries <= 0:
            raise ValueError("Device recovery timeout and consecutive recovery limit must be positive")
        if config.quorum_warmup_timeout_ms <= 0:
            raise ValueError("Quorum warmup timeout must be positive")
        self.process_group = ProcessGroupHCCLEx(
            timedelta(milliseconds=config.process_group_timeout_ms),
            warmup_timeout=timedelta(milliseconds=config.quorum_warmup_timeout_ms),
            recovery_timeout=timedelta(seconds=config.device_recovery_timeout_seconds),
        )
        self._manager = ManagerEx(
            pg=self.process_group,
            min_replica_size=config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=False,
            replica_id=f"torchtitan_ft_{config.replica_id}",
            init_sync=True,
            quorum_timeout=timedelta(seconds=config.quorum_timeout_seconds),
        )
        self.replicate_pg = torchft.process_group.ManagedProcessGroup(self._manager)
        self.replicate_pg.register("dp_replicate")

    def maybe_set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        @torch.compiler.disable(reason="TorchFT collective completion must stay outside compiled autograd")
        def all_reduce_hook(output: torch.Tensor) -> None:
            if self.replicate_pg.size() <= 1:
                return
            work = dist.all_reduce(output, group=self.replicate_pg, op=dist.ReduceOp.AVG, async_op=True)
            self.process_group.wait_for_completion(work)

        num_fsdp_modules = _install_all_reduce_hooks(model_parts, all_reduce_hook)
        if num_fsdp_modules == 0:
            raise ValueError("NPU TorchFT requires FSDP-sharded model parts")

    @property
    def loss_sync_pg(self):
        return self.replicate_pg
