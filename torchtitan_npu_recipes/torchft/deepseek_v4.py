# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous TorchFT recipes for DeepSeek-V4 Flash on Ascend."""

from __future__ import annotations

import os
from dataclasses import fields
from importlib import import_module
from typing import TYPE_CHECKING, Any

from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu_recipes.models.deepseek_v4 import _enable_indexer_aux_loss, _flash_config

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


def _torchft_flash_config(
    base: Trainer.Config,
    *,
    expert_parallel_degree: int,
    data_parallel_shard_degree: int = 64,
    global_batch_size: int = 1024,
) -> TrainerEx.Config:
    # Check the optional dependency before importing upstream TorchFT modules.
    import_module("torchtitan_npu.extensions.torchtitan.experiments.torchft")

    from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager

    from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import (
        TorchFTOptimEx,
        TorchFTOptimizersContainerEx,
    )
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import FaultTolerantTrainerEx

    def config_values(source: object, target: type) -> dict[str, Any]:
        return {
            item.name: getattr(source, item.name) for item in fields(target) if item.init and hasattr(source, item.name)
        }

    values = config_values(base, FaultTolerantTrainerEx.Config)
    checkpointer = (
        TorchFTCheckpointManager.Config()
        if base.checkpointer is None
        else TorchFTCheckpointManager.Config(**config_values(base.checkpointer, TorchFTCheckpointManager.Config))
    )
    checkpointer.enable_ft_dataloader_checkpoints = True
    values["checkpointer"] = checkpointer
    optim_values = config_values(base.optim, TorchFTOptimEx.Config)
    optim_values["optimizer"] = TorchFTOptimizersContainerEx.Config(
        **config_values(base.optim.optimizer, TorchFTOptimizersContainerEx.Config)
    )
    values["optim"] = TorchFTOptimEx.Config(**optim_values)
    config = FaultTolerantTrainerEx.Config(**values)
    assert isinstance(config.model, DeepSeekV4Model.Config)
    # The reference selector needs its distillation objective for Muon to see its gradients.
    _enable_indexer_aux_loss(config.model)
    config.fault_tolerance.group_size = 2
    config.fault_tolerance.replica_id = int(os.environ.get("TORCHFT_REPLICA_ID", "0"))
    config.fault_tolerance.min_replica_size = int(os.environ.get("TORCHFT_MIN_REPLICA_SIZE", "2"))
    config.training.num_tokens_per_train_step = global_batch_size * config.model.max_context_length
    config.training.disable_cuda_graphs = True
    if (
        config.parallelism.expert_parallel_degree != expert_parallel_degree
        or config.parallelism.data_parallel_shard_degree != data_parallel_shard_degree
    ):
        raise ValueError("Construct the base model with the TorchFT replica topology")
    return config


def deepseek_v4_flash_torchft() -> TrainerEx.Config:
    """Full Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(
        _flash_config(
            "deepseek_v4_flash",
            parallelism=ParallelismConfig(
                expert_parallel_degree=64, data_parallel_shard_degree=64, fsdp_reshard_after_forward="always"
            ),
            per_doc_alignment=int(os.environ.get("DSV4_PER_DOC_ALIGNMENT", 128)),
        ),
        expert_parallel_degree=64,
    )
