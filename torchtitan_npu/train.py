# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/train.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from functools import wraps

import torch
import torch_npu
import torchtitan.train as titan_train
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

init_logger()


def get_grad_accumulation_steps(trainer) -> int:
    ga_steps = trainer.gradient_accumulation_steps
    if trainer.parallel_dims.pp_enabled:
        local_bs = trainer.config.training.local_batch_size
        micro_bs = trainer.config.parallelism.pipeline_parallel_microbatch_size
        return ga_steps * (local_bs // micro_bs)
    return ga_steps


def _patch_train_step_for_dsa_indexer_loss():
    _original_train_step = titan_train.Trainer.train_step

    def wrapper_train_step(self, *args, **kwargs):
        # Execute the original train_step (which includes gradient accumulation loops)
        result = _original_train_step(self, *args, **kwargs)

        # Handle indexer loss tracking. ``Trainer`` exposes ``model_config``
        # (set in ``Trainer.__init__``); the legacy ``self.model_args``
        # attribute does not exist on the upstream Trainer.
        model_cfg = getattr(self, "model_config", None)
        if model_cfg is not None and getattr(model_cfg, "enable_indexer_loss", False):
            # Import dynamically to avoid circular dependencies
            from torchtitan_npu.models.common.dsa_indexer_loss import (
                DSAIndexerLossLoggingHelper,
            )

            # Align logging frequency with the core metrics processor
            if self.metrics_processor.should_log(self.step):
                DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics(total_acc_steps=get_grad_accumulation_steps(self))
            else:
                # Crucial: Clear tracker silently if this step is not being logged
                # to prevent runaway accumulation of losses across steps.
                DSAIndexerLossLoggingHelper.clean_loss_in_tracker()

        return result

    # Apply the monkey patch
    titan_train.Trainer.train_step = wrapper_train_step


def _patch_init_for_dsa_set_loss_scale():
    _original = titan_train.Trainer.__init__

    @wraps(_original)
    def wrapper_init(self, config: Trainer.Config):
        _original(self, config)

        loss_degree = 1
        if getattr(self.parallel_dims, "dp_cp_enabled", False):
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                loss_degree = loss_mesh.size()

        scale = 1.0 / float(get_grad_accumulation_steps(self) * loss_degree)
        scale_tensor = torch.tensor(scale, device=self.device, dtype=torch.float32)

        from torchtitan_npu.models.common.dsa_indexer_loss import (
            DSAIndexerLossAutoScaler,
        )

        DSAIndexerLossAutoScaler.set_loss_scale(scale_tensor)

    titan_train.Trainer.__init__ = wrapper_init


def _patch_for_train_npu_memory():
    _original = titan_train.Trainer.train

    def wrapper_train(self):
        torch.npu.empty_cache()  # pyrefly: ignore[missing-attribute]
        memory_ratio = self.config.training.torch_npu_memory_ratio
        if not (0.0 < memory_ratio <= 1.0):
            logger.warning(
                f"torch_npu_memory_ratio {memory_ratio} is invalid "
                "(must be in (0.0, 1.0]), falling back to default value 1.0"
            )
            memory_ratio = 1.0
        torch.npu.set_per_process_memory_fraction(  # pyrefly: ignore[missing-attribute]
            memory_ratio
        )
        logger.info(f"[NPU Memory Config] Set process memory usage upper limit to {memory_ratio}")
        return _original(self)

    titan_train.Trainer.train = wrapper_train


def _patch_for_garbage_collection_run():
    """
    Apply monkey patch to GarbageCollection.run method.
    To prevent NPUCachingAllocator from caching to much memory after checkpoint
    loading, resulting in OOM later for NPUWorkspaceAllocator / HCCL.
    """
    from torchtitan.tools import utils

    original_run = utils.GarbageCollection.run

    @wraps(original_run)
    def patched_run(self, step_count: int):
        """Patched version with NPU cache clearing at step 1."""
        if step_count == 1:
            # Clear NPU cache at step 1
            gc.collect()
            # pyrefly: ignore [missing-attribute]
            torch.npu.empty_cache()
            logger.info(f"[NPU] Cleared NPU cache at step {step_count}")

        # Call original method
        return original_run(self, step_count)

    # Apply the patch
    utils.GarbageCollection.run = patched_run
    logger.info("[PATCH] Successfully patched GarbageCollection.run method")


def _patch_for_parallel_dims_build_mesh():
    """
    Apply monkey patch to ParallelDims.build_mesh method.
    To avoid FSDP and EP from using the same ProcessGroup, and therefore
    makes AG/RS communication blocking EP A2A and becomes non-overlapped.
    """
    from torchtitan.distributed import ParallelDims

    _original_build_mesh = ParallelDims.build_mesh

    @wraps(_original_build_mesh)
    def patched_build_mesh(self):
        """Patched version which guarantees FSDP and EP uses different PG."""
        world_mesh = _original_build_mesh(self)

        sparse_dims = ("pp", "dp_replicate", "efsdp", "ep", "etp")
        sparse_degrees = tuple(self._meshes[dim].size() for dim in sparse_dims)
        backend_override = {
            dim: "fake"
            if dim != "ep"
            # Provide a non-None Option to force Pytorch to create
            # a new ProcessGroup for EP communication.
            else torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            for dim in sparse_dims
        }

        sparse_mesh = world_mesh._unflatten(
            0,
            sparse_degrees,
            sparse_dims,
            backend_override,
        )

        self._global_meshes["sparse"] = sparse_mesh
        self._meshes["ep"] = sparse_mesh["ep"]
        return world_mesh

    # Apply the patch
    ParallelDims.build_mesh = patched_build_mesh
    logger.info("[PATCH] Successfully patched ParallelDims.build_mesh method")
