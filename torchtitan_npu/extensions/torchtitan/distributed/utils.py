# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed initialization for NPU runtimes predating per-edge PP groups.

The fallback follows ebdfc8's initialization policy for PP=1. Remove it once
the supported torch_npu stack exposes pipeline_per_edge_p2p. This module is
the sole extension owner of torchtitan.distributed.utils.init_distributed.
"""

import inspect
import logging
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torchtitan.distributed.utils
from torchtitan.config import CommConfig
from torchtitan.distributed.parallelism_context import DistributedTopology
from torchtitan.tools.utils import device_type

logger = logging.getLogger(__name__)
_original_init_distributed = torchtitan.distributed.utils.init_distributed


def init_distributed(
    comm_config: CommConfig,
    enable_cpu_backend: bool = False,
    base_folder: str = "",
    ranks: list[int] | None = None,
    *,
    pipeline_parallel_degree: int = 1,
) -> DistributedTopology:
    if hasattr(dist_config, "pipeline_per_edge_p2p") or dist.is_initialized():
        return _original_init_distributed(
            comm_config,
            enable_cpu_backend,
            base_folder,
            ranks,
            pipeline_parallel_degree=pipeline_parallel_degree,
        )
    if pipeline_parallel_degree > 1 or comm_config.backend == "real_pp_fake_spmd":
        raise NotImplementedError("Pipeline parallelism requires a PyTorch runtime with pipeline_per_edge_p2p support")

    torch.autograd.set_multithreading_enabled(False)
    if comm_config.backend == "fake":
        world_size = torchtitan.distributed.utils._env_int("NGPU")
        rank = torchtitan.distributed.utils._fake_logical_rank(world_size, pipeline_parallel_degree)
        if not 0 <= rank < world_size:
            raise ValueError(f"Fake rank must be in [0, {world_size}), got {rank}")
        torchtitan.distributed.utils.init_fake_mode(world_size, rank=rank)
        return DistributedTopology(world_size)

    def set_env(name, value):
        if name in os.environ:
            logger.warning("ENV[%s] = %s will be overridden to %s based on job config", name, os.environ[name], value)
        os.environ[name] = value

    set_env("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    set_env("TORCH_FR_BUFFER_SIZE", str(comm_config.trace_buf_size))
    if comm_config.trace_buf_size > 0:
        set_env("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
        folder = os.path.join(base_folder, comm_config.save_traces_folder)
        os.makedirs(folder, exist_ok=True)
        set_env("TORCH_FR_DUMP_TEMP_FILE", os.path.join(folder, comm_config.save_traces_file_prefix))
    backend = dist.Backend.default_device_backend_map.get(device_type, "nccl")
    if enable_cpu_backend:
        backend = f"{device_type}:{backend},cpu:gloo"
    group_kwargs: dict[str, Any] = {}
    if "_ranks" in inspect.signature(dist.init_process_group).parameters:
        group_kwargs["_ranks"] = ranks if ranks is not None else []
    elif ranks is not None:
        raise NotImplementedError("Explicit process-group ranks require a newer PyTorch runtime")
    dist.init_process_group(
        backend=backend, timeout=timedelta(seconds=comm_config.init_timeout_seconds), **group_kwargs
    )
    return DistributedTopology(dist.get_world_size())


def apply() -> None:
    torchtitan.distributed.utils.init_distributed = init_distributed


apply()
