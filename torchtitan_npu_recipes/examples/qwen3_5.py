# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complete Qwen3.5 debug configurations selected by the example launchers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer


def _configure_resources(config: Trainer.Config) -> None:
    from torchtitan.components.data import GrainDataLoader, HuggingFaceStreamingSource, SingleDatasetConfig

    sibling_checkout = Path(__file__).resolve().parents[2].parent / "torchtitan"
    upstream_checkout = Path(os.environ.get("TORCHTITAN_REPO") or os.environ.get("TORCHTITAN_DIR") or sibling_checkout)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH") or str(upstream_checkout / "tests/assets/tokenizer")
    if not (os.environ.get("QWEN_DATASET_PATH") or os.environ.get("DATASET_PATH")):
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        assert isinstance(config.dataloader.dataset, SingleDatasetConfig)
        source = config.dataloader.dataset.source
        assert isinstance(source, HuggingFaceStreamingSource.Config)
        source.path = str(upstream_checkout / "tests/assets/cc12m_test")


def _configure_overrides(config: Trainer.Config, *, use_npu_moe_dispatcher: bool | None) -> None:
    if "OVERRIDE_IMPORTS" in os.environ:
        raise ValueError("Set config.override.imports in a Python recipe or use --override instead of OVERRIDE_IMPORTS")
    if use_npu_moe_dispatcher is None:
        use_npu_moe_dispatcher = os.environ.get("ENABLE_NPU_MOE_DISPATCHER", "0") == "1"
    config.override.imports = ["torchtitan_npu.override.qwen3_5.gated_delta.npu"]
    if use_npu_moe_dispatcher:
        config.override.imports.append("torchtitan_npu.override.common.token_dispatcher.asc")


def qwen3_5_debugmodel_1p_cpt_512(*, use_npu_moe_dispatcher: bool | None = None) -> Trainer.Config:
    from torchtitan.config.parallelism import ParallelismConfig

    from torchtitan_npu_recipes.models.qwen3_5 import qwen35_debugmodel

    config = qwen35_debugmodel(seq_len=512)
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        data_parallel_replicate_degree=1,
        expert_parallel_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.num_tokens_per_train_step = 512
    config.training.max_context_length = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    config.debug.print_config = True
    _configure_resources(config)
    _configure_overrides(config, use_npu_moe_dispatcher=use_npu_moe_dispatcher)
    return config


def qwen3_5_debugmodel_moe_8p_cpt_512(*, use_npu_moe_dispatcher: bool | None = None) -> Trainer.Config:
    from torchtitan.config.parallelism import ParallelismConfig

    from torchtitan_npu.models.qwen3_5 import model_registry
    from torchtitan_npu_recipes.models.qwen3_5 import qwen35_debugmodel_moe

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("NGPU", "8")))
    if world_size <= 0 or world_size % 8:
        raise ValueError("Qwen3.5 MoE requires WORLD_SIZE (or NGPU) to be a positive multiple of 8")
    config = qwen35_debugmodel_moe(seq_len=512)
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=world_size // 8,
        expert_parallel_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    config.model = model_registry("debugmodel_moe", seq_len=512, parallelism=config.parallelism)
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.training.num_tokens_per_train_step = -1
    config.training.max_context_length = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    config.debug.print_config = True
    _configure_resources(config)
    _configure_overrides(config, use_npu_moe_dispatcher=use_npu_moe_dispatcher)
    return config
