# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 debug model recipes and small-scale launcher configurations."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING

from torchtitan.components.data import GrainDataLoader, HuggingFaceStreamingSource
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS, VisionProcessor
from torchtitan.config import derive
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan_recipes.tests.models import qwen3_5 as _upstream_config_registry

from torchtitan_npu.models.qwen3_5 import NPUQwen35Model, model_registry

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer


def _multimodal_dataloader() -> GrainDataLoader.Config:
    """Build the multimodal loader for DATASET, honouring DATASET_PATH.

    The dataset, collator and shuffle policy are the upstream multimodal ones;
    this constructs the loader directly instead of patching a text loader.
    """
    dataset_name = os.environ.get("DATASET", "cc12m-test")
    if dataset_name not in MM_DATASETS:
        raise ValueError(f"Unknown DATASET {dataset_name!r}; expected one of {sorted(MM_DATASETS)}")
    dataset = copy.deepcopy(MM_DATASETS[dataset_name])
    dataset_path = os.environ.get("QWEN_DATASET_PATH") or os.environ.get("DATASET_PATH")
    if dataset_path is not None and not dataset_path.strip():
        raise ValueError("QWEN_DATASET_PATH / DATASET_PATH must be non-empty when set")
    if dataset_path:
        source = dataset.source
        assert isinstance(source, HuggingFaceStreamingSource.Config)
        source.path = dataset_path
    processor = dataset.processor
    assert isinstance(processor, VisionProcessor.Config)
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            build_mrope_positions=True,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
        ),
        streaming_shuffle_buffer_size=128,
    )


def qwen35_debugmodel(seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel(seq_len=seq_len)
    config.model = derive(config.model, NPUQwen35Model.Config)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_debugmodel_varlen_attn(seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN) -> Trainer.Config:
    config = qwen35_debugmodel(seq_len=seq_len)
    config.model = model_registry("debugmodel", parallelism=config.parallelism, seq_len=seq_len, attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.model = derive(config.model, NPUQwen35Model.Config)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_debugmodel_moe_float8_lora(seq_len: int | None = 512) -> Trainer.Config:
    """LoRA on NPU block-FP8 weights with block-32 activation quantization."""
    from torchtitan.config.transform import (
        LoRATransform,
        apply_transforms,
    )
    from torchtitan.models.common.linear import GroupedLinear, Linear

    from torchtitan_npu.extensions.adaptor.torchao.param_swaps import block_fp8_param_swap
    from torchtitan_npu.extensions.adaptor.torchao.recipes import NpuQuantizeConverter

    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.model = model_registry(
        "debugmodel_moe",
        parallelism=config.parallelism,
        seq_len=seq_len,
        converters=[
            NpuQuantizeConverter.Config(
                base_config=block_fp8_param_swap(),
                unified_npu_quant_modules=(Linear.Config, GroupedLinear.Config),
            )
        ],
    )
    return apply_transforms(
        config,
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["w13", "w2"],
            )
        ],
    )


def _configure_resources(config: Trainer.Config) -> None:
    from torchtitan.components.data import SingleDatasetConfig

    sibling_checkout = Path(__file__).resolve().parents[3].parent / "torchtitan"
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
