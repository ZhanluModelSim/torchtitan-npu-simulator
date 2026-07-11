# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, CompileConfig
from torchtitan.experiments.vlm.configs import MultiModalTrainerConfig
from torchtitan.experiments.vlm.datasets.mm_datasets import (
    HuggingFaceMultiModalDataLoader,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ParallelismConfig,
    ProfilingConfig,
    TrainingConfig,
)
from torchtitan_npu.converters.registry import get_model_converter_config

from . import model_registry

VLM_TOKENIZER_ASSETS_PATH = "./tests/assets/tokenizer/vlm_tokenizer"


def vlm_debugmodel_npu() -> MultiModalTrainerConfig:
    return MultiModalTrainerConfig(
        hf_assets_path=VLM_TOKENIZER_ASSETS_PATH,
        model_spec=model_registry("debugmodel"),
        model_converters=ModelConvertersContainer.Config(converters=[get_model_converter_config("npu_vlm")]),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceMultiModalDataLoader.Config(
            dataset="cc12m-test",
            max_patches_per_image=1024,
            max_images_per_batch=64,
        ),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=8e-4,
            implementation="for-loop",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def vlm_debugmodel_npu_compile() -> MultiModalTrainerConfig:
    return replace(vlm_debugmodel_npu(), compile=CompileConfig(enable=True))
