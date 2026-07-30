# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for Kimi K3."""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ActivationCheckpointConfig, DebugConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry


def _default_converters() -> list:
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]


def kimi_k3_smoketest() -> TrainerConfig:
    """Minimal smoketest: 4-layer debug model, 2 steps, random weights."""
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debug"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=[]),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def kimi_k3_16layer_reduced() -> TrainerConfig:
    """16-layer reduced model for single-node validation (matches MindSpeed-MM A3 config)."""
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("16layer_reduced"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=4,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        profiling=ProfilingConfig(enable_profiling=False),
    )
