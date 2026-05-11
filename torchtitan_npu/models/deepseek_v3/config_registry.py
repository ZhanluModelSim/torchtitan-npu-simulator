# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters.registry import get_npu_converter_config
from torchtitan_npu.patches.optimizer.swap_optimizer import SwapOptimizersContainer

from . import model_registry


def deepseek_v3_671b_debug() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="/mnt/share/hanboyou/DeepSeek-V3.1-Base",
        model_spec=model_registry("671B_debug_16die"),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                get_npu_converter_config("npu_rms_norm"),
                get_npu_converter_config("npu_rope"),
                get_npu_converter_config("npu_permute"),
                get_npu_converter_config("npu_gmm"),
            ],
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=SwapOptimizersContainer.Config(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=4,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=1,
            expert_parallel_degree=2,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
            enable_custom_context_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="/mnt/models/DeepSeek-V3",
            interval=500,
            last_save_model_only=True,
            load_only=True,
            initial_load_in_hf=True,
            initial_load_path="/mnt/share/c00878090/20260330/torchtitan-npu-dsv3/dsv3_4layers_0401_2/",
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        profiling=ProfilingConfig(
            enable_profiling=False,
            profile_step_start=5,
            profile_step_end=6,
            profile_ranks=[0],
            profile_record_shapes=True,
            profile_with_memory=False,
            profile_with_stack=False,
            enable_online_parse=True,
        ),
    )


def deepseek_v3_671b_16die_debug() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry("671B_debug_16die"),
        metrics=MetricsProcessor.Config(log_freq=10),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=SwapOptimizersContainer.Config(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            max_norm=1.0,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=2,
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=4,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=500,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        profiling=ProfilingConfig(
            enable_profiling=False,
            profile_step_start=5,
            profile_step_end=6,
            profile_ranks=[0],
            profile_record_shapes=True,
            profile_with_memory=False,
            profile_with_stack=False,
            enable_online_parse=True,
        ),
    )


def deepseek_v3_671b_61layers_4k_128die() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry("671B_debug_128die"),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="enwiki-eod"),
        optimizer=SwapOptimizersContainer.Config(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=1024,
            seq_len=4096,
            max_norm=1.0,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=4,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=128,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=500,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        profiling=ProfilingConfig(
            enable_profiling=False,
            profile_step_start=5,
            profile_step_end=6,
            profile_ranks=[0],
            profile_record_shapes=True,
            profile_with_memory=False,
            profile_with_stack=False,
            enable_online_parse=True,
        ),
    )


def deepseek_v3_smoketest() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("671B_debug"),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=SwapOptimizersContainer.Config(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
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
            enable_custom_context_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        profiling=ProfilingConfig(
            enable_profiling=False,
        ),
    )
