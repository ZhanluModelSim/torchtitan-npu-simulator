# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
    ]


def deepseek_v32_671b_16npus() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./assets/hf/DeepSeek-V3.2",
        model_spec=model_registry("_v32_671b_61layers_16experts"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            weight_decay=0.01,
            beta2=0.999,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=5,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=2048,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            interval=10000,
            last_save_model_only=False,
            export_dtype="float32",
            async_mode="disabled",
            load_only=True,
            initial_load_path="./checkpoint/DeepSeek-V3.2",
            initial_load_in_hf=False,
            initial_load_in_hf_quantized=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        profiling=ProfilingConfig(
            enable_profiling=False,
            enable_online_parse=False,
            profile_ranks=[0],
            profile_step_start=6,
            profile_step_end=7,
            profile_record_shapes=True,
            profile_with_memory=True,
        ),
    )


def deepseek_v32_671b_128npus() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./assets/hf/DeepSeek-V3.2",
        model_spec=model_registry("_v32_671b_61layers_256experts"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="enwiki-eod"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=7.3e-6,
            eps=1e-6,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=512,
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
            expert_parallel_degree=64,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            interval=10000,
            last_save_model_only=True,
            export_dtype="float32",
            async_mode="disabled",
            load_only=True,
            initial_load_path="./checkpoint/DeepSeek-V3.2",
            initial_load_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def deepseek_v32_671b_32k_128npus() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./assets/hf/DeepSeek-V3.2",
        model_spec=model_registry("_v32_671b_61layers_256experts"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="enwiki-eod"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=7.3e-6,
            eps=1e-6,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=128,
            seq_len=32768,
            max_norm=1.0,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=4,
            pipeline_parallel_degree=1,
            expert_parallel_degree=64,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=8,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            interval=10000,
            last_save_model_only=True,
            export_dtype="float32",
            async_mode="disabled",
            load_only=True,
            initial_load_path="./checkpoint/DeepSeek-V3.2",
            initial_load_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )
