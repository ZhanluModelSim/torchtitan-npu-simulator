# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

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
    TrainingConfig,
)
from torchtitan_npu.config.configs import (
    TrainerConfig as NpuTrainerConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry
from .config_overrides import (
    DeepSeekV32ModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """DeepSeek V3.2 config with stable CLI model overrides."""

    model_overrides: DeepSeekV32ModelOverrides = field(default_factory=DeepSeekV32ModelOverrides)

    def __post_init__(self) -> None:
        baseline_mtp = self.model_spec.model.num_mtp_modules
        override_mtp = self.model_overrides.num_mtp_modules
        training_mtp = self.training.num_mtp_modules
        if override_mtp != baseline_mtp:
            if training_mtp not in {baseline_mtp, override_mtp}:
                raise ValueError("model_overrides.num_mtp_modules conflicts with training.num_mtp_modules")
            if training_mtp == baseline_mtp:
                self.training.num_mtp_modules = override_mtp
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _model_spec_with_overrides(flavor: str):
    return build_model_spec_with_overrides(model_registry(flavor))


def _default_converters(*, enable_fused_moe: bool = True) -> list:
    converters = [
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]
    if enable_fused_moe:
        converters.extend(
            [
                get_model_converter_config("npu_moe_dispatch"),
                get_model_converter_config("npu_gmm"),
            ]
        )
    return converters


def deepseek_v32_671b_4layers_debug() -> TrainerConfig:
    model_spec, model_overrides = _model_spec_with_overrides("671B_debug_4_layers")
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
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
            local_batch_size=4,
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


def deepseek_v32_671b_61layers_4k_128die() -> TrainerConfig:
    model_spec, model_overrides = _model_spec_with_overrides("671B_debug_128die")
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
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


def deepseek_v32_671b_61layers_32k_128die() -> TrainerConfig:
    model_spec, model_overrides = _model_spec_with_overrides("671B_debug_128die")
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
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


def deepseek_v32_smoketest() -> TrainerConfig:
    """Small single-rank recipe that still contains dense and MoE layers."""
    model_spec, model_overrides = _model_spec_with_overrides("smoketest")
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
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
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
            dtype="float32",
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
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
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def deepseek_v32_tp_smoketest() -> TrainerConfig:
    """Two-rank TP recipe using the ATen MoE path instead of NPU GMM."""
    config = deepseek_v32_smoketest()
    config.model_converters = ModelConvertersContainer.Config(converters=_default_converters(enable_fused_moe=False))
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.disable_loss_parallel = True
    return config
