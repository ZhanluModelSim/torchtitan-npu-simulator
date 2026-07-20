# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig as NpuTrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """DeepSeek V4 config with a stable MXFP8 FQN CLI override."""

    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        _apply_mxfp8_fqns_override(self.model_converters, self.mxfp8_fqns)


def _apply_mxfp8_fqns_override(
    model_converters: ModelConvertersContainer.Config,
    fqns: list[str] | None,
) -> None:
    if fqns is None:
        return

    mxfp8_configs = [
        converter
        for converter in model_converters.converters
        if isinstance(converter, MXFP8Converter.Config)
    ]
    if len(mxfp8_configs) != 1:
        raise ValueError(
            "mxfp8_fqns requires exactly one MXFP8 converter, "
            f"but found {len(mxfp8_configs)}"
        )
    mxfp8_configs[0].fqns = list(fqns)


def _default_converters(*, enable_mxfp8: bool) -> list:
    converters = [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
    ]
    if enable_mxfp8:
        converters.append(
            MXFP8Converter.Config(
                recipe_name="mxfp8_rceil",
                fqns=[
                    "pre_attention.wq_a",
                    "pre_attention.wq_b",
                    "pre_attention.wkv",
                    "pre_attention.indexer.wq_b",
                    "pre_attention.indexer.weights_proj",
                    "post_attention.wo_a",
                    "post_attention.wo_b",
                    "moe.experts",
                    "moe.shared_experts",
                    "e_proj",
                    "h_proj",
                ],
            )
        )
    return converters


def _adamw_optimizer() -> OptimizerConfig:
    return OptimizerConfig(
        name="AdamW",
        lr=1e-5,
        eps=1e-6,
        swap_optimizer=True,
    )


def _lr_scheduler(*, warmup_steps: int) -> LRSchedulersContainer.Config:
    return LRSchedulersContainer.Config(
        warmup_steps=warmup_steps,
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.01,
    )


def _parallelism(*, expert_parallel_degree: int) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        fsdp_reshard_after_forward="always",
        tensor_parallel_degree=1,
        enable_async_tensor_parallel=False,
        pipeline_parallel_degree=1,
        pipeline_parallel_schedule="1F1B",
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def _profiling() -> ProfilingConfig:
    return ProfilingConfig(
        enable_profiling=False,
        enable_online_parse=False,
        profile_ranks=[0],
        profile_step_start=6,
        profile_step_end=7,
        profile_record_shapes=True,
        profile_with_memory=True,
        enable_memory_snapshot=False,
        save_memory_snapshot_folder="memory_snapshot",
    )


def _flash_baseline(*, enable_mxfp8: bool) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv4_tokenizer",
        model_spec=model_registry("v4_flash_baseline"),
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_adamw_optimizer(),
        lr_scheduler=_lr_scheduler(warmup_steps=400),
        training=TrainingConfig(
            global_batch_size=1024,
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
            num_mtp_modules=1,
        ),
        parallelism=_parallelism(expert_parallel_degree=32),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            initial_load_in_hf=False,
            initial_load_path="/data/models/dsv4_flash_bf16",
            interval=10000,
            last_save_model_only=True,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=_profiling(),
    )


def deepseek_v4_flash_baseline_bf16() -> TrainerConfig:
    return _flash_baseline(enable_mxfp8=False)


def deepseek_v4_flash_baseline_mxfp8() -> TrainerConfig:
    return _flash_baseline(enable_mxfp8=True)


def _pro_baseline(*, enable_mxfp8: bool) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseek_v4_pro_tokenizer",
        model_spec=model_registry("v4_pro_baseline"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_adamw_optimizer(),
        lr_scheduler=_lr_scheduler(warmup_steps=400),
        training=TrainingConfig(
            global_batch_size=384,
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
            num_mtp_modules=1,
        ),
        parallelism=_parallelism(expert_parallel_degree=64),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            initial_load_in_hf=True,
            initial_load_path="/data/models/deepseek-v4-pro-bfloat16",
            interval=10000,
            last_save_model_only=True,
            load_only=True,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        profiling=_profiling(),
    )


def deepseek_v4_pro_baseline_bf16() -> TrainerConfig:
    return _pro_baseline(enable_mxfp8=False)


def deepseek_v4_pro_baseline_mxfp8() -> TrainerConfig:
    return _pro_baseline(enable_mxfp8=True)


def _pro_20t_baseline(*, enable_mxfp8: bool) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv4_tokenizer",
        model_spec=model_registry("v4_pro_20t_baseline"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=True),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_adamw_optimizer(),
        lr_scheduler=_lr_scheduler(warmup_steps=4),
        training=TrainingConfig(
            global_batch_size=2048,
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
            num_mtp_modules=1,
        ),
        parallelism=_parallelism(expert_parallel_degree=256),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=True,
            initial_load_in_hf=False,
            initial_load_path="/data/models/dsv4_bf16",
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def deepseek_v4_pro_20t_baseline_bf16() -> TrainerConfig:
    return _pro_20t_baseline(enable_mxfp8=False)


def deepseek_v4_pro_20t_baseline_mxfp8() -> TrainerConfig:
    return _pro_20t_baseline(enable_mxfp8=True)


def deepseek_v4_smoketest() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv4_tokenizer",
        model_spec=model_registry("smoketest"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=[]),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
            num_mtp_modules=0,
        ),
        parallelism=_parallelism(expert_parallel_degree=1),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )
