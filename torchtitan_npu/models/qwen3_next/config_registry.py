# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for Qwen3-Next."""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ActivationCheckpointConfig, CommConfig, CompileConfig, DebugConfig
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


def _default_converters() -> list:
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    optimizer: OptimizerConfig,
    lr_scheduler: LRSchedulersContainer.Config,
    parallelism: ParallelismConfig,
    activation_checkpoint: ActivationCheckpointConfig,
    print_config: bool,
) -> NpuTrainerConfig:
    return NpuTrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/qwen3-tokenizer",
        model_spec=model_registry(flavor),
        debug=DebugConfig(print_config=print_config),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters()
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def _parallelism(*, expert_parallel_degree: int = 1) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def qwen3_next_smoketest() -> NpuTrainerConfig:
    """Minimal 4-layer recipe (3 GDN + 1 gated attention); local debugging."""
    return _trainer_config(
        flavor="debugmodel",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
            # MindSpeed fsdp2 example uses reduce_dtype bf16
            mixed_precision_reduce="bfloat16",
        ),
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
        parallelism=_parallelism(expert_parallel_degree=1),
        # Measured fsdp2 config runs with recompute: False
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        print_config=True,
    )


def qwen3_next_80b_a3b() -> NpuTrainerConfig:
    """Qwen3-Next-80B-A3B baseline mirroring the measured fsdp2 run
    (pretrain_qwen3_next_16k_fsdp2_A3.yaml): FSDP=8, EP=2, eFSDP=4,
    recompute disabled, reduce bf16.

    Override via CLI, e.g. ``--parallelism.expert_parallel_degree 8``.
    """
    return _trainer_config(
        flavor="80B-A3B",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16384,
            max_norm=1.0,
            steps=1,
            # MindSpeed fsdp2 example uses reduce_dtype bf16
            mixed_precision_reduce="bfloat16",
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(expert_parallel_degree=2),
        # Measured config runs with recompute: False
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        print_config=True,
    )
