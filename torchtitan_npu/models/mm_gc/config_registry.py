# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for mm_gc."""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
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
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters.kernels.mm_gc_mxfp8 import MMGcMXFP8Converter

from . import model_registry


def _default_converters(*, enable_mxfp8: bool) -> list:
    """mm_gc runs without NPU fused-op converters by default; MXFP8 covers
    every matmul (attention/dense/MoE projections + shared expert) and the
    routed-expert grouped matmul, scoped by module-path FQNs. The SLA2 block
    router and the MoE head router always stay fp32."""
    if not enable_mxfp8:
        return []
    return [MMGcMXFP8Converter.Config()]


def _mxfp8_fqns_converter(fqns: list[str]) -> MMGcMXFP8Converter.Config:
    return MMGcMXFP8Converter.Config(fqns=fqns)


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    activation_checkpoint: ActivationCheckpointConfig,
    print_config: bool,
    enable_mxfp8: bool = False,
) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_registry(flavor),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
        ),
        debug=DebugConfig(print_config=print_config),
        comm=CommConfig(trace_buf_size=0),
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
        training=training,
        parallelism=parallelism,
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def _single_rank_parallelism(expert_parallel_degree: int = 1) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def mm_gc_smoketest() -> TrainerConfig:
    """Minimal 6-layer recipe; single rank, debug flavor, AC selective."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        print_config=True,
    )


def mm_gc_reduced() -> TrainerConfig:
    """16-layer reduced model for meta/simulator validation."""
    return _trainer_config(
        flavor="reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        print_config=False,
    )


def mm_gc_baseline() -> TrainerConfig:
    """Full enlarged-spec baseline (96 layers, 32 heads x 1024 experts)."""
    return _trainer_config(
        flavor="full",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        print_config=True,
    )


def mm_gc_smoketest_mxfp8() -> TrainerConfig:
    """Debug recipe with MXFP8 on all matmuls + routed-expert grouped MM."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        print_config=True,
        enable_mxfp8=True,
    )


def mm_gc_reduced_mxfp8() -> TrainerConfig:
    """Reduced spec with MXFP8; single-rank meta validation."""
    return _trainer_config(
        flavor="reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        print_config=False,
        enable_mxfp8=True,
    )


def mm_gc_baseline_mxfp8() -> TrainerConfig:
    """Full enlarged-spec baseline with MXFP8 (A5 hardware target)."""
    return _trainer_config(
        flavor="full",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
        ),
        parallelism=_single_rank_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        print_config=True,
        enable_mxfp8=True,
    )
