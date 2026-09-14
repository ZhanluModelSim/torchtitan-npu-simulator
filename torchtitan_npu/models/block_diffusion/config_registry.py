# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training recipes for the native Block Diffusion model."""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
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

from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
)

from . import model_registry
from .data import BlockDiffusionDataLoader


def _converters() -> list:
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
    ]


def _parallelism(
    *,
    dp_shard: int = 1,
    tp: int = 1,
    ep: int = 1,
    etp: int = 1,
    cp: int = 1,
    pp: int = 1,
) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=dp_shard,
        tensor_parallel_degree=tp,
        pipeline_parallel_degree=pp,
        expert_parallel_degree=ep,
        expert_tensor_parallel_degree=etp,
        context_parallel_degree=cp,
    )


def _trainer_config(
    *,
    flavor: str,
    seq_len: int,
    steps: int,
    parallelism: ParallelismConfig,
    ac_mode: str,
) -> TrainerConfig:
    model_spec = model_registry(flavor)
    model_config = model_spec.model
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=BlockDiffusionDataLoader.Config(
            dataset="c4_test",
            block_size=model_config.block_size,
            mask_token_id=model_config.mask_token_id,
        ),
        optimizer=OptimizerConfig(name="AdamW", lr=1e-4, eps=1e-8),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=seq_len,
            max_norm=1.0,
            steps=steps,
        ),
        parallelism=parallelism,
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode=ac_mode),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def block_diffusion_smoketest() -> TrainerConfig:
    """Small MoE recipe for local construction and one-step validation."""

    return _trainer_config(
        flavor="debug",
        seq_len=32,
        steps=2,
        parallelism=_parallelism(),
        ac_mode="none",
    )


def block_diffusion_dense_smoketest() -> TrainerConfig:
    """Dense counterpart used to validate the non-MoE architecture path."""

    return _trainer_config(
        flavor="dense_debug",
        seq_len=16,
        steps=2,
        parallelism=_parallelism(),
        ac_mode="full",
    )


def block_diffusion_reduced() -> TrainerConfig:
    """Reduced structural model for meta parallel-combination validation."""

    return _trainer_config(
        flavor="reduced",
        seq_len=256,
        steps=20,
        parallelism=_parallelism(),
        ac_mode="full",
    )


def block_diffusion_baseline() -> TrainerConfig:
    """Full 10.05T structural baseline using its intended TP x EP layout."""

    return _trainer_config(
        flavor="full",
        seq_len=4096,
        steps=2000,
        parallelism=_parallelism(dp_shard=64, tp=8, ep=512),
        ac_mode="full",
    )
