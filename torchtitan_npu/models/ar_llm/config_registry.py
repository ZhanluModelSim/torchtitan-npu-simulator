# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for ar_llm (DeepSeekV4-Sparse)."""

from dataclasses import dataclass, field

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
    TrainerConfig as NpuTrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry
from .config_overrides import (
    ArLlmModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """ar_llm config with stable model CLI overrides."""

    model_overrides: ArLlmModelOverrides = field(default_factory=ArLlmModelOverrides)

    def __post_init__(self) -> None:
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _default_converters() -> list:
    """npu_rms_norm/npu_rope have simulator shims; model-specific fused ops
    (KDA, CSA/HCA, mHC, LatentMoE GMM) are captured through their own paths,
    so no NPU converter is required for the meta acceptance."""
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]


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


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    activation_checkpoint: ActivationCheckpointConfig,
) -> TrainerConfig:
    model_spec, model_overrides = build_model_spec_with_overrides(
        model_registry(flavor)
    )
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters()
        ),
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
        parallelism=_parallelism(),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def ar_llm_debug() -> TrainerConfig:
    """Minimal 6-layer (one CSA/HCA/KDA unit) recipe for local debugging."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def ar_llm_reduced() -> TrainerConfig:
    """12-layer reduced spec for meta single-step and AC toggle tests."""
    return _trainer_config(
        flavor="reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def ar_llm_50t() -> TrainerConfig:
    """Official 50T spec (meta/capacity validation only, never real training)."""
    return _trainer_config(
        flavor="50t",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=65536,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def ar_llm_100t() -> TrainerConfig:
    """Official 100T spec (meta/capacity validation only, never real training)."""
    return _trainer_config(
        flavor="100t",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=65536,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )
