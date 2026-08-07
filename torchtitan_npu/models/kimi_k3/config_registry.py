# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for Kimi K3."""

from dataclasses import dataclass, field

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
    """Kimi K3 config with an optional MXFP8 target-FQN override."""

    mxfp8_fqns: list[str] | None = field(default=None)

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
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_kimi_k3_moe"),
    ]
    if enable_mxfp8:
        converters.append(
            MXFP8Converter.Config(
                recipe_name="mxfp8_rceil",
                fqns=["moe.experts", "moe.shared_experts"],
            )
        )
    return converters


def _parallelism(*, expert_parallel_degree: int = 128) -> ParallelismConfig:
    """Return the Kimi K3 baseline parallel layout.

    Kimi K3 is an MoE model, so the production baseline shards parameters
    through FSDP and distributes routed experts across 128 EP ranks. Other
    dimensions stay disabled unless a recipe explicitly opts into them.
    """
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
    optimizer: OptimizerConfig,
    lr_scheduler: LRSchedulersContainer.Config,
    parallelism: ParallelismConfig,
    activation_checkpoint: ActivationCheckpointConfig,
    enable_mxfp8: bool,
    print_config: bool,
) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_registry(flavor),
        debug=DebugConfig(print_config=print_config),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
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


def _baseline(*, enable_mxfp8: bool) -> TrainerConfig:
    """Full Kimi K3 baseline with FSDP auto-sharding and EP=128."""
    return _trainer_config(
        flavor="full",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
        ),
        optimizer=OptimizerConfig(
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
        parallelism=_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        enable_mxfp8=enable_mxfp8,
        print_config=True,
    )


def kimi_k3_baseline_bf16() -> TrainerConfig:
    return _baseline(enable_mxfp8=False)


def kimi_k3_baseline_mxfp8() -> TrainerConfig:
    return _baseline(enable_mxfp8=True)


def kimi_k3_smoketest() -> TrainerConfig:
    """Minimal 4-layer recipe; EP=1 is intentional for local debugging."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
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
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        enable_mxfp8=False,
        print_config=True,
    )


def kimi_k3_16layer_reduced() -> TrainerConfig:
    """16-layer reduced model for single-node validation (matches MindSpeed-MM A3 config)."""
    return _trainer_config(
        flavor="16layer_reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
        ),
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
        parallelism=_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        enable_mxfp8=False,
        print_config=True,
    )
