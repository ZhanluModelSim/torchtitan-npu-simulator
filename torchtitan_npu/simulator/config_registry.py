# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin simulator wrappers around the DeepSeek-V4 training configs."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torchtitan_npu.models.deepseek_v4 import config_registry as _model_configs
from torchtitan_npu.models.deepseek_v4.config_overrides import (
    DeepSeekV4ModelOverrides,
    apply_model_overrides,
)
from torchtitan_npu.simulator.trainer import SimulationConfig, SimulationTrainerConfig

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass(kw_only=True, slots=True)
class DeepSeekV4SimulationTrainerConfig(SimulationTrainerConfig):
    model_overrides: DeepSeekV4ModelOverrides = dataclasses.field(
        default_factory=DeepSeekV4ModelOverrides
    )
    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )
        _model_configs._apply_mxfp8_fqns_override(
            self.model_converters,
            self.mxfp8_fqns,
        )


def _simulation_config(
    factory: Callable[[], _model_configs.TrainerConfig],
    *,
    output_name: str,
) -> DeepSeekV4SimulationTrainerConfig:
    base_config = factory()
    base_fields = {field.name: getattr(base_config, field.name) for field in dataclasses.fields(base_config)}
    # Simulator capture requires eager dispatch. This must be disabled before
    # entry.py performs its compile dependency checks.
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return DeepSeekV4SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def deepseek_v4_flash_baseline_bf16() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_flash_baseline_bf16,
        output_name="deepseek_v4_flash_baseline_bf16",
    )


def deepseek_v4_flash_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_flash_baseline_mxfp8,
        output_name="deepseek_v4_flash_baseline_mxfp8",
    )


def deepseek_v4_pro_baseline_bf16() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_pro_baseline_bf16,
        output_name="deepseek_v4_pro_baseline_bf16",
    )


def deepseek_v4_pro_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_pro_baseline_mxfp8,
        output_name="deepseek_v4_pro_baseline_mxfp8",
    )


def deepseek_v4_pro_20t_baseline_bf16() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_pro_20t_baseline_bf16,
        output_name="deepseek_v4_pro_20t_baseline_bf16",
    )


def deepseek_v4_pro_20t_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_pro_20t_baseline_mxfp8,
        output_name="deepseek_v4_pro_20t_baseline_mxfp8",
    )


def deepseek_v4_smoketest() -> SimulationTrainerConfig:
    return _simulation_config(
        _model_configs.deepseek_v4_smoketest,
        output_name="deepseek_v4_smoketest",
    )


# ---------------------------------------------------------------------------
# Kimi K3 simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.kimi_k3.config_registry import (  # noqa: E402
    _default_converters as _kimi_k3_default_converters,
    kimi_k3_smoketest as _kimi_k3_smoketest,
)


def kimi_k3_simulate() -> SimulationTrainerConfig:
    """Kimi K3 debug model: FSDP2 + CP=2 + EP=2, PP=1, TP=1."""
    base_config = _kimi_k3_smoketest()
    base_config.parallelism.data_parallel_shard_degree = 4
    base_config.parallelism.tensor_parallel_degree = 1
    base_config.parallelism.pipeline_parallel_degree = 1
    base_config.parallelism.expert_parallel_degree = 2
    base_config.parallelism.context_parallel_degree = 2
    base_config.parallelism.expert_tensor_parallel_degree = 1
    base_fields = {field.name: getattr(base_config, field.name) for field in dataclasses.fields(base_config)}
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir="./simulator_output/kimi_k3_simulate"),
    )


def kimi_k3_full_simulate() -> SimulationTrainerConfig:
    """Kimi K3 full 2.8T model: 93 layers, 896 experts, FSDP2+CP=4+EP=128."""
    from torchtitan_npu.models.kimi_k3 import model_registry as kimi_k3_model_registry
    from torchtitan_npu.models.kimi_k3.config_registry import TrainerConfig as K3TrainerConfig
    from torchtitan_npu.config.configs import (
        CheckpointConfig,
        OptimizerConfig,
        ParallelismConfig,
        ProfilingConfig,
        TrainingConfig,
    )
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.config import ActivationCheckpointConfig, DebugConfig
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from torchtitan.protocols.model_converter import ModelConvertersContainer

    base_config = K3TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=kimi_k3_model_registry("full"),
        debug=DebugConfig(print_config=False),
        model_converters=ModelConvertersContainer.Config(
            converters=_kimi_k3_default_converters()
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(name="AdamW", lr=2.2e-4, eps=1e-8),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000, decay_ratio=0.8, decay_type="cosine", min_lr_factor=0.1,
        ),
        training=TrainingConfig(local_batch_size=1, seq_len=4096, max_norm=1.0, steps=1),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1024,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=128,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=4,
        ),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        profiling=ProfilingConfig(enable_profiling=False),
    )
    base_fields = {field.name: getattr(base_config, field.name) for field in dataclasses.fields(base_config)}
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir="./simulator_output/kimi_k3_full_simulate"),
    )
