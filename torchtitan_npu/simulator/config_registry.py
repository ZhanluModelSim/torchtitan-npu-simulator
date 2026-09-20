# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin simulator wrappers around production model training configs."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torchtitan_npu.models.deepseek_v4 import config_registry as _model_configs
from torchtitan_npu.models.deepseek_v4.config_overrides import (
    DeepSeekV4ModelOverrides,
    apply_model_overrides,
)
from torchtitan_npu.config.configs import TrainerConfig
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

from torchtitan_npu.models.kimi_k3 import config_registry as _kimi_k3_configs  # noqa: E402
from torchtitan_npu.models.kimi_k3.config_overrides import (  # noqa: E402
    KimiK3ModelOverrides,
    apply_model_overrides as apply_kimi_k3_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class KimiK3SimulationTrainerConfig(SimulationTrainerConfig):
    """Kimi K3 simulator config with stable model and MXFP8 CLI overrides."""

    model_overrides: KimiK3ModelOverrides = dataclasses.field(
        default_factory=KimiK3ModelOverrides
    )
    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        self.model_spec = apply_kimi_k3_model_overrides(
            self.model_spec,
            self.model_overrides,
        )
        _kimi_k3_configs._apply_mxfp8_fqns_override(
            self.model_converters,
            self.mxfp8_fqns,
        )


def _kimi_k3_simulation_config(
    factory: Callable[[], TrainerConfig],
    *,
    output_name: str,
) -> KimiK3SimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return KimiK3SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def kimi_k3_baseline_bf16() -> KimiK3SimulationTrainerConfig:
    return _kimi_k3_simulation_config(
        _kimi_k3_configs.kimi_k3_baseline_bf16,
        output_name="kimi_k3_baseline_bf16",
    )


def kimi_k3_baseline_mxfp8() -> KimiK3SimulationTrainerConfig:
    return _kimi_k3_simulation_config(
        _kimi_k3_configs.kimi_k3_baseline_mxfp8,
        output_name="kimi_k3_baseline_mxfp8",
    )


def kimi_k3_smoketest() -> KimiK3SimulationTrainerConfig:
    return _kimi_k3_simulation_config(
        _kimi_k3_configs.kimi_k3_smoketest,
        output_name="kimi_k3_smoketest",
    )


# ---------------------------------------------------------------------------
# ar_llm (DeepSeekV4-Sparse) simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.ar_llm import config_registry as _ar_llm_configs  # noqa: E402
from torchtitan_npu.models.ar_llm.config_overrides import (  # noqa: E402
    ArLlmModelOverrides,
    apply_model_overrides as apply_ar_llm_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class ArLlmSimulationTrainerConfig(SimulationTrainerConfig):
    """ar_llm simulator config with stable model CLI overrides."""

    model_overrides: ArLlmModelOverrides = dataclasses.field(
        default_factory=ArLlmModelOverrides
    )
    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        self.model_spec = apply_ar_llm_model_overrides(
            self.model_spec,
            self.model_overrides,
        )
        _ar_llm_configs._apply_mxfp8_fqns_override(
            self.model_converters,
            self.mxfp8_fqns,
        )


def _ar_llm_simulation_config(
    factory: Callable[[], TrainerConfig],
    *,
    output_name: str,
) -> ArLlmSimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return ArLlmSimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def ar_llm_smoketest() -> ArLlmSimulationTrainerConfig:
    return _ar_llm_simulation_config(
        _ar_llm_configs.ar_llm_smoketest,
        output_name="ar_llm_smoketest",
    )


def ar_llm_50t() -> ArLlmSimulationTrainerConfig:
    return _ar_llm_simulation_config(
        _ar_llm_configs.ar_llm_50t,
        output_name="ar_llm_50t",
    )


def ar_llm_50t_mxfp8() -> ArLlmSimulationTrainerConfig:
    return _ar_llm_simulation_config(
        _ar_llm_configs.ar_llm_50t_mxfp8,
        output_name="ar_llm_50t_mxfp8",
    )


def ar_llm_100t() -> ArLlmSimulationTrainerConfig:
    return _ar_llm_simulation_config(
        _ar_llm_configs.ar_llm_100t,
        output_name="ar_llm_100t",
    )


def ar_llm_100t_mxfp8() -> ArLlmSimulationTrainerConfig:
    return _ar_llm_simulation_config(
        _ar_llm_configs.ar_llm_100t_mxfp8,
        output_name="ar_llm_100t_mxfp8",
    )


# ---------------------------------------------------------------------------
# glm5_next (GLM-5.3-Flash) simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.glm5_next import config_registry as _glm5_next_configs  # noqa: E402
from torchtitan_npu.models.glm5_next.config_overrides import (  # noqa: E402
    Glm5NextModelOverrides,
    apply_model_overrides as apply_glm5_next_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class Glm5NextSimulationTrainerConfig(SimulationTrainerConfig):
    """glm5_next simulator config with stable model CLI overrides."""

    model_overrides: Glm5NextModelOverrides = dataclasses.field(
        default_factory=Glm5NextModelOverrides
    )
    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        self.model_spec = apply_glm5_next_model_overrides(
            self.model_spec,
            self.model_overrides,
        )
        _glm5_next_configs._apply_mxfp8_fqns_override(
            self.model_converters,
            self.mxfp8_fqns,
        )


def _glm5_next_simulation_config(
    factory: Callable[[], TrainerConfig],
    *,
    output_name: str,
    target_npu_device_type: str | None = None,
) -> Glm5NextSimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    simulation_kwargs = {}
    if target_npu_device_type is not None:
        simulation_kwargs["target_npu_device_type"] = target_npu_device_type
    return Glm5NextSimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(
            output_dir=f"./simulator_output/{output_name}",
            **simulation_kwargs,
        ),
    )


def glm5_next_smoketest() -> Glm5NextSimulationTrainerConfig:
    return _glm5_next_simulation_config(
        _glm5_next_configs.glm5_next_smoketest,
        output_name="glm5_next_smoketest",
    )


def glm5_next_baseline() -> Glm5NextSimulationTrainerConfig:
    return _glm5_next_simulation_config(
        _glm5_next_configs.glm5_next_baseline,
        output_name="glm5_next_baseline",
    )


def glm5_next_baseline_mxfp8() -> Glm5NextSimulationTrainerConfig:
    return _glm5_next_simulation_config(
        _glm5_next_configs.glm5_next_baseline_mxfp8,
        output_name="glm5_next_baseline_mxfp8",
        target_npu_device_type="A5",
    )


def glm5_next_baseline_mm() -> Glm5NextSimulationTrainerConfig:
    return _glm5_next_simulation_config(
        _glm5_next_configs.glm5_next_baseline_mm,
        output_name="glm5_next_baseline_mm",
    )


def glm5_next_baseline_mm_mxfp8() -> Glm5NextSimulationTrainerConfig:
    return _glm5_next_simulation_config(
        _glm5_next_configs.glm5_next_baseline_mm_mxfp8,
        output_name="glm5_next_baseline_mm_mxfp8",
        target_npu_device_type="A5",
    )


# ---------------------------------------------------------------------------
# mm_gc (SLA2 + Multi-Head MoE) simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.mm_gc import config_registry as _mm_gc_configs  # noqa: E402


def _mm_gc_simulation_config(
    factory: Callable[[], TrainerConfig],
    *,
    output_name: str,
) -> SimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def mm_gc_smoketest() -> SimulationTrainerConfig:
    return _mm_gc_simulation_config(
        _mm_gc_configs.mm_gc_smoketest,
        output_name="mm_gc_smoketest",
    )


def mm_gc_baseline() -> SimulationTrainerConfig:
    return _mm_gc_simulation_config(
        _mm_gc_configs.mm_gc_baseline,
        output_name="mm_gc_baseline",
    )


def mm_gc_baseline_mxfp8() -> SimulationTrainerConfig:
    return _mm_gc_simulation_config(
        _mm_gc_configs.mm_gc_baseline_mxfp8,
        output_name="mm_gc_baseline_mxfp8",
    )
