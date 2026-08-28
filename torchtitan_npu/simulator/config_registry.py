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
# MAGI-2-preview simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.magi2_preview import config_registry as _magi2_configs  # noqa: E402
from torchtitan_npu.models.magi2_preview.config_overrides import (  # noqa: E402
    Magi2PreviewModelOverrides,
    apply_model_overrides as apply_magi2_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class Magi2SimulationTrainerConfig(SimulationTrainerConfig):
    """MAGI-2-preview simulator config with stable model CLI overrides."""

    model_overrides: Magi2PreviewModelOverrides = dataclasses.field(
        default_factory=Magi2PreviewModelOverrides
    )

    def __post_init__(self) -> None:
        self.model_spec = apply_magi2_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _magi2_simulation_config(
    factory: Callable[[], TrainerConfig],
    *,
    output_name: str,
) -> Magi2SimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(base_config.compile, enable=False)
    return Magi2SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def magi2_preview_smoketest() -> Magi2SimulationTrainerConfig:
    return _magi2_simulation_config(
        _magi2_configs.magi2_preview_smoketest,
        output_name="magi2_preview_smoketest",
    )
