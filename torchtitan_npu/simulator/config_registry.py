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
# DeepSeek V3.2 simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.deepseek_v32 import (  # noqa: E402
    config_registry as _deepseek_v32_configs,
)
from torchtitan_npu.models.deepseek_v32.config_overrides import (  # noqa: E402
    DeepSeekV32ModelOverrides,
    apply_model_overrides as apply_deepseek_v32_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class DeepSeekV32SimulationTrainerConfig(SimulationTrainerConfig):
    """DeepSeek V3.2 simulator config with stable model overrides."""

    model_overrides: DeepSeekV32ModelOverrides = dataclasses.field(
        default_factory=DeepSeekV32ModelOverrides
    )

    def __post_init__(self) -> None:
        baseline_mtp = self.model_spec.model.num_mtp_modules
        override_mtp = self.model_overrides.num_mtp_modules
        training_mtp = self.training.num_mtp_modules
        if override_mtp != baseline_mtp:
            if training_mtp not in {baseline_mtp, override_mtp}:
                raise ValueError(
                    "model_overrides.num_mtp_modules conflicts with "
                    "training.num_mtp_modules"
                )
            if training_mtp == baseline_mtp:
                self.training.num_mtp_modules = override_mtp
        self.model_spec = apply_deepseek_v32_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _deepseek_v32_simulation_config(
    factory: Callable[[], _deepseek_v32_configs.TrainerConfig],
    *,
    output_name: str,
) -> DeepSeekV32SimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
    }
    base_fields["compile"] = dataclasses.replace(
        base_config.compile,
        enable=False,
    )
    return DeepSeekV32SimulationTrainerConfig(
        **base_fields,
        simulation=SimulationConfig(
            output_dir=f"./simulator_output/{output_name}"
        ),
    )


def deepseek_v32_smoketest() -> DeepSeekV32SimulationTrainerConfig:
    return _deepseek_v32_simulation_config(
        _deepseek_v32_configs.deepseek_v32_smoketest,
        output_name="deepseek_v32_smoketest",
    )


def deepseek_v32_tp_smoketest() -> DeepSeekV32SimulationTrainerConfig:
    return _deepseek_v32_simulation_config(
        _deepseek_v32_configs.deepseek_v32_tp_smoketest,
        output_name="deepseek_v32_tp_smoketest",
    )


def deepseek_v32_671b_4layers_debug() -> DeepSeekV32SimulationTrainerConfig:
    return _deepseek_v32_simulation_config(
        _deepseek_v32_configs.deepseek_v32_671b_4layers_debug,
        output_name="deepseek_v32_671b_4layers_debug",
    )


def deepseek_v32_671b_61layers_4k_128die(
) -> DeepSeekV32SimulationTrainerConfig:
    return _deepseek_v32_simulation_config(
        _deepseek_v32_configs.deepseek_v32_671b_61layers_4k_128die,
        output_name="deepseek_v32_671b_61layers_4k_128die",
    )


def deepseek_v32_671b_61layers_32k_128die(
) -> DeepSeekV32SimulationTrainerConfig:
    return _deepseek_v32_simulation_config(
        _deepseek_v32_configs.deepseek_v32_671b_61layers_32k_128die,
        output_name="deepseek_v32_671b_61layers_32k_128die",
    )


# ---------------------------------------------------------------------------
# GLM-5.2 simulator configs
# ---------------------------------------------------------------------------

from torchtitan_npu.models.glm5_2 import (  # noqa: E402
    config_registry as _glm5_2_configs,
)
from torchtitan_npu.models.glm5_2.config_overrides import (  # noqa: E402
    GLM52ModelOverrides,
    apply_model_overrides as apply_glm5_2_model_overrides,
)


@dataclasses.dataclass(kw_only=True, slots=True)
class GLM52SimulationTrainerConfig(SimulationTrainerConfig):
    """GLM-5.2 simulator config with stable model overrides."""

    model_overrides: GLM52ModelOverrides = dataclasses.field(
        default_factory=GLM52ModelOverrides
    )

    def __post_init__(self) -> None:
        baseline_mtp = self.model_spec.model.num_mtp_modules
        override_mtp = self.model_overrides.num_mtp_modules
        training_mtp = self.training.num_mtp_modules
        if override_mtp != baseline_mtp:
            if training_mtp not in {baseline_mtp, override_mtp}:
                raise ValueError(
                    "model_overrides.num_mtp_modules conflicts with "
                    "training.num_mtp_modules"
                )
            if training_mtp == baseline_mtp:
                self.training.num_mtp_modules = override_mtp
        self.model_spec = apply_glm5_2_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _glm5_2_simulation_config(
    factory: Callable[[], _glm5_2_configs.TrainerConfig],
    *,
    output_name: str,
) -> GLM52SimulationTrainerConfig:
    base_config = factory()
    base_fields = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
        if field.name != "model_overrides"
    }
    base_fields["compile"] = dataclasses.replace(
        base_config.compile,
        enable=False,
    )
    return GLM52SimulationTrainerConfig(
        **base_fields,
        model_overrides=base_config.model_overrides,
        simulation=SimulationConfig(output_dir=f"./simulator_output/{output_name}"),
    )


def glm5_2_smoketest() -> GLM52SimulationTrainerConfig:
    return _glm5_2_simulation_config(
        _glm5_2_configs.glm5_2_smoketest,
        output_name="glm5_2_smoketest",
    )


def glm5_2_tp_smoketest() -> GLM52SimulationTrainerConfig:
    return _glm5_2_simulation_config(
        _glm5_2_configs.glm5_2_tp_smoketest,
        output_name="glm5_2_tp_smoketest",
    )


def glm5_2_78layers_1mtp() -> GLM52SimulationTrainerConfig:
    return _glm5_2_simulation_config(
        _glm5_2_configs.glm5_2_78layers_1mtp,
        output_name="glm5_2_78layers_1mtp",
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
