# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin simulator wrappers around the DeepSeek-V4 training configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from torchtitan_npu.models.deepseek_v4.config_registry import (
    TrainerConfig as DeepSeekV4TrainerConfig,
    _apply_mxfp8_fqns_override,
    deepseek_v4_flash_baseline_bf16 as _deepseek_v4_flash_baseline_bf16,
    deepseek_v4_flash_baseline_mxfp8 as _deepseek_v4_flash_baseline_mxfp8,
    deepseek_v4_pro_20t_baseline_bf16 as _deepseek_v4_pro_20t_baseline_bf16,
    deepseek_v4_pro_20t_baseline_mxfp8 as _deepseek_v4_pro_20t_baseline_mxfp8,
    deepseek_v4_pro_baseline_bf16 as _deepseek_v4_pro_baseline_bf16,
    deepseek_v4_pro_baseline_mxfp8 as _deepseek_v4_pro_baseline_mxfp8,
    deepseek_v4_smoketest as _deepseek_v4_smoketest,
)
from torchtitan_npu.simulator.trainer import SimulationConfig, SimulationTrainerConfig


@dataclasses.dataclass(kw_only=True, slots=True)
class DeepSeekV4SimulationTrainerConfig(SimulationTrainerConfig):
    mxfp8_fqns: list[str] | None = None

    def __post_init__(self) -> None:
        _apply_mxfp8_fqns_override(self.model_converters, self.mxfp8_fqns)


def _simulation_config(
    factory: Callable[[], DeepSeekV4TrainerConfig],
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
        _deepseek_v4_flash_baseline_bf16,
        output_name="deepseek_v4_flash_baseline_bf16",
    )


def deepseek_v4_flash_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_flash_baseline_mxfp8,
        output_name="deepseek_v4_flash_baseline_mxfp8",
    )


def deepseek_v4_pro_baseline_bf16() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_pro_baseline_bf16,
        output_name="deepseek_v4_pro_baseline_bf16",
    )


def deepseek_v4_pro_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_pro_baseline_mxfp8,
        output_name="deepseek_v4_pro_baseline_mxfp8",
    )


def deepseek_v4_pro_20t_baseline_bf16() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_pro_20t_baseline_bf16,
        output_name="deepseek_v4_pro_20t_baseline_bf16",
    )


def deepseek_v4_pro_20t_baseline_mxfp8() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_pro_20t_baseline_mxfp8,
        output_name="deepseek_v4_pro_20t_baseline_mxfp8",
    )


def deepseek_v4_smoketest() -> SimulationTrainerConfig:
    return _simulation_config(
        _deepseek_v4_smoketest,
        output_name="deepseek_v4_smoketest",
    )
