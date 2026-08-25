# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configs for the GLM-5.2 TTNS model."""

from dataclasses import dataclass, field
import dataclasses

from torchtitan_npu.config.configs import TrainerConfig as NpuTrainerConfig
from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.models.deepseek_v32.config_registry import (
    deepseek_v32_671b_4layers_debug,
    deepseek_v32_smoketest,
    deepseek_v32_tp_smoketest,
)

from . import model_registry
from .config_overrides import (
    GLM52ModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    model_overrides: GLM52ModelOverrides = field(default_factory=GLM52ModelOverrides)

    def __post_init__(self) -> None:
        baseline_mtp = self.model_spec.model.num_mtp_modules
        override_mtp = self.model_overrides.num_mtp_modules
        training_mtp = self.training.num_mtp_modules
        if override_mtp != baseline_mtp:
            if training_mtp not in {baseline_mtp, override_mtp}:
                raise ValueError("model_overrides.num_mtp_modules conflicts with training.num_mtp_modules")
            if training_mtp == baseline_mtp:
                self.training.num_mtp_modules = override_mtp
        self.model_spec = apply_model_overrides(self.model_spec, self.model_overrides)


def _model_spec_with_overrides(flavor: str):
    return build_model_spec_with_overrides(model_registry(flavor))


def _default_converters(*, enable_fused_moe: bool = True) -> list:
    converters = [
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]
    if enable_fused_moe:
        converters.extend(
            [
                get_model_converter_config("npu_moe_dispatch"),
                get_model_converter_config("npu_gmm"),
            ]
        )
    return converters


def _from_base_config(base_factory, model_spec, model_overrides):
    base_config = base_factory()
    values = {
        field.name: getattr(base_config, field.name)
        for field in dataclasses.fields(base_config)
        if field.name != "model_overrides"
    }
    values["model_spec"] = model_spec
    values["training"] = dataclasses.replace(
        base_config.training,
        num_mtp_modules=model_overrides.num_mtp_modules,
    )
    # Keep the GLM defaults visible in --print-config while allowing the
    # base DSV3.2 factory to supply the repository's standard data/checkpoint
    # and optimizer settings.
    return TrainerConfig(**values, model_overrides=model_overrides)


def glm5_2_smoketest() -> TrainerConfig:
    model_spec, model_overrides = _model_spec_with_overrides("smoketest")
    return _from_base_config(
        deepseek_v32_smoketest,
        model_spec,
        model_overrides,
    )


def glm5_2_78layers_1mtp() -> TrainerConfig:
    model_spec, model_overrides = _model_spec_with_overrides("78layers_1mtp")
    return _from_base_config(
        deepseek_v32_671b_4layers_debug,
        model_spec,
        model_overrides,
    )


def glm5_2_tp_smoketest() -> TrainerConfig:
    """Two-rank TP recipe using the ATen MoE path, matching DSV3.2."""
    model_spec, model_overrides = _model_spec_with_overrides("smoketest")
    return _from_base_config(
        deepseek_v32_tp_smoketest,
        model_spec,
        model_overrides,
    )


__all__ = [
    "TrainerConfig",
    "glm5_2_smoketest",
    "glm5_2_tp_smoketest",
    "glm5_2_78layers_1mtp",
]
