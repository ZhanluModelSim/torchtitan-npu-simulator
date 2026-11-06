# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global quantization config registry (pure declaration layer).

Configures each model family with pure data: recipe matrix, filter specs,
prequantize defaults, target/skip types and overrides.  ``ModelType`` is the
single source of truth for model identity.  This module never depends on the
component execution layer; which components a model owns is resolved by the
engine from the ``components`` registry at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ..filters import ConfigFilterFn, match_config_fqn_suffix
from . import deepseek_v4 as _deepseek_v4
from . import deepseek_v4_1 as _deepseek_v4_1

if TYPE_CHECKING:
    from ..recipes import FilterSpec


class ModelType(StrEnum):
    """Canonical model-family identity (single source of truth).

    Used for the config registry keys, the model-spec name maps, the components'
    ``models`` declaration and ``RecipeContext.model_type`` so the model token is
    never duplicated as a hand-written string.  Declared here (before the family
    modules below) so neither the family config modules nor ``components`` hit an
    import cycle when they import it back from this package.
    """

    V4 = "v4"
    V4_1 = "v41"


class ModelQuantConfig:
    """One model's quantization declaration (pure data; no component objects)."""

    def __init__(
        self,
        *,
        recipe_matrix: dict,
        filters: dict[str, ConfigFilterFn],
        prequantize_default_fqns: dict[str, tuple[str, ...]],
        unified_npu_quant_modules: tuple[type, ...] | None = None,
        specific_npu_quant_modules: dict[type, type] | None = None,
    ) -> None:
        self.recipe_matrix = recipe_matrix
        self.filters = filters
        self.prequantize_default_fqns = prequantize_default_fqns
        self.unified_npu_quant_modules = unified_npu_quant_modules
        self.specific_npu_quant_modules = specific_npu_quant_modules


def _build_filters(
    filter_specs: dict[str, FilterSpec],
    extra_filters: dict[str, ConfigFilterFn] | None = None,
) -> dict[str, ConfigFilterFn]:
    out: dict[str, ConfigFilterFn] = dict(extra_filters or {})
    for name, spec in filter_specs.items():
        out[name] = match_config_fqn_suffix(*spec["suffixes"])
    return out


# Built & exposed (same dict objects the engine + facade/tests share).
_DSV4_CONFIG_FILTERS = _build_filters(_deepseek_v4.V4_FILTER_SPECS)
_DSV41_CONFIG_FILTERS = _build_filters(_deepseek_v4_1.V41_FILTER_SPECS, extra_filters=_deepseek_v4_1.V41_EXTRA_FILTERS)


_MODEL_REGISTRY: dict[ModelType, ModelQuantConfig] = {
    ModelType.V4: ModelQuantConfig(
        recipe_matrix=_deepseek_v4.DEEPSEEK_V4_RECIPE_MATRIX,
        filters=_DSV4_CONFIG_FILTERS,
        prequantize_default_fqns=_deepseek_v4.V4_PREQUANTIZE_DEFAULT_FQNS,
        unified_npu_quant_modules=_deepseek_v4.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=_deepseek_v4.SPECIFIC_NPU_QUANTIZED_MODULES,
    ),
    ModelType.V4_1: ModelQuantConfig(
        recipe_matrix=_deepseek_v4_1.DEEPSEEK_V41_RECIPE_MATRIX,
        filters=_DSV41_CONFIG_FILTERS,
        prequantize_default_fqns=_deepseek_v4_1.V41_PREQUANTIZE_DEFAULT_FQNS,
        unified_npu_quant_modules=_deepseek_v4_1.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=_deepseek_v4_1.SPECIFIC_NPU_QUANTIZED_MODULES,
    ),
}

_NAME_TO_TYPE: dict[str, ModelType] = {
    "deepseek_v4": ModelType.V4,
    "deepseek_v4_1": ModelType.V4_1,
}


def get_model_config(model_type: ModelType | str) -> ModelQuantConfig:
    try:
        kind = model_type if isinstance(model_type, ModelType) else ModelType(model_type)
        return _MODEL_REGISTRY[kind]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported DeepSeek model type {model_type!r}; expected 'v4' or 'v41'") from exc


def _model_type_for_config(model_config) -> ModelType:
    for base in type(model_config).__mro__:
        if base.__module__.startswith("torchtitan_npu.models.deepseek_v4_1."):
            return ModelType.V4_1
        if base.__module__.startswith("torchtitan_npu.models.deepseek_v4."):
            return ModelType.V4
    raise ValueError(
        f"TorchAO-NPU quantization supports DeepSeek V4 and V4.1 model configs, got {type(model_config).__qualname__!r}"
    )
