# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU quant inference: generic fixed pipeline template + converters.

Owns the shared recipe contract types (``RecipeContext`` / ``ParamSwapSpec`` /
``FilterSpec``), the dynamic quantized-module class factory, and the converter
classes (``NpuQuantizeConverter`` / ``KVNormFakeQuantConverter`` /
``_check_prequantize_policy``) so callers and the family config modules import
everything from here.  ``configs`` is imported lazily in the methods below to
avoid an import cycle (configs -> deepseek/components -> recipes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict, cast

import tyro
from torchao.core.config import AOBaseConfig  # noqa: TC002
from torchao.quantization.quant_api import quantize_
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.config import derive
from torchtitan.tools.logging import logger

from .filters import ConfigFilterFn, _match_fqn_suffix, _replace_config
from .param_swaps import _is_prequantizing_block_mx, _param_swap_without_prequantize

if TYPE_CHECKING:
    from torchao_npu.configs import ParamSwapConfig
    from torchtitan.protocols.module import Module

    from torchtitan_npu.config.configs import KVNormQuantizationConfig, LIQuantization

    from .configs import ModelType


@dataclass(frozen=True)
class RecipeContext:
    """Typed, strongly-checked recipe/extension inputs (no bare ``**kwargs``)."""

    model_type: ModelType
    model_compile_enabled: bool
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"]
    enable_mxfp4_qat: bool = False
    dst_type_max: float = 0.0
    enable_fsdp_prequantize: bool = False
    enable_hif8_save_quant_codes: bool = False
    enable_sparse_attention_quantization: bool = False
    li_quantization: LIQuantization | None = None
    kv_norm_quantization: object | None = None
    unified_npu_quant_modules: Annotated[tuple[type, ...] | None, tyro.conf.Suppress] = None
    specific_npu_quant_modules: dict[type, type] | None = None
    model_config: object | None = None
    override_config: object | None = None


@dataclass(frozen=True)
class ParamSwapSpec:
    """One param-swap specification: concrete swap config + filter (+ optional scope)."""

    swap_config: ParamSwapConfig
    filter: ConfigFilterFn
    scope: str | None = None


class FilterSpec(TypedDict):
    """How one config-tree scope's filter is described declaratively."""

    suffixes: tuple[str, ...]


_SUPPORTED_RECIPES = (
    "all_mxfp8",
    "all_hif8",
    "mix",
    "all_block_fp8",
    "W4A8_MXFP",
    "W8A8_MXFP8",
)


_npu_quantized_module_cache: dict[type[Module], type[Module]] = {}


def _get_npu_quantized_module_cls(parent_cls: type[Module]) -> type[Module]:
    """Create a quantized subclass while preserving host behavior.

    The generated class inherits the concrete config owner so custom forward
    methods and config fields remain intact. Standard GroupedExperts and
    AscGroupedExperts use the dedicated subclass selected by NpuQuantizeConverter.
    """

    if parent_cls in _npu_quantized_module_cache:
        return _npu_quantized_module_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class NpuQuantizedModule(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc, valid-type]
            # TorchAO configs contain Python objects and are supplied by the
            # recipe, not by Tyro CLI parsing or config serialization.
            _torchao_npu_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            if config._torchao_npu_config is None:
                raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
            quantize_(
                self,
                config._torchao_npu_config,
                filter_fn=lambda candidate, _fqn: candidate is self,
            )

    NpuQuantizedModule.__name__ = f"NpuQuantized{parent_cls.__name__}"
    NpuQuantizedModule.__qualname__ = f"NpuQuantized{parent_cls.__name__}"
    _npu_quantized_module_cache[parent_cls] = NpuQuantizedModule
    return NpuQuantizedModule


def _quantization_converter(
    base_config: AOBaseConfig | None,
    filter_fn: ConfigFilterFn | None = None,
    *,
    model_compile_enabled: bool,
    replacement_config_type: type | None = None,
    replacement_kwargs: dict[str, object] | None = None,
    prequantize_fqns: tuple[str, ...] | None = None,
    unified_npu_quant_modules: tuple[type, ...] | None = None,
    specific_npu_quant_modules: dict[type, type] | None = None,
    recurse: bool = False,
) -> NpuQuantizeConverter.Config:
    return NpuQuantizeConverter.Config(
        base_config=base_config,
        filter_fn=filter_fn,
        model_compile_enabled=model_compile_enabled,
        replacement_config_type=replacement_config_type,
        replacement_kwargs=replacement_kwargs or {},
        prequantize_fqns=prequantize_fqns,
        unified_npu_quant_modules=unified_npu_quant_modules,
        specific_npu_quant_modules=specific_npu_quant_modules,
        recurse=recurse,
    )


def _resolve_prequantize_fqns(
    user_fqns: list[str] | None,
    *,
    enabled: bool,
    prequantize_default_fqns: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]] | None:
    if not enabled:
        return None
    if user_fqns is None:
        logger.info(
            "fsdp_prequantize whitelist = recipe default (%s)",
            {scope: list(fqns) for scope, fqns in prequantize_default_fqns.items()},
        )
        return dict(prequantize_default_fqns)
    logger.info("fsdp_prequantize whitelist = user (%d patterns)", len(user_fqns))
    resolved = tuple(user_fqns)
    return dict.fromkeys(prequantize_default_fqns, resolved)


class NpuQuantizeConverter(QuantizationConverter):
    """Replace selected quantizable module or metadata config nodes."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        base_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None
        filter_fn: Annotated[ConfigFilterFn | None, tyro.conf.Suppress] = None
        replacement_config_type: Annotated[type | None, tyro.conf.Suppress] = None
        replacement_kwargs: Annotated[dict[str, object], tyro.conf.Suppress] = field(default_factory=dict)
        require_match: bool = True
        unified_npu_quant_modules: Annotated[tuple[type, ...] | None, tyro.conf.Suppress] = None
        """Config-node types this converter may unify (param-swap).
        An error will be raised if this parameter is not provided.".
        """
        specific_npu_quant_modules: Annotated[dict[type, type] | None, tyro.conf.Suppress] = None
        """Per-model map: config-node type -> dedicated quantized module class.

        Overrides the generic unified factory for specific nodes (e.g. routed
        experts), which otherwise get a dynamically generated subclass.
        """
        recurse: Annotated[bool, tyro.conf.Suppress] = False
        """Descend into matched configs too; a FQN-filtered replacement of one node needs it."""
        prequantize_fqns: Annotated[tuple[str, ...] | None, tyro.conf.Suppress] = None
        """Whitelist of FQN suffix patterns that keep ``fsdp_prequantize=True``.
        Matched nodes keep the flag, unmatched nodes get an explicit per-node
        ``fsdp_prequantize=False``. ``None`` disables the policy (the base
        config's flag applies to every node). Shard-alignment violations are
        caught at runtime by the fail-fast guard in
        :class:`BlockMXTrainingWeightWrapperTensor`.
        """

        def __post_init__(self) -> None:
            if self.base_config is None and self.replacement_config_type is None:
                raise ValueError("NpuQuantizeConverter.Config requires base_config or replacement_config_type")

    def __init__(self, config: Config) -> None:
        self.config = config
        self.matched_prequantize_fqns: set[str] = set()

    def _apply_prequantize_policy(self, base_config: AOBaseConfig, fqn: str) -> AOBaseConfig:
        if self.config.prequantize_fqns is None:
            return base_config
        weight_config = getattr(base_config, "weight_config", None)
        if not _is_prequantizing_block_mx(weight_config):
            return base_config

        patterns = self.config.prequantize_fqns
        matched = [pattern for pattern in patterns if _match_fqn_suffix(fqn, pattern)]
        if not matched:
            logger.info("[Converter] %s: fsdp_prequantize=disabled (not in whitelist)", fqn)
            return _param_swap_without_prequantize(base_config)

        logger.info("[Converter] %s: fsdp_prequantize=enabled (whitelist %s)", fqn, matched[0])
        self.matched_prequantize_fqns.update(matched)
        return base_config

    def convert(self, model_config):
        converted = 0
        matches = []
        seen_fqns = set()
        unified_npu_quant_modules = self.config.unified_npu_quant_modules
        specific_npu_quant_modules = self.config.specific_npu_quant_modules

        specific_npu_quant_modules_config = (
            _get_configs(specific_npu_quant_modules) if specific_npu_quant_modules else ()
        )
        if unified_npu_quant_modules is None:
            raise ValueError(
                "NpuQuantizeConverter requires unified_npu_quant_modules: "
                "cannot determine config-node types to configure"
            )
        for config_type in unified_npu_quant_modules:
            for match in model_config.traverse(config_type, recurse=self.config.recurse):
                if match[0] not in seen_fqns:
                    matches.append(match)
                    seen_fqns.add(match[0])
        filtered = 0
        idempotent_skipped = 0
        for fqn, config, parent, attr in matches:
            if self.config.filter_fn is not None and not self.config.filter_fn(config, fqn):
                filtered += 1
                continue
            if self.config.replacement_config_type is not None:
                replacement = derive(config, self.config.replacement_config_type, **self.config.replacement_kwargs)
                replacement_name = self.config.replacement_config_type.__qualname__
            else:
                if self.config.base_config is None:
                    raise ValueError(
                        f"NpuQuantizeConverter has no base_config and no replacement_config_type at {fqn!r}"
                    )
                parent_cls = cast("type[Module] | None", type(config)._owner)
                if parent_cls is None:
                    raise TypeError(f"Config at {fqn!r} has no owning module class")
                # Config-node types that are already the result of quantization (either a
                # generated class or a model-declared quantized-class override) -- converting
                # them again is a no-op, so we skip them for idempotence.
                if (
                    specific_npu_quant_modules_config and isinstance(config, specific_npu_quant_modules_config)
                ) or parent_cls in _npu_quantized_module_cache.values():
                    idempotent_skipped += 1
                    continue
                quantized_cls = (specific_npu_quant_modules or {}).get(type(config))
                if quantized_cls is None:
                    quantized_cls = _get_npu_quantized_module_cls(parent_cls)
                assert hasattr(quantized_cls, "Config"), f"{quantized_cls} missing Config attribute"
                node_base_config = self._apply_prequantize_policy(self.config.base_config, fqn)
                replacement = derive(
                    config,
                    quantized_cls.Config,
                    _torchao_npu_config=node_base_config,
                )
                replacement_name = f"{quantized_cls.__qualname__}.Config"
            model_config = _replace_config(model_config, parent, attr, replacement)
            converted += 1
            logger.info(
                "[Converter] %s.%s: model_spec.model.%s %s -> %s",
                type(self).__module__,
                type(self).__qualname__,
                fqn,
                type(config).__qualname__,
                replacement_name,
            )

        if converted == 0 and self.config.require_match:
            raise ValueError("NpuQuantizeConverter did not match any config nodes")
        logger.info(
            "Converted %d config node(s) for torchao-npu%s",
            converted,
            f" ({filtered} filtered, {idempotent_skipped} already-quantized skipped)"
            if filtered or idempotent_skipped
            else "",
        )
        return model_config


def _get_configs(modules_dict):
    configs = set()
    for cls in modules_dict.values():
        if not hasattr(cls, "Config"):
            raise ValueError(f"Module '{cls.__name__}' configuration error, need to configure 'Config' subclass.")
        configs.add(cls.Config)
    return tuple(configs)


def _recipe_converters(
    recipe: str,
    *,
    model_type: ModelType,
    model_config=None,
    override_config=None,
    enable_sparse_attention_quantization: bool = False,
    enable_mxfp4_qat: bool,
    dst_type_max: float,
    enable_fsdp_prequantize: bool,
    model_compile_enabled: bool,
    li_quantization: LIQuantization | None = None,
    kv_norm_quantization: KVNormQuantizationConfig | None = None,
    fsdp_prequantize_fqns: list[str] | None = None,
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = "fp8",
    enable_hif8_save_quant_codes: bool = False,
) -> list[QuantizationConverter.Config]:
    from .components import component_names_for_model, instantiate_component_for_model_and_name
    from .configs import get_model_config

    model_quant_config = get_model_config(model_type)
    filters = model_quant_config.filters
    if filters is None:
        raise ValueError(f"model type {model_type!r} found, but filters config is missing")
    prequantize_fqns_by_scope = _resolve_prequantize_fqns(
        fsdp_prequantize_fqns,
        enabled=enable_fsdp_prequantize,
        prequantize_default_fqns=model_quant_config.prequantize_default_fqns,
    )
    ctx = RecipeContext(
        model_type=model_type,
        model_compile_enabled=model_compile_enabled,
        enable_mxfp4_qat=enable_mxfp4_qat,
        dst_type_max=dst_type_max,
        enable_fsdp_prequantize=enable_fsdp_prequantize,
        enable_hif8_save_quant_codes=enable_hif8_save_quant_codes,
        enable_sparse_attention_quantization=enable_sparse_attention_quantization,
        li_quantization=li_quantization,
        fsdp_prequantize_dtype=fsdp_prequantize_dtype,
        kv_norm_quantization=kv_norm_quantization,
        unified_npu_quant_modules=model_quant_config.unified_npu_quant_modules,
        specific_npu_quant_modules=model_quant_config.specific_npu_quant_modules,
        model_config=model_config,
        override_config=override_config,
    )

    recipe_spec_func = model_quant_config.recipe_matrix.get(recipe)
    if recipe_spec_func is None:
        supported = tuple(model_quant_config.recipe_matrix)
        raise ValueError(f"recipe for model {model_type.value!r} must be one of {supported}, got {recipe!r}")
    if recipe in ("W4A8_MXFP", "W8A8_MXFP8") and enable_fsdp_prequantize and fsdp_prequantize_dtype == "mxfp4":
        raise ValueError(
            f"Qwen3.5 recipe {recipe} supports only FP8 FSDP-prequantized dense weights; "
            "remove --extension.quantization.fsdp-prequantize-dtype mxfp4 or set it to fp8"
        )
    param_swap_specs = recipe_spec_func(filters, ctx)

    def _scope_fqns(param_swap_spec: ParamSwapSpec) -> tuple[str, ...] | None:
        if param_swap_spec.scope is None or prequantize_fqns_by_scope is None:
            return None
        if not _is_prequantizing_block_mx(param_swap_spec.swap_config.weight_config):
            return None
        return prequantize_fqns_by_scope.get(param_swap_spec.scope)

    converters: list[QuantizationConverter.Config] = []
    for param_swap_spec in param_swap_specs:
        converters.append(
            _quantization_converter(
                param_swap_spec.swap_config,
                param_swap_spec.filter,
                model_compile_enabled=model_compile_enabled,
                prequantize_fqns=_scope_fqns(param_swap_spec),
                unified_npu_quant_modules=model_quant_config.unified_npu_quant_modules,
                specific_npu_quant_modules=model_quant_config.specific_npu_quant_modules,
            )
        )

    triggered_components = []
    for component_name in component_names_for_model(model_type):
        component = instantiate_component_for_model_and_name(model_type, component_name)
        if component.should_trigger(filters, ctx):
            triggered_components.append(component_name)
            converters.extend(component.build_converters(filters, ctx))

    logger.info(
        "recipe=%s model=%s: built %d converter(s); components triggered=%s",
        recipe,
        model_type.value,
        len(converters),
        triggered_components or "none",
    )
    return converters
