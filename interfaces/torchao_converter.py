# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU converter integration hosted by ``torchtitan-npu``.

TorchTitan builds models from a tree of ``Module.Config`` objects.  Quantization
must therefore be expressed in that tree before the model is built: a converter
replaces selected configs with configs whose modules install the torchao-npu
parameter wrappers in ``__init__``.  This keeps the wrappers in place before
TP/EP/FSDP and optimizer construction without modifying TorchTitan's trainer.

Only this adapter lives in ``torchtitan_npu``.  Quantization configs, parameter
wrappers and NPU kernels continue to come from the separately installed
``torchao_npu`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Protocol, cast

import torch
import tyro
from torchao.core.config import AOBaseConfig  # noqa: TC002
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockQuantizeConfig, MXQuantizeConfig
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.config import derive
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.utils import validate_converter_order
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec
    from torchtitan.protocols.module import Module

    from torchtitan_npu.config.configs import QuantizationExtensionConfig


class ConfigFilterFn(Protocol):
    """Predicate over a TorchTitan model-config node and its config-tree FQN."""

    def __call__(self, config: Module.Config, fqn: str) -> bool: ...


def any_config_filter(*filters: ConfigFilterFn) -> ConfigFilterFn:
    """Combine config-tree filters with logical OR."""

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(filter_fn(config, fqn) for filter_fn in filters)

    return _filter


def match_config_fqn_suffix(*suffixes: str) -> ConfigFilterFn:
    """Match TorchTitan config nodes whose config-tree FQN has a suffix."""

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(fqn.endswith(suffix) for suffix in suffixes)

    return _filter


def _replace_config(model_config, parent: object | None, attr: str | int | None, replacement):
    """Replace one config-tree node, including the uncommon root-node case."""

    if parent is None:
        return replacement
    if isinstance(parent, list):
        assert isinstance(attr, int)
        parent[attr] = replacement
    else:
        assert isinstance(attr, str)
        setattr(parent, attr, replacement)
    return model_config


_npu_quantized_module_cache: dict[type[Module], type[Module]] = {}


def _get_npu_quantized_module_cls(parent_cls: type[Module]) -> type[Module]:
    """Create a parameter-quantized subclass while preserving host behavior.

    Both Linear and GroupedExperts can have model-specific subclasses.  The
    generated class inherits the concrete config owner instead of replacing it
    with a fixed implementation, so custom forward methods and config fields
    (for example DeepSeek-V4's SwiGLU clamp) remain intact.
    """

    if parent_cls in _npu_quantized_module_cache:
        return _npu_quantized_module_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class NpuQuantizedModule(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc, valid-type]
            # ParamSwapConfig contains Python objects and is supplied by the
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


class NpuQuantizeConverter(QuantizationConverter):
    """Quantize selected Linear, BatchedLinear, or GroupedExperts config nodes.

    A recipe may instantiate this converter more than once with different
    parameter-swap policies.  For example, one instance can select attention
    and shared-expert Linear nodes for MX training while another selects routed
    GroupedExperts nodes for Block FP8 training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        base_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None
        filter_fn: Annotated[ConfigFilterFn | None, tyro.conf.Suppress] = None
        require_match: bool = True

        def __post_init__(self) -> None:
            if self.base_config is None:
                raise ValueError(
                    "NpuQuantizeConverter.Config.base_config must be provided "
                    "programmatically; it cannot be set via CLI."
                )

    def __init__(self, config: Config) -> None:
        self.config = config

    def convert(self, model_config):
        assert self.config.base_config is not None
        converted = 0
        matches = list(model_config.traverse(Linear.Config))
        matches.extend(model_config.traverse(BatchedLinear.Config))
        matches.extend(model_config.traverse(GroupedExperts.Config))
        for fqn, config, parent, attr in matches:
            parent_cls = cast("type[Module] | None", type(config)._owner)
            if parent_cls in _npu_quantized_module_cache.values():
                continue
            if self.config.filter_fn is not None and not self.config.filter_fn(config, fqn):
                continue
            if parent_cls is None:
                raise TypeError(f"Config at {fqn!r} has no owning module class")

            quantized_cls = _get_npu_quantized_module_cls(parent_cls)
            replacement = derive(
                config,
                quantized_cls.Config,
                _torchao_npu_config=self.config.base_config,
            )
            model_config = _replace_config(model_config, parent, attr, replacement)
            converted += 1
            logger.info(
                "[Converter] %s.%s: model_spec.model.%s %s -> %s",
                type(self).__module__,
                type(self).__qualname__,
                fqn,
                type(config).__qualname__,
                f"{quantized_cls.__qualname__}.Config",
            )

        if converted == 0 and self.config.require_match:
            raise ValueError(
                "NpuQuantizeConverter did not match any Linear, BatchedLinear, or GroupedExperts config nodes"
            )
        logger.info("Converted %d config node(s) for torchao-npu", converted)
        return model_config


# DeepSeek-V4 config-tree filters for the current model hierarchy.
is_attention = match_config_fqn_suffix(
    ".attention.wq_a",
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
)

is_shared_expert = match_config_fqn_suffix(
    ".moe.shared_experts.w1",
    ".moe.shared_experts.w2",
    ".moe.shared_experts.w3",
)

is_routed_expert = match_config_fqn_suffix(".moe.routed_experts.inner_experts")


_SUPPORTED_RECIPES = ("all_mxfp8", "mix", "all_block_fp8")


def _mxfp8_param_swap() -> ParamSwapConfig:
    mx_config = MXQuantizeConfig()
    return ParamSwapConfig(
        weight_config=mx_config,
        activation_config=mx_config,
    )


def _block_fp8_param_swap(
    *,
    enable_mxfp4_qat: bool = False,
    dst_type_max: float = 0.0,
) -> ParamSwapConfig:
    mxfp4_config = None
    if enable_mxfp4_qat:
        mxfp4_config = MXQuantizeConfig(
            elem_dtype=torch.float4_e2m1fn_x2,
            dst_type_max=dst_type_max,
        )
    return ParamSwapConfig(
        weight_config=BlockQuantizeConfig(
            mxfp4_fake_quantize_config=mxfp4_config,
        ),
        activation_config=MXQuantizeConfig(),
    )


def _quantization_converter(
    base_config: ParamSwapConfig,
    filter_fn: ConfigFilterFn,
    *,
    model_compile_enabled: bool,
) -> NpuQuantizeConverter.Config:
    return NpuQuantizeConverter.Config(
        base_config=base_config,
        filter_fn=filter_fn,
        model_compile_enabled=model_compile_enabled,
    )


def _recipe_converters(
    recipe: str,
    *,
    enable_mxfp4_qat: bool,
    dst_type_max: float,
    model_compile_enabled: bool,
) -> list[NpuQuantizeConverter.Config]:
    dense_filter = any_config_filter(is_attention, is_shared_expert)

    if recipe == "all_mxfp8":
        return [
            _quantization_converter(
                _mxfp8_param_swap(),
                any_config_filter(dense_filter, is_routed_expert),
                model_compile_enabled=model_compile_enabled,
            )
        ]

    routed_config = _block_fp8_param_swap(
        enable_mxfp4_qat=enable_mxfp4_qat,
        dst_type_max=dst_type_max,
    )
    if recipe == "mix":
        dense_config = _mxfp8_param_swap()
    elif recipe == "all_block_fp8":
        dense_config = _block_fp8_param_swap()
    else:
        raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got {recipe!r}")

    return [
        _quantization_converter(
            dense_config,
            dense_filter,
            model_compile_enabled=model_compile_enabled,
        ),
        _quantization_converter(
            routed_config,
            is_routed_expert,
            model_compile_enabled=model_compile_enabled,
        ),
    ]


def apply_quantization_converter(
    model_spec: ModelSpec | None,
    quantization_config: QuantizationExtensionConfig,
    *,
    model_compile_enabled: bool,
) -> ModelSpec:
    """Apply the selected TorchAO-NPU recipe to an existing BF16 model spec.

    This runs during ``TrainerEx`` construction and before base ``Trainer``
    initialization.  The regular model registry therefore remains responsible
    only for constructing the high-precision config tree, while this function
    performs the optional low-precision config replacement.
    """

    if model_spec is None:
        raise ValueError("TorchAO-NPU quantization requires model_spec to be configured")
    if not quantization_config.enable_quantized_training:
        return model_spec

    converters = _recipe_converters(
        quantization_config.recipe,
        enable_mxfp4_qat=quantization_config.enable_mxfp4_qat,
        dst_type_max=quantization_config.dst_type_max,
        model_compile_enabled=model_compile_enabled,
    )
    validate_converter_order(converters)

    model_config = model_spec.model
    for converter_config in converters:
        model_config = converter_config.build().convert(model_config)

    logger.info(
        "Applied TorchAO-NPU recipe=%s, mxfp4_qat=%s, dst_type_max=%s",
        quantization_config.recipe,
        quantization_config.enable_mxfp4_qat,
        quantization_config.dst_type_max,
    )
    return replace(model_spec, model=model_config)


__all__ = [
    "ConfigFilterFn",
    "NpuQuantizeConverter",
    "any_config_filter",
    "apply_quantization_converter",
    "is_attention",
    "is_routed_expert",
    "is_shared_expert",
    "match_config_fqn_suffix",
]
