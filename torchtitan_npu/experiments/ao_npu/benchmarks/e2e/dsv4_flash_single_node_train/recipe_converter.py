# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipe-level quantization converter.

Wraps multiple ``NpuQuantizeConverter`` invocations behind a single Config
with recipe-level parameters. The recipe, dst_type_max, and feature flags
are all set from Config fields.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
from torchao.quantization.quant_api import quantize_
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.quantization.module_utils import (
    capture_module_attrs,
    inject_module_protocol,
    verify_module_protocol,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

from torchtitan_npu.experiments.ao_npu.torchao_npu.configs import ParamSwapConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.interfaces.torchtitan import (
    is_attention,
    is_routed_expert,
    is_shared_expert,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.filters import (
    any_filter,
    match_fqn_suffix,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)

# All attention/shared expert/e_proj/h_proj modules
_mx_filter = any_filter(
    is_attention,
    is_shared_expert,
    match_fqn_suffix(".e_proj", ".h_proj"),
)

_SUPPORTED_RECIPES = ("all_mxfp8", "mix", "all_block_fp8")


class RecipeQuantizeConverter(QuantizationConverter):
    """Single converter that applies a full quantization recipe.

    Instead of configuring multiple ``NpuQuantizeConverter`` instances in the
    config_registry, this converter encapsulates the recipe logic internally.

    See :class:`Config` for the accepted arguments.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        _quantization_type: ClassVar[str] = "recipe_quantize_converter"

        recipe: str = "mix"
        """One of ``all_mxfp8``, ``mix``, ``all_block_fp8``."""

        dst_type_max: float = 7.0
        """Target dtype max for MXFP4 QAT weight fake-quantize (0.0 = auto-infer). Not used for FP8 activations."""

        enable_quantized_training: bool = True
        """If False, skips quantization entirely (returns base model unchanged)."""

        enable_mxfp4_qat: bool = True
        """If True and recipe=mix, routed experts use MXFP4 fake-quantize weights.
        If False, routed experts use plain BlockQuantizeConfig."""

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        if config.recipe not in _SUPPORTED_RECIPES:
            raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got '{config.recipe}'")
        self.recipe = config.recipe
        self.dst_type_max = config.dst_type_max
        self.enable_quantized_training = config.enable_quantized_training
        self.enable_mxfp4_qat = config.enable_mxfp4_qat
        logger.info(
            f"RecipeQuantizeConverter: recipe={self.recipe}, "
            f"enable_quantized_training={self.enable_quantized_training}, "
            f"enable_mxfp4_qat={self.enable_mxfp4_qat}"
        )

    def convert(self, model: nn.Module):
        if not self.enable_quantized_training:
            logger.info("RecipeQuantizeConverter: skipping (enable_quantized_training=False)")
            return

        verify_module_protocol(model, nn.Linear, Linear)
        saved_attrs = capture_module_attrs(model, ["_init_mean", "_init_std"], nn_module_cls=nn.Linear)

        mx = MXQuantizeConfig()

        if self.recipe == "all_mxfp8":
            self._apply_recipe_all_mxfp8(model, mx)
        elif self.recipe == "mix":
            self._apply_recipe_mix(model, mx)
        elif self.recipe == "all_block_fp8":
            self._apply_recipe_all_block_fp8(model, mx)

        inject_module_protocol(model, Linear, saved_attrs)
        verify_module_protocol(model, nn.Linear, Linear)
        logger.info(f"Applied recipe quantize wrapping ({self.recipe})")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        pass

    # ------------------------------------------------------------------
    # Per-recipe apply functions (private)
    # ------------------------------------------------------------------

    def _apply_recipe_all_mxfp8(self, model: nn.Module, mx: MXQuantizeConfig):
        config = ParamSwapConfig(
            weight_config=mx,
            activation_config=mx,
        )
        quantize_(
            model,
            config,
            filter_fn=any_filter(_mx_filter, is_routed_expert),
        )

    def _apply_recipe_mix(self, model: nn.Module, mx: MXQuantizeConfig):
        config_mx = ParamSwapConfig(
            weight_config=mx,
            activation_config=mx,
        )
        quantize_(model, config_mx, filter_fn=_mx_filter)

        if self.enable_mxfp4_qat:
            routed_weight = BlockQuantizeConfig(
                mxfp4_fake_quantize_config=MXQuantizeConfig(
                    elem_dtype=torch.float4_e2m1fn_x2,
                    dst_type_max=self.dst_type_max,
                ),
            )
        else:
            routed_weight = BlockQuantizeConfig()

        config_block = ParamSwapConfig(
            weight_config=routed_weight,
            activation_config=mx,
        )
        quantize_(model, config_block, filter_fn=is_routed_expert)

    def _apply_recipe_all_block_fp8(self, model: nn.Module, mx: MXQuantizeConfig):
        config_block_mx = ParamSwapConfig(
            weight_config=BlockQuantizeConfig(),
            activation_config=mx,
        )
        quantize_(model, config_block_mx, filter_fn=_mx_filter)

        if self.enable_mxfp4_qat:
            routed_weight = BlockQuantizeConfig(
                mxfp4_fake_quantize_config=MXQuantizeConfig(
                    elem_dtype=torch.float4_e2m1fn_x2,
                    dst_type_max=self.dst_type_max,
                ),
            )
        else:
            routed_weight = BlockQuantizeConfig()

        config_block_routed = ParamSwapConfig(
            weight_config=routed_weight,
            activation_config=mx,
        )
        quantize_(model, config_block_routed, filter_fn=is_routed_expert)
