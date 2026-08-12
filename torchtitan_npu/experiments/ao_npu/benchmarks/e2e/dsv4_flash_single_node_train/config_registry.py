# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2E benchmark config using RecipeQuantizeConverter.

Recipe parameters are controlled via environment variables:
    RECIPE                    - quantization recipe: all_mxfp8, mix, all_block_fp8 (default mix)
    ENABLE_QUANTIZED_TRAINING - "true" enables quantize converters; "false" uses base precision (default true)
    ENABLE_MXFP4_QAT          - "true" enables mxfp4_fake_quantize for routed experts (default true)
    DST_TYPE_MAX              - target dtype max for MXFP4 QAT weight fake-quantize
    BASE_MODEL                - base config: "flash" or "1b" (default flash)

The `BASE_MODEL` env var selects the base config (flash or 1b).
"""

import os
from dataclasses import replace

from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import TrainerConfig
from torchtitan_npu.experiments.ao_npu.benchmarks.e2e.dsv4_flash_single_node_train.recipe_converter import (
    RecipeQuantizeConverter,
)
from torchtitan_npu.models.deepseek_v4.config_registry import (
    debug_deepseek_v4_flash_single_node,
    debug_deepseek_v4_single_node_1b,
)

_BASE_CONFIGS = {
    "flash": debug_deepseek_v4_flash_single_node,
    "1b": debug_deepseek_v4_single_node_1b,
}


def _get_env_bool(name: str, default: str) -> bool:
    val = os.environ.get(name, default).lower()
    if val not in ("true", "false"):
        raise ValueError(f"Environment variable {name} must be 'true' or 'false', got '{val!r}'")
    return val == "true"


def debug_deepseek_v4_flash_single_node_train() -> TrainerConfig:
    base_name = os.environ.get("BASE_MODEL", "flash")
    if base_name not in _BASE_CONFIGS:
        raise ValueError(f"BASE_MODEL must be one of {_BASE_CONFIGS.keys()}, got '{base_name}'")
    base = _BASE_CONFIGS[base_name]()

    recipe = os.environ.get("RECIPE", "mix")
    dst_type_max = float(os.environ.get("DST_TYPE_MAX", "0.0"))
    enable_quantized_training = _get_env_bool("ENABLE_QUANTIZED_TRAINING", "true")
    enable_mxfp4_qat = _get_env_bool("ENABLE_MXFP4_QAT", "true")

    return replace(
        base,
        model_converters=ModelConvertersContainer.Config(
            converters=base.model_converters.converters  # noqa: RUF005
            + [
                RecipeQuantizeConverter.Config(
                    recipe=recipe,
                    dst_type_max=dst_type_max,
                    enable_quantized_training=enable_quantized_training,
                    enable_mxfp4_qat=enable_mxfp4_qat,
                )
            ]
        ),
    )
