# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-specific TorchAO-NPU recipe declarations."""

from __future__ import annotations

from torchtitan.models.common.linear import Linear

from ..param_swaps import block_fp8_param_swap
from ..recipes import FilterSpec, ParamSwapSpec, RecipeContext


class Qwen35FilterKeys:
    DENSE_MODEL = "dense_model"


# Linear projections used by Qwen3.5 dense-model recipes. Qwen3.5 MoE modules,
# including routed and shared experts, are intentionally outside this scope.
_DENSE_MODEL_PROJECTIONS = (
    ".attention.wq",
    ".attention.wk",
    ".attention.wv",
    ".attention.wo",
    ".delta_net.in_proj_q",
    ".delta_net.in_proj_k",
    ".delta_net.in_proj_v",
    ".delta_net.in_proj_z",
    ".delta_net.in_proj_a",
    ".delta_net.in_proj_b",
    ".delta_net.out_proj",
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3",
)

QWEN35_FILTER_SPECS: dict[str, FilterSpec] = {
    Qwen35FilterKeys.DENSE_MODEL: {"suffixes": _DENSE_MODEL_PROJECTIONS},
}

QWEN35_PREQUANTIZE_DEFAULT_FQNS: dict[str, tuple[str, ...]] = {
    Qwen35FilterKeys.DENSE_MODEL: _DENSE_MODEL_PROJECTIONS,
}

UNIFIED_NPU_QUANTIZED_MODULES = (Linear.Config,)
SPECIFIC_NPU_QUANTIZED_MODULES: dict[type, type] = {}


def _w4a8_mxfp_dense_model(filters: dict, ctx: RecipeContext) -> ParamSwapSpec:
    return ParamSwapSpec(
        swap_config=block_fp8_param_swap(
            enable_mxfp4_qat=True,
            dst_type_max=ctx.dst_type_max,
            enable_fsdp_prequantize=ctx.enable_fsdp_prequantize,
            fsdp_prequantize_dtype="fp8",
        ),
        filter=filters[Qwen35FilterKeys.DENSE_MODEL],
        scope=Qwen35FilterKeys.DENSE_MODEL,
    )


def _w8a8_mxfp8_dense_model(filters: dict, ctx: RecipeContext) -> ParamSwapSpec:
    return ParamSwapSpec(
        swap_config=block_fp8_param_swap(
            enable_fsdp_prequantize=ctx.enable_fsdp_prequantize,
            fsdp_prequantize_dtype="fp8",
        ),
        filter=filters[Qwen35FilterKeys.DENSE_MODEL],
        scope=Qwen35FilterKeys.DENSE_MODEL,
    )


QWEN35_RECIPE_MATRIX = {
    "W4A8_MXFP": lambda filters, ctx: [_w4a8_mxfp_dense_model(filters, ctx)],
    "W8A8_MXFP8": lambda filters, ctx: [_w8a8_mxfp8_dense_model(filters, ctx)],
}
