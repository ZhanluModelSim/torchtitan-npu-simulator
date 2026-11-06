# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 recipe matrix + filter specs + quantized-module config.

Hosts V4's ``UNIFIED_NPU_QUANTIZED_MODULES`` /``SPECIFIC_NPU_QUANTIZED_MODULES``
and the DeepSeek-family shared config-node types
and routed-expert row builder.  V4.1 re-declares its own copies in
:mod:`.deepseek_v4_1` so a V4.1-only divergence is a local edit.
"""

from __future__ import annotations

from torchtitan.models.common.linear import GroupedLinear, Linear

from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer as DeepSeekV4LightningIndexer
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from ..filters import any_config_filter
from ..param_swaps import _hif8_param_swap, block_fp8_param_swap, mxfp8_param_swap
from ..recipes import FilterSpec, ParamSwapSpec, RecipeContext

# Projection owners retain their forward and sharding behavior through the
# generic parameter-swap subclass; the activation override is independent.
SPECIFIC_NPU_QUANTIZED_MODULES: dict[type, type] = {}

# Per-model config-node types that may be param-swapped, and those to skip.
UNIFIED_NPU_QUANTIZED_MODULES = (
    Linear.Config,
    BatchedLinear.Config,
    GroupedLinear.Config,
    DeepSeekV4LightningIndexer.Config,
)

_DENSE = (
    ".attention.wq_a",
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
    ".moe.shared_experts.w1",
    ".moe.shared_experts.w2",
    ".moe.shared_experts.w3",
)
_ROUTED = (".moe.routed_experts.w13", ".moe.routed_experts.w2")


class V4FilterKeys:
    """Canonical DeepSeek-V4 filter names (avoid hand-written string keys)."""

    DENSE = "dense"
    ROUTED_EXPERT = "routed_expert"
    LIGHTNING_INDEXER = "lightning_indexer"
    LIGHTNING_INDEXER_METADATA = "lightning_indexer_metadata"


V4_FILTER_SPECS: dict[str, FilterSpec] = {
    V4FilterKeys.DENSE: {"suffixes": _DENSE},
    V4FilterKeys.ROUTED_EXPERT: {"suffixes": _ROUTED},
    V4FilterKeys.LIGHTNING_INDEXER: {"suffixes": (".attention.lightning_indexer",)},
}

V4_PREQUANTIZE_DEFAULT_FQNS: dict[str, tuple[str, ...]] = {
    V4FilterKeys.DENSE: _DENSE,
    V4FilterKeys.ROUTED_EXPERT: _ROUTED,
}


def _routed_expert_block_fp8(filters: dict, ctx: RecipeContext) -> ParamSwapSpec:
    return ParamSwapSpec(
        swap_config=block_fp8_param_swap(
            enable_mxfp4_qat=ctx.enable_mxfp4_qat,
            dst_type_max=ctx.dst_type_max,
            enable_fsdp_prequantize=ctx.enable_fsdp_prequantize,
            fsdp_prequantize_dtype=ctx.fsdp_prequantize_dtype,
        ),
        filter=filters[V4FilterKeys.ROUTED_EXPERT],
        scope=V4FilterKeys.ROUTED_EXPERT,
    )


DEEPSEEK_V4_RECIPE_MATRIX = {
    "all_mxfp8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=mxfp8_param_swap(),
            filter=any_config_filter(filters[V4FilterKeys.DENSE], filters[V4FilterKeys.ROUTED_EXPERT]),
        ),
    ],
    "all_hif8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=_hif8_param_swap(ctx.enable_hif8_save_quant_codes),
            filter=any_config_filter(filters[V4FilterKeys.DENSE], filters[V4FilterKeys.ROUTED_EXPERT]),
        ),
    ],
    "mix": lambda filters, ctx: [
        ParamSwapSpec(swap_config=mxfp8_param_swap(), filter=filters[V4FilterKeys.DENSE], scope=V4FilterKeys.DENSE),
        _routed_expert_block_fp8(filters, ctx),
    ],
    "all_block_fp8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=block_fp8_param_swap(
                enable_fsdp_prequantize=ctx.enable_fsdp_prequantize,
                fsdp_prequantize_dtype="fp8",
            ),
            filter=filters[V4FilterKeys.DENSE],
            scope=V4FilterKeys.DENSE,
        ),
        _routed_expert_block_fp8(filters, ctx),
    ],
}
