# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 recipe matrix + filter specs + quantized-module config.

Owns its own ``DEEPSEEK_V41_RECIPE_MATRIX`` and its own
``UNIFIED_NPU_QUANTIZED_MODULES`` / ``SPECIFIC_NPU_QUANTIZED_MODULES``
so a V4.1-only divergence is a local edit.
Reuses the DeepSeek-family shared config-node types and the routed-expert row
builder declared in :mod:`.deepseek_v4`.
"""

from __future__ import annotations

from torchtitan.models.common.linear import GroupedLinear, Linear

from torchtitan_npu.models.deepseek_v4_1.attention import (
    CompressedSparseInnerAttention2 as DeepSeekV41CompressedSparseInnerAttention2,
)
from torchtitan_npu.models.deepseek_v4_1.compressor import Compressor as DeepSeekV41Compressor
from torchtitan_npu.models.deepseek_v4_1.indexer import Selector as DeepSeekV41Selector
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from ..filters import _match_fqn_suffix, any_config_filter
from ..param_swaps import _hif8_param_swap, block_fp8_param_swap, mxfp8_param_swap
from ..recipes import FilterSpec, ParamSwapSpec
from .deepseek_v4 import _DENSE, _ROUTED, _routed_expert_block_fp8


def _is_v41_kv_source(config, fqn: str) -> bool:
    return (
        _match_fqn_suffix(fqn, ".attention.compressor")
        and isinstance(config, DeepSeekV41Compressor.Config)
        and config.is_source
    )


class V41FilterKeys:
    """Canonical DeepSeek-V4.1 filter names (avoid hand-written string keys)."""

    DENSE = "dense"
    ROUTED_EXPERT = "routed_expert"
    LIGHTNING_INDEXER = "lightning_indexer"
    SPARSE_ATTENTION = "sparse_attention"
    COMPRESSOR = "compressor"


V41_FILTER_SPECS: dict[str, FilterSpec] = {
    V41FilterKeys.DENSE: {"suffixes": _DENSE},
    V41FilterKeys.ROUTED_EXPERT: {"suffixes": _ROUTED},
    V41FilterKeys.LIGHTNING_INDEXER: {"suffixes": (".attention.indexer.selector",)},
    V41FilterKeys.SPARSE_ATTENTION: {"suffixes": (".attention.inner_attention",)},
}

V41_EXTRA_FILTERS = {V41FilterKeys.COMPRESSOR: _is_v41_kv_source}

V41_PREQUANTIZE_DEFAULT_FQNS: dict[str, tuple[str, ...]] = {
    V41FilterKeys.DENSE: _DENSE,
    V41FilterKeys.ROUTED_EXPERT: _ROUTED,
}

SPECIFIC_NPU_QUANTIZED_MODULES: dict[type, type] = {}

UNIFIED_NPU_QUANTIZED_MODULES = (
    Linear.Config,
    BatchedLinear.Config,
    GroupedLinear.Config,
    DeepSeekV41CompressedSparseInnerAttention2.Config,
    DeepSeekV41Compressor.Config,
    DeepSeekV41Selector.Config,
)

DEEPSEEK_V41_RECIPE_MATRIX = {
    "all_mxfp8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=mxfp8_param_swap(),
            filter=any_config_filter(filters[V41FilterKeys.DENSE], filters[V41FilterKeys.ROUTED_EXPERT]),
        ),
    ],
    "all_hif8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=_hif8_param_swap(ctx.enable_hif8_save_quant_codes),
            filter=any_config_filter(filters[V41FilterKeys.DENSE], filters[V41FilterKeys.ROUTED_EXPERT]),
        ),
    ],
    "mix": lambda filters, ctx: [
        ParamSwapSpec(swap_config=mxfp8_param_swap(), filter=filters[V41FilterKeys.DENSE], scope=V41FilterKeys.DENSE),
        _routed_expert_block_fp8(filters, ctx),
    ],
    "all_block_fp8": lambda filters, ctx: [
        ParamSwapSpec(
            swap_config=block_fp8_param_swap(
                enable_fsdp_prequantize=ctx.enable_fsdp_prequantize,
                fsdp_prequantize_dtype="fp8",
            ),
            filter=filters[V41FilterKeys.DENSE],
            scope=V41FilterKeys.DENSE,
        ),
        _routed_expert_block_fp8(filters, ctx),
    ],
}
