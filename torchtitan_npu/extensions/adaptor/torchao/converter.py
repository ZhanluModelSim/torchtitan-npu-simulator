# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU converter integration hosted by ``torchtitan-npu`` (facade).

This module is the public entry (``apply_quantization_converter``) and a thin
facade re-exporting the split-out implementation (converter classes, the generic
recipe template, the config registry, family configs).  It is loaded lazily by
the quantized-training / quantized-export paths only.

NOTE: the large re-export block below is a deliberate compatibility surface --
existing callers and CPU tests read these symbols off the ``converter`` module
(which used to be a single monolithic file).  Keep it stable or update those
consumers in the same change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torchao_npu.quantization.quant_configs import (  # noqa: F401  (re-exported)
    BlockMXQuantizeConfig,
    FP8QuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan.config import derive  # noqa: F401  (re-exported)
from torchtitan.config.transform import validate_converter_compatibility
from torchtitan.models.common.linear import GroupedLinear, Linear  # noqa: F401  (re-exported for tests)
from torchtitan.protocols.module import Module  # noqa: F401  (re-exported for tests)

import torchtitan_npu  # noqa: F401  (apply package patches before binding model classes)
from torchtitan_npu.models.deepseek_v4_1.indexer import Selector as DeepSeekV41Selector  # noqa: F401  (re-exported)

from .components.deepseek_v4 import (
    _prepare_quant_lightning_indexer_inputs,  # noqa: F401  (re-exported)
)
from .configs import (
    _DSV4_CONFIG_FILTERS,  # noqa: F401  (re-exported for tests)
    _DSV41_CONFIG_FILTERS,  # noqa: F401  (re-exported for tests)
    ModelType,  # noqa: F401  (re-exported for tests)
    _model_type_for_config,
)
from .configs.deepseek_v4 import (  # noqa: F401  (re-exported for tests)
    SPECIFIC_NPU_QUANTIZED_MODULES,
    UNIFIED_NPU_QUANTIZED_MODULES,
    V4FilterKeys,
)
from .filters import (  # noqa: F401
    ConfigFilterFn,
    _match_fqn_suffix,
    _replace_config,
    any_config_filter,
    match_config_fqn_suffix,
)
from .param_swaps import block_fp8_param_swap, mxfp8_param_swap
from .recipes import (
    NpuQuantizeConverter,
    _npu_quantized_module_cache,  # noqa: F401  (re-exported for tests)
    _recipe_converters,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torchtitan.config.transform.quantization import QuantizationConverter
    from torchtitan.protocols.model import BaseModel

    from torchtitan_npu.config.configs import QuantizationExtensionConfig


_AC_BLOCK_OPS_PATCHED = [False]


# extension.quantization.save-block-ops-level levels -> the transformer-block ops
# added to the selective-AC save list, grouped by the module they belong to
# and ordered by critical-path ms per saved GiB (best first).
_BLOCK_SAVE_OPS = (
    # 1: attention
    (1, "cann_ops_transformer.sparse_flash_mla.default"),
    (1, "cann_ops_transformer.lightning_indexer.default"),
    # 2: mHC
    (2, "npu.npu_mhc_pre.default"),
    (2, "cann_ops_transformer.mhc_post.default"),
    (2, "npu.npu_mhc_sinkhorn.default"),
    # 3: MoE routing
    (3, "npu.npu_moe_token_permute.default"),
    (3, "torchtitan_npu.npu_moe_re_routing.default"),
    (3, "torchtitan_npu.npu_moe_token_unpermute.default"),
)


def _resolve_torch_op(path: str):
    """``torch.ops`` lookup by dotted path, or None if not registered yet."""
    obj = torch.ops
    try:
        for part in path.split("."):
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _save_block_ops_under_selective_ac(level: int) -> None:
    """extension.quantization.save-block-ops-level: keep the attention / mHC /
    MoE-routing results across the selective-AC boundary instead of
    re-deriving them in the backward.

    torchtitan's ``_get_default_save_ops()`` doesn't know about custom NPU
    ops, so the min-cut partitioner always recomputes them. ``_BLOCK_SAVE_OPS``
    orders them best-first by profiled win, an upper bound rather than a
    guarantee -- a recomputed kernel off the critical path may already be
    overlapped.

    NOT bit-exact (forward and recompute don't compile to the same fused
    kernel), so this needs a loss A/B before being trusted.

    Resolution is deferred to call time (inside the returned closure, not
    at patch time) because some ops register only once model modules have
    imported ``cann_ops_transformer``.
    """
    if level <= 0 or _AC_BLOCK_OPS_PATCHED[0]:
        return
    try:
        from torchtitan.distributed import activation_checkpoint as ac
    except ImportError:
        return
    wanted = tuple(path for lvl, path in _BLOCK_SAVE_OPS if lvl <= level)
    if not wanted:
        return
    _AC_BLOCK_OPS_PATCHED[0] = True
    orig = ac._get_default_save_ops

    def _get_default_save_ops():
        ops = orig()
        added, missing = [], []
        for path in wanted:
            op = _resolve_torch_op(path)
            if op is None:
                # Not registered in this build/config (e.g. the lightning
                # indexer is off for this model). A miss is a lost
                # optimization, never a correctness problem.
                missing.append(path)
            else:
                ops.add(op)
                added.append(path)
        logger.info(
            "save_block_ops_level=%d: added %s to the selective-AC save list%s",
            level,
            ", ".join(added) or "nothing",
            f" (not registered: {', '.join(missing)})" if missing else "",
        )
        return ops

    ac._get_default_save_ops = _get_default_save_ops


def _check_prequantize_policy(
    converters: list[object],
    prequantize_fqns: tuple[str, ...],
    *,
    strict: bool,
) -> None:
    """Validate whitelist coverage after conversion.

    ``strict`` distinguishes the two whitelist sources. A *user-provided*
    whitelist is a promise that every pattern exists, so a pattern matching no
    pre-quantizing node is a typo or stale name and must fail fast. The
    *recipe default* whitelist enumerates the full projection set including
    optional modules, so a zero-match default entry is a legitimate absent module
    and only logs.
    """
    if not prequantize_fqns:
        return
    matched: set[str] = set()
    for converter in converters:
        matched |= getattr(converter, "matched_prequantize_fqns", set())
    unmatched = [pattern for pattern in prequantize_fqns if pattern not in matched]
    if not unmatched:
        return
    if not strict:
        logger.info(
            "fsdp_prequantize default whitelist entries not present in this model, skipped: %s",
            unmatched,
        )
        return
    raise ValueError(
        f"fsdp_prequantize_fqns patterns matched no quantized weight: {unmatched}. "
        f"Check the spelling against the model's quantized projections "
        f"(all patterns: {list(prequantize_fqns)})."
    )


def _validate_fsdp_prequantize_policy(
    *,
    converter_configs: list[QuantizationConverter.Config],
    built_converters: list[object],
    enable_fsdp_prequantize: bool,
    recipe: str,
    strict: bool,
) -> None:
    """Validate the fsdp-prequantize config against what was actually quantized.

    1. Union the prequantize whitelist patterns declared by every converter config.
    2. Fail fast if prequantize is enabled but the recipe pre-quantizes no scope.
    3. Verify every declared pattern matched a real node (via
       :func:`_check_prequantize_policy`): fail-fast for a user whitelist, log-only
       for the recipe-default whitelist.
    """
    effective_fqns = tuple(
        dict.fromkeys(
            fqn
            for converter_config in converter_configs
            for fqn in (getattr(converter_config, "prequantize_fqns", None) or ())
        )
    )
    if enable_fsdp_prequantize and not effective_fqns:
        raise ValueError(
            f"enable_fsdp_prequantize=True but recipe {recipe!r} pre-quantizes no "
            "weight scope; use all_block_fp8 (or mix, which pre-quantizes routed experts only) "
            "or disable enable_fsdp_prequantize."
        )
    _check_prequantize_policy(built_converters, effective_fqns, strict=strict)


def apply_quantization_converter(
    model_config: BaseModel.Config | None,
    quantization_config: QuantizationExtensionConfig,
    *,
    model_compile_enabled: bool,
    override_config=None,
) -> BaseModel.Config:
    """Apply the selected TorchAO-NPU recipe to an existing BF16 model config."""

    if model_config is None:
        raise ValueError("TorchAO-NPU quantization requires model to be configured")
    # Recipe-independent, so it is installed here rather than in
    # _hif8_param_swap: these are block ops, not quantized ops, and MXFP8 wants
    # them saved just as much as HiF8 does. Off by default -- at level 0
    # nothing is patched and the partition is the one we ship today.
    _save_block_ops_under_selective_ac(quantization_config.save_block_ops_level)
    if not quantization_config.enable_quantized_training:
        logger.info("enable_quantized_training=False; skipping TorchAO-NPU quantization")
        return model_config
    quantization_config.validate()

    model_type = _model_type_for_config(model_config)
    converter_configs = _recipe_converters(
        quantization_config.recipe,
        model_type=model_type,
        enable_sparse_attention_quantization=quantization_config.enable_sparse_attention_quantization,
        enable_mxfp4_qat=quantization_config.enable_mxfp4_qat,
        dst_type_max=quantization_config.dst_type_max,
        enable_fsdp_prequantize=quantization_config.enable_fsdp_prequantize,
        fsdp_prequantize_dtype=quantization_config.fsdp_prequantize_dtype,
        model_compile_enabled=model_compile_enabled,
        li_quantization=quantization_config.li_quantization,
        kv_norm_quantization=quantization_config.kv_norm_quantization,
        fsdp_prequantize_fqns=quantization_config.fsdp_prequantize_fqns,
        enable_hif8_save_quant_codes=quantization_config.enable_hif8_save_quant_codes,
        model_config=model_config,
        override_config=override_config,
    )
    validate_converter_compatibility(list(converter_configs))

    built_converters: list[object] = []
    for converter_config in converter_configs:
        converter = converter_config.build()
        built_converters.append(converter)
        model_config = converter.convert(model_config)

    # Aggregate the per-node whitelist decisions. Only patterns actually handed
    # to a pre-quantizing scope participate (e.g. "mix" runs dense as plain MX,
    # so dense patterns are not passed there and must not be reported as
    # unmatched). Zero-match handling is source-dependent: user whitelists fail
    # fast on typos, the recipe default tolerates absent optional modules.
    # Shard-alignment validation depends on the runtime FSDP sharding and is
    # left to the fail-fast guard in the wrapper tensor.
    _validate_fsdp_prequantize_policy(
        converter_configs=converter_configs,
        built_converters=built_converters,
        enable_fsdp_prequantize=quantization_config.enable_fsdp_prequantize,
        recipe=quantization_config.recipe,
        strict=quantization_config.fsdp_prequantize_fqns is not None,
    )

    logger.info(
        "Applied TorchAO-NPU recipe=%s, sparse_attention_quantization=%s, mxfp4_qat=%s, "
        "li_quantization=%s, kv_norm_quantization=%s, kv_norm_block_size=%s, "
        "dst_type_max=%s, enable_fsdp_prequantize=%s, fsdp_prequantize_dtype=%s",
        quantization_config.recipe,
        quantization_config.enable_sparse_attention_quantization,
        quantization_config.enable_mxfp4_qat,
        quantization_config.li_quantization,
        quantization_config.kv_norm_quantization.format,
        quantization_config.kv_norm_quantization.block_size,
        quantization_config.dst_type_max,
        quantization_config.enable_fsdp_prequantize,
        quantization_config.fsdp_prequantize_dtype,
    )
    return model_config


__all__ = [
    "ConfigFilterFn",
    "NpuQuantizeConverter",
    "any_config_filter",
    "apply_quantization_converter",
    "block_fp8_param_swap",
    "match_config_fqn_suffix",
    "mxfp8_param_swap",
]
