# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter-swap / quantization-config builders for the torchao-npu recipes."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal

import torch
from torchao_npu.configs import ParamSwapConfig
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)

logger = logging.getLogger(__name__)

# extension.quantization.enable-hif8-save-quant-codes ops added to the
# selective-AC save list.
_QUANT_CODE_SAVE_OPS = (
    "npu.npu_quantize.default",
    "npu.npu_dynamic_quant.default",
    "npu.npu_grouped_matmul.default",
    "npu.npu_quant_matmul.default",
)

_AC_QUANT_CODES_PATCHED = [False]


def _save_quant_codes_under_selective_ac(enable: bool) -> None:
    """extension.quantization.enable-hif8-save-quant-codes: keep the quantized
    matmul's operands and result across the selective-AC boundary instead of
    re-deriving them in the backward.

    Must be called at model-setup time, before parallelization wraps the
    blocks: torchtitan reads ``_get_default_save_ops`` when it installs
    activation checkpointing.
    """
    if not enable or _AC_QUANT_CODES_PATCHED[0]:
        return
    try:
        from torchtitan.distributed import activation_checkpoint as ac
    except ImportError:
        return
    _AC_QUANT_CODES_PATCHED[0] = True
    orig = ac._get_default_save_ops

    def _get_default_save_ops():
        ops = orig()
        added, missing = [], []
        for path in _QUANT_CODE_SAVE_OPS:
            obj = torch.ops
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except (AttributeError, RuntimeError):
                missing.append(path)
                continue
            ops.add(obj)
            added.append(path)
        logger.info(
            "enable_hif8_save_quant_codes: added %s to the selective-AC save list%s",
            ", ".join(added) or "nothing",
            f" (not registered: {', '.join(missing)})" if missing else "",
        )
        return ops

    ac._get_default_save_ops = _get_default_save_ops


def mxfp8_param_swap() -> ParamSwapConfig:
    mx_config = MXQuantizeConfig()
    return ParamSwapConfig(
        weight_config=mx_config,
        activation_config=mx_config,
    )


def block_fp8_param_swap(
    *,
    enable_mxfp4_qat: bool = False,
    dst_type_max: float = 0.0,
    enable_fsdp_prequantize: bool = False,
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = "fp8",
) -> ParamSwapConfig:
    if fsdp_prequantize_dtype not in ("fp8", "mxfp4"):
        raise ValueError(f"fsdp_prequantize_dtype must be 'fp8' or 'mxfp4', got {fsdp_prequantize_dtype!r}")
    mxfp4_config = None
    if enable_mxfp4_qat:
        mxfp4_config = MXQuantizeConfig(
            elem_dtype=torch.float4_e2m1fn_x2,
            dst_type_max=dst_type_max,
        )
    return ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(
            mxfp4_fake_quantize_config=mxfp4_config,
            fsdp_prequantize=enable_fsdp_prequantize,
            fsdp_prequantize_dtype=fsdp_prequantize_dtype,
        ),
        activation_config=MXQuantizeConfig(),
    )


def _is_prequantizing_block_mx(weight_config: object) -> bool:
    """Whether a weight config is a Block MX config with pre-quantize enabled."""
    return isinstance(weight_config, BlockMXQuantizeConfig) and weight_config.fsdp_prequantize


def _param_swap_without_prequantize(param_swap: ParamSwapConfig) -> ParamSwapConfig:
    """Copy a ``ParamSwapConfig`` with the node-local ``fsdp_prequantize`` off.

    Used for weights outside the whitelist: the BF16 communication path is
    then an explicit configuration decision, not an implicit runtime fallback.
    Every constructor field is carried over so the copy cannot silently drop
    state if ``ParamSwapConfig`` grows new parameters.
    """
    weight_config = param_swap.weight_config
    assert isinstance(weight_config, BlockMXQuantizeConfig)
    return ParamSwapConfig(
        base_config=param_swap.base_config,
        activation_config=param_swap.activation_config,
        weight_config=replace(weight_config, fsdp_prequantize=False),
        step=param_swap.step,
        params_filter_fn=param_swap.params_filter_fn,
    )


def _hif8_param_swap(save_quant_codes: bool) -> ParamSwapConfig:
    # Current/constant tensor scaling only: every quantize call recomputes
    # its scale via npu_dynamic_quant, no persisted per-parameter state.
    hif8_config = HiF8QuantizeConfig()
    _save_quant_codes_under_selective_ac(save_quant_codes)
    return ParamSwapConfig(
        weight_config=hif8_config,
        activation_config=hif8_config,
    )
