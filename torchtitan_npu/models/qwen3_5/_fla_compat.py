# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Make the optional FLA imports in TorchTitan Qwen3.5 lazy and explicit.

TorchTitan keeps the Qwen3.5 model in the upstream package, but importing that
module imports Flash Linear Attention (FLA) unconditionally.  The NPU adapter
uses its own Triton kernel, so FLA is not required for that path.  These small
placeholder modules let the upstream model be imported without pretending that
the FLA kernels are available at runtime.
"""

# Dynamic module attributes are intentional: these modules only exist as
# import-time placeholders when the optional FLA dependency is unavailable.
# ruff: noqa: B010

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any

import torch


def _unavailable_fla_kernel(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError(
        "Qwen3.5 FLA kernels are unavailable. Use the NPU override "
        "torchtitan_npu.override.qwen3_5.gated_delta.npu or install FLA."
    )


class _UnavailableFlaAutogradFunction(torch.autograd.Function):
    """Autograd-compatible placeholder for FLA Function classes."""

    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        return _unavailable_fla_kernel(*args, **kwargs)


def _has_fla() -> bool:
    required_modules = (
        "fla.modules.conv.triton.ops",
        "fla.modules.conv.causal_conv1d",
        "fla.ops.gated_delta_rule",
        "fla.ops.gated_delta_rule.chunk",
        "fla.ops.gated_delta_rule.fused_recurrent",
    )
    try:
        return all(importlib.util.find_spec(name) is not None for name in required_modules)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _temporary_fla_modules() -> dict[str, types.ModuleType]:
    fla = types.ModuleType("fla")
    fla.__path__ = []
    modules = types.ModuleType("fla.modules")
    modules.__path__ = []
    conv = types.ModuleType("fla.modules.conv")
    conv.__path__ = []
    conv_triton = types.ModuleType("fla.modules.conv.triton")
    conv_triton.__path__ = []
    conv_triton_ops = types.ModuleType("fla.modules.conv.triton.ops")
    setattr(conv_triton_ops, "CausalConv1dFunction", _UnavailableFlaAutogradFunction)
    causal_conv1d = types.ModuleType("fla.modules.conv.causal_conv1d")
    setattr(causal_conv1d, "causal_conv1d", _unavailable_fla_kernel)
    ops = types.ModuleType("fla.ops")
    ops.__path__ = []
    gated_delta_rule = types.ModuleType("fla.ops.gated_delta_rule")
    setattr(gated_delta_rule, "chunk_gated_delta_rule", _unavailable_fla_kernel)
    setattr(gated_delta_rule, "fused_recurrent_gated_delta_rule", _unavailable_fla_kernel)
    gated_delta_rule_chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    setattr(gated_delta_rule_chunk, "ChunkGatedDeltaRuleFunction", _UnavailableFlaAutogradFunction)
    gated_delta_rule_fused_recurrent = types.ModuleType("fla.ops.gated_delta_rule.fused_recurrent")
    setattr(
        gated_delta_rule_fused_recurrent,
        "FusedRecurrentFunction",
        _UnavailableFlaAutogradFunction,
    )

    setattr(fla, "modules", modules)
    setattr(modules, "conv", conv)
    setattr(conv, "triton", conv_triton)
    setattr(conv, "causal_conv1d", causal_conv1d)
    setattr(conv_triton, "ops", conv_triton_ops)
    setattr(fla, "ops", ops)
    setattr(ops, "gated_delta_rule", gated_delta_rule)
    setattr(gated_delta_rule, "chunk", gated_delta_rule_chunk)
    setattr(gated_delta_rule, "fused_recurrent", gated_delta_rule_fused_recurrent)
    return {
        "fla": fla,
        "fla.modules": modules,
        "fla.modules.conv": conv,
        "fla.modules.conv.triton": conv_triton,
        "fla.modules.conv.triton.ops": conv_triton_ops,
        "fla.modules.conv.causal_conv1d": causal_conv1d,
        "fla.ops": ops,
        "fla.ops.gated_delta_rule": gated_delta_rule,
        "fla.ops.gated_delta_rule.chunk": gated_delta_rule_chunk,
        "fla.ops.gated_delta_rule.fused_recurrent": gated_delta_rule_fused_recurrent,
    }


def ensure_qwen3_5_importable():
    """Import upstream Qwen3.5, isolating optional FLA placeholders."""
    module_name = "torchtitan.models.qwen3_5"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if _has_fla():
        return importlib.import_module(module_name)

    created: dict[str, types.ModuleType] = {}
    for name, module in _temporary_fla_modules().items():
        if name not in sys.modules:
            sys.modules[name] = module
            created[name] = module
    try:
        return importlib.import_module(module_name)
    finally:
        for name, module in reversed(tuple(created.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]


__all__ = ["ensure_qwen3_5_importable"]
