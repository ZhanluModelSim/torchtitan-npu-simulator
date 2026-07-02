# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reversible class-attribute patch that swaps the `npu_smla` model converter's target
implementation class for the simulator's shape-only shims (SimNpuSparseAttention/
SimNpuLiCompute/SimNpuLiLoss), instead of SimulationTrainer stripping this converter out
entirely. Mirrors mhc_converter.py's apply_mhc_shims()/unapply_mhc_shims() exactly --
NpuSMLAModelConfig is the real, already-registered converter-config class from
torchtitan_npu.converters.kernels.npu_smla (zero modification to that file: this module only
reassigns its `model_converter` class attribute at runtime, under simulation)."""

from __future__ import annotations

import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.npu_smla import NpuSMLAModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.deepseek_v4.model import LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute, SimNpuLiLoss, SimNpuSparseAttention

_original_smla_converter: type | None = None


class SimSMLAConverter(ModelCustomConverter):
    """Replaces every SparseAttention/LiCompute/LiLoss submodule with the corresponding Sim*
    shim -- never selects the real fused (A5) or JIT-compiled (non-A5) implementation (see
    design doc §2: neither path can execute under simulation)."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, SparseAttention):
                replace_module_with_name(model, name, SimNpuSparseAttention(module))
            if isinstance(module, LiCompute):
                replace_module_with_name(model, name, SimNpuLiCompute(module))
            if isinstance(module, LiLoss):
                replace_module_with_name(model, name, SimNpuLiLoss(module))


def apply_smla_shims() -> None:
    """Patch NpuSMLAModelConfig.model_converter to point at SimSMLAConverter. Idempotent: the
    `is None` guard below means only the *first* call saves the pre-patch "original" value;
    every subsequent call is a no-op for that bookkeeping, so unapply_smla_shims() always
    restores the value active before the very first apply_smla_shims() call."""
    global _original_smla_converter
    if _original_smla_converter is None:
        _original_smla_converter = NpuSMLAModelConfig.model_converter
    NpuSMLAModelConfig.model_converter = SimSMLAConverter


def unapply_smla_shims() -> None:
    """Restore the original converter class. Safe to call even if apply_smla_shims() was
    never called (no-op), and safe to call more than once (idempotent)."""
    global _original_smla_converter
    if _original_smla_converter is not None:
        NpuSMLAModelConfig.model_converter = _original_smla_converter
        _original_smla_converter = None
