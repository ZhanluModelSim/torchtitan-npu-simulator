# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reversible class-attribute patch that swaps the `npu_mhc_pre`/`npu_mhc_post` model
converters' target implementation classes for the simulator's shape-only shims
(SimHcPre/SimHcHead/SimHcPost), instead of SimulationTrainer stripping these converters out
entirely. Mirrors meta_env.py's established "patch a class attribute, track the original for a
symmetric unpatch" pattern -- MHCPrePostModelConfig/MHCPostModelConfig are the real, singleton,
already-registered converter-config classes from torchtitan_npu.converters.kernels.mhc_prepost
(zero modification to that file itself: this module only reassigns their `model_converter`
class attribute at runtime, under simulation)."""

from __future__ import annotations

import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.mhc_prepost import MHCPostModelConfig, MHCPrePostModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcHead, SimHcPost, SimHcPre

_original_mhc_pre_converter: type | None = None
_original_mhc_post_converter: type | None = None


class SimMHCPreConverter(ModelCustomConverter):
    """Replaces every `HcPre` submodule with `SimHcPre` -- never selects the
    real fused/Triton implementation (see design doc §2: neither path can
    execute under simulation)."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPre):
                replace_module_with_name(model, name, SimHcPre(module))


class SimMHCPostConverter(ModelCustomConverter):
    """Replaces every `HcPost` submodule with `SimHcPost` and every
    `HcHead` submodule with `SimHcHead`."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPost):
                replace_module_with_name(model, name, SimHcPost(module))
            if isinstance(module, HcHead):
                replace_module_with_name(model, name, SimHcHead(module))


def apply_mhc_shims() -> None:
    """Patch MHCPrePostModelConfig.model_converter / MHCPostModelConfig.model_converter to
    point at the Sim* converters above. Idempotent: the `is None` guards below mean only the
    *first* call saves the pre-patch "original" value; every subsequent call is a no-op for
    that bookkeeping, so unapply_mhc_shims() always restores the value active before the very
    first apply_mhc_shims() call, regardless of how many times apply was called in between."""
    global _original_mhc_pre_converter, _original_mhc_post_converter
    if _original_mhc_pre_converter is None:
        _original_mhc_pre_converter = MHCPrePostModelConfig.model_converter
    if _original_mhc_post_converter is None:
        _original_mhc_post_converter = MHCPostModelConfig.model_converter
    MHCPrePostModelConfig.model_converter = SimMHCPreConverter
    MHCPostModelConfig.model_converter = SimMHCPostConverter


def unapply_mhc_shims() -> None:
    """Restore the original converter classes. Safe to call even if
    apply_mhc_shims() was never called (no-op), and safe to call more than
    once (idempotent)."""
    global _original_mhc_pre_converter, _original_mhc_post_converter
    if _original_mhc_pre_converter is not None:
        MHCPrePostModelConfig.model_converter = _original_mhc_pre_converter
        _original_mhc_pre_converter = None
    if _original_mhc_post_converter is not None:
        MHCPostModelConfig.model_converter = _original_mhc_post_converter
        _original_mhc_post_converter = None
