# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
MXFP8 quantization configuration for activation checkpoint (AC) optimization.

Provides global state tracking (AC mode, compile flag) and decision functions
used by NpuMXFP8MM / NpuMXFP8GroupedMM to select the appropriate quantization
strategy:

- AC disabled: dual-axis quant in forward, save bwd data on ctx (zero bwd overhead).
- AC enabled + compile disabled: stack bridge passes bwd data from recomputation
  forward to backward. With scale_alg=1, dual-axis and single-axis produce
  equivalent forward output, so AC recomputation is numerically consistent.
- AC enabled + compile enabled: re-quantize in backward (Dynamo-safe).
"""

import contextlib
import os
import threading

from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchao_npu.activation_checkpoint_state import is_in_recomputation

_ac_mode: str | None = None
_compile_enabled: bool | None = None
_recomputation_detection_initialized: bool = False
_quant_mode_logged: bool = False


def _ensure_recomputation_detection() -> None:
    """Lazy-init recomputation detection on first use."""
    global _recomputation_detection_initialized
    if _recomputation_detection_initialized:
        return
    from torchtitan_npu.patches.torchao_npu.activation_checkpoint_state import (
        enable_recomputation_detection,
    )

    enable_recomputation_detection()
    _recomputation_detection_initialized = True


def is_mxfp8_dual_axis_forward() -> bool:
    """Check whether MXFP8 dual-axis forward quantization is enabled.

    Controlled by the environment variable MXFP8_DUAL_AXIS_FORWARD.
    Defaults to True (dual-axis enabled). Set to '0' or 'false' to disable.
    """
    val = os.environ.get("MXFP8_DUAL_AXIS_FORWARD", "1")
    return val.lower() not in ("0", "false", "no")


def get_ac_mode() -> str:
    """Get AC mode, lazily reading from Trainer config on first call."""
    global _ac_mode
    if _ac_mode is not None:
        return _ac_mode
    try:
        from torchtitan_npu.patches.torchtitan._trainer_config_stash import get_trainer_config

        config = get_trainer_config()
        if config is not None and hasattr(config, "activation_checkpoint"):
            ac = config.activation_checkpoint
            if ac is not None:
                _ac_mode = ac.mode
                return _ac_mode
    except ImportError:
        pass  # Config stash not available yet, use default
    _ac_mode = "none"
    return _ac_mode


def is_compile_enabled() -> bool:
    """Check if model is compiled, lazily reading from Trainer config.

    Returns True only when compile is enabled AND 'model' is in compile
    components. When only loss is compiled (model not compiled), the MXFP8
    forward path is not traced by Dynamo, so the stack bridge mechanism
    works normally and we don't need to fall back to re-quantization.
    """
    global _compile_enabled
    if _compile_enabled is not None:
        return _compile_enabled
    try:
        from torchtitan_npu.patches.torchtitan._trainer_config_stash import get_trainer_config

        config = get_trainer_config()
        if config is not None and hasattr(config, "compile"):
            compile_cfg = config.compile
            if compile_cfg is not None and compile_cfg.enable:
                components = getattr(compile_cfg, "components", None)
                _compile_enabled = components is None or "model" in components
                return _compile_enabled
    except ImportError:
        pass  # Config stash not available yet, use default
    _compile_enabled = False
    return _compile_enabled


_ac_context = threading.local()


@contextlib.contextmanager
def ac_enabled_context(enabled: bool):
    old_value = getattr(_ac_context, "enabled", False)
    _ac_context.enabled = enabled
    try:
        yield
    finally:
        _ac_context.enabled = old_value


def is_ac_enabled() -> bool:
    return getattr(_ac_context, "enabled", False)


def should_save_bwd_quant_for_mx() -> bool:
    """Decide whether to save bwd quant data in forward.

    When mxfp8_dual_axis_forward is False (default), always return False
    to use single-axis quant throughout.
    When True, apply four-scenario AC-aware handling.
    """
    global _quant_mode_logged
    if not _quant_mode_logged:
        _quant_mode_logged = True
        if is_mxfp8_dual_axis_forward():
            logger.info("[Quant] MXFP8 dual-axis forward quantization enabled (via MXFP8_DUAL_AXIS_FORWARD)")
        else:
            logger.info("[Quant] MXFP8 dual-axis forward quantization disabled (single-axis only)")

    if not is_mxfp8_dual_axis_forward():
        return False
    if is_compile_enabled() and is_ac_enabled():
        return False
    if is_ac_enabled():
        return is_in_recomputation()
    return True
