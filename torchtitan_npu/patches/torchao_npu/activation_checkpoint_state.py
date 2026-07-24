# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Detect activation checkpoint recomputation state and provide stack bridge
for MXFP8 quant data between recomputation forward and backward.
"""

import threading
from functools import wraps
from typing import Any

_recomputation_state = threading.local()
_detection_enabled: bool = False

# Stacks bridge quant data from recomputation forward to backward.
# Two separate stacks for NpuMXFP8MM and NpuMXFP8GroupedMM.
_mx_quant_stack: list[dict[str, Any]] = []
_gmm_quant_stack: list[dict[str, Any]] = []


def push_mx_quant_data(data: dict[str, Any]) -> None:
    _mx_quant_stack.append(data)


def pop_mx_quant_data() -> dict[str, Any] | None:
    return _mx_quant_stack.pop() if _mx_quant_stack else None


def push_gmm_quant_data(data: dict[str, Any]) -> None:
    _gmm_quant_stack.append(data)


def pop_gmm_quant_data() -> dict[str, Any] | None:
    return _gmm_quant_stack.pop() if _gmm_quant_stack else None


def is_in_recomputation() -> bool:
    """Return True during AC recomputation forward.

    Lazily initializes recomputation detection on first call when
    MXFP8 dual-axis forward is enabled.
    """
    # Lazy import to avoid circular dependency with mx_ac_config
    from torchtitan_npu.patches.torchao_npu.mx_ac_config import (
        _ensure_recomputation_detection,
        is_mxfp8_dual_axis_forward,
    )

    if is_mxfp8_dual_axis_forward():
        _ensure_recomputation_detection()
    if not _detection_enabled:
        return False
    return getattr(_recomputation_state, "in_recomp", False)


def enable_recomputation_detection() -> None:
    """Patch torch.utils.checkpoint to set recomputation flag.

    Preserves @torch._disable_dynamo on the patched function so that
    recompute remains outside the Dynamo graph, matching PyTorch's
    original checkpoint compile behavior.
    """
    global _detection_enabled
    if _detection_enabled:
        return

    import torch
    from torch.utils.checkpoint import _run_fn_with_dynamo_disabled

    _orig_run_fn = _run_fn_with_dynamo_disabled.__wrapped__  # pyrefly: ignore [missing-attribute]

    @wraps(_orig_run_fn)
    @torch._disable_dynamo  # Preserve original compile behavior
    def _patched_run_fn(fn, *args, **kwargs):
        if fn.__name__ == "recompute_fn":
            old_value = getattr(_recomputation_state, "in_recomp", False)
            _recomputation_state.in_recomp = True
            try:
                return _orig_run_fn(fn, *args, **kwargs)
            finally:
                _recomputation_state.in_recomp = old_value
        return _orig_run_fn(fn, *args, **kwargs)

    import torch.utils.checkpoint as checkpoint_module

    checkpoint_module._run_fn_with_dynamo_disabled = _patched_run_fn
    _detection_enabled = True
