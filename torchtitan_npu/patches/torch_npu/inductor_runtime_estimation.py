# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU fallback for PyTorch Inductor runtime estimation helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Conservative HBM bandwidth used only by graph scheduling roofline estimates.
# It avoids Triton CUDA-driver probing on NPU; it does not change generated ops.
_NPU_DRAM_GBPS = 1200

# Idempotency flags are kept as module state rather than function attributes.
_DRAM_PATCHED = False
_STANDALONE_COMPILE_PATCHED = False


def _patch_dram_bandwidth() -> None:
    global _DRAM_PATCHED
    import torch
    import torch._inductor.utils
    import torch.utils._runtime_estimation

    if _DRAM_PATCHED:
        return

    original_device = getattr(torch._inductor.utils, "get_device_dram_gbps", lambda device=None: None)
    original_gpu = getattr(torch._inductor.utils, "get_gpu_dram_gbps", lambda: _NPU_DRAM_GBPS)

    def is_npu(device: torch.device | str | None) -> bool:
        if device is None:
            device = torch.accelerator.current_accelerator(check_available=True)
        return device is not None and torch.device(device).type == "npu"

    def get_device_dram_gbps(device: torch.device | str | None = None) -> float | None:
        if is_npu(device):
            return _NPU_DRAM_GBPS
        return original_device(device)

    def get_gpu_dram_gbps() -> float:
        if is_npu(None):
            return _NPU_DRAM_GBPS
        return original_gpu()

    torch._inductor.utils.get_device_dram_gbps = get_device_dram_gbps  # pyrefly: ignore [missing-attribute]
    # Runtime estimation imports this helper by value before this patch runs.
    torch.utils._runtime_estimation.get_device_dram_gbps = get_device_dram_gbps  # pyrefly: ignore [missing-attribute]
    # The legacy helper is still used by Inductor's scheduler and pad_mm.
    torch._inductor.utils.get_gpu_dram_gbps = get_gpu_dram_gbps  # pyrefly: ignore [bad-assignment]
    _DRAM_PATCHED = True
    logger.info(
        "[NPU Runtime Estimation Patch] using %s GB/s DRAM bandwidth fallback",
        _NPU_DRAM_GBPS,
    )


def _patch_standalone_compile_deepcopy() -> None:
    global _STANDALONE_COMPILE_PATCHED
    import torch._inductor

    if _STANDALONE_COMPILE_PATCHED:
        return

    original = torch._inductor.standalone_compile

    def standalone_compile_npu(*args, **kwargs):
        kwargs.setdefault("donate_graph_module", True)
        return original(*args, **kwargs)

    torch._inductor.standalone_compile = standalone_compile_npu
    _STANDALONE_COMPILE_PATCHED = True
    logger.info("[NPU Runtime Estimation Patch] enabled standalone_compile donate_graph_module")


def apply() -> None:
    try:
        import torch
    except ImportError as exc:
        logger.warning("[NPU Runtime Estimation Patch] skip patch: %s", exc)
        return

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return

    _patch_dram_bandwidth()
    _patch_standalone_compile_deepcopy()


apply()
