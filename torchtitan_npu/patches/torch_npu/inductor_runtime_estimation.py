# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU fallback for PyTorch Inductor runtime estimation helpers."""

from __future__ import annotations

from torchtitan.tools.logging import logger

# Conservative HBM bandwidth used only by graph scheduling roofline estimates.
# It avoids Triton CUDA-driver probing on NPU; it does not change generated ops.
_NPU_DRAM_GBPS = 1200

# Idempotency flags. Kept as module state instead of attributes on the patched
# functions: pyrefly rejects FunctionType attribute writes and pyrefly types
# ``get_gpu_dram_gbps`` as ``_lru_cache_wrapper[int]`` (bad-assignment), while
# ruff rejects the setattr() workaround (B010).
_DRAM_PATCHED = False
_STANDALONE_COMPILE_PATCHED = False


def _patch_dram_bandwidth() -> None:
    global _DRAM_PATCHED
    import torch._inductor.utils as inductor_utils
    import torch.utils._runtime_estimation as runtime_estimation

    if _DRAM_PATCHED:
        return

    def get_npu_dram_gbps() -> int:
        return _NPU_DRAM_GBPS

    inductor_utils.get_gpu_dram_gbps = get_npu_dram_gbps  # pyrefly: ignore [bad-assignment]
    runtime_estimation.get_gpu_dram_gbps = get_npu_dram_gbps  # pyrefly: ignore [bad-assignment]
    _DRAM_PATCHED = True
    logger.info(
        "[NPU Runtime Estimation Patch] using %s GB/s DRAM bandwidth fallback",
        _NPU_DRAM_GBPS,
    )


def _patch_standalone_compile_deepcopy() -> None:
    global _STANDALONE_COMPILE_PATCHED
    import torch._inductor as inductor

    if _STANDALONE_COMPILE_PATCHED:
        return

    original = inductor.standalone_compile

    def standalone_compile_npu(*args, **kwargs):
        kwargs.setdefault("donate_graph_module", True)
        return original(*args, **kwargs)

    inductor.standalone_compile = standalone_compile_npu
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
