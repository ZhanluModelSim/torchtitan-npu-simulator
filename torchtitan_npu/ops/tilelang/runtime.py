# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy TileLang runtime and source-aware kernel cache helpers."""

from __future__ import annotations

import hashlib
import importlib
import os
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import torch


def kernel_source_fingerprint() -> str:
    """Invalidate compiled binaries whenever a vendored kernel changes."""

    kernel_dir = Path(__file__).parent
    digest = hashlib.sha256()
    for filename in ("moe_reduce_fused.py", "moe_reduce_fused_bwd.py"):
        path = kernel_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def configure_tilelang_cache() -> Path:
    """Give every rank and kernel source revision an independent disk cache."""

    cache_root = Path(
        os.getenv(
            "TORCHTITAN_NPU_TILELANG_CACHE_ROOT",
            "/tmp/torchtitan_npu_tilelang_cache",
        )
    )
    job_id = os.getenv("TORCHELASTIC_RUN_ID") or os.getenv("MASTER_PORT") or "default"
    rank_id = os.getenv("RANK") or f"pid_{os.getpid()}"
    source_id = kernel_source_fingerprint()
    cache_dir = cache_root / job_id / f"source_{source_id}" / f"rank_{rank_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TileLang reads this at import time. The setter below also covers an
    # environment where another component imported TileLang first.
    os.environ["TILELANG_CACHE_DIR"] = str(cache_dir)
    return cache_dir


@lru_cache(maxsize=1)
def lazy_load_tilelang() -> tuple[ModuleType, ModuleType]:
    """Load TileLang and the vendored kernels only after the opt-in is used."""

    cache_dir = configure_tilelang_cache()
    tilelang = importlib.import_module("tilelang")
    tilelang.cache.set_cache_dir(str(cache_dir))
    forward_module = importlib.import_module("torchtitan_npu.ops.tilelang.moe_reduce_fused")
    backward_module = importlib.import_module("torchtitan_npu.ops.tilelang.moe_reduce_fused_bwd")
    return forward_module, backward_module


@cache
def get_cached_forward_kernel(hidden: int, num_topk: int, dtype: torch.dtype):
    forward_module, _ = lazy_load_tilelang()
    return forward_module.get_reduce_fused_kernel(
        hidden=hidden,
        num_topk=num_topk,
        in_dtype=dtype,
        out_dtype=dtype,
        with_sf=False,
        with_weights=False,
        with_x_sf=False,
    )


@cache
def get_cached_backward_kernel(hidden: int, num_topk: int, dtype: torch.dtype):
    _, backward_module = lazy_load_tilelang()
    return backward_module.get_reduce_fused_backward_kernel(
        hidden=hidden,
        num_topk=num_topk,
        in_dtype=dtype,
        out_dtype=dtype,
        with_sf=False,
        with_weights=False,
        with_x_sf=False,
    )


def require_raw_storage(tensor: torch.Tensor, name: str) -> None:
    """Reject an opaque tensor instead of silently introducing a device copy."""

    if tensor.numel() == 0:
        return
    try:
        data_ptr = tensor.data_ptr()
    except Exception as error:
        raise RuntimeError(
            f"TileLang custom-op zero-copy path cannot access {name}: {type(error).__name__}: {error}"
        ) from error
    if data_ptr == 0:
        raise RuntimeError(f"TileLang custom-op zero-copy path received a null data pointer for {name}")
