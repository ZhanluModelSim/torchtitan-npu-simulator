# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sole owner for patching PyTorch composable FSDP ``foreach_reduce``.

PyTorch's ``_fsdp_param_group`` binds ``foreach_reduce`` by value at import time,
so both ``_fsdp_collectives.foreach_reduce`` and
``_fsdp_param_group.foreach_reduce`` must be updated together. Centralizing all
writes here keeps multiple NPU features (CPU-offload gradient accumulation and
compute-stream RS buffer allocation) composable regardless of activation order
or repeated ``apply()`` / ``install()`` calls, and restores the untouched
upstream function whenever no layer is active.

Importing this module has no side effects.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed.fsdp._fully_shard._fsdp_param_group
from torch.distributed.fsdp._fully_shard import _fsdp_collectives

if TYPE_CHECKING:
    from collections.abc import Callable

    ForeachReduceFn = Callable[..., Any]
    ForeachReduceLayerFactory = Callable[[ForeachReduceFn], ForeachReduceFn]

# Fixed execution order from outermost wrapper to innermost wrapper -> upstream:
# - ``cpu_offload``: handles CPU-canonical offloaded parameter groups directly
#   and delegates all other groups to the next inner layer.
# - ``compute_stream``: redirects reduce-scatter output allocation to the
#   compute stream while preserving native communication streams.
LAYER_ORDER: tuple[str, ...] = ("cpu_offload", "compute_stream")

ORIGINAL_FOREACH_REDUCE: ForeachReduceFn = _fsdp_collectives.foreach_reduce
FOREACH_REDUCE_SIGNATURE = inspect.signature(ORIGINAL_FOREACH_REDUCE)

_ACTIVE_LAYERS: dict[str, ForeachReduceLayerFactory] = {}


def _sync() -> ForeachReduceFn:
    target: ForeachReduceFn = ORIGINAL_FOREACH_REDUCE
    for name in reversed(LAYER_ORDER):
        factory = _ACTIVE_LAYERS.get(name)
        if factory is not None:
            target = factory(target)
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce = target
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = target
    return target


def set_layer(name: str, factory: ForeachReduceLayerFactory | None) -> ForeachReduceFn:
    """Register, replace, or remove one named ``foreach_reduce`` wrapper layer."""
    if name not in LAYER_ORDER:
        raise ValueError(f"Unknown foreach_reduce layer {name!r}; expected one of {LAYER_ORDER}")
    if factory is None:
        _ACTIVE_LAYERS.pop(name, None)
    else:
        _ACTIVE_LAYERS[name] = factory
    return _sync()


def is_layer_active(name: str) -> bool:
    """Return whether the named ``foreach_reduce`` layer is currently active."""
    return name in _ACTIVE_LAYERS
