# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit float32 projection policy for Meta router autocast regions.

Real-NPU profiling confirms the reference model runs with an fp32 main chain
(linear/mm/bmm compute in fp32) and bf16 only inside the MoE path where the
model code casts explicitly before the grouped GEMM. Meta capture therefore
only needs the reference's own ``autocast(dtype=float32)`` router regions;
there is no outer bf16 autocast to simulate. The mode still reads the innermost
region from a stack so nested regions behave like torch autocast.
"""

from contextlib import nullcontext
from functools import wraps

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

_PROJECTIONS = {
    F.linear,
    torch.mm,
    torch.matmul,
    torch.bmm,
    torch.addmm,
    torch.Tensor.mm,
    torch.Tensor.matmul,
    torch.Tensor.bmm,
    torch.Tensor.addmm,
}

_CAST_STACK: list[torch.dtype] = []


def _cast(value, dtype):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value if value.dtype == dtype else value.to(dtype)
    if isinstance(value, tuple):
        return tuple(_cast(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast(item, dtype) for key, item in value.items()}
    return value


class _ProjectionCast(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _PROJECTIONS and _CAST_STACK:
            args, kwargs = _cast(args, _CAST_STACK[-1]), _cast(kwargs, _CAST_STACK[-1])
        return func(*args, **kwargs)


def install_meta_autocast(scope):
    original = torch.autocast

    @wraps(original)
    def autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type != "meta":
            return original(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        if not enabled:
            return nullcontext()
        if dtype != torch.float32:
            raise ValueError("Meta autocast currently supports explicit float32 router projections only")
        # Casting at the Python projection boundary keeps cast autograd nodes.
        # A dispatcher-only cast would occur below autograd and lose those edges.
        mode = _ProjectionCast()

        class _Region:
            def __enter__(self):
                _CAST_STACK.append(dtype)
                mode.__enter__()

            def __exit__(self, *exc):
                mode.__exit__(*exc)
                _CAST_STACK.pop()

        return _Region()

    scope.replace(torch, "autocast", autocast)


def install_meta_checkpoint_recompute_cast(scope):
    """Re-apply the active projection cast around non-reentrant checkpoint recompute.

    The autograd engine clears the torch function mode stack while it runs
    backward nodes, so a recompute triggered from an unpack hook would lose an
    active projection region. No-op unless a region is active on the calling
    thread; today only the model's own fp32 router region can be.
    """
    import torch.utils.checkpoint as checkpoint_module

    def run_fn_with_cast(fn, *args, **kwargs):
        if _CAST_STACK:
            dtype = _CAST_STACK[-1]
            mode = _ProjectionCast()
            mode.__enter__()
            _CAST_STACK.append(dtype)
            try:
                return fn(*args, **kwargs)
            finally:
                mode.__exit__(None, None, None)
                _CAST_STACK.pop()
        return fn(*args, **kwargs)

    scope.replace(checkpoint_module, "_run_fn_with_dynamo_disabled", run_fn_with_cast)
