# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit float32 projection policy for Meta router autocast regions."""

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


def _float32(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(torch.float32)
    if isinstance(value, tuple):
        return tuple(_float32(item) for item in value)
    if isinstance(value, list):
        return [_float32(item) for item in value]
    if isinstance(value, dict):
        return {key: _float32(item) for key, item in value.items()}
    return value


class _Float32Projections(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _PROJECTIONS:
            args, kwargs = _float32(args), _float32(kwargs)
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
        return _Float32Projections()

    scope.replace(torch, "autocast", autocast)
