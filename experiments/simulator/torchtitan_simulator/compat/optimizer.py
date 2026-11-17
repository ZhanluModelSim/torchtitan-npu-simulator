# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Permit Meta operands while preserving native optimizer dispatch."""

from functools import wraps
from importlib import import_module


def install_meta_optimizer(scope):
    """Allow native fused optimizers on storage-free Meta operands."""
    optimizer = import_module("torch.optim.optimizer")
    original_check = optimizer._device_dtype_check_for_fused

    @wraps(original_check)
    def check(parameter, *args, **kwargs):
        if parameter.device.type == "meta" and parameter.is_floating_point():
            return
        return original_check(parameter, *args, **kwargs)

    for module_name in ("torch.optim.optimizer", "torch.optim.adam"):
        owner = import_module(module_name)
        if getattr(owner, "_device_dtype_check_for_fused", None) is original_check:
            scope.replace(owner, "_device_dtype_check_for_fused", check)
