# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep asynchronous NPU-to-CPU DeviceCopy results synchronized.

PyTorch marks asynchronous D2H outputs pinned only for devices accepted by its
GPU predicate. NPU stays outside that global predicate because other Inductor
paths use it to select Triton. Keep ``prims.device_put`` in the AscendC graph and
mark only asynchronous NPU-to-CPU results pinned so ``DeviceCopy.codegen()``
emits event synchronization before host reads.
"""

import functools
from importlib.machinery import PathFinder
from typing import Any, cast

import torch
import torch_npu
from torch._inductor import ir as inductor_ir

_original_device_copy_create = inductor_ir.DeviceCopy.create


@functools.wraps(_original_device_copy_create)
def _patched_device_copy_create(cls, x, device, non_blocking, *args, **kwargs):
    result = _original_device_copy_create(x, device, non_blocking, *args, **kwargs)
    source_device = x.get_device()
    if (
        source_device is not None
        and source_device.type == "npu"
        and device.type == "cpu"
        and non_blocking
        and isinstance(result, inductor_ir.DeviceCopy)
        and isinstance(result.layout, inductor_ir.Layout)
    ):
        # DeviceCopy.codegen() emits D2H event synchronization for pinned layouts.
        result.layout.is_pinned = True
    return result


def apply() -> None:
    if not torch_npu.npu.is_available():
        return

    # Resolving dotted names with importlib.util.find_spec would initialize
    # _inductor, whose backend loader can fail before an absent ascendc is found.
    path = torch_npu.__path__
    for name in ("_inductor", "ascendc", "lowering", "common"):
        spec = PathFinder.find_spec(name, path)
        if spec is None:
            return
        path = spec.submodule_search_locations
        if name != "common" and path is None:
            return

    from torch_npu._inductor.ascendc.lowering.common import (  # pyrefly: ignore [missing-import]
        _LoweringGuard,
        exclude,
    )

    cast("Any", inductor_ir.DeviceCopy).create = classmethod(_patched_device_copy_create)
    _LoweringGuard.support(
        torch.ops.prims.device_put,
        cast("tuple[torch.dtype]", (*exclude(), torch.int64)),
    )


apply()
