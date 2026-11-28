# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture-compatibility probe for the optional TorchAO-NPU MX path.

Meta capture prices quantized training by observing the dispatched ops: the
MX quantized matmul must route to ``npu.npu_quant_matmul`` for the captured
workload to carry the quantized operators the tests (and the cost model)
assert on. Some torchao-npu builds fall back to the plain ``aten.mm`` path
under meta dispatch instead, which silently degrades every mxfp8 capture to
its unquantized spelling. ``mx_capture_routes_to_quant_matmul`` detects that
with a one-shot smoke capture so tests can skip instead of asserting against
a build that cannot honor them.
"""

from __future__ import annotations

import functools


@functools.cache
def mx_capture_routes_to_quant_matmul() -> bool:
    """Return whether an MXTensor matmul dispatches to ``npu_quant_matmul``.

    The probe runs once per process on meta tensors (no device work). A
    torchao-npu build that cannot import, or that degrades the matmul to
    ``aten.mm``, reports ``False``.
    """
    try:
        import torch
        from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig
        from torchao_npu.quantized_tensors.mx_tensor import MXTensor
    except ImportError:
        return False

    ops: list[str] = []

    class _Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(str(func))
            return func(*args, **(kwargs or {}))

    config = BlockMXQuantizeConfig(block_size=32)
    try:
        with _Recorder():
            lhs = MXTensor.from_hp(torch.randn(8, 32, device="meta", dtype=torch.bfloat16), config, axis=-1)
            rhs = MXTensor.from_hp(torch.randn(32, 8, device="meta", dtype=torch.bfloat16), config, axis=-2)
            lhs @ rhs
    except Exception:
        return False
    return any("quant_matmul" in op for op in ops)
