# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom op used by the static SDC checksum graph pass."""

import torch
import torch_npu
from torch._library.effects import EffectType


# The torch-npu checksum recomputation may update ``output`` in place, so compiled
# consumers must see that aliasing instead of treating this as a pure observer.
@torch.library.custom_op(
    "torchtitan_npu::sdc_matmul_checksum",
    mutates_args={"output"},
    device_types="npu",
)
def sdc_matmul_checksum(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
) -> None:
    checker = torch_npu.asd.asd.matmul_check
    # The op stays in the compiled graph; gradient strikes open this torch-npu gate
    # at runtime, avoiding graph mutation and recompilation.
    if checker.checksum_enable:
        # Preserve failures from every instrumented matmul in the graph.
        checker.checksum_result.logical_or_(torch_npu.matmul_checksum(left, right, output))


@sdc_matmul_checksum.register_fake
def _sdc_matmul_checksum_fake(
    _left: torch.Tensor,
    _right: torch.Tensor,
    _output: torch.Tensor,
) -> None:
    return None


# A side-effect-only checksum must not be dropped or reordered by Inductor.
sdc_matmul_checksum.register_effect(EffectType.ORDERED)
