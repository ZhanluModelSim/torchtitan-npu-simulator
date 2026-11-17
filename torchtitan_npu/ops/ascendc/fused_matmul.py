# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
import torch.library

from torchtitan_npu.override import _IS_A5

_DISPATCH_KEY = "PrivateUse1"

cann_ops_nn: Any = None
if _IS_A5:
    cann_ops_nn = import_module("cann_ops_nn")


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    # 连续 tensor 直接返回，不触发 aten.contiguous
    return x if x.is_contiguous() else x.contiguous()


def _npu_baddbmm(
    self: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta=1,
    alpha=1,
) -> torch.Tensor:
    if self.dim() == 3:
        batch1 = _maybe_contiguous(batch1)
        batch2 = _maybe_contiguous(batch2)
        return cann_ops_nn.fused_matmul(batch1, batch2, x3=self, alpha=alpha, beta=beta, fused_op_type="add")
    result = torch.bmm(batch1, batch2).mul_(alpha)
    if beta != 0:
        result.add_(self, alpha=beta)
    return result


def _register_impl(op_name: str, fn) -> None:
    try:
        torch.library.impl(op_name, _DISPATCH_KEY)(fn)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to register {op_name!r} for dispatch key {_DISPATCH_KEY!r}") from exc


if _IS_A5:
    _register_impl("aten::baddbmm", _npu_baddbmm)
