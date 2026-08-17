# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt TorchTitan's RL batch-invariant switch for Ascend NPU."""

import os

import torch
import torchtitan.distributed.utils as dist_utils


def _set_batch_invariance(enable: bool) -> None:
    if not enable or dist_utils.is_in_batch_invariant_mode():
        return

    npu_module = getattr(torch, "npu", None)
    set_deterministic_level = getattr(npu_module, "set_deterministic_level", None)
    if not callable(set_deterministic_level):
        raise RuntimeError(
            "NPU batch_invariant requires torch.npu.set_deterministic_level(3). "
            "Please install a compatible CANN & torch_npu version."
        )

    os.environ["HCCL_DETERMINISTIC"] = "strict"
    set_deterministic_level(3)


dist_utils.set_batch_invariance = _set_batch_invariance
