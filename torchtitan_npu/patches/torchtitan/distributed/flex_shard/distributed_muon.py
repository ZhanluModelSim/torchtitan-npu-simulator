# Pending upstream issue: https://github.com/pytorch/torchtitan/issues/2183
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Adapt DistributedMuon storage admission from CUDA to NPU.

TorchTitan's upstream check currently admits CUDA storage only. This temporary
patch changes that backend gate to NPU while preserving the DTensor and
one-device-per-process invariants.

Remove this module after the TorchTitan dependency includes the PR.
"""

import torch
import torchtitan.distributed.flex_shard.distributed_muon
from torch.distributed.tensor import DTensor


def _validate_parameter_storage(self) -> torch.device:
    local_devices = set()
    for group in self.param_groups:
        for param in group["params"]:
            if not isinstance(param, DTensor):
                raise TypeError("DistributedMuon requires DTensor parameters")
            local_device = param.to_local().device
            if local_device.type != "npu":
                raise ValueError("DistributedMuon requires NPU parameters")
            local_devices.add(local_device)
    if len(local_devices) != 1:
        raise ValueError("DistributedMuon requires one device per process")
    return local_devices.pop()


def apply() -> None:
    torchtitan.distributed.flex_shard.distributed_muon.DistributedMuon._validate_parameter_storage = (
        _validate_parameter_storage
    )


apply()
