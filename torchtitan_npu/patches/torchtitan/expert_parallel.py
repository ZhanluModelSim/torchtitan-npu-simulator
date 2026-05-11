# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/distributed/expert_parallel.py

The NPU GMM converter fuses w1+w3 into w13 and sets w1/w3 to None.
The upstream TensorParallel._partition_fn and ExpertTensorParallel._partition_fn
unconditionally access module.w1 / module.w3, which crashes when those are None.
This patch adds None-guards and w13 sharding support.
"""

import torch
import torch.nn as nn

import torchtitan.distributed.expert_parallel as ep_module
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Shard


def _distribute_w13_interleaved(w13_param, device_mesh):
    """Distribute w13=[w1|w3] so each rank gets [w1_shard|w3_shard].

    Shard(1) on w13 would give contiguous chunks that break the w1|w3
    boundary (e.g. rank 0 gets only w1 data, rank N-1 gets only w3 data).
    npu_swiglu expects [w1_out|w3_out] on every rank, so we must split,
    shard separately, then re-concatenate local shards.
    """
    if isinstance(w13_param, DTensor):
        w13_full = w13_param.full_tensor()
    else:
        w13_full = w13_param.data if isinstance(w13_param, nn.Parameter) else w13_param

    w1_full, w3_full = torch.chunk(w13_full, 2, dim=1)
    w1_dt = distribute_tensor(w1_full, device_mesh, [Shard(1)])
    w3_dt = distribute_tensor(w3_full, device_mesh, [Shard(1)])
    w1_local = w1_dt.to_local()
    w3_local = w3_dt.to_local()
    w13_local = torch.cat([w1_local, w3_local], dim=1)
    return DTensor.from_local(w13_local, device_mesh, [Shard(1)], run_check=False)


def _tp_partition_fn(self, name, module, device_mesh):
    # w1 has shape (experts, out_dim, in_dim)
    if module.w1 is not None:
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )

    # w2 has shape (experts, in_dim, out_dim)
    module.register_parameter(
        "w2",
        nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
    )

    # w3 has shape (experts, out_dim, in_dim)
    if module.w3 is not None:
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )

    # w13 has shape (experts, out_dim*2, in_dim), fused w1 and w3 for GMM
    if getattr(module, "w13", None) is not None:
        w13_dt = _distribute_w13_interleaved(module.w13, device_mesh)
        module.register_parameter("w13", nn.Parameter(w13_dt))


def _distribute_w13_interleaved_etp(w13_param, device_mesh):
    """Distribute w13=[w1|w3] under ETP with [Shard(0), Shard(1)] placements.

    Same w1|w3 boundary issue as _distribute_w13_interleaved but for 2D mesh.
    """
    if isinstance(w13_param, DTensor):
        w13_full = w13_param.full_tensor()
    else:
        w13_full = w13_param.data if isinstance(w13_param, nn.Parameter) else w13_param

    w1_full, w3_full = torch.chunk(w13_full, 2, dim=1)
    w1_dt = distribute_tensor(w1_full, device_mesh, [Shard(0), Shard(1)])
    w3_dt = distribute_tensor(w3_full, device_mesh, [Shard(0), Shard(1)])
    w1_local = w1_dt.to_local()
    w3_local = w3_dt.to_local()
    w13_local = torch.cat([w1_local, w3_local], dim=1)
    return DTensor.from_local(
        w13_local, device_mesh, [Shard(0), Shard(1)], run_check=False
    )


def _etp_partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
    if mod.w1 is not None:
        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)])),
        )

    mod.register_parameter(
        "w2",
        nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)])),
    )

    if mod.w3 is not None:
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)])),
        )

    if getattr(mod, "w13", None) is not None:
        w13_dt = _distribute_w13_interleaved_etp(mod.w13, device_mesh)
        mod.register_parameter("w13", nn.Parameter(w13_dt))


_PARTITION_FN = "_partition_fn"
setattr(ep_module.TensorParallel, _PARTITION_FN, _tp_partition_fn)
if hasattr(ep_module, "ExpertTensorParallel"):
    setattr(ep_module.ExpertTensorParallel, _PARTITION_FN, _etp_partition_fn)
