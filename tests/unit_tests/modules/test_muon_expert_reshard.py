# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical regression for Muon expert matrix shards using real CPU collectives."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from tests.testing.muon_reshard import check_expert_updates
from torchtitan_npu.patches.optimizer.muon_optimizer import (
    ParamType,
    _expert_matrix_shard_axis,
    get_param_type,
)


def _distributed_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("efsdp", "ep"))
        check_expert_updates(mesh, (Shard(1), Shard(0)), (4, 6, 5))
        # Fewer experts than EFSDP ranks, uneven matrix shards and BF16 transport.
        check_expert_updates(mesh, (Shard(1), Shard(0)), (2, 5, 7), transpose=True, dtype=torch.bfloat16)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep", "efsdp"))
        check_expert_updates(mesh, (Shard(0), Shard(1)), (6, 5, 7))
        mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("fsdp",))
        # EP=1, including an empty row shard and empty compute-owner ranks.
        check_expert_updates(mesh, (Shard(1),), (1, 3, 5))
        # Complete per-expert matrices retain the local fast path, also at EP=1.
        check_expert_updates(mesh, (Shard(0),), (5, 6, 4))
        mesh = init_device_mesh("cpu", (2, 2, 1), mesh_dim_names=("dp_replicate", "efsdp", "ep"))
        check_expert_updates(mesh, (Replicate(), Shard(1), Shard(0)), (3, 5, 7))

        invalid_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("etp",))
        parameter = distribute_tensor(torch.ones(4, 8, 5), invalid_mesh, (Shard(1),))
        with pytest.raises(ValueError, match="expert tensor parallelism"):
            _expert_matrix_shard_axis(parameter, "moe.experts.w1")
    finally:
        dist.destroy_process_group()


def test_expert_alltoall_matches_unsharded_muon(tmp_path):
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)


def test_expert_classification_does_not_require_ep():
    parameter = torch.nn.Parameter(torch.ones(2, 3, 4))
    assert get_param_type(parameter, "moe.experts.w1", True, False) == ParamType.Expert
