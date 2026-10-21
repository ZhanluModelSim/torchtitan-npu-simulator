# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from tests.testing.muon_reshard import check_expert_updates

pytestmark = pytest.mark.smoke


def _npu_worker(rank, init_method):
    torch.npu.set_device(rank)
    torch.npu.matmul.allow_hf32 = False
    dist.init_process_group("hccl", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("fsdp",))
        check_expert_updates(mesh, (Shard(1),), (4, 8, 16))
        check_expert_updates(mesh, (Shard(1),), (1, 5, 7), transpose=True, dtype=torch.bfloat16)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group()


def test_expert_matrix_reshard_on_two_npus(tmp_path):
    if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
        pytest.skip("requires two real NPU devices")
    mp.spawn(_npu_worker, args=(f"file://{tmp_path / 'hccl_store'}",), nprocs=2, join=True)
