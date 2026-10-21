# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan_npu.patches.optimizer.muon_optimizer import DistributedMuon
from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta, unpatch_device_type_to_meta


def _capture_worker(process_index, tmp_path):
    # DeviceMesh caches are process-global: use a fresh process per logical rank.
    rank = (0, 1, 256, 257)[process_index]
    torch.set_num_threads(1)
    dist.init_process_group("fake", rank=rank, world_size=512, init_method=f"file://{tmp_path}/store_{rank}")
    try:
        patch_device_type_to_meta()
        mesh = DeviceMesh("meta", torch.arange(512).reshape(2, 256), mesh_dim_names=("efsdp", "ep"))

        def expert_tensor():
            return DTensor.from_local(
                torch.empty(1, 8, 8, device="meta"),
                mesh,
                (Shard(1), Shard(0)),
                shape=torch.Size((256, 16, 8)),
                stride=(128, 8, 1),
                run_check=False,
            )

        parameter = torch.nn.Parameter(expert_tensor())
        parameter.grad = expert_tensor()
        parallel_dims = SimpleNamespace(
            world_mesh=mesh,
            fsdp_enabled=True,
            ep_enabled=True,
            tp_enabled=False,
            dp_replicate_enabled=False,
            get_optional_mesh=lambda name: mesh[name] if name in mesh.mesh_dim_names else None,
        )
        optimizer = DistributedMuon([parameter], ["moe.experts.w1"], parallel_dims)
        capture = OpDispatchCapture(phase_provider=lambda: "optimizer")
        with capture, capture_fake_collectives(capture_process_rank=rank) as recorder:
            optimizer.step()

        exchanges = [event for event in recorder.events if event.comm_primitive == "all_to_all"]
        assert len(exchanges) == 2
        assert all(event.world_size == 2 for event in exchanges)
        assert all(event.raw_group_name == mesh.get_group("efsdp").group_name for event in exchanges)
        assert parameter.to_local().shape == (1, 8, 8)
        assert parameter.placements == (Shard(1), Shard(0))
        assert capture.build_nodes()
    finally:
        unpatch_device_type_to_meta()
        dist.destroy_process_group()


def test_ep256_efsdp2_muon_records_both_alltoalls(tmp_path):
    mp.spawn(_capture_worker, args=(str(tmp_path),), nprocs=4, join=True)
