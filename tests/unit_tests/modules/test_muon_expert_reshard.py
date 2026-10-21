# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical regression for Muon expert matrix shards using real CPU collectives."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    DistributedMuon,
    ParamType,
    _expert_matrix_shard_axis,
    get_param_type,
)


def expert_parallel_dims(mesh):
    names = mesh.mesh_dim_names
    return SimpleNamespace(
        world_mesh=mesh,
        fsdp_enabled=True,
        ep_enabled="ep" in names and mesh["ep"].size() > 1,
        tp_enabled=False,
        dp_replicate_enabled="dp_replicate" in names,
        get_optional_mesh=lambda name: mesh[name] if name in names else None,
    )


def check_expert_updates(mesh, placements, shape, *, transpose=False, dtype=torch.float32):
    torch.manual_seed(73)
    reference = torch.randn(shape).to(mesh.device_type).requires_grad_()
    parameter = torch.nn.Parameter(distribute_tensor(reference.detach().clone(), mesh, placements))
    optimizer_kwargs = dict(
        lr=0.03,
        weight_decay=0.1,
        momentum=0.8,
        nesterov=not transpose,
        communication_dtype=dtype,
        experts_weights_layout="G-D_in-D_out" if transpose else "G-D_out-D_in",
    )
    optimizer = DistributedMuon([parameter], ["moe.experts.w1"], expert_parallel_dims(mesh), **optimizer_kwargs)
    assert optimizer.expert_params == [parameter]
    momentum = torch.zeros_like(reference)
    matrix_axis = _expert_matrix_shard_axis(parameter, "moe.experts.w1")
    inputs = torch.randn(shape[0], 2, shape[1]).to(mesh.device_type)
    original_alltoall = dist.all_to_all_single

    for step in range(3):
        if step == 2:
            # Resume momentum in its original shard layout before the next update.
            saved_state = deepcopy(optimizer.state_dict())
            optimizer = DistributedMuon([parameter], ["moe.experts.w1"], expert_parallel_dims(mesh), **optimizer_kwargs)
            optimizer.load_state_dict(saved_state)
        loss = torch.bmm(inputs, reference).square().mean()
        loss.backward()
        parameter.grad = distribute_tensor(reference.grad, mesh, placements)
        # Unsharded reference: momentum/NS sees every row of every expert.
        with torch.no_grad():
            momentum.lerp_(reference.grad, 0.2)
            effective = momentum if transpose else reference.grad.add(momentum, alpha=0.8)
            if matrix_axis is not None:
                effective = effective.to(dtype)
            update = DistributedMuon.lmo(
                effective,
                eps=1e-7,
                backend_steps=5,
                transpose_experts=transpose,
                adjust_lr_fn="original",
            ).float()
            reference.mul_(1 - 0.03 * 0.1).add_(update, alpha=-0.03)
        with patch.object(dist, "all_to_all_single", wraps=original_alltoall) as collectives:
            optimizer.step()
        if matrix_axis is None:
            assert collectives.call_count == 0
        else:
            assert collectives.call_count == 2
            expected_group = mesh.get_group(matrix_axis)
            assert all(call.kwargs["group"] is expected_group for call in collectives.call_args_list)
        actual = parameter.full_tensor().detach()
        torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            torch.bmm(inputs, actual).square().mean().cpu(),
            torch.bmm(inputs, reference).square().mean().cpu(),
            rtol=0,
            atol=0,
        )
        assert optimizer.state[parameter]["momentum_buffer"].placements == parameter.placements
        reference.grad = None

    # Missing gradients still participate in EFSDP collectives; no weight decay.
    parameter.grad = None
    optimizer.step()
    torch.testing.assert_close(parameter.full_tensor().cpu(), reference.cpu(), rtol=0, atol=0)


def _distributed_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("efsdp", "ep"))
        # FSDP-over-EP uses strided expert ownership, with complete local matrices.
        check_expert_updates(mesh, (_StridedShard(0, split_factor=2), Shard(0)), (8, 6, 5))
        check_expert_updates(mesh, (_StridedShard(0, split_factor=2), Shard(0)), (12, 5, 7), transpose=True)
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

        # Recognizing strided expert ownership must not enable strided matrix shards.
        invalid_parameter = DTensor.from_local(
            torch.ones(2, 2, 5),
            init_device_mesh("cpu", (4,), mesh_dim_names=("efsdp",)),
            (_StridedShard(1, split_factor=2),),
            shape=torch.Size((2, 8, 5)),
            stride=(40, 5, 1),
            run_check=False,
        )
        with pytest.raises(ValueError, match="expert tensor parallelism"):
            _expert_matrix_shard_axis(invalid_parameter, "moe.experts.w1")
    finally:
        dist.destroy_process_group()


def test_expert_alltoall_matches_unsharded_muon(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2, 3, 4))
    assert get_param_type(parameter, "moe.experts.w1", True, False) == ParamType.Expert
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)
