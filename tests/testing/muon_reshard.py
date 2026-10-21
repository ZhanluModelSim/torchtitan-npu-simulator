# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU/NPU numerical oracle for expert Muon redistribution."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor

from torchtitan_npu.patches.optimizer.muon_optimizer import DistributedMuon, _expert_matrix_shard_axis


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
