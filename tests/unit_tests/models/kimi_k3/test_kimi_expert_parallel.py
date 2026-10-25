# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi latent MoE composition over real two-rank Gloo all-to-all."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.distributed.utils import set_spmd_backend

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


def _cpu_grouped_mm(*, A, B_t, offs):
    # Keep the production BF16 inputs/outputs; only the unavailable grouped
    # CPU primitive is expressed with ordinary per-expert matrix multiplies.
    starts = [0, *offs.tolist()]
    return torch.cat([A[starts[e] : starts[e + 1]] @ B_t[e] for e in range(B_t.shape[0])])


def _latent_ep_worker(rank: int, rendezvous: str):
    set_spmd_backend("partial_dtensor")
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(313)
            config = kimi_k3_debugmodel().model_spec.model.layers[1].moe
            config.router._debug_force_load_balance = False
            module = config.build()
            module.init_states()
            full_x = torch.randn(2, 8, 256)
        reference = copy.deepcopy(module)
        reference.routed_experts.inner_experts._grouped_mm = _cpu_grouped_mm
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        module.routed_experts.token_dispatcher.wire_meshes(ep_mesh=mesh)
        module.routed_experts.token_dispatcher.sp_size = 1
        experts = module.routed_experts.inner_experts
        experts._grouped_mm = _cpu_grouped_mm
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            parameter = getattr(experts, name)
            setattr(experts, name, torch.nn.Parameter(parameter.detach().chunk(2, dim=0)[rank].clone()))
        local_x = full_x[rank : rank + 1].clone().requires_grad_()
        ref_x = full_x.clone().requires_grad_()
        full_dy = torch.linspace(-0.3, 0.7, full_x.numel()).reshape_as(full_x)

        output = module(local_x)
        expected = reference(ref_x)
        output.backward(full_dy[rank : rank + 1])
        expected.backward(full_dy)

        torch.testing.assert_close(output, expected[rank : rank + 1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(local_x.grad, ref_x.grad[rank : rank + 1], rtol=0.03, atol=8e-4)
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            expected_grad = reference.get_parameter(name).grad
            if ".inner_experts." in name:
                expected_grad = expected_grad.chunk(2, dim=0)[rank]
            else:
                dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=0.05, atol=1e-3, msg=name)
    finally:
        dist.destroy_process_group()


def test_latent_moe_ep_dispatch_matches_unsharded_output_input_and_all_parameter_gradients(tmp_path):
    mp.spawn(_latent_ep_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
