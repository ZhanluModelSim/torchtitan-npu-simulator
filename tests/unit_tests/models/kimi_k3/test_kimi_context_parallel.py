# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real Gloo validation of Kimi's full-context CP attention protocol."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


def _mla_context_worker(rank: int, rendezvous: str):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.set_num_threads(1)
        config = kimi_k3_debugmodel().model_spec.model.layers[3].attention
        with torch.random.fork_rng():
            torch.manual_seed(997)
            attention = config.build()
            attention.init_states()
            attention = attention.double()
            full_x = torch.randn(2, 16, 256, dtype=torch.float64)
        reference = copy.deepcopy(attention)
        attention._cp_group = dist.group.WORLD
        attention._cp_rank, attention._cp_size = rank, 2
        local_x = full_x.chunk(2, dim=1)[rank].contiguous().requires_grad_()
        ref_x = full_x.clone().requires_grad_()
        full_dy = torch.linspace(-0.7, 1.3, full_x.numel(), dtype=torch.float64).reshape_as(full_x)
        local_dy = full_dy.chunk(2, dim=1)[rank].contiguous()

        output = attention(local_x)
        expected = reference(ref_x)
        output.backward(local_dy)
        expected.backward(full_dy)

        torch.testing.assert_close(output, expected.chunk(2, dim=1)[rank], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(local_x.grad, ref_x.grad.chunk(2, dim=1)[rank], rtol=1e-10, atol=1e-11)
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(
                parameter.grad, reference.get_parameter(name).grad, rtol=1e-10, atol=1e-11, msg=name
            )
    finally:
        dist.destroy_process_group()


def test_mla_context_gather_local_slice_matches_full_sequence_output_and_all_gradients(tmp_path):
    mp.spawn(_mla_context_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
