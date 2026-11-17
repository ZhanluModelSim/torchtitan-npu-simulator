# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two-rank FX replay checks for the chunk-loss RS coalescing patch."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.experiments.graph_trainer.common_utils import accumulate_param_grads_, compute_annotated_loss
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

from torchtitan_npu.models.deepseek_v4.config_registry import graph_trainer_deepseek_v4_flash_43layers_16experts

pytestmark = pytest.mark.cpu


def _dense_reference(hidden, labels, weight, num_chunks, valid_tokens):
    hidden = hidden.detach().clone().requires_grad_()
    weight = weight.detach().clone().requires_grad_()
    losses = []
    for rank_hidden, rank_labels in zip(hidden, labels, strict=True):
        # Separate compute-dtype casts accumulate into an FP32 master gradient;
        # this reference uses no chunk-loss wrapper or distributed collectives.
        chunk_losses = [
            F.cross_entropy(
                F.linear(h, weight.to(hidden.dtype)).float().flatten(0, 1),
                target.flatten(),
                reduction="sum",
                ignore_index=-100,
            )
            / valid_tokens
            for h, target in zip(
                rank_hidden.chunk(num_chunks, dim=1), rank_labels.chunk(num_chunks, dim=1), strict=True
            )
        ]
        losses.append(torch.stack(chunk_losses).sum())
    hidden_grad, weight_grad = torch.autograd.grad(torch.stack(losses).sum(), (hidden, weight))
    return losses, hidden_grad, weight_grad


def _assert_coalesced_microbatch_gradients(rank, rendezvous, dtype):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(13, 4, generator=generator) * 0.3
        hidden = torch.randn(2, 1, 16, 4, generator=generator).to(dtype)
        labels = torch.randint(13, (2, 1, 16), generator=generator)
        labels[0, 0, 1] = -100
        labels[1, 0, 12] = -100
        valid_tokens = (labels != -100).sum()
        cfg = graph_trainer_deepseek_v4_flash_43layers_16experts()
        cfg.loss.num_chunks = 8
        loss_fn = cfg.loss.build(compile_config=None)
        head = torch.nn.Linear(4, 13, bias=False)
        with torch.no_grad():
            head.weight.copy_(weight)
        data_parallel(
            head,
            mesh,
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
        )
        parameter = head._parameters["weight"]
        loss_fn.set_lm_head(head)

        def step(local_hidden, targets, token_count):
            loss = compute_annotated_loss(loss_fn, local_hidden, targets, {"global_valid_tokens": token_count})
            grads = torch.autograd.grad(loss * 0.25, (local_hidden, head._parameters["weight"]))
            return loss, *grads

        traced = minimal_fx_tracer(step, module=head)(
            hidden[rank].detach().requires_grad_(),
            labels[rank],
            valid_tokens,
        )
        # The loss-level patch must coalesce RS before any graph optimization.
        actual_step = run_traced(traced, module=head)
        assert parameter.grad is None

        optimizer = torch.optim.SGD(head.parameters(), lr=0.1, foreach=False, fused=False)
        parameter.grad = torch.full_like(parameter, 0.125)
        # The patch must preserve both an existing gradient and a fresh step.
        for cycle in range(2):
            expected_accumulated = torch.full_like(parameter.to_local(), 0.125 if cycle == 0 else 0.0)
            for scale in (1.0, 0.7, 1.3):
                current_hidden = hidden * scale
                expected_loss, expected_hidden_grad, expected_weight_grad = _dense_reference(
                    current_hidden,
                    labels,
                    weight,
                    8,
                    valid_tokens,
                )
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    loss, hidden_grad, weight_grad = actual_step(
                        current_hidden[rank].detach().requires_grad_(),
                        labels[rank],
                        valid_tokens,
                    )
                rs_count = sum(
                    event.count
                    for event in prof.key_averages()
                    if event.key == "_c10d_functional::reduce_scatter_tensor"
                )
                assert rs_count == 1
                torch.testing.assert_close(loss, expected_loss[rank], rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(hidden_grad, expected_hidden_grad[rank] * 0.25, rtol=2e-6, atol=2e-7)
                local_grad = expected_weight_grad.chunk(2, dim=0)[rank] * 0.25
                torch.testing.assert_close(weight_grad.to_local(), local_grad, rtol=3e-6, atol=3e-7)
                assert loss_fn.lm_head is head
                # Producing explicit gradients must not clear or mutate live .grad.
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad.to_local(), expected_accumulated, rtol=3e-6, atol=3e-7)
                else:
                    assert torch.count_nonzero(expected_accumulated) == 0

                accumulate_param_grads_((parameter,), (weight_grad,))
                expected_accumulated += local_grad
                torch.testing.assert_close(parameter.grad.to_local(), expected_accumulated, rtol=3e-6, atol=3e-7)

            optimizer.step()
            torch.testing.assert_close(
                parameter.to_local(),
                weight.chunk(2, dim=0)[rank] - 0.1 * expected_accumulated,
            )
            weight = parameter.full_tensor().detach().clone()
            optimizer.zero_grad(set_to_none=True)
            assert parameter.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_coalesced_chunk_loss_uses_one_rs_and_preserves_microbatch_gradients(tmp_path, dtype):
    mp.start_processes(
        _assert_coalesced_microbatch_gradients,
        args=(str(tmp_path / "rendezvous"), dtype),
        nprocs=2,
        join=True,
        start_method="fork",
    )
