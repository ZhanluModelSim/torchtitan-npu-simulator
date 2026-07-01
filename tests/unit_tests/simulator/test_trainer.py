# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan_npu.simulator.trainer import run_simulation_step


@pytest.fixture
def fake_world():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29913"
    dist.init_process_group("fake", rank=0, world_size=4)
    yield
    dist.destroy_process_group()


def _build_parallel_dims(world_size: int):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=world_size, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=world_size
    )
    parallel_dims.build_mesh()
    return parallel_dims


def test_run_simulation_step_produces_complete_workload_graph(fake_world):
    parallel_dims = _build_parallel_dims(4)

    model = nn.Linear(8, 8, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        pred = model(input_dict["input"])
        loss = pred.sum() / global_valid_tokens
        loss.backward()
        return loss

    input_dict = {"input": torch.randn(2, 8, device="meta")}
    labels = torch.randint(0, 10, (2, 8), device="meta")

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict=input_dict,
        labels=labels,
        optimizer_step=optimizer.step,
        lr_scheduler_step=lr_scheduler.step,
        local_batch_size=2,
        seq_len=8,
    )

    assert graph.num_iterations == 1
    assert "forward" in graph.step_templates
    assert "backward" in graph.step_templates
    schedule = graph.iteration.schedule
    assert len(schedule.instances) == 4  # world_size
    assert schedule.annotations["rank_table"]["world_size"] == 4


def test_run_simulation_step_captures_optimizer_phase(fake_world):
    parallel_dims = _build_parallel_dims(4)
    model = nn.Linear(4, 4, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        loss = model(input_dict["input"]).sum() / global_valid_tokens
        loss.backward()
        return loss

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict={"input": torch.randn(2, 4, device="meta")},
        labels=torch.randint(0, 10, (2, 4), device="meta"),
        optimizer_step=optimizer.step,
        lr_scheduler_step=lambda: None,
        local_batch_size=2,
        seq_len=4,
    )
    assert "optimizer" in graph.step_templates
    optimizer_ops = graph.step_templates["optimizer"].nodes
    assert len(optimizer_ops) > 0


def test_simulation_trainer_config_build_dispatches_to_simulation_trainer():
    # Regression test for the Configurable._owner auto-wiring mechanism
    # this design relies on (verified against the pinned torchtitan
    # source): `SimulationTrainerConfig().build()` must construct a
    # SimulationTrainer, not a plain Trainer, even though
    # `SimulationTrainer.Config = SimulationTrainerConfig` uses simple
    # attribute assignment rather than nested `class Config:` syntax.
    from torchtitan_npu.simulator.trainer import SimulationTrainerConfig

    assert SimulationTrainerConfig._owner.__name__ == "SimulationTrainer"
