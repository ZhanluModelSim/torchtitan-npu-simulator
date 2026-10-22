# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torchtitan_simulator.runtime import meta_runtime
from torchtitan_simulator.step import run_simulation_step


def test_meta_linear_adamw_step_builds_graphs_and_local_parameter_memory():
    model = torch.nn.Linear(3, 2, bias=False, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    inputs = torch.ones(4, 3, device="meta", requires_grad=True)
    dims = SimpleNamespace(
        world_size=1,
        pp=1,
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        ep=1,
        pp_enabled=False,
        dp_enabled=False,
    )

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        del labels
        loss = model(input_dict["input"]).sum() / global_valid_tokens
        loss.backward()
        return loss

    with meta_runtime():
        workload = run_simulation_step(
            model_parts=[model],
            parallel_dims=dims,
            forward_backward_step=forward_backward_step,
            input_dict={"input": inputs},
            labels=torch.zeros(4, device="meta"),
            optimizer_step=optimizer.step,
            lr_scheduler_step=lambda: None,
            local_batch_size=4,
            seq_len=1,
            local_valid_tokens=4,
        )

    assert inputs.grad is not None and inputs.grad.shape == (4, 3)
    assert model.weight.grad is not None and model.weight.grad.shape == (2, 3)
    assert set(workload.step_templates) == {"s0_F", "s0_B", "s0_OPTIMIZER"}
    assert all(template.nodes for template in workload.step_templates.values())
    memory = workload.iteration.schedule.annotations["memory_plan"]
    assert memory.persistent_param_bytes == 24
    assert memory.peak_active_bytes >= 24
