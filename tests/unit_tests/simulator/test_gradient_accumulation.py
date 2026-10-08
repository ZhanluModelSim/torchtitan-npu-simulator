# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan_npu.simulator.meta_env import unpatch_device_type_to_meta
from torchtitan_npu.simulator.trainer import run_simulation_step


def _dims():
    return SimpleNamespace(
        pp_enabled=False, dp_enabled=False, pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, world_size=1
    )


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_gradient_accumulation_executes_real_batches_and_one_optimizer_step(count, device):
    torch.manual_seed(0)
    reference = nn.Linear(4, 2, device="cpu")
    model = nn.Linear(4, 2, device=device)
    model.load_state_dict(reference.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    batches = []
    for mb in range(count):
        labels = torch.tensor([0, 1, -100] if mb == 1 else [0, 1, 0])
        batches.append(({"input": torch.arange(12, dtype=torch.float32).reshape(3, 4) + mb}, labels))
    denominator = float(sum((labels != -100).sum().item() for _, labels in batches))
    for inputs, labels in batches:
        (nn.functional.cross_entropy(reference(inputs["input"]), labels, reduction="sum") / denominator).backward()
    reference_optimizer.step()
    prepared = [({"input": inputs["input"].to(device)}, labels.to(device)) for inputs, labels in batches]
    calls = []
    optimizer_calls = []
    scheduler_calls = []

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        calls.append(global_valid_tokens)
        loss = nn.functional.cross_entropy(model(input_dict["input"]), labels, reduction="sum") / global_valid_tokens
        loss.backward()

    def optimizer_step():
        optimizer_calls.append(True)
        optimizer.step()

    try:
        graph = run_simulation_step(
            model_parts=[model],
            parallel_dims=_dims(),
            forward_backward_step=forward_backward_step,
            input_dict=prepared[0][0],
            labels=prepared[0][1],
            microbatches=prepared,
            global_valid_tokens=denominator,
            optimizer_step=optimizer_step,
            lr_scheduler_step=lambda: scheduler_calls.append(True),
            local_batch_size=3,
            seq_len=4,
            gradient_accumulation=count,
            num_micro_batches=count,
        )
    finally:
        unpatch_device_type_to_meta()
    assert calls == [denominator] * count
    assert len(optimizer_calls) == len(scheduler_calls) == 1
    compute = [action for action in graph.schedule_plan.actions if action.action_type == "COMPUTE"]
    assert [(action.comp_type, action.mb_idx) for action in compute] == [
        (kind, mb) for mb in range(count) for kind in ("F", "B")
    ]
    assert len([action for action in graph.schedule_plan.actions if action.action_type == "OPTIMIZER"]) == 1
    memory = graph.iteration.schedule.annotations["memory_plan"]
    assert {event.pp_mb_idx for event in memory.raw_events if event.phase == "forward"} == set(range(count))
    if device == "cpu":
        for actual, expected in zip(model.parameters(), reference.parameters()):
            assert torch.equal(actual.grad, expected.grad)
            assert torch.equal(actual, expected)


def test_ga_requires_actual_microbatch_inputs():
    model = nn.Linear(4, 2, device="meta")
    with pytest.raises(ValueError, match="microbatches"):
        run_simulation_step(
            model_parts=[model],
            parallel_dims=_dims(),
            forward_backward_step=lambda **_: None,
            input_dict={"input": torch.ones(3, 4, device="meta")},
            labels=torch.ones(3, dtype=torch.int64, device="meta"),
            optimizer_step=lambda: None,
            lr_scheduler_step=lambda: None,
            local_batch_size=3,
            seq_len=4,
            gradient_accumulation=2,
        )


def test_ga_reenters_tp_context_and_clips_accumulated_gradients_once():
    from torchtitan_npu.simulator.capture.comm_events import _default_collective_context

    torch.manual_seed(1)
    model = nn.Linear(4, 2)
    reference = nn.Linear(4, 2)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    batches = [
        ({"input": torch.arange(12, dtype=torch.float32).reshape(3, 4) + mb}, torch.tensor([0, 1, 0]))
        for mb in range(3)
    ]
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    reference_optimizer.zero_grad()
    for inputs, labels in batches:
        (nn.functional.cross_entropy(reference(inputs["input"]), labels, reduction="sum") / 9).backward()
    reference_norm = nn.utils.clip_grad_norm_(reference.parameters(), max_norm=0.25)
    reference_optimizer.step()
    group = object()
    dims = _dims()
    dims.get_optional_mesh = lambda name: SimpleNamespace(get_group=lambda: group) if name == "tp" else None
    calls = []
    norms = []

    def zero_grad():
        calls.append("zero")
        optimizer.zero_grad()

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        assert _default_collective_context.get() == ("tp", group)
        calls.append("batch")
        (
            nn.functional.cross_entropy(model(input_dict["input"]), labels, reduction="sum") / global_valid_tokens
        ).backward()

    def clip():
        calls.append("clip")
        norms.append(nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25))

    def update():
        calls.append("optimizer")
        optimizer.step()

    try:
        run_simulation_step(
            model_parts=[model],
            parallel_dims=dims,
            forward_backward_step=forward_backward_step,
            input_dict=batches[0][0],
            labels=batches[0][1],
            microbatches=batches,
            optimizer_zero_grad=zero_grad,
            clip_grad_norm=clip,
            optimizer_step=update,
            lr_scheduler_step=lambda: calls.append("scheduler"),
            local_batch_size=3,
            seq_len=4,
            gradient_accumulation=3,
        )
    finally:
        unpatch_device_type_to_meta()
    assert calls == ["zero", "batch", "batch", "batch", "clip", "optimizer", "scheduler"]
    assert torch.equal(norms[0], reference_norm)
    assert _default_collective_context.get() is None
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual.grad, expected.grad)
        assert torch.equal(actual, expected)


def test_meta_labels_require_explicit_valid_token_count():
    model = nn.Linear(4, 2, device="meta")
    try:
        with pytest.raises(ValueError, match="valid.*token"):
            run_simulation_step(
                model_parts=[model], parallel_dims=_dims(),
                forward_backward_step=lambda **_: None,
                input_dict={"input": torch.ones(3, 4, device="meta")},
                labels=torch.ones(3, dtype=torch.int64, device="meta"),
                optimizer_step=lambda: None, lr_scheduler_step=lambda: None,
                local_batch_size=3, seq_len=4,
            )
    finally:
        unpatch_device_type_to_meta()


def test_gradient_clipping_is_a_single_step_operation_not_a_backward_template():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = ({"input": torch.ones(3, 4)}, torch.tensor([0, 1, 0]))

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        (model(input_dict["input"]).sum() / global_valid_tokens).backward()

    try:
        graph = run_simulation_step(
            model_parts=[model], parallel_dims=_dims(),
            forward_backward_step=forward_backward_step,
            input_dict=batch[0], labels=batch[1], microbatches=[batch, batch],
            gradient_accumulation=2, optimizer_step=optimizer.step,
            lr_scheduler_step=lambda: None, local_batch_size=3, seq_len=4,
            clip_grad_norm=lambda: nn.utils.clip_grad_norm_(model.parameters(), 0.25),
        )
    finally:
        unpatch_device_type_to_meta()
    norm_nodes = [node for step in graph.step_templates.values() for node in step.nodes.values()
                  if "norm" in node.annotations.get("raw_op_type", "")]
    assert norm_nodes
    assert all(node.annotations["comp_type"] == "OPTIMIZER" for node in norm_nodes)
