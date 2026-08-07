# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchtitan.config as torchtitan_config
from torchtitan.components.optimizer import OptimizersContainer

with (
    patch.dict(
        sys.modules,
        {"torch_npu": MagicMock(), "torchtitan_npu.patches": MagicMock()},
    ),
    patch.multiple(
        torchtitan_config,
        derive=lambda cfg, target: target,
        override=lambda **kwargs: lambda function: function,
        create=True,
    ),
):
    from torchtitan_npu.override.common import optimizer as product_optimizer


@pytest.fixture
def optimizer_module(monkeypatch):
    allocations = []

    def allocate(size, *, device):
        allocations.append((torch.Size(size), torch.device(device)))
        return torch.empty(size, device=device)

    monkeypatch.setattr(product_optimizer.torch_npu, "empty_with_swapped_memory", allocate)
    return product_optimizer, allocations


@pytest.mark.cpu
def test_make_swap_allocates_nonempty_tensor_and_optionally_copies(optimizer_module):
    module, allocations = optimizer_module
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    make_swap = getattr(module, "_make_swap")
    empty = make_swap(source)
    copied = make_swap(source, copy_data=True)

    assert len(allocations) == 2
    assert empty.shape == source.shape
    assert torch.equal(copied, source)
    assert copied.data_ptr() != source.data_ptr()


@pytest.mark.cpu
def test_make_swap_uses_regular_empty_tensor_for_zero_size_input(optimizer_module):
    module, allocations = optimizer_module
    source = torch.empty(0, 4)

    result = getattr(module, "_make_swap")(source, copy_data=True)

    assert result.shape == source.shape
    assert result.numel() == 0
    assert allocations == [(torch.Size([0, 4]), torch.device("cpu"))]


@pytest.mark.cpu
def test_swap_state_hook_initializes_only_missing_states_with_gradients(optimizer_module):
    module, allocations = optimizer_module
    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(2))
    skipped = torch.nn.Parameter(torch.ones(2))
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    state = {first: {}, second: {"sentinel": object()}, skipped: {}}
    optimizer = types.SimpleNamespace(
        param_groups=[{"params": [first, second, skipped]}],
        state=state,
    )

    getattr(module, "_swap_state_init_hook")(optimizer, (), {})

    assert set(state[first]) == {"step", "exp_avg", "exp_avg_sq"}
    assert state[first]["step"].item() == 0
    assert torch.count_nonzero(state[first]["exp_avg"]) == 0
    assert torch.count_nonzero(state[first]["exp_avg_sq"]) == 0
    assert state[second] == {"sentinel": state[second]["sentinel"]}
    assert state[skipped] == {}
    assert len(allocations) == 2


@pytest.mark.cpu
def test_state_dict_returns_independent_cpu_moment_snapshot(optimizer_module, monkeypatch):
    module, _ = optimizer_module
    param = torch.nn.Parameter(torch.ones(2))
    live_exp_avg = torch.tensor([1.0, 2.0])
    live_exp_avg_sq = torch.tensor([3.0, 4.0])
    step = torch.tensor(5.0)
    optimizer = types.SimpleNamespace(
        state={param: {"exp_avg": live_exp_avg, "exp_avg_sq": live_exp_avg_sq, "step": step}}
    )
    flat = {
        "state.weight.exp_avg": live_exp_avg,
        "state.weight.exp_avg_sq": live_exp_avg_sq,
        "state.weight.step": step,
    }
    container = module.SwapOptimizersContainer.__new__(module.SwapOptimizersContainer)
    container.optimizers = [optimizer]
    setattr(container, "_swap_load_targets", {})
    monkeypatch.setattr(OptimizersContainer, "state_dict", lambda self: dict(flat))

    snapshot = container.state_dict()

    assert snapshot["state.weight.exp_avg"].device.type == "cpu"
    assert snapshot["state.weight.exp_avg"] is live_exp_avg
    assert snapshot["state.weight.exp_avg_sq"] is live_exp_avg_sq
    assert snapshot["state.weight.step"] is step
    assert optimizer.state[param]["exp_avg"] is live_exp_avg
    live_exp_avg.add_(10)
    assert torch.equal(snapshot["state.weight.exp_avg"], torch.tensor([11.0, 12.0]))
    snapshot["state.weight.exp_avg_sq"].zero_()
    assert torch.equal(live_exp_avg_sq, torch.zeros(2))


@pytest.mark.cpu
def test_empty_state_uses_regular_staging_without_replacing_live_swap_targets(optimizer_module, monkeypatch):
    module, _ = optimizer_module
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = types.SimpleNamespace(state={})
    target_exp_avg = torch.zeros(2)
    target_exp_avg_sq = torch.zeros(2)
    container = module.SwapOptimizersContainer.__new__(module.SwapOptimizersContainer)
    container.optimizers = [optimizer]
    setattr(container, "_swap_load_targets", {})

    def materialize():
        optimizer.state[param] = {
            "exp_avg": target_exp_avg,
            "exp_avg_sq": target_exp_avg_sq,
            "step": torch.tensor(0.0),
        }
        return {
            "state.weight.exp_avg": target_exp_avg,
            "state.weight.exp_avg_sq": target_exp_avg_sq,
            "state.weight.step": optimizer.state[param]["step"],
        }

    monkeypatch.setattr(OptimizersContainer, "state_dict", lambda self: materialize())

    load_targets = container.state_dict()

    assert load_targets["state.weight.exp_avg"] is target_exp_avg
    assert load_targets["state.weight.exp_avg_sq"] is target_exp_avg_sq
    assert optimizer.state[param]["exp_avg"] is target_exp_avg
    assert optimizer.state[param]["exp_avg_sq"] is target_exp_avg_sq

    checkpoint = {
        "state.weight.exp_avg": torch.tensor([1.5, 2.5]),
        "state.weight.exp_avg_sq": torch.tensor([3.5, 4.5]),
        "state.weight.step": torch.tensor(7.0),
    }
    loaded = {}
    monkeypatch.setattr(
        OptimizersContainer,
        "load_state_dict",
        lambda self, state: loaded.update(state),
    )
    container.load_state_dict(checkpoint)

    loaded_exp_avg = loaded.get("state.weight.exp_avg")
    checkpoint_exp_avg = checkpoint.get("state.weight.exp_avg")
    assert loaded_exp_avg is not None
    assert checkpoint_exp_avg is not None
    assert loaded_exp_avg is checkpoint_exp_avg

    loaded_exp_avg_sq = loaded.get("state.weight.exp_avg_sq")
    checkpoint_exp_avg_sq = checkpoint.get("state.weight.exp_avg_sq")
    assert loaded_exp_avg_sq is not None
    assert checkpoint_exp_avg_sq is not None
    assert loaded_exp_avg_sq is checkpoint_exp_avg_sq
    torch.testing.assert_close(target_exp_avg, torch.zeros(2))
    torch.testing.assert_close(target_exp_avg_sq, torch.zeros(2))
    assert getattr(container, "_swap_load_targets") == {}


@pytest.mark.cpu
def test_inherited_load_path_restores_state_without_mutating_checkpoint(optimizer_module, monkeypatch):
    module, _ = optimizer_module
    checkpoint = {
        "state.weight.exp_avg": torch.tensor([1.5, 2.5]),
        "state.weight.exp_avg_sq": torch.tensor([3.5, 4.5]),
        "state.weight.step": torch.tensor(7.0),
    }
    original = {key: value.clone() for key, value in checkpoint.items()}
    loaded = {}
    container = module.SwapOptimizersContainer.__new__(module.SwapOptimizersContainer)
    container.optimizers = [types.SimpleNamespace(state={})]
    setattr(container, "_swap_load_targets", {})

    def load(state_dict):
        loaded.update({key: value.clone() for key, value in state_dict.items()})

    monkeypatch.setattr(OptimizersContainer, "load_state_dict", lambda self, state: load(state))
    container.load_state_dict(checkpoint)

    assert all(torch.equal(checkpoint[key], original[key]) for key in checkpoint)
    assert all(torch.equal(loaded[key], original[key]) for key in loaded)
