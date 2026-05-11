# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    ADAMW_STATE_KEYS,
    build_muon_hybrid_optimizers,
    MUON_STATE_KEYS,
    MuonVirtualAllocator,
    VirtualMuonHybridOptimizersContainer,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _init_single_rank_process_group():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method="tcp://localhost:29501", rank=0, world_size=1
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def npu_parallel_dims():
    from unittest.mock import patch

    from tests.testing.parallel_dims import build_parallel_dims

    with patch("torchtitan.distributed.parallel_dims.device_type", "npu"):
        pd = build_parallel_dims()
        pd.build_mesh()
    return pd


def _muon_config(**overrides):
    base = dict(
        name="Muon",
        lr=1e-3,
        weight_decay=0.01,
        muon_lr=None,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
        muon_hybrid_ns=False,
        virtual_allocator=False,
        swap_optimizer=False,
        extra_param_group_split_rules=None,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        implementation="for-loop",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _virtual_muon_config(**overrides):
    overrides.setdefault("virtual_allocator", True)
    return _muon_config(**overrides)


def _simple_model(npu_device):
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.LayerNorm(64),
    ).to(npu_device)


def _step_model(model, npu_device, in_features=32):
    x = torch.randn(2, in_features, device=npu_device)
    loss = model(x).sum()
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# Non-virtual Muon smoke tests
# ---------------------------------------------------------------------------


def test_muon_two_steps_loss_decreases(npu_device, npu_parallel_dims):
    torch.manual_seed(42)
    model = _simple_model(npu_device)
    container = build_muon_hybrid_optimizers([model], _muon_config(), npu_parallel_dims)

    losses = []
    for _ in range(3):
        loss = _step_model(model, npu_device)
        container.step()
        container.zero_grad()
        losses.append(loss.item())

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert losses[-1] < losses[0], f"Loss should decrease over steps: {losses}"


# ---------------------------------------------------------------------------
# Virtual Muon smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def virtual_container(npu_device, npu_parallel_dims):
    torch.manual_seed(42)
    model = _simple_model(npu_device)
    container = build_muon_hybrid_optimizers(
        [model],
        _virtual_muon_config(),
        npu_parallel_dims,
        virtual_allocator=True,
    )
    return model, container


def test_virtual_muon_builds_virtual_container(npu_device, npu_parallel_dims):
    model = _simple_model(npu_device)

    container = build_muon_hybrid_optimizers(
        [model],
        _virtual_muon_config(),
        npu_parallel_dims,
        virtual_allocator=True,
    )

    assert isinstance(container, VirtualMuonHybridOptimizersContainer)
    assert len(container.optimizers) == 2
    assert container.virtual_allocator is not None
    assert isinstance(container.virtual_allocator, MuonVirtualAllocator)


def test_virtual_muon_step_updates_params(npu_device, npu_parallel_dims):
    torch.manual_seed(42)
    model = _simple_model(npu_device)

    container = build_muon_hybrid_optimizers(
        [model],
        _virtual_muon_config(),
        npu_parallel_dims,
        virtual_allocator=True,
    )

    orig_linear_weight = model[0].weight.data.clone()

    _step_model(model, npu_device)
    container.step()
    container.zero_grad()

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert not torch.equal(model[0].weight.data, orig_linear_weight)


def _assert_state_keys_on_cpu(state, keys):
    for key in keys:
        if key in state and state[key] is not None:
            assert state[key].device.type == "cpu", (
                f"State '{key}' should be on CPU after step, "
                f"got {state[key].device}"
            )


def _assert_param_states_on_cpu(optim, keys):
    for group in optim.param_groups:
        for p in group["params"]:
            state = optim.state.get(p, {})
            if state:
                _assert_state_keys_on_cpu(state, keys)


def _assert_states_on_cpu(container):
    for optim_idx, optim in enumerate(container.optimizers):
        keys = MUON_STATE_KEYS if optim_idx == 0 else ADAMW_STATE_KEYS
        _assert_param_states_on_cpu(optim, keys)


def test_virtual_muon_states_on_cpu_after_step(virtual_container, npu_device):
    model, container = virtual_container

    for _ in range(2):
        _step_model(model, npu_device)
        container.step()
        container.zero_grad()

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    _assert_states_on_cpu(container)


def test_virtual_muon_swap_stats_nonzero_after_step(virtual_container, npu_device):
    model, container = virtual_container

    for _ in range(2):
        _step_model(model, npu_device)
        container.step()
        container.zero_grad()

    va = container.virtual_allocator
    assert va is not None
    assert va.actually_swap_size > 0, "Swap size should be > 0 after two steps"
    assert va.theoretical_swap_size > 0, "Theoretical swap size should be > 0"


def test_virtual_muon_state_dict_roundtrip(virtual_container, npu_device):
    model, container = virtual_container

    _step_model(model, npu_device)
    container.step()

    sd = container.state_dict()
    assert len(sd) > 0

    container.load_state_dict(sd)

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def test_virtual_muon_two_steps_loss_decreases(virtual_container, npu_device):
    model, container = virtual_container

    losses = []
    for _ in range(3):
        loss = _step_model(model, npu_device)
        container.step()
        container.zero_grad()
        losses.append(loss.item())

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert losses[-1] < losses[0], f"Loss should decrease over steps: {losses}"


# ---------------------------------------------------------------------------
# MuonVirtualAllocator smoke tests
# ---------------------------------------------------------------------------


def test_virtual_allocator_copy2swap_and_copy2device(npu_device):
    va = MuonVirtualAllocator(pp_rank=0, pp_size=1)

    src = torch.randn(4, 8, device=npu_device)

    swap = va.copy2swap(src)
    assert swap.device.type == "cpu"
    assert swap.shape == src.shape

    restored = va.copy2device(swap, ref_tensor=None)
    assert restored.device == npu_device
    assert torch.allclose(restored.cpu(), src.cpu(), atol=1e-6)
