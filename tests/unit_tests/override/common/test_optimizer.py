# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for swap allocation and lazy optimizer-state initialization."""

from __future__ import annotations

import types

import pytest
import torch
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig

from torchtitan_npu.override.common import optimizer as product_optimizer

make_swap = vars(product_optimizer)["_make_swap"]
swap_state_init_hook = vars(product_optimizer)["_swap_state_init_hook"]

pytestmark = pytest.mark.cpu


@pytest.fixture
def swap_allocations(monkeypatch):
    allocations = []

    def allocate(size, *, dtype, device):
        allocations.append((torch.Size(size), dtype, torch.device(device)))
        return torch.empty(size, dtype=dtype, device=device)

    monkeypatch.setattr(product_optimizer.torch_npu, "empty_with_swapped_memory", allocate)
    return allocations


def test_make_swap_allocates_nonempty_tensor(swap_allocations):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    result = make_swap(source)

    assert swap_allocations == [(source.size(), source.dtype, source.device)]
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert result.data_ptr() != source.data_ptr()


def test_make_swap_uses_regular_empty_tensor_for_zero_size_input(swap_allocations):
    source = torch.empty(0, 4)

    result = make_swap(source)

    assert result.shape == source.shape
    assert result.numel() == 0
    assert swap_allocations == []


def test_swap_state_hook_initializes_only_missing_states_with_gradients(
    swap_allocations,
):
    missing = torch.nn.Parameter(torch.ones(2))
    existing = torch.nn.Parameter(torch.ones(2))
    no_grad = torch.nn.Parameter(torch.ones(2))
    missing.grad = torch.ones_like(missing)
    existing.grad = torch.ones_like(existing)
    sentinel = object()
    state = {missing: {}, existing: {"sentinel": sentinel}, no_grad: {}}
    optimizer = types.SimpleNamespace(
        param_groups=[{"params": [missing, existing, no_grad]}],
        state=state,
    )

    swap_state_init_hook(optimizer, (), {})

    assert set(state[missing]) == {"step", "exp_avg", "exp_avg_sq"}
    assert state[missing]["step"].item() == 0
    assert state[missing]["step"].dtype == torch.float32
    assert torch.count_nonzero(state[missing]["exp_avg"]) == 0
    assert torch.count_nonzero(state[missing]["exp_avg_sq"]) == 0
    assert state[existing] == {"sentinel": sentinel}
    assert state[no_grad] == {}
    assert swap_allocations == [(missing.size(), missing.dtype, missing.device)] * 2


def test_virtual_derives_optimizer_config():
    cfg = OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.03})],
    )
    result = product_optimizer.virtual(cfg)

    assert isinstance(result, product_optimizer.VirtualOptimizersContainer.Config)
    assert result.implementation == "for-loop"
    assert result.param_groups[0].optimizer_kwargs["lr"] == 0.03


@pytest.mark.parametrize("override_name", ["cpu_offload", "swap_optimizer"])
def test_cpu_offload_preserves_sparse_training_and_checkpoint(monkeypatch, override_name):
    from copy import deepcopy
    from dataclasses import dataclass

    from torchtitan.config import Configurable, OverrideConfig, apply_overrides

    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
        HostSparseOptimizersContainer,
    )
    from torchtitan_npu.extensions.distributed import grad_accum, grad_clip
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

    @dataclass(kw_only=True)
    class RootConfig(Configurable.Config):
        optimizer: OptimizersContainer.Config

    # Replace only the NPU resource setup; keep container construction, factory,
    # SparseAdam, lifecycle and checkpoint serialization on the production path.
    monkeypatch.setattr(product_optimizer.torch_npu.npu, "current_device", lambda: 0)
    monkeypatch.setattr(
        product_optimizer,
        "CpuStaging",
        lambda *args, **kwargs: types.SimpleNamespace(
            device=torch.device("cpu"),
            wait=lambda: None,
            close=lambda: None,
        ),
    )
    for module, names in (
        (grad_accum, ("install", "register_cpu_offload_hooks", "unregister_cpu_offload_hooks", "clear")),
        (grad_clip, ("install", "clear")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, lambda: None)

    def build():
        table = HostEngramTable.Config(
            vocab_size=16,
            layer_id=0,
            ngram_orders=(2,),
            num_heads=1,
            head_vocab_sizes=(11,),
            embedding_dim=4,
            num_embeddings=12,
            require_token_id_map=False,
            pin_memory=False,
        ).build()
        with torch.no_grad():
            table.weight.copy_(torch.arange(48).reshape(12, 4) / 16)
        root = RootConfig(
            optimizer=HostSparseOptimizersContainer.Config(
                _cpu_offload=True,
                implementation="fused",
                param_groups=[
                    ParamGroupConfig(pattern=".*", optimizer_name="SparseAdam", optimizer_kwargs={"lr": 0.05})
                ],
            )
        )
        apply_overrides(OverrideConfig(imports=[f"torchtitan_npu.override.common.optimizer.{override_name}"]), root)
        optimizer = root.optimizer.build(model_parts=[table])
        assert isinstance(optimizer, product_optimizer.CpuOffloadOptimizersContainer)
        assert isinstance(optimizer, HostSparseOptimizersContainer)
        assert type(optimizer.optimizers[0]) is torch.optim.SparseAdam
        assert optimizer._clip_channel._preserve_coefficient_dtype
        return table, optimizer

    table, optimizer = build()
    restored_optimizer = None
    refreshes = []
    monkeypatch.setattr(
        table, "refresh_lookup_storage", lambda rows: refreshes.append((rows.clone(), table.weight.detach().clone()))
    )
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    reference_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    try:
        # Checkpointing before the first gradient must initialize sparse state
        # without attempting a dense-gradient SparseAdam step.
        optimizer.state_dict()
        for rows in ([0, 7, 7], [1, 7, 11]):
            ids = torch.tensor(rows)
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            table._distributed_lookup(ids).sum().backward()
            reference(ids).sum().backward()
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
        torch.testing.assert_close(refreshes[-1][0], torch.tensor([1, 7, 11]))
        torch.testing.assert_close(refreshes[-1][1], table.weight, rtol=0, atol=0)
        state = deepcopy(optimizer.state_dict())
        restored, restored_optimizer = build()
        restored.load_state_dict(table.state_dict())
        restored_optimizer.load_state_dict(state)
        original_state = optimizer.optimizers[0].state[table.weight]
        loaded_state = restored_optimizer.optimizers[0].state[restored.weight]
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(loaded_state[key], original_state[key], rtol=0, atol=0)
        assert loaded_state["step"] == original_state["step"] == 2
        restored_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        ids = torch.tensor([2, 7])
        restored._distributed_lookup(ids).sum().backward()
        reference(ids).sum().backward()
        restored_optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(restored.weight, reference.weight, rtol=0, atol=0)
        optimizer.zero_grad()
        assert table.pending_sparse_grad() is None and table.weight.grad is None
    finally:
        optimizer.close()
        if restored_optimizer is not None:
            restored_optimizer.close()
        product_optimizer.clip_state.set_active_channel(None)


@pytest.mark.parametrize("cached", [False, True])
def test_cpu_offload_sparse_clip_uses_one_global_coefficient(monkeypatch, cached):
    from torchtitan.distributed import utils as dist_utils

    from torchtitan_npu.models.deepseek_v4_1.config_registry import DeepSeekV41Trainer
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

    table = HostEngramTable.Config(
        vocab_size=8,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(7,),
        embedding_dim=1,
        num_embeddings=8,
        require_token_id_map=False,
        pin_memory=False,
    ).build()
    table.accumulate_sparse_gradient(torch.tensor([0]), torch.tensor([[4.0]]))
    dense = torch.nn.Parameter(torch.zeros(1))
    dense.grad = torch.tensor([3.0])
    channel = product_optimizer.clip_state.GradientClipChannel(preserve_coefficient_dtype=True)
    optimizer = object.__new__(product_optimizer.CpuOffloadHostSparseOptimizersContainer)
    optimizer._closed = True
    optimizer._clip_channel = channel
    optimizer.model_parts = [table]
    trainer = object.__new__(DeepSeekV41Trainer)
    trainer.optimizers = optimizer

    def clip(parameters, max_norm, norm_type, **kwargs):
        if cached:
            channel.publish([(dense, dense.grad, dense.grad.clone())], torch.tensor(max_norm / (3.0 + 1e-6)), None)
            return torch.tensor(3.0)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", clip)
    assert trainer.clip_grad_norm([dense, table.weight], 1.0).item() == pytest.approx(5.0)
    if cached:
        assert dense.grad.item() == 3.0
        actual = channel.apply_pending(channel.take_gradient(dense, dense.grad))
    else:
        actual = dense.grad
    torch.testing.assert_close(actual, torch.tensor([0.6]))
    torch.testing.assert_close(table.pending_sparse_grad().values(), torch.tensor([[0.8]]))
    channel.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("preserve_dtype", [False, True], ids=["dense-only", "host-sparse"])
def test_offload_clip_preserves_scaling_policy_across_steps(dtype, preserve_dtype):
    gradient = torch.linspace(-2, 2, 2049, dtype=dtype)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    coefficient = torch.tensor(0.1234567, dtype=torch.float32)
    channel = product_optimizer.clip_state.GradientClipChannel(preserve_coefficient_dtype=preserve_dtype)
    # Exercise correction, identity correction and no sparse gradient on the
    # same channel: clearing a step must not change its coefficient precision.
    for correction in (0.8765432, 1.0, None):
        expected = gradient.clone()
        torch._foreach_mul_([expected], coefficient if preserve_dtype else coefficient.to(dtype))
        if correction is not None:
            expected.mul_(correction)
        channel.publish([(parameter, gradient, gradient.clone())], coefficient.clone(), None)
        if correction is not None:
            assert channel.rescale_pending(correction)
        actual = channel.apply_pending(channel.take_gradient(parameter, gradient))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        channel.clear_pending()
