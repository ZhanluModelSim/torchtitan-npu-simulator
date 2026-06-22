# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import types
from dataclasses import fields
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.components.lr_scheduler import LRSchedulersContainer

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    _build_adamw_kwargs,
    _build_muon_kwargs,
    _get_muon_lr_config,
    _split_parameters_for_muon,
    build_muon_lr_schedulers,
    MuonHybridOptimizersContainer,
    MuonLRSchedulersContainer,
    zeropower_via_newtonschulz5,
)
from torchtitan_npu.patches.optimizer.swap_optimizer import unwrap_dtensor


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16, bias=True)
        self.embed = nn.Linear(4, 8, bias=False)
        self.norm = nn.LayerNorm(16)
        self.expert_weight = nn.Parameter(torch.randn(4, 8, 16))


def _build_container(muon_optimizer_config, cpu_parallel_dims):
    model = _DummyModel()
    opt_config = muon_optimizer_config()
    target_fields = {f.name for f in fields(MuonHybridOptimizersContainer.Config)}
    cfg = MuonHybridOptimizersContainer.Config(
        **{k: v for k, v in opt_config.__dict__.items() if k in target_fields}
    )
    return cfg.build(model_parts=[model], parallel_dims=cpu_parallel_dims), model


# --- TestSplitParametersForMuon ---


def test_2d_params_go_to_muon():
    model = _DummyModel()
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("linear.weight" in n for n in muon_names)
    assert not any("linear.weight" in n for n in adamw_names)


def test_excluded_2d_params_go_to_adamw():
    model = _DummyModel()
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("embed.weight" in n for n in adamw_names)
    assert not any("embed.weight" in n for n in muon_names)


def test_1d_params_go_to_adamw():
    model = _DummyModel()
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("linear.bias" in n for n in adamw_names)
    assert any("norm.weight" in n for n in adamw_names)
    assert any("norm.bias" in n for n in adamw_names)


def test_3d_params_go_to_muon():
    model = _DummyModel()
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("expert_weight" in n for n in muon_names)
    assert not any("expert_weight" in n for n in adamw_names)


def test_lm_head_excluded():
    model = nn.Module()
    model.lm_head = nn.Linear(8, 100, bias=False)
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("lm_head" in n for n in adamw_names)
    assert not any("lm_head" in n for n in muon_names)


def test_output_excluded():
    model = nn.Module()
    model.output_proj = nn.Linear(8, 100, bias=False)
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert any("output" in n for n in adamw_names)
    assert not any("output" in n for n in muon_names)


def test_no_grad_params_excluded():
    model = nn.Module()
    model.frozen = nn.Linear(4, 4, bias=False)
    frozen = cast(nn.Linear, model.frozen)
    frozen.weight.requires_grad = False
    muon_params, muon_names, adamw_params, adamw_names = _split_parameters_for_muon(
        [model]
    )
    assert len(muon_params) == 0
    assert len(adamw_params) == 0


# --- TestGetMuonLrConfig ---


def test_original_mode_with_muon_lr():
    config = types.SimpleNamespace(muon_adjust_lr_fn="original", muon_lr=1e-2)
    muon_lr, fn = _get_muon_lr_config(config, base_lr=3e-4)
    assert muon_lr == 1e-2
    assert fn == "original"


def test_original_mode_without_muon_lr():
    config = types.SimpleNamespace(muon_adjust_lr_fn="original", muon_lr=None)
    muon_lr, fn = _get_muon_lr_config(config, base_lr=3e-4)
    assert muon_lr == 3e-4
    assert fn == "original"


def test_match_rms_adamw_ignores_muon_lr():
    config = types.SimpleNamespace(muon_adjust_lr_fn="match_rms_adamw", muon_lr=1e-2)
    muon_lr, fn = _get_muon_lr_config(config, base_lr=3e-4)
    assert muon_lr == 3e-4
    assert fn == "match_rms_adamw"


def test_match_rms_adamw_without_muon_lr():
    config = types.SimpleNamespace(muon_adjust_lr_fn="match_rms_adamw", muon_lr=None)
    muon_lr, fn = _get_muon_lr_config(config, base_lr=3e-4)
    assert muon_lr == 3e-4
    assert fn == "match_rms_adamw"


# --- TestBuildKwargs ---


def test_build_muon_kwargs_original():
    config = types.SimpleNamespace(
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=10,
        eps=1e-7,
        muon_hybrid_ns=True,
    )
    kwargs = _build_muon_kwargs(
        muon_lr=1e-2,
        weight_decay=0.1,
        optimizer_config=config,
        muon_adjust_lr_fn="original",
    )
    assert kwargs["lr"] == 1e-2
    assert kwargs["weight_decay"] == 0.1
    assert kwargs["momentum"] == 0.95
    assert kwargs["nesterov"] is True
    assert kwargs["ns_steps"] == 10
    assert kwargs["eps"] == 1e-7
    assert kwargs["adjust_lr_fn"] == "original"
    assert kwargs["hybrid_ns"] is True


def test_build_adamw_kwargs_fused():
    config = types.SimpleNamespace(
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        implementation="fused",
    )
    kwargs = _build_adamw_kwargs(lr=3e-4, weight_decay=0.01, optimizer_config=config)
    assert kwargs["lr"] == 3e-4
    assert kwargs["betas"] == (0.9, 0.95)
    assert kwargs["fused"] is True
    assert kwargs["foreach"] is False


def test_build_adamw_kwargs_invalid_implementation():
    config = types.SimpleNamespace(
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        implementation="invalid",
    )
    with pytest.raises(ValueError, match="Invalid implementation"):
        _build_adamw_kwargs(lr=3e-4, weight_decay=0.01, optimizer_config=config)


# --- TestNewtonSchulz ---


def test_output_shape_2d():
    torch.manual_seed(42)
    grad = torch.randn(16, 8)
    result = zeropower_via_newtonschulz5(grad, steps=5)
    assert result.shape == grad.shape


def test_output_is_approximately_orthogonal():
    torch.manual_seed(42)
    grad = torch.randn(8, 8)
    result = zeropower_via_newtonschulz5(grad, steps=10)
    eye = result @ result.T
    identity = torch.eye(8)
    diag = torch.diag(eye)
    assert (diag > 0.4).all(), f"Diagonal values too small: {diag}"
    off_diag = eye - torch.diag(diag)
    assert (
        off_diag.abs().max() < 0.5
    ), f"Off-diagonal values too large: {off_diag.abs().max()}"


def test_3d_input():
    torch.manual_seed(42)
    grad = torch.randn(3, 16, 8)
    result = zeropower_via_newtonschulz5(grad, steps=5)
    assert result.shape == grad.shape


def test_hybrid_ns_runs():
    torch.manual_seed(42)
    grad = torch.randn(8, 8)
    result = zeropower_via_newtonschulz5(grad, steps=10, hybrid_ns=True)
    assert result.shape == grad.shape
    assert torch.isfinite(result).all()


def test_hybrid_ns_differs_from_standard():
    torch.manual_seed(42)
    grad = torch.randn(16, 8)
    result_standard = zeropower_via_newtonschulz5(grad, steps=10, hybrid_ns=False)
    result_hybrid = zeropower_via_newtonschulz5(grad, steps=10, hybrid_ns=True)
    assert not torch.allclose(result_standard, result_hybrid, atol=1e-6)


def test_steps_too_large_raises():
    grad = torch.randn(4, 4)
    with pytest.raises(ValueError, match="must be < 100"):
        zeropower_via_newtonschulz5(grad, steps=100)


def test_1d_input_raises():
    grad = torch.randn(16)
    with pytest.raises(ValueError, match="2D or 3D"):
        zeropower_via_newtonschulz5(grad, steps=5)


def test_preserves_dtype():
    grad = torch.randn(8, 8, dtype=torch.float32)
    result = zeropower_via_newtonschulz5(grad, steps=5)
    assert result.dtype == grad.dtype


# --- TestMuonHybridOptimizersContainer ---


def test_container_type(muon_optimizer_config, cpu_parallel_dims):
    container, _ = _build_container(muon_optimizer_config, cpu_parallel_dims)
    assert isinstance(container, MuonHybridOptimizersContainer)


def test_has_two_sub_optimizers(muon_optimizer_config, cpu_parallel_dims):
    container, _ = _build_container(muon_optimizer_config, cpu_parallel_dims)
    assert len(container.optimizers) == 2
    assert container.muon_optimizer is container.optimizers[0]
    assert container.adamw_optimizer is container.optimizers[1]


def test_step_updates_params(muon_optimizer_config, cpu_parallel_dims):
    container, model = _build_container(muon_optimizer_config, cpu_parallel_dims)
    orig_weight = model.linear.weight.data.clone()
    x = torch.randn(2, 4)
    out = model.embed(x)
    out.sum().backward()
    container.step()
    assert torch.equal(
        model.linear.weight.data, orig_weight
    ), "Muon parameters without gradients should not be updated"


def test_step_updates_muon_params_with_grad(muon_optimizer_config, cpu_parallel_dims):
    container, model = _build_container(muon_optimizer_config, cpu_parallel_dims)
    orig_weight = model.linear.weight.data.clone()
    x = torch.randn(2, 8)
    out = model.linear(x)
    out.sum().backward()
    container.step()
    assert not torch.equal(
        model.linear.weight.data, orig_weight
    ), "Muon optimizer step should update Muon-managed parameters with gradients"


def test_zero_grad_clears_gradients(muon_optimizer_config, cpu_parallel_dims):
    container, model = _build_container(muon_optimizer_config, cpu_parallel_dims)
    x = torch.randn(2, 4)
    out = model.embed(x)
    out.sum().backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
    container.zero_grad()
    all_none = all(p.grad is None for p in model.parameters())
    assert all_none


def test_iter_yields_sub_optimizers(muon_optimizer_config, cpu_parallel_dims):
    container, _ = _build_container(muon_optimizer_config, cpu_parallel_dims)
    optimizers = list(container)
    assert len(optimizers) == 2


def test_state_dict_roundtrip(muon_optimizer_config, cpu_parallel_dims):
    container, model = _build_container(muon_optimizer_config, cpu_parallel_dims)
    x = torch.randn(2, 4)
    out = model.embed(x)
    out.sum().backward()
    container.step()
    sd = container.state_dict()
    assert len(sd) > 0
    container.load_state_dict(sd)


# --- TestNpuOptimizerDispatcher ---


def test_muon_with_swap_and_virtual_raises():
    from torchtitan_npu.config.configs import OptimizerConfig

    optimizer_config = OptimizerConfig(
        name="Muon",
        swap_optimizer=True,
        virtual_optimizer=True,
    )
    with pytest.raises(
        ValueError, match="Cannot enable both virtual_optimizer and swap_optimizer"
    ):
        optimizer_config.build(
            model_parts=[],
            parallel_dims=None,
            ft_manager=None,
        )


def test_muon_with_virtual_raises():
    from torchtitan_npu.config.configs import OptimizerConfig

    optimizer_config = OptimizerConfig(
        name="Muon",
        virtual_optimizer=True,
        virtual_optimizer_size=1.0,
    )
    with pytest.raises(
        ValueError, match="Muon does not support virtual_optimizer"
    ):
        optimizer_config.build(
            model_parts=[],
            parallel_dims=None,
            ft_manager=None,
        )


def test_muon_routes_correctly(muon_optimizer_config, cpu_parallel_dims, monkeypatch):
    from torchtitan_npu.config.configs import OptimizerConfig
    from torchtitan_npu.patches.torchtitan import _trainer_config_stash

    model = _DummyModel()
    opt_config = muon_optimizer_config().to_namespace()
    optimizer_config = OptimizerConfig(
        name=opt_config.name,
        lr=opt_config.lr,
        weight_decay=opt_config.weight_decay,
        beta1=opt_config.beta1,
        beta2=opt_config.beta2,
        eps=opt_config.eps,
        implementation=opt_config.implementation,
    )

    monkeypatch.setattr(
        _trainer_config_stash, "_active_parallel_dims", cpu_parallel_dims
    )
    result = optimizer_config.build(
        model_parts=[model],
        ft_manager=None,
    )
    assert isinstance(result, MuonHybridOptimizersContainer)


# --- TestSwapUtils ---


def test_unwrap_dtensor_plain_tensor():
    t = torch.randn(2, 2)
    assert unwrap_dtensor(t) is t


# --- TestMuonLRScheduler ---


def _build_optimizers(muon_optimizer_config, cpu_parallel_dims, **config_overrides):
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(**config_overrides)
    target_fields = {f.name for f in fields(MuonHybridOptimizersContainer.Config)}
    cfg = MuonHybridOptimizersContainer.Config(
        **{k: v for k, v in opt_config.__dict__.items() if k in target_fields}
    )
    return cfg.build(model_parts=[model], parallel_dims=cpu_parallel_dims)


def test_creates_two_independent_schedulers(
    muon_optimizer_config, lr_scheduler_config, cpu_parallel_dims
):
    optimizers = _build_optimizers(
        muon_optimizer_config, cpu_parallel_dims, muon_adjust_lr_fn="original"
    )

    lr_config = lr_scheduler_config().to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    assert isinstance(schedulers, MuonLRSchedulersContainer)
    assert len(schedulers.schedulers) == 2
    assert isinstance(schedulers.schedulers[0], LambdaLR)
    assert isinstance(schedulers.schedulers[1], LambdaLR)


def test_step_updates_both_schedulers(muon_optimizer_config, cpu_parallel_dims):
    optimizers = _build_optimizers(muon_optimizer_config, cpu_parallel_dims)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    initial_epochs = [s.last_epoch for s in schedulers.schedulers]

    schedulers.step()

    for i, s in enumerate(schedulers.schedulers):
        assert (
            s.last_epoch == initial_epochs[i] + 1
        ), f"Scheduler {i} should have incremented last_epoch"


def test_state_dict_saves_first_scheduler_only(
    muon_optimizer_config, cpu_parallel_dims
):
    optimizers = _build_optimizers(muon_optimizer_config, cpu_parallel_dims)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    for _ in range(5):
        schedulers.step()

    state = schedulers.state_dict()

    assert "last_epoch" in state
    assert state["last_epoch"] == 5


def test_load_state_dict_applies_to_both_schedulers(
    muon_optimizer_config, cpu_parallel_dims
):
    optimizers = _build_optimizers(muon_optimizer_config, cpu_parallel_dims)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    state = {"last_epoch": 10}

    schedulers.load_state_dict(state)

    assert schedulers.schedulers[0].last_epoch == 10
    assert schedulers.schedulers[1].last_epoch == 10


def test_checkpoint_preserves_independent_base_lr(
    muon_optimizer_config, lr_scheduler_config, cpu_parallel_dims
):
    optimizers = _build_optimizers(
        muon_optimizer_config,
        cpu_parallel_dims,
        lr=2.2e-4,
        muon_lr=1e-2,
        muon_adjust_lr_fn="original",
    )

    lr_config = lr_scheduler_config(warmup_steps=2, decay_ratio=0.8).to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    muon_scheduler = schedulers.schedulers[0]
    adamw_scheduler = schedulers.schedulers[1]

    initial_muon_base_lr = muon_scheduler.base_lrs[0]
    initial_adamw_base_lr = adamw_scheduler.base_lrs[0]

    assert initial_muon_base_lr == 1e-2
    assert initial_adamw_base_lr == 2.2e-4

    for _ in range(6):
        schedulers.step()

    saved_state = schedulers.state_dict()

    optimizers2 = _build_optimizers(
        muon_optimizer_config,
        cpu_parallel_dims,
        lr=2.2e-4,
        muon_lr=1e-2,
        muon_adjust_lr_fn="original",
    )
    schedulers2 = build_muon_lr_schedulers(optimizers2, lr_config, training_steps)

    schedulers2.load_state_dict(saved_state)

    muon_scheduler2 = schedulers2.schedulers[0]
    adamw_scheduler2 = schedulers2.schedulers[1]

    assert (
        muon_scheduler2.base_lrs[0] == initial_muon_base_lr
    ), f"Muon base_lr not preserved: {muon_scheduler2.base_lrs[0]} != {initial_muon_base_lr}"
    assert (
        adamw_scheduler2.base_lrs[0] == initial_adamw_base_lr
    ), f"AdamW base_lr not preserved: {adamw_scheduler2.base_lrs[0]} != {initial_adamw_base_lr}"

    assert (
        schedulers2.schedulers[0].last_epoch == 6
    ), f"Muon scheduler last_epoch should be 6, got {schedulers2.schedulers[0].last_epoch}"
    assert (
        schedulers2.schedulers[1].last_epoch == 6
    ), f"AdamW scheduler last_epoch should be 6, got {schedulers2.schedulers[1].last_epoch}"


def test_match_rms_adamw_uses_standard_scheduler(
    muon_optimizer_config, lr_scheduler_config, cpu_parallel_dims
):
    optimizers = _build_optimizers(
        muon_optimizer_config, cpu_parallel_dims, muon_adjust_lr_fn="match_rms_adamw"
    )

    lr_config = lr_scheduler_config().to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    assert isinstance(
        schedulers, LRSchedulersContainer
    ), f"match_rms_adamw should use standard LRSchedulersContainer, got {type(schedulers)}"


def test_lr_scheduler_config_build_routes_muon_original(
    lr_scheduler_config, monkeypatch
):
    sentinel = object()
    captured = {}

    lr_namespace = lr_scheduler_config().to_namespace()
    lr_config = LRSchedulersContainer.Config(
        warmup_steps=lr_namespace.warmup_steps,
        decay_ratio=lr_namespace.decay_ratio,
        decay_type=lr_namespace.decay_type,
        min_lr_factor=lr_namespace.min_lr_factor,
    )

    def fake_build_muon_lr_schedulers(optimizers, lr_scheduler_config, training_steps):
        captured["optimizers"] = optimizers
        captured["lr_scheduler_config"] = lr_scheduler_config
        captured["training_steps"] = training_steps
        return sentinel

    optimizers = object.__new__(MuonHybridOptimizersContainer)
    optimizers.muon_adjust_lr_fn = "original"

    monkeypatch.setattr(
        "torchtitan_npu.patches.optimizer.muon_optimizer.build_muon_lr_schedulers",
        fake_build_muon_lr_schedulers,
    )

    schedulers = lr_config.build(optimizers=optimizers, training_steps=10)

    assert schedulers is sentinel
    assert captured["optimizers"] is optimizers
    assert captured["lr_scheduler_config"] is lr_config
    assert captured["training_steps"] == 10


# --- TestSwapMuonOptimizer ---


def _make_swap_muon_optimizer_config():
    from torchtitan_npu.config.configs import OptimizerConfig

    return OptimizerConfig(
        name="Muon",
        swap_optimizer=True,
        virtual_optimizer=False,
        swap_optimizer_times=8,
        swap_merge_buckets=4,
        lr=1e-3,
        weight_decay=0.01,
        muon_lr=None,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=5,
        muon_adjust_lr_fn="original",
        muon_hybrid_ns=False,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        implementation="for-loop",
        extra_param_group_split_rules=None,
    )


def test_muon_swap_optimizer_routing_and_config(monkeypatch, cpu_parallel_dims):
    from torchtitan_npu.config.configs import OptimizerConfig
    from torchtitan_npu.patches.optimizer import swap_optimizer as swap_mod
    from torchtitan_npu.patches.optimizer.optimizer_selector import (
        NpuOptimizerDispatcher,
    )
    from torchtitan_npu.patches.torchtitan import _trainer_config_stash

    sentinel = object()
    recorded = {}

    def fake_build(self, model_parts=None, parallel_dims=None, ft_manager=None, **kw):
        recorded["swap_optimizer_times"] = self.swap_optimizer_times
        recorded["swap_merge_buckets"] = self.swap_merge_buckets
        recorded["model_parts"] = model_parts
        return sentinel

    monkeypatch.setattr(
        swap_mod.SwapMuonHybridOptimizersContainer.Config,
        "build",
        fake_build,
    )

    monkeypatch.setattr(
        _trainer_config_stash,
        "_active_parallel_dims",
        cpu_parallel_dims,
    )
    monkeypatch.setattr(
        OptimizerConfig,
        "build",
        NpuOptimizerDispatcher.dispatch_build,
    )
    result = _make_swap_muon_optimizer_config().build(
        model_parts=[],
        ft_manager=None,
    )

    assert result is sentinel
    assert recorded["swap_optimizer_times"] == 8
    assert recorded["swap_merge_buckets"] == 4


def test_swap_muon_state_lifecycle(monkeypatch):
    from torchtitan_npu.patches.optimizer.swap_optimizer import SwapMuonState

    p = torch.randn(4, 4)

    original_zeros_like = torch.zeros_like

    def zeros_like_no_pin(input, *, pin_memory=False, device=None, **kwargs):
        return original_zeros_like(input, device=device or input.device, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", zeros_like_no_pin)

    import torchtitan_npu.patches.optimizer.swap_optimizer as swap_mod

    monkeypatch.setattr(swap_mod.torch, "zeros_like", zeros_like_no_pin)

    class _FakeStream:
        def record_event(self):
            return None

    class _FakeDeviceModule:
        Stream = _FakeStream

        @staticmethod
        def current_stream():
            return _FakeStream()

    fake_device = _FakeDeviceModule()
    swap_state = SwapMuonState(p, fake_device)

    momentum_buffer = torch.randn(4, 4)
    state = {"momentum_buffer": momentum_buffer}
    swap_state.optim_state = state

    swap_state.init_from_momentum_buffer(momentum_buffer)

    assert swap_state.cpu_momentum is not None
    assert torch.allclose(swap_state.cpu_momentum, momentum_buffer)
    assert state["momentum_buffer"] is None
    assert swap_state.on_device is False

    swap_state.swap_to_device(stream=None)
    assert state["momentum_buffer"] is not None
    assert torch.allclose(state["momentum_buffer"], swap_state.cpu_momentum)
    assert swap_state.on_device

    state["momentum_buffer"].fill_(1.0)

    swap_state.swap_to_host(stream=None)
    assert torch.all(swap_state.cpu_momentum == 1.0)
    assert state["momentum_buffer"] is None
    assert swap_state.on_device is False


def test_swap_merge_buckets_scheduling():
    from torchtitan_npu.patches.optimizer.muon_optimizer import (
        DistributedMuon,
        SwapMergeContext,
    )

    opt = DistributedMuon.__new__(DistributedMuon)

    opt._swap_merge_buckets = 4
    assert opt._swap_merge_buckets == 4

    total_buckets = 10
    swap_merge_buckets = opt._swap_merge_buckets
    num_merge_groups = math.ceil(total_buckets / swap_merge_buckets)
    assert num_merge_groups == 3

    groups = []
    for merge_idx in range(num_merge_groups):
        start = merge_idx * swap_merge_buckets
        end = min(start + swap_merge_buckets, total_buckets)
        groups.append((start, end))

    assert groups[0] == (0, 4)
    assert groups[1] == (4, 8)
    assert groups[2] == (8, 10)

    swap_ctx = SwapMergeContext(
        merge_buckets=swap_merge_buckets,
        use_swap=True,
        to_device_stream=None,
        to_host_stream=None,
    )
    assert swap_ctx.merge_buckets == 4
    assert swap_ctx.use_swap is True

    opt._swap_merge_buckets = 1
    total_buckets = 5
    num_merge_groups = math.ceil(total_buckets / opt._swap_merge_buckets)
    assert num_merge_groups == 5


def test_swap_muon_hybrid_checkpoint_roundtrip(monkeypatch):
    from torchtitan_npu.patches.optimizer.swap_optimizer import (
        SwapMuonHybridOptimizersContainer,
        SwapMuonState,
    )

    original_zeros_like = torch.zeros_like

    def zeros_like_no_pin(input, *, pin_memory=False, device=None, **kwargs):
        return original_zeros_like(input, device=device or input.device, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", zeros_like_no_pin)
    import torchtitan_npu.patches.optimizer.swap_optimizer as swap_mod

    monkeypatch.setattr(swap_mod.torch, "zeros_like", zeros_like_no_pin)

    container = SwapMuonHybridOptimizersContainer.__new__(
        SwapMuonHybridOptimizersContainer
    )
    container._muon_swap_states = {}

    p = torch.randn(4, 4)
    state = {"momentum_buffer": None}
    swap_state = SwapMuonState(p, torch)
    swap_state.optim_state = state

    initial_buf = torch.randn(4, 4)
    swap_state.init_from_momentum_buffer(initial_buf)
    container._muon_swap_states[id(p)] = swap_state

    fake_muon_optim = types.SimpleNamespace(state={p: state})

    serialized = container._serialize_momentum_buffer(p, fake_muon_optim)
    assert serialized is not None
    assert swap_state.cpu_momentum is not None
    assert torch.allclose(serialized, swap_state.cpu_momentum)

    container2 = SwapMuonHybridOptimizersContainer.__new__(
        SwapMuonHybridOptimizersContainer
    )
    container2._muon_swap_states = {}

    p2 = torch.randn(4, 4)
    state2 = {"momentum_buffer": torch.randn(4, 4)}
    swap_state2 = SwapMuonState(p2, torch)
    swap_state2.optim_state = state2
    swap_state2.on_device = True
    container2._muon_swap_states[id(p2)] = swap_state2

    fake_muon_optim2 = types.SimpleNamespace(state={p2: state2})

    container2._load_momentum_from_state_dict(
        swap_state2, serialized, fake_muon_optim2, p2
    )

    assert swap_state2.cpu_momentum is not None
    assert torch.allclose(swap_state2.cpu_momentum, serialized)
    assert swap_state2.on_device is False
    assert state2["momentum_buffer"] is None
    assert swap_state2.buf_shape == p2.shape
    assert swap_state2.buf_dtype == p2.dtype
