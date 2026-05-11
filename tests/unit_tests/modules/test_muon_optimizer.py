# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types

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
    ADAMW_STATE_KEYS,
    build_muon_hybrid_optimizers,
    build_muon_lr_schedulers,
    MUON_STATE_KEYS,
    MuonHybridOptimizersContainer,
    MuonLRSchedulersContainer,
    zeropower_via_newtonschulz5,
)
from torchtitan_npu.patches.optimizer.virtual_allocator import (
    ALL_VIRTUAL_KEYS,
    is_swap_device,
    unwrap_dtensor,
)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16, bias=True)
        self.embed = nn.Linear(4, 8, bias=False)
        self.norm = nn.LayerNorm(16)
        self.expert_weight = nn.Parameter(torch.randn(4, 8, 16))


def _build_container(muon_optimizer_config, cpu_parallel_dims, virtual=False):
    model = _DummyModel()
    opt_config = muon_optimizer_config().to_namespace()
    return (
        build_muon_hybrid_optimizers(
            [model],
            opt_config,
            cpu_parallel_dims,
            virtual_allocator=virtual,
        ),
        model,
    )


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
    model.frozen.weight.requires_grad = False
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
    assert not torch.equal(
        model.linear.weight.data, orig_weight
    ), "Muon optimizer step should update parameters"


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


# --- TestBuildOptimizersWrapper ---


def test_muon_with_swap_optimizer_raises():
    import torchtitan.components.optimizer as tt_optimizer

    optimizer_config = types.SimpleNamespace(
        name="Muon",
        swap_optimizer=True,
        virtual_optimizer=False,
    )
    with pytest.raises(ValueError, match="does not support swap_optimizer"):
        tt_optimizer.build_optimizers(
            model_parts=[],
            optimizer_config=optimizer_config,
            parallel_dims=None,
            ft_manager=None,
        )


def test_muon_routes_correctly(muon_optimizer_config, cpu_parallel_dims):
    import torchtitan.components.optimizer as tt_optimizer

    model = _DummyModel()
    opt_config = muon_optimizer_config().to_namespace()
    result = tt_optimizer.build_optimizers(
        model_parts=[model],
        optimizer_config=opt_config,
        parallel_dims=cpu_parallel_dims,
        ft_manager=None,
    )
    assert isinstance(result, MuonHybridOptimizersContainer)


# --- TestVirtualUtils ---


def test_muon_state_keys():
    assert MUON_STATE_KEYS == ["momentum_buffer"]


def test_adamw_state_keys():
    assert ADAMW_STATE_KEYS == ["exp_avg", "exp_avg_sq"]


def test_all_virtual_keys():
    assert set(ALL_VIRTUAL_KEYS) == {"momentum_buffer", "exp_avg", "exp_avg_sq"}


def test_is_swap_device():
    assert is_swap_device(torch.device("cpu"))
    assert not is_swap_device(torch.device("meta"))


def test_unwrap_dtensor_plain_tensor():
    t = torch.randn(2, 2)
    assert unwrap_dtensor(t) is t


# --- TestMuonLRScheduler ---


def _build_optimizers(muon_optimizer_config, cpu_parallel_dims, **config_overrides):
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(**config_overrides).to_namespace()
    return build_muon_hybrid_optimizers([model], opt_config, cpu_parallel_dims)


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
