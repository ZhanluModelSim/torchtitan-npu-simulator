# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU contracts for the explicit Muon and AdamW state-swap override policy."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, OverrideConfig, apply_overrides

from torchtitan_npu.override.common.muon_state_swap import (
    MuonSwapCheckpointManager,
    MuonSwapOptimizersContainer,
    _install_adamw_swap_adapter,
)
from torchtitan_npu.override.common import muon_state_swap as product_swap


pytestmark = pytest.mark.cpu


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )


def test_muon_swap_override_replaces_optimizer_and_checkpoint_configs() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.muon_state_swap.muon_state_swap",
                "torchtitan_npu.override.common.muon_state_swap.muon_state_swap_checkpoint",
            ]
        ),
        config,
    )

    assert len(replacements) == 2
    assert isinstance(config.optimizer, MuonSwapOptimizersContainer.Config)
    assert isinstance(config.checkpoint, MuonSwapCheckpointManager.Config)


def test_muon_swap_conflicts_with_virtual_optimizer_override() -> None:
    config = _Root.Config()

    with pytest.raises(ValueError, match="both claim node 'optimizer'"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_npu.override.common.optimizer.virtual",
                    "torchtitan_npu.override.common.muon_state_swap.muon_state_swap",
                ]
            ),
            config,
        )


def test_muon_swap_rejects_checkpoint_and_state_dict_access() -> None:
    manager = object.__new__(MuonSwapCheckpointManager)
    with pytest.raises(ValueError, match="does not support checkpoint save or load"):
        manager.__init__(config=MuonSwapCheckpointManager.Config(enable=True))

    optimizers = object.__new__(MuonSwapOptimizersContainer)
    with pytest.raises(RuntimeError, match="does not support optimizer checkpoint save"):
        optimizers.state_dict()
    with pytest.raises(RuntimeError, match="does not support optimizer checkpoint load"):
        optimizers.load_state_dict({})


def test_muon_swap_has_no_unowned_cleanup_facade() -> None:
    source = product_swap.__file__
    assert source is not None
    contents = Path(source).read_text()

    assert "close_swap_state" not in contents
    assert "_torchtitan_npu_close_swap_state" not in contents
    assert "_torchtitan_npu_swap_error" not in contents


def test_adamw_swap_hooks_offload_moments_after_first_step(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    events = []
    phases = {}

    def register_tensor(tensor, name) -> None:
        events.append(("register", name, tensor))

    def execute(name, action) -> None:
        events.append(("execute", name, action))
        if action == "D2H":
            phases[name] = "D2H"

    fake_swap_api = SimpleNamespace(
        register_tensor=register_tensor,
        execute=execute,
        get_handle_phase=lambda name: phases.get(name),
        remove_tensor=lambda name: events.append(("remove", name)),
    )
    monkeypatch.setattr(product_swap, "swap_api", fake_swap_api)
    _install_adamw_swap_adapter(optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    names = {
        f"adamw.{id(parameter)}.exp_avg",
        f"adamw.{id(parameter)}.exp_avg_sq",
    }
    assert {event[1] for event in events if event[0] == "register"} == names
    assert not hasattr(optimizer, "_torchtitan_npu_swap_names")
    assert [event[2] for event in events if event[0] == "execute"] == [
        "D2H",
        "D2H",
    ]

    events.clear()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "H2D",
        "WAIT_DEVICE",
        "WAIT_DEVICE",
        "D2H",
        "D2H",
    ]
