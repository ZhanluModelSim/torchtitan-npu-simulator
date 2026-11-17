# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.components.optimizer import ParamGroupConfig

from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx


def make_optimizer(accepted):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    manager = Mock()
    manager._pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    manager.should_commit.return_value = accepted
    ft = SimpleNamespace(manager=manager, process_group=manager._pg, use_async_quorum=False)
    config = TorchFTOptimizersContainerEx.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 0.1,
                    "betas": (0.0, 0.0),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            )
        ],
        implementation="for-loop",
    )
    optimizer = config.build(model_parts=[model], ft_manager=ft)
    return model, optimizer, manager


def test_accepted_step_updates_parameter_once_and_advances_scheduler_once(monkeypatch):
    model, optimizer, manager = make_optimizer(True)
    monkeypatch.setattr("torch_npu.npu.current_stream", lambda: Mock(synchronize=Mock()))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizers[0], step_size=1, gamma=0.5)
    hooks = []
    optimizer.register_step_pre_hook(lambda *_: hooks.append("step"))

    assert optimizer.hccl_recovery is manager._pg.recovery
    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    scheduler.step()

    torch.testing.assert_close(model.weight, torch.tensor([[0.9]]))
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.05
    assert hooks == ["step"]
    manager.should_commit.assert_called_once_with()
    manager._begin_step.assert_called_once_with()


def test_rejected_step_skips_optimizer_and_preserves_upstream_scheduler_advancement():
    model, optimizer, manager = make_optimizer(False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizers[0], step_size=1, gamma=0.5)
    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()

    optimizer.step()
    with pytest.warns(UserWarning, match=r"lr_scheduler\.step"):
        scheduler.step()

    torch.testing.assert_close(model.weight, torch.ones(1, 1))
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.05
    assert not optimizer.hccl_recovery.optimizer_update_completed
    manager.should_commit.assert_called_once_with()
