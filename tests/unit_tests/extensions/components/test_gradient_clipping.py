# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchtitan.components.optim import AdamW, LRSchedulersContainer

from torchtitan_npu.config import OptimConfig
from torchtitan_npu.extensions.components.gradient_clipping import GradientClippingOptim
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer


def _parallelism_context():
    return SimpleNamespace(
        get_optional_mesh=lambda _name: None,
        pp_enabled=False,
        ep_enabled=False,
    )


def test_optimization_uses_container_clipping_before_update(monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, -2.0]]))
    expected = torch.nn.Parameter(model.weight.detach().clone())
    reference = torch.optim.AdamW([expected], lr=0.1, eps=0.01, weight_decay=0.0, fused=False)
    grad = torch.tensor([[3.0, 4.0]])
    expected.grad = grad.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_([expected], 1.0)
    reference.step()
    calls = []
    original_clip = HostSparseOptimizersContainer.clip_grad_norm

    def record_clip(container, parameters, max_norm, **kwargs):
        calls.append(max_norm)
        return original_clip(container, parameters, max_norm, **kwargs)

    monkeypatch.setattr(HostSparseOptimizersContainer, "clip_grad_norm", record_clip)
    config = OptimConfig(
        optimizer=HostSparseOptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", lr=0.1, eps=0.01, weight_decay=0.0, fused=False)]
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0),
        max_norm=1.0,
    )
    optim = config.build(
        model_parts=[model],
        parallelism_context=_parallelism_context(),
        training_steps=10,
        pp_has_last_stage=True,
    )
    model.weight.grad = grad.clone()

    norm = optim.step(torch.tensor(1.0), current_step=1)

    assert isinstance(optim, GradientClippingOptim)
    assert calls == [1.0]
    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(model.weight, expected)


def test_nonfinite_loss_does_not_commit_optimizer_update():
    model = torch.nn.Linear(2, 1, bias=False)
    before = model.weight.detach().clone()
    config = OptimConfig(
        optimizer=HostSparseOptimizersContainer.Config(optimizers=[AdamW.Config(pattern=".*", lr=0.1, fused=False)])
    )
    optim = config.build(
        model_parts=[model],
        parallelism_context=_parallelism_context(),
        training_steps=10,
        pp_has_last_stage=True,
    )
    model.weight.grad = torch.ones_like(model.weight)

    with pytest.raises(RuntimeError, match="Loss or gradient norm is not finite"):
        optim.step(torch.tensor(float("nan")), current_step=1)

    torch.testing.assert_close(model.weight, before)
