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
    MuonHybridOptimizersContainer,
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
        extra_param_group_split_rules=None,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        implementation="for-loop",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


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
    config = _muon_config()
    cfg = MuonHybridOptimizersContainer.Config(**config.__dict__)
    container = cfg.build(model_parts=[model], parallel_dims=npu_parallel_dims)

    losses = []
    for _ in range(3):
        loss = _step_model(model, npu_device)
        container.step()
        container.zero_grad()
        losses.append(loss.item())

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert losses[-1] < losses[0], f"Loss should decrease over steps: {losses}"
