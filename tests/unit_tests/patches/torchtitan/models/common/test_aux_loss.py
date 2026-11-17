# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts of the backported auxiliary-loss framework.

Covers `torchtitan_npu.patches.torchtitan.models.common.aux_loss`: the injected
gradient and the per-instance metric of ``inject``, the optimizer pre-hook that
rolls instances into metric groups, the log-time reduction, and the two package
patches that call into it (``Trainer.train_step`` and ``MetricsProcessor.log``).
The multi-rank collectives themselves are NPU-side: the fake mesh here returns no
mesh at all, so only the single-process divisor and tag are asserted.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import (
    AuxLoss,
    _zero_aux_losses,
    collect_aux_loss_metrics,
)

_COEFF = 0.25
_DENOMINATOR = 8.0


class DemoLoss(AuxLoss):
    """Aux-loss module with no product path: the framework's own contract is under test."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass


class OtherDemoLoss(AuxLoss):
    """A second metric group, so the roll-up has more than one key to keep apart."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass


def _no_mesh_parallel_dims() -> SimpleNamespace:
    """ParallelDims stand-in with no mesh: the single-process code path of the reduction."""
    return SimpleNamespace(get_optional_mesh=lambda name: None)


@pytest.fixture(autouse=True)
def _clean_aux_loss_state():
    """The registry and the step denominator are process state; restore them around every test."""
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def test_inject_is_identity_and_scales_gradient_and_register_by_coeff_and_denominator():
    """Hand-computed from the definition: gradient ``coeff/denominator``, register ``raw/denominator``."""
    AuxLoss.set_step_denominator(torch.tensor(_DENOMINATOR, dtype=torch.float64))
    loss = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    carrier = torch.randn(3, dtype=torch.float64, requires_grad=True)
    raw_sum = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)

    out = loss.inject(raw_sum, carrier=carrier)

    assert torch.equal(out, carrier)
    assert float(loss.instance_acc) == pytest.approx(6.0 / _DENOMINATOR)
    out.sum().backward()
    assert torch.equal(carrier.grad, torch.ones_like(carrier))
    assert float(raw_sum.grad) == pytest.approx(_COEFF / _DENOMINATOR)


def test_inject_refuses_a_forward_before_the_step_denominator_is_set():
    """The framework cannot normalize a step it was never told the token count of."""
    loss = DemoLoss(DemoLoss.Config(coeff=_COEFF))

    with pytest.raises(ValueError, match="set_step_denominator"):
        loss.inject(torch.zeros(()), carrier=torch.zeros(1))


def test_the_step_hook_rolls_instances_into_their_groups_and_zeroes_them():
    """Two instances of one class share a group; the next step starts from zero, not from the total."""
    first = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    second = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    other = OtherDemoLoss(OtherDemoLoss.Config(coeff=_COEFF))
    first.instance_acc.add_(1.5)
    second.instance_acc.add_(0.5)
    other.instance_acc.add_(4.0)

    _zero_aux_losses([nn.ModuleList([first, second]), other])

    assert set(AuxLoss.group_acc) == {("batch", "demo_loss"), ("batch", "other_demo_loss")}
    assert float(AuxLoss.group_acc[("batch", "demo_loss")]) == pytest.approx(2.0)
    assert float(AuxLoss.group_acc[("batch", "other_demo_loss")]) == pytest.approx(4.0)
    assert float(first.instance_acc) == 0.0
    assert float(second.instance_acc) == 0.0

    _zero_aux_losses([nn.ModuleList([first])])

    assert set(AuxLoss.group_acc) == {("batch", "demo_loss")}
    assert float(AuxLoss.group_acc[("batch", "demo_loss")]) == 0.0


def test_collect_returns_each_groups_mean_over_its_build_time_instance_count():
    """Two layers of one group average to their mean, which is the divisor the metric carries."""
    first = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    second = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    first.instance_acc.add_(1.0)
    second.instance_acc.add_(3.0)
    _zero_aux_losses([nn.ModuleList([first, second])])

    metrics = collect_aux_loss_metrics(_no_mesh_parallel_dims())

    assert metrics == {"demo_loss/mean": pytest.approx(2.0)}


def test_the_trainer_patch_sets_the_step_denominator_inside_train_step():
    """The pinned method has no such call; the installed one does, and it is this module's class."""
    import inspect

    import torchtitan.trainer as trainer_module
    from torchtitan.trainer import Trainer

    assert "set_step_denominator" not in inspect.getsource(trainer_module)
    assert "set_step_denominator" in Trainer.train_step.__code__.co_names
    assert Trainer.train_step.__globals__["AuxLoss"] is AuxLoss


def test_the_metrics_patch_merges_the_aux_metric_into_the_logged_step_metrics(monkeypatch):
    """The consumer is the real log call: the aux metric reaches it in ``extra_metrics``."""
    from torchtitan.components.metrics import MetricsProcessor

    from torchtitan_npu.patches.torchtitan.components import metrics as metrics_patch

    # Another NPU patch wraps the method after this one, so the replacement is
    # confirmed through the wrapper chain rather than by identity.
    wrappers, installed = [], MetricsProcessor.log
    while installed is not None:
        wrappers.append(installed)
        installed = getattr(installed, "__wrapped__", None)
    assert metrics_patch.patched_log in wrappers
    logged: dict = {}
    monkeypatch.setattr(
        metrics_patch,
        "original_log",
        lambda self, step, *args, **kwargs: logged.update({"step": step, **kwargs}),
    )
    loss = DemoLoss(DemoLoss.Config(coeff=_COEFF))
    loss.instance_acc.add_(0.5)
    _zero_aux_losses([nn.ModuleList([loss])])
    processor = SimpleNamespace(parallel_dims=_no_mesh_parallel_dims())

    metrics_patch.patched_log(processor, 3, 1.0, 1.0, 0.0, extra_metrics={"lr": 0.1})

    assert logged["step"] == 3
    assert logged["extra_metrics"] == {"lr": 0.1, "demo_loss/mean": pytest.approx(0.5)}
