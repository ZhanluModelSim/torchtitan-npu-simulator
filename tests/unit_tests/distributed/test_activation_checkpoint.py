# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from torch.utils._python_dispatch import is_in_torch_dispatch_mode
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from torchtitan.distributed import activation_checkpoint as titan_ac

from torchtitan_npu.distributed.activation_checkpoint import (
    create_scoped_selective_checkpoint_contexts,
    extend_selective_ac_save_ops,
    retain_op_output,
)

_GROUPED_MM_ATTR = "_grouped_mm"


def _get_titan_sac_save_ops():
    return vars(titan_ac)["_get_save_ops"]()


def _get_grouped_mm_op():
    return getattr(torch.ops.aten, _GROUPED_MM_ATTR).default


def _native_mm_save_ops():
    return {torch.ops.aten.mm.default}


@pytest.fixture
def cpu_checkpoint_device_type():
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        yield
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)


def test_extend_selective_ac_save_ops_is_scoped(monkeypatch):
    native_op = torch.ops.aten.mm.default
    original_get_save_ops = _native_mm_save_ops
    monkeypatch.setattr(titan_ac, "_get_save_ops", original_get_save_ops)

    grouped_mm_op = _get_grouped_mm_op()
    with extend_selective_ac_save_ops({grouped_mm_op}):
        assert _get_titan_sac_save_ops() == {
            native_op,
            grouped_mm_op,
        }

    assert vars(titan_ac)["_get_save_ops"] is original_get_save_ops


def test_extend_selective_ac_save_ops_restores_policy_after_error(monkeypatch):
    original_get_save_ops = _native_mm_save_ops
    monkeypatch.setattr(titan_ac, "_get_save_ops", original_get_save_ops)

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        extend_selective_ac_save_ops({_get_grouped_mm_op()}),
    ):
        assert _get_grouped_mm_op() in _get_titan_sac_save_ops()
        raise RuntimeError("expected failure")

    assert vars(titan_ac)["_get_save_ops"] is original_get_save_ops


def test_scoped_selective_ac_is_narrow_and_preserves_gradients(
    cpu_checkpoint_device_type,
):
    observations = []
    mm_op = torch.ops.aten.mm.default

    def observed_mm(x, weight):
        observations.append(("retained", is_in_torch_dispatch_mode()))
        return torch.mm(x, weight)

    def function(x, weight, *, scoped):
        observations.append(("before", is_in_torch_dispatch_mode()))
        x = torch.sigmoid(x)
        x = retain_op_output({mm_op}, observed_mm, x, weight) if scoped else torch.mm(x, weight)
        observations.append(("after", is_in_torch_dispatch_mode()))
        return torch.sigmoid(x)

    base_x = torch.randn(8, 8, device="cpu")
    base_weight = torch.randn(8, 8, device="cpu")

    def gradients(scoped):
        x = base_x.detach().clone().requires_grad_()
        weight = base_weight.detach().clone().requires_grad_()
        kwargs = {"use_reentrant": False, "scoped": scoped}
        if scoped:
            kwargs["context_fn"] = partial(create_scoped_selective_checkpoint_contexts, {mm_op})
        output = checkpoint(function, x, weight, **kwargs)
        output.sum().backward()
        return x.grad, weight.grad

    full_grads = gradients(False)
    observations.clear()
    scoped_grads = gradients(True)

    assert all(active for label, active in observations if label == "retained")
    assert not any(active for label, active in observations if label != "retained")
    assert all(torch.allclose(actual, expected) for actual, expected in zip(scoped_grads, full_grads, strict=True))
