# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Policy boundary regression; real FSDP communication is covered by the GA audit."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan_simulator.compat.loss import install_chunked_loss_policy_restore
from torchtitan_simulator.compat.patches import PatchScope


class PolicyHead(FSDPModule, torch.nn.Module):
    """Expose only the FSDP policy boundary, including multiple param groups."""

    def __new__(cls, groups):
        return object.__new__(cls)

    def __init__(self, groups):
        torch.nn.Module.__init__(self)
        self.state = SimpleNamespace(_fsdp_param_groups=groups, _auto_reshard_after_forward=True)

    def _get_fsdp_state(self):
        return self.state


def group(mesh, backward=False, reduce=False, all_reduce=True):
    return SimpleNamespace(
        post_forward_mesh_info=mesh,
        reshard_after_backward=backward,
        reduce_grads=reduce,
        all_reduce_grads=all_reduce,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_restore_each_group_and_recursive_policy_on_success_or_failure(monkeypatch, fail):
    original_mesh = object()  # Preserve the mesh object, not a lossy bool policy.
    first = group(None)
    second = group(original_mesh, backward=True, reduce=True, all_reduce=False)
    child_group = group(object())
    head = PolicyHead([first, second])
    head.child = PolicyHead([child_group])
    saved = [(item, vars(item).copy()) for item in (first, second, child_group)]
    marker = object()

    def native_call(self, pred, **kwargs):
        assert self.lm_head is head and pred is marker and kwargs == {"example": 4}
        for module in head.modules():
            module.state._auto_reshard_after_forward = False
            for item in module.state._fsdp_param_groups:
                item.post_forward_mesh_info = None
                item.reshard_after_backward = False
        for item in head.state._fsdp_param_groups:
            item.reduce_grads = True
            item.all_reduce_grads = True
        if fail:
            raise ValueError("native loss failure")
        return marker

    monkeypatch.setattr(ChunkedLossWrapper, "__call__", native_call)
    loss = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=2))
    loss.set_lm_head(head)
    with PatchScope() as scope:
        install_chunked_loss_policy_restore(scope)
        if fail:
            with pytest.raises(ValueError, match="native loss failure"):
                loss(marker, example=4)
        else:
            assert loss(marker, example=4) is marker
        for item, before in saved:
            assert vars(item) == before
        assert head.state._auto_reshard_after_forward
        assert head.child.state._auto_reshard_after_forward
    assert ChunkedLossWrapper.__call__ is native_call


def test_native_non_fsdp_loss_preserves_outputs_and_gradients():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        head = torch.nn.Linear(4, 8, bias=False)
        loss = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=2))
        loss.set_lm_head(head)
        hidden = torch.randn(1, 4, 4, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, 4]])
        expected, _ = loss(hidden, labels)
        expected.backward()
        expected_hidden_grad = hidden.grad.clone()
        expected_head_grad = head.weight.grad.clone()
        hidden.grad = None
        head.zero_grad(set_to_none=True)
        original = ChunkedLossWrapper.__call__
        with PatchScope() as scope:
            install_chunked_loss_policy_restore(scope)
            actual, _ = loss(hidden, labels)
            actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden.grad, expected_hidden_grad)
        torch.testing.assert_close(head.weight.grad, expected_head_grad)
        assert ChunkedLossWrapper.__call__ is original
