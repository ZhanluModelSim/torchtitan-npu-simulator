# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused partial-RoPE under ``torch.compile``: CPU fullgraph AOT contract.

The real wrapper (functional autograd.Function over the two native mutator
schemas, emulated on CPU by ``rope_test_utils``; its lazy CANN import
is pre-resolved by ``register_cpu_kernels`` before compiling, so the
first-call-in-a-fresh-process import is covered by device runs) is compiled with
``backend="aot_eager", fullgraph=True`` and must survive the AOT joint trace
of *both* the forward clone+mutator and the backward mutator, matching the
independent workaround reference.  These are CPU contract tests only: they
prove the wrapper's compile behavior, not NPU kernel execution (the A5
native path is validated on device separately).
"""

import pytest
import torch

from tests.unit_tests.rope_test_utils import (
    register_cpu_kernels,
    split_workaround_reference,
)
from torchtitan_npu.override.common.rope import AscPartialComplexRoPE

SPLIT, ROTARY_DIM = 4, 4
# Positions are non-zero so a lost rotation cannot hide behind identity.
POSITIONS = torch.arange(1, 4).unsqueeze(0)


def _fresh_pair():
    torch.manual_seed(1234)
    x = torch.randn(1, 3, 2, SPLIT + ROTARY_DIM, dtype=torch.float32, requires_grad=True)
    return x, x.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("inverse", [False, True])
def test_partial_rope_aot_eager_matches_reference(inverse):
    register_cpu_kernels()
    fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=ROTARY_DIM, max_seq_len=16, split=SPLIT))
    compiled = torch.compile(
        lambda q: fused(q, positions=POSITIONS, inverse=inverse),
        backend="aot_eager",
        fullgraph=True,
    )

    x_compiled, x_ref = _fresh_pair()
    snap_compiled, snap_ref = x_compiled.detach().clone(), x_ref.detach().clone()

    out_compiled = compiled(x_compiled)
    out_ref = split_workaround_reference(x_ref, SPLIT, ROTARY_DIM, POSITIONS, inverse=inverse)
    torch.testing.assert_close(out_compiled, out_ref, rtol=1e-5, atol=1e-6)
    # The untouched prefix channels ride along in the clone.
    assert torch.equal(out_compiled[..., :SPLIT], snap_compiled[..., :SPLIT])

    grad_out = torch.randn_like(out_compiled)
    grad_compiled = torch.autograd.grad(out_compiled, x_compiled, grad_out)[0]
    grad_ref = torch.autograd.grad(out_ref, x_ref, grad_out)[0]
    torch.testing.assert_close(grad_compiled, grad_ref, rtol=1e-5, atol=1e-6)

    # The caller's tensors are never mutated by the fused path.
    assert torch.equal(x_compiled.detach(), snap_compiled)
    assert torch.equal(x_ref.detach(), snap_ref)


@pytest.mark.parametrize("inverse", [False, True])
def test_partial_rope_compile_preserves_saved_activation(inverse):
    """A producer whose backward saves its output feeds the fused module
    through a view — the recomputed/replayed forward must see the same
    activation the eager pass saw."""
    register_cpu_kernels()
    fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=ROTARY_DIM, max_seq_len=16, split=SPLIT))
    weight = torch.randn(1, 3, 2, SPLIT + ROTARY_DIM)

    def model(x):
        y = x.exp()
        rotated = fused(y.view_as(y), positions=POSITIONS, inverse=inverse)
        return (rotated * weight).square().sum() / 2

    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    torch.manual_seed(4321)
    x = torch.randn(1, 3, 2, SPLIT + ROTARY_DIM, dtype=torch.float32, requires_grad=True)

    loss = compiled_model(x)
    grad = torch.autograd.grad(loss, x)[0]

    # Eager reference on a value-equal copy: apply the workaround rotation to
    # the same exp() activation and the same weighted loss.
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = x_ref.exp()
    rotated_ref = split_workaround_reference(y_ref, SPLIT, ROTARY_DIM, POSITIONS, inverse=inverse)
    loss_ref = (rotated_ref * weight).square().sum() / 2
    grad_ref = torch.autograd.grad(loss_ref, x_ref)[0]

    torch.testing.assert_close(loss, loss_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, grad_ref, rtol=1e-5, atol=1e-6)
