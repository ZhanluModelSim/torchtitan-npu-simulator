# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pass-level CPU integration: recomputed forward mutations stay faithful.

Traces the *real* fused partial-RoPE wrapper (functional autograd.Function,
CPU-emulated native mutator kernels) into a joint forward+backward FX graph,
tags it the way GraphTrainer's memory policy does, and checks that the
``functionalize_recompute_mutations_pass`` + native
``selective_activation_remat_pass`` pipeline keeps the replayed rotation in
the backward-facing dataflow.

This is a pass-level integration test: the ``autograd_backward`` /
``recompute`` metadata that the full GraphTrainer tracer produces is provided
explicitly here on a fixed-structure fixture, so no GraphTrainer trainer or
NPU device is involved.

The pass passes a torch keyword (propagate_input_mutations) newer than the
repo pin; on such builds the pass-driven cases below skip while the negative
control keeps running unconditionally.
"""

import inspect

import pytest
import torch
from torch._subclasses.functional_tensor import dispatch_functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)

from tests.unit_tests.rope_test_utils import (
    KERNEL_CALLS,
    register_cpu_kernels,
    reset_kernel_counts,
    split_workaround_reference,
)
from torchtitan_npu.override.common.rope import AscPartialComplexRoPE
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer.functionalize_recompute_mutations import (
    functionalize_recompute_mutations_pass,
)

# The pass passes propagate_input_mutations=True, a keyword newer than the
# repo's pinned torch (2.14.0.dev20260719); check environments resolve such
# builds, where the pass-driven cases below cannot run.
_PROPAGATES_INPUT_MUTATIONS = "propagate_input_mutations" in inspect.signature(dispatch_functionalize).parameters

SPLIT, ROTARY_DIM = 4, 4
POSITIONS = torch.arange(1, 4).unsqueeze(0)
# Fixed, unequal channel weights so a dropped rotation visibly shifts the
# gradient instead of cancelling out.
WEIGHT = torch.arange(1.0, SPLIT + ROTARY_DIM + 1.0).view(1, 1, 1, -1)


def _build_joint(inverse):
    fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=ROTARY_DIM, max_seq_len=16, split=SPLIT))

    def joint(x):
        y = fused(x, positions=POSITIONS, inverse=inverse)
        loss = (y * WEIGHT).square().sum()
        (grad,) = torch.autograd.grad(loss, x)
        return loss, grad

    return joint


def _trace_and_tag(inverse):
    """Capture the joint graph and apply GraphTrainer-style metadata."""
    torch.manual_seed(2026)
    x = torch.randn(1, 3, 2, SPLIT + ROTARY_DIM, requires_grad=True)
    gm = make_fx(_build_joint(inverse))(x)

    # The fixed fixture has a single loss reduction; everything after it is
    # the backward region (GraphTrainer's tracer marks this natively).
    in_backward = False
    for node in gm.graph.nodes:
        if in_backward and node.op != "output":
            node.meta["autograd_backward"] = True
        if node.target == torch.ops.aten.sum.default:
            in_backward = True
        # Recompute the whole forward region, mutator included — tagging only
        # the clone would let a saved mutator output pass without ever being
        # recomputed.
        if not in_backward and node.op == "call_function":
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    return gm, x


def _mutator_nodes(gm, backward):
    return [
        n
        for n in gm.graph.nodes
        if "inplace_partial_rotary_mul" in str(n.target) and bool(n.meta.get("autograd_backward")) == backward
    ]


@pytest.mark.skipif(
    not _PROPAGATES_INPUT_MUTATIONS,
    reason=(
        "installed torch predates dispatch_functionalize(propagate_input_mutations=...); "
        "the functionalization backport requires it (see its module docstring)"
    ),
)
@pytest.mark.parametrize("inverse", [False, True])
def test_partial_rope_joint_graph_recompute_matches_reference(inverse):
    register_cpu_kernels()
    gm, x = _trace_and_tag(inverse)

    # The traced graph really carries the wrapper's shape: a forward clone
    # plus one forward and one backward mutator call.
    clones = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
    assert clones, "no forward clone in the traced graph"
    assert _mutator_nodes(gm, backward=False), "no forward mutator in the traced graph"
    assert _mutator_nodes(gm, backward=True), "no backward mutator in the traced graph"

    reset_kernel_counts()
    expected = gm(x)
    base_calls = dict(KERNEL_CALLS)
    x_snapshot = x.detach().clone()

    fixed = selective_activation_remat_pass(functionalize_recompute_mutations_pass(gm, (x,)))
    fixed.graph.lint()
    assert any("recomputed" in n.name and "auto_functionalized" in str(n.target) for n in fixed.graph.nodes), (
        "the mutation was not recomputed through auto_functionalized_v2"
    )

    reset_kernel_counts()
    actual = fixed(x)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)

    # Independent math reference (workaround path), so a wrapper bug cannot
    # hide behind comparing two copies of the same mistake.
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = split_workaround_reference(x_ref, SPLIT, ROTARY_DIM, POSITIONS, inverse=inverse)
    loss_ref = (y_ref * WEIGHT).square().sum()
    grad_ref = torch.autograd.grad(loss_ref, x_ref)[0]
    torch.testing.assert_close(actual[0], loss_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1], grad_ref, rtol=1e-5, atol=1e-6)

    # The forward rotation really runs twice (original + recompute) and the
    # backward exactly once (no double replay).
    assert KERNEL_CALLS["forward"] == base_calls["forward"] + 1
    assert KERNEL_CALLS["backward"] == base_calls["backward"]
    assert torch.equal(x.detach(), x_snapshot), "the caller's input was polluted"


def test_native_sar_without_functionalization_mismatches():
    """Negative control: the fixture reproduces the historical bug.

    The native SAR alone replays the clone but drops the users-less forward
    mutator, so the backward reads an unrotated clone and the gradient is
    wrong.  Remove this test together with the local
    functionalize_recompute_mutations backport once the pinned TorchTitan
    ships an equivalent pass (the mismatch then disappears by design).
    """
    register_cpu_kernels()
    gm, x = _trace_and_tag(inverse=False)
    expected = gm(x)

    buggy = selective_activation_remat_pass(gm)
    buggy.graph.lint()
    actual = buggy(x)

    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)
