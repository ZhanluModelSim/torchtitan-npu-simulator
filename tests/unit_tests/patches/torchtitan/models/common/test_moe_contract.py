# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU coverage for score ordering and gated activation contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def patches():
    # Import in an isolated process when running this file: package startup requires
    # the CANN runtime, but these two Python implementations are CPU-capable.
    root = Path(__file__).resolve().parents[6] / "torchtitan_npu/patches/torchtitan/models/common"
    import torchtitan.models.common.config_utils as config_utils
    import torchtitan.models.common.moe as moe_package
    import torchtitan.models.common.moe.moe as moe_impl
    import torchtitan.models.common.moe.token_dispatcher as dispatch_impl

    patcher = pytest.MonkeyPatch()
    for module, names in (
        (
            config_utils,
            (
                "RoutedExperts",
                "LocalTokenDispatcher",
                "AllToAllTokenDispatcher",
                "DeepEPTokenDispatcher",
                "make_routed_experts_config",
                "make_ffn_config",
            ),
        ),
        (moe_package, ("RoutedExperts",)),
        (moe_impl, ("RoutedExperts", "LocalTokenDispatcher")),
        (dispatch_impl, ("LocalTokenDispatcher", "AllToAllTokenDispatcher", "DeepEPTokenDispatcher")),
    ):
        for name in names:
            if hasattr(module, name):
                patcher.setattr(module, name, getattr(module, name))
            else:
                patcher.setattr(module, name, None, raising=False)
    modules = []
    for name, file in (("_moe_dispatch", "token_dispatcher.py"), ("_moe_patch", "moe.py")):
        spec = importlib.util.spec_from_file_location(name, root / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    yield modules
    patcher.undo()
    sys.modules.pop("_moe_dispatch", None)
    sys.modules.pop("_moe_patch", None)


def test_clamped_activation_uses_fused_gate_up_and_backward(patches):
    _, moe = patches
    gate_up = torch.tensor([[[-2.0, 2.0], [3.0, -4.0]], [[1.0, 0.5], [-2.0, 2.0]]], requires_grad=True)
    reference = gate_up.detach().clone().requires_grad_()
    actual = moe.LimitedSwiGLU.Config(swiglu_limit=1.0).build()(gate_up)
    gate, up = reference.unbind(-2)
    expected = F.silu(gate.clamp(max=1.0)) * up.clamp(-1.0, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(gate_up.grad, reference.grad, rtol=0, atol=0)


def test_ep_permutation_aligns_scores_with_empty_expert(patches, monkeypatch):
    dispatcher, _ = patches
    obj = dispatcher.AllToAllTokenDispatcher(dispatcher.AllToAllTokenDispatcher.Config(num_experts=4, top_k=1))
    monkeypatch.setattr(
        dispatcher.AllToAllTokenDispatcher, "ep_mesh", property(lambda self: SimpleNamespace(size=lambda: 2))
    )
    tokens = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    # Rank-major counts: rank 0 -> (1,1), rank 1 -> (2,0).
    routed, indices, counts = obj._permute(tokens, torch.tensor([1, 1, 2, 0]))
    torch.testing.assert_close(routed.flatten(), torch.tensor([0.0, 2.0, 3.0, 1.0]))
    torch.testing.assert_close(scores[indices], torch.tensor([0.1, 0.3, 0.4, 0.2]))
    torch.testing.assert_close(counts, torch.tensor([3, 1]))


def test_local_absorption_multiplies_scores_only_once(patches):
    dispatcher, _ = patches
    cls = dispatcher.LocalTokenDispatcher
    obj = cls(cls.Config(num_experts=3, top_k=2, absorb_router_scores=True))
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    scores = torch.tensor([[0.25, 0.75], [0.5, 0.5]], requires_grad=True)
    ids = torch.tensor([[1, 0], [0, 1]])
    routed, counts, metadata = obj.dispatch(x, scores, ids, torch.tensor([2, 2, 0]))
    torch.testing.assert_close(metadata.routed_scores_R, torch.tensor([0.75, 0.5, 0.25, 0.5]))
    result = obj.combine(routed * metadata.routed_scores_R[:, None], metadata, x)
    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(scores.grad, x.detach().sum(-1, keepdim=True).expand_as(scores))
    torch.testing.assert_close(counts, torch.tensor([2, 2, 0]))


def test_routed_experts_scores_before_w2_with_bfloat16_rounding_and_padding(patches):
    dispatcher, moe = patches
    from torchtitan.models.common.linear import GroupedLinear

    cfg = moe._RouterScoreAbsorbingRoutedExperts.Config(
        w13=GroupedLinear.Config(group_size=2, in_features=2, out_features=2, num_linears=2),
        w2=GroupedLinear.Config(group_size=2, in_features=2, out_features=2),
        token_dispatcher=dispatcher.LocalTokenDispatcher.Config(num_experts=2, top_k=1, absorb_router_scores=True),
        activation_fn=moe.LimitedSwiGLU.Config(swiglu_limit=1.0),
    )
    experts = cfg.build()
    captured = []

    class Up(torch.nn.Module):
        def forward(self, x, offsets):
            assert offsets.tolist() == [1, 2]
            return torch.stack((x + 1, x * 2), dim=-2)

    class Down(torch.nn.Module):
        def forward(self, x, offsets):
            captured.append(x)
            return x

    experts.w13, experts.w2 = Up(), Down()
    x = torch.tensor([[0.4, -0.8], [1.1, 0.25]], dtype=torch.bfloat16, requires_grad=True)
    scores = torch.tensor([[0.123], [0.679]], requires_grad=True)
    out = experts(x, scores, torch.tensor([[1], [0]]), torch.tensor([1, 1]), padding_mask_T=torch.tensor([False, True]))
    ordered = x[[1, 0]]
    gate = (ordered + 1).clamp(max=1.0)
    up = (ordered * 2).clamp(-1.0, 1.0)
    hidden = F.silu(gate) * up
    expected = (hidden.float() * scores[[1, 0]]).to(torch.bfloat16)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(out, expected[[1, 0]], rtol=0, atol=0)
    out.sum().backward()
    assert scores.grad is not None and x.grad is not None


def test_ep_dispatch_transports_and_permutes_scores_once(patches, monkeypatch):
    dispatcher, _ = patches
    cls = dispatcher.AllToAllTokenDispatcher
    monkeypatch.setattr(cls, "ep_mesh", property(lambda self: SimpleNamespace(size=lambda: 2)))
    obj = cls(cls.Config(num_experts=4, top_k=1, absorb_router_scores=True))
    monkeypatch.setattr(obj, "_token_count_exchange", lambda counts, group, size: counts)
    monkeypatch.setattr(
        obj,
        "_sync_token_count_exchange",
        lambda counts, exchanged, size: (torch.tensor([1, 1, 2, 0]), torch.tensor([2, 2]), torch.tensor([2, 2])),
    )
    calls = []

    def exchange(tensor, group, output_splits, input_splits):
        calls.append(tensor)
        return tensor

    monkeypatch.setattr(obj, "_dispatch_token_exchange", exchange)
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    scores = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    routed, counts, metadata = obj.dispatch(x, scores, torch.arange(4).reshape(-1, 1), torch.ones(4, dtype=torch.int64))
    torch.testing.assert_close(routed.flatten(), torch.tensor([0.0, 2.0, 3.0, 1.0]))
    torch.testing.assert_close(metadata.routed_scores_R, torch.tensor([0.1, 0.3, 0.4, 0.2]))
    torch.testing.assert_close(counts, torch.tensor([3, 1]))
    assert len(calls) == 2
    assert metadata.input_splits == [2, 2] and metadata.output_splits == [2, 2]


def test_cann_activation_packs_gate_up_and_preserves_gradient(monkeypatch):
    path = Path(__file__).resolve().parents[6] / "torchtitan_npu/override/common/swiglu_group/ascendc.py"
    spec = importlib.util.spec_from_file_location("_v04_asc_activation", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    seen = []

    def fake_op(packed, limit):
        seen.append((packed, limit))
        gate, up = packed.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)

    monkeypatch.setattr(module, "swiglu_group_activation", fake_op)
    gate_up = torch.tensor([[[2.0, -1.0], [3.0, -2.0]]], requires_grad=True)
    ref = gate_up.detach().clone().requires_grad_()
    actual = module.AscSwiGLU.Config(swiglu_limit=1.0).build()(gate_up, offsets=torch.tensor([1]))
    expected = F.silu(ref[:, 0].clamp(max=1.0)) * ref[:, 1].clamp(-1.0, 1.0)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(gate_up.grad, ref.grad)
    torch.testing.assert_close(seen[0][0], gate_up.flatten(-2))
    assert seen[0][1] == 1.0
