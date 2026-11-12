# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchtitan.distributed.spmd_types import set_spmd_meshes
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE

from torchtitan_npu.ops.ascendc import moe_re_routing, moe_token_unpermute
from torchtitan_npu.override.common import token_dispatcher as npu_token_dispatcher
from torchtitan_npu.patches.torchtitan.models.common import moe as moe_patch
from torchtitan_npu.patches.torchtitan.models.common import (
    token_dispatcher as dispatcher_patch,
)
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


def test_deepseek_v4_enables_score_absorption_for_standard_dispatch():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")
    model = model_spec.model if hasattr(model_spec, "model") else model_spec
    routed_configs = [
        layer.feed_forward.routed_experts for layer in model.layers if isinstance(layer.feed_forward, MoE.Config)
    ]

    assert routed_configs
    assert all(config.token_dispatcher.absorb_router_scores for config in routed_configs)


@pytest.mark.skip(reason="DeepSeek-V3.2 folded-token migration is deferred past v0.4.0")
def test_deepseek_v3_2_enables_score_absorption():
    from torchtitan_npu.models.deepseek_v3_2 import model_registry

    model_spec = model_registry("debugmodel")
    model = model_spec.model if hasattr(model_spec, "model") else model_spec
    routed_configs = [
        layer.feed_forward.routed_experts for layer in model.layers if isinstance(layer.feed_forward, MoE.Config)
    ]

    assert routed_configs
    assert all(config.token_dispatcher.absorb_router_scores for config in routed_configs)


def test_standard_dispatch_preserves_default_api_and_exposes_aligned_scores():
    default_dispatcher = AllToAllTokenDispatcher(AllToAllTokenDispatcher.Config(num_experts=2, top_k=2))
    absorbed_dispatcher = AllToAllTokenDispatcher(
        AllToAllTokenDispatcher.Config(num_experts=2, top_k=2, absorb_router_scores=True)
    )
    x_TD = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    scores_TK = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    expert_ids_TK = torch.tensor([[1, 0], [0, 1]])
    counts_E = torch.tensor([2, 2])

    default_result = default_dispatcher.dispatch(x_TD, scores_TK, expert_ids_TK, counts_E)
    absorbed_result = absorbed_dispatcher.dispatch(x_TD, scores_TK, expert_ids_TK, counts_E)

    assert len(default_result) == 3
    assert default_result[2].routed_scores_R is None
    assert len(absorbed_result) == 3
    routed_input_RD, _, metadata = absorbed_result
    routed_scores_R = metadata.routed_scores_R
    assert routed_scores_R is not None
    torch.testing.assert_close(routed_input_RD, default_result[0])
    torch.testing.assert_close(routed_scores_R, metadata.topk_scores_experts_sorted_N)

    expert_output_RD = routed_input_RD * 1.5
    baseline = default_dispatcher.combine(
        expert_output_RD,
        default_result[2],
        x_TD,
    )
    candidate = absorbed_dispatcher.combine(
        expert_output_RD * routed_scores_R.to(expert_output_RD.dtype).reshape(-1, 1),
        metadata,
        x_TD,
    )
    torch.testing.assert_close(candidate, baseline, atol=0.02, rtol=0.02)


def test_upstream_dispatcher_config_rejects_absorption_option():
    with pytest.raises(TypeError, match="unexpected keyword argument 'absorb_router_scores'"):
        dispatcher_patch.TorchTitanLocalTokenDispatcher.Config(
            num_experts=2,
            top_k=1,
            absorb_router_scores=True,
        )


def test_standard_dispatchers_reuse_torchtitan_helpers():
    assert issubclass(LocalTokenDispatcher, dispatcher_patch.TorchTitanLocalTokenDispatcher)
    assert issubclass(AllToAllTokenDispatcher, dispatcher_patch.TorchTitanAllToAllTokenDispatcher)
    # The traceable stable-argsort workaround now lives in
    # override/common/token_dispatcher and is opted in at compile time only,
    # so the patch keeps the upstream _local_reorder verbatim.
    assert LocalTokenDispatcher._local_reorder is dispatcher_patch.TorchTitanLocalTokenDispatcher._local_reorder
    assert AllToAllTokenDispatcher._local_reorder is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._local_reorder
    assert (
        AllToAllTokenDispatcher._token_count_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._token_count_exchange
    )
    # Patch override: _sync_token_count_exchange forces synchronous D2H
    # copies for the split lists (non_blocking=False) -- under torch.compile
    # the inductor scheduler may hoist the non-blocking input_splits reads
    # above the blocking output_splits copy and read uninitialized memory.
    assert (
        AllToAllTokenDispatcher._sync_token_count_exchange
        is not dispatcher_patch.TorchTitanAllToAllTokenDispatcher._sync_token_count_exchange
    )
    assert (
        AllToAllTokenDispatcher._dispatch_token_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._dispatch_token_exchange
    )
    assert (
        AllToAllTokenDispatcher._combine_token_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._combine_token_exchange
    )
    assert AllToAllTokenDispatcher._permute is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._permute
    assert AllToAllTokenDispatcher._unpermute is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._unpermute


def test_compile_friendly_local_reorder_matches_upstream():
    """The compile-time opt-in reorder is the exact upstream permutation."""
    pytest.importorskip("torch_npu")
    from torchtitan_npu.override.common.token_dispatcher import (
        configure_dispatchers_for_compile,
    )

    x_TD = torch.randn(8, 3)
    scores_TK = torch.rand(8, 2)
    expert_ids_TK = torch.tensor([[3, 1], [0, 0], [2, 1], [1, 3], [0, 2], [3, 3], [1, 0], [2, 2]])

    for cls in (LocalTokenDispatcher, AllToAllTokenDispatcher):
        config = cls.Config(num_experts=4, top_k=2)
        compiled, eager = cls(config), cls(config)
        original = cls._local_reorder
        configure_dispatchers_for_compile(compiled, eager_all_to_all=False)
        assert cls._local_reorder is original
        assert compiled._local_reorder.__func__ is not original
        assert eager._local_reorder.__func__ is original
        patched = compiled._local_reorder(x_TD, scores_TK, expert_ids_TK)
        upstream = eager._local_reorder(x_TD, scores_TK, expert_ids_TK)
        for got, want in zip(patched, upstream, strict=True):
            torch.testing.assert_close(got, want)


def test_v41_compile_keeps_variable_splits_eager_and_model_local(monkeypatch):
    import torch._inductor.config as inductor_config

    from torchtitan_npu.models.deepseek_v4_1.parallelize import _apply_compile_v4_1

    class SplitDispatcher(AllToAllTokenDispatcher):
        @property
        def ep_mesh(self):
            return object()

        def dispatch(self, x, count):
            assert not torch.compiler.is_compiling()
            return x.split((count, x.numel() - count))

        def combine(self, left, right):
            assert not torch.compiler.is_compiling()
            return torch.cat((left, right))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dispatcher = SplitDispatcher(SplitDispatcher.Config(num_experts=2, top_k=1))

        def forward(self, x, count):
            left, right = self.dispatcher.dispatch(x, count)
            return self.dispatcher.combine(left.square(), right * 3)

    # Run the production compile assembly with AOTAutograd on CPU.
    compile_module = torch.nn.Module.compile
    monkeypatch.setattr(
        torch.nn.Module, "compile", lambda module, **kw: compile_module(module, **(kw | {"backend": "aot_eager"}))
    )
    monkeypatch.setattr(inductor_config, "do_not_emit_runtime_assertions", False)
    for setting in ("capture_scalar_outputs", "skip_fwd_side_effects_in_bwd_under_checkpoint"):
        monkeypatch.setattr(torch._dynamo.config, setting, False)
    model = torch.nn.Module()
    compiled, eager = Block(), Block()
    model.layers = torch.nn.ModuleList([compiled])
    _apply_compile_v4_1(model, compile_config=SimpleNamespace(backend="inductor"))
    assert inductor_config.do_not_emit_runtime_assertions is True
    assert eager.dispatcher.dispatch.__func__ is SplitDispatcher.dispatch
    for count in (1, 3):
        x = torch.arange(4.0, requires_grad=True)
        actual = compiled(x, count)
        torch.testing.assert_close(actual, eager(x, count))
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.cat((2 * x[:count], torch.full_like(x[count:], 3))))


def test_ep_dispatch_transports_scores_in_router_dtype(monkeypatch):
    dispatcher = AllToAllTokenDispatcher(
        AllToAllTokenDispatcher.Config(num_experts=2, top_k=1, absorb_router_scores=True)
    )
    fake_ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    monkeypatch.setattr(type(dispatcher), "ep_mesh", property(lambda self: fake_ep_mesh))
    exchanged = []

    monkeypatch.setattr(
        dispatcher,
        "_token_count_exchange",
        lambda counts, group, ep_size: counts,
    )
    monkeypatch.setattr(
        dispatcher,
        "_sync_token_count_exchange",
        lambda counts, exchanged_counts, ep_size: (
            torch.tensor([1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1, 1]),
        ),
    )

    def fake_exchange(tensor, group, output_splits, input_splits):
        exchanged.append(tensor.clone())
        return tensor.flip(0)

    monkeypatch.setattr(dispatcher, "_dispatch_token_exchange", fake_exchange)
    permuted_indices = torch.tensor([1, 0])
    monkeypatch.setattr(
        dispatcher,
        "_permute",
        lambda routed_input, counts: (routed_input[permuted_indices], permuted_indices, counts),
    )

    x_TD = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    scores_TK = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    result = dispatcher.dispatch(x_TD, scores_TK, torch.tensor([[0], [1]]), torch.tensor([1, 1]))

    assert len(exchanged) == 2
    assert exchanged[1].dtype == scores_TK.dtype
    expected_scores = scores_TK.flatten().flip(0)[permuted_indices]
    assert result[2].routed_scores_R is not None
    torch.testing.assert_close(result[2].routed_scores_R, expected_scores)


def _build_routed_experts(*, absorb: bool):
    cfg = moe_patch._RouterScoreAbsorbingRoutedExperts.Config(
        w13=GroupedLinear.Config(group_size=2, in_features=4, out_features=4, num_linears=2),
        w2=GroupedLinear.Config(group_size=2, in_features=4, out_features=4),
        activation_fn=moe_patch.LimitedSwiGLU.Config(swiglu_limit=7.0),
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1, absorb_router_scores=absorb),
    )
    return cfg.build()


@pytest.mark.parametrize("absorb", [False, True])
def test_routed_experts_applies_scores_at_expected_boundary(absorb):
    experts = _build_routed_experts(absorb=absorb)
    inputs = []

    class GateUp(torch.nn.Module):
        def forward(self, x, offsets):
            return torch.stack((torch.ones_like(x), torch.full_like(x, 2)), dim=-2)

    class Down(torch.nn.Module):
        def forward(self, x, offsets):
            inputs.append(x)
            return x

    experts.w13, experts.w2 = GateUp(), Down()
    x = torch.randn(4, 4, dtype=torch.bfloat16)
    scores = torch.tensor([[0.25], [0.5], [0.75], [1.0]])
    expert_ids = torch.tensor([[0], [1], [0], [1]])
    out = experts(x, scores, expert_ids, torch.tensor([2, 2]), padding_mask_T=torch.tensor([False, False, False, True]))
    hidden = F.silu(torch.ones_like(x)) * 2
    expected = (hidden.float() * scores[[0, 2, 1, 3]]).bfloat16() if absorb else hidden
    torch.testing.assert_close(inputs[0], expected)
    assert out.shape == x.shape


# The tests above cover the patched interfaces and control flow with small
# fakes. The float64 contract tests below independently verify the numerical
# claim behind score absorption: moving a scalar router score from after W2 to
# immediately before the bias-free linear W2 preserves outputs and gradients.
_REFERENCE_DTYPE = torch.float64


@dataclass
class _ExpertWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor


@dataclass
class _PathResult:
    output: torch.Tensor
    grads: tuple[torch.Tensor, ...]


def _weights(num_experts: int, dim: int = 3, hidden: int = 4) -> _ExpertWeights:
    generator = torch.Generator().manual_seed(4095)
    return _ExpertWeights(
        w1=torch.randn(
            num_experts,
            hidden,
            dim,
            dtype=_REFERENCE_DTYPE,
            generator=generator,
            requires_grad=True,
        ),
        w2=torch.randn(
            num_experts,
            dim,
            hidden,
            dtype=_REFERENCE_DTYPE,
            generator=generator,
            requires_grad=True,
        ),
        w3=torch.randn(
            num_experts,
            hidden,
            dim,
            dtype=_REFERENCE_DTYPE,
            generator=generator,
            requires_grad=True,
        ),
    )


def _clone_weights(weights: _ExpertWeights) -> _ExpertWeights:
    return _ExpertWeights(*(value.detach().clone().requires_grad_() for value in (weights.w1, weights.w2, weights.w3)))


def _expert_forward(
    routed_input: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    weights: _ExpertWeights,
    routed_scores: torch.Tensor | None = None,
    *,
    expert_offset: int = 0,
) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, count in enumerate(tokens_per_expert.tolist()):
        end = start + count
        x = routed_input[start:end]
        global_expert = expert_offset + expert
        gate = x @ weights.w1[global_expert].transpose(-2, -1)
        up = x @ weights.w3[global_expert].transpose(-2, -1)
        hidden = F.silu(gate) * up
        if routed_scores is not None:
            hidden = hidden * routed_scores[start:end, None]
        outputs.append(hidden @ weights.w2[global_expert].transpose(-2, -1))
        start = end
    return torch.cat(outputs, dim=0) if outputs else routed_input.new_empty((0, routed_input.shape[-1]))


def _run_local_path(
    x,
    scores,
    expert_ids,
    counts,
    weights,
    *,
    pre_w2: bool,
) -> _PathResult:
    dispatcher = LocalTokenDispatcher(LocalTokenDispatcher.Config(num_experts=3, top_k=2, absorb_router_scores=True))
    routed, _, metadata = dispatcher.dispatch(x, scores, expert_ids, counts)
    aligned_scores = metadata.routed_scores_R
    assert aligned_scores is not None

    # Check the central dispatcher contract explicitly: each score must follow
    # the same stable expert-order permutation as its routed token.
    sorted_assignments = torch.argsort(expert_ids.reshape(-1), stable=True)
    expected_scores = scores.reshape(-1)[sorted_assignments]
    torch.testing.assert_close(aligned_scores, expected_scores, rtol=1e-12, atol=1e-12)

    routed_output = _expert_forward(
        routed,
        counts,
        weights,
        aligned_scores if pre_w2 else None,
    )
    if not pre_w2:
        routed_output = routed_output * aligned_scores[:, None]
    output = dispatcher.combine(
        routed_output,
        metadata,
        x,
    )
    output.square().sum().backward()
    grads = (
        x.grad.detach().clone(),
        scores.grad.detach().clone(),
        *(p.grad.detach().clone() for p in (weights.w1, weights.w2, weights.w3)),
    )
    return _PathResult(output.detach(), grads)


def _reference_local_oracle(x, scores, expert_ids, weights, pre_w2):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(scores.shape[1]):
            e = int(expert_ids[t, k])
            hidden = F.silu(x[t] @ weights.w1[e].T) * (x[t] @ weights.w3[e].T)
            if pre_w2:
                hidden = hidden * scores[t, k]
                out[t] += hidden @ weights.w2[e].T
            else:
                out[t] += (hidden @ weights.w2[e].T) * scores[t, k]
    return out


def _reference_local_path_result(x, scores, expert_ids, weights, pre_w2):
    xo = x.detach().clone().requires_grad_()
    so = scores.detach().clone().requires_grad_()
    wo = _clone_weights(weights)
    out = _reference_local_oracle(xo, so, expert_ids, wo, pre_w2)
    out.square().sum().backward()
    return _PathResult(
        out.detach(),
        (
            xo.grad.detach().clone(),
            so.grad.detach().clone(),
            wo.w1.grad.detach().clone(),
            wo.w2.grad.detach().clone(),
            wo.w3.grad.detach().clone(),
        ),
    )


def test_local_dispatch_pre_w2_matches_post_w2_in_float64():
    dispatcher_patch.apply()
    x = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [1.1, 0.3, -0.8],
            [-0.6, 0.9, 0.5],
            [0.4, -1.2, 0.2],
        ],
        dtype=_REFERENCE_DTYPE,
        requires_grad=True,
    )
    scores = torch.tensor(
        [[0.17, 0.83], [0.61, 0.39], [0.28, 0.72], [0.94, 0.06]],
        dtype=_REFERENCE_DTYPE,
        requires_grad=True,
    )
    expert_ids = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 0]])
    counts = torch.bincount(expert_ids.reshape(-1), minlength=3)
    weights = _weights(3)

    # Independent autograd graphs make gradient equality meaningful instead of
    # accidentally comparing two views of the same accumulated gradients.
    baseline = _run_local_path(
        x.detach().clone().requires_grad_(),
        scores.detach().clone().requires_grad_(),
        expert_ids,
        counts,
        _clone_weights(weights),
        pre_w2=False,
    )
    candidate = _run_local_path(
        x.detach().clone().requires_grad_(),
        scores.detach().clone().requires_grad_(),
        expert_ids,
        counts,
        _clone_weights(weights),
        pre_w2=True,
    )
    oracle_post = _reference_local_path_result(x, scores, expert_ids, weights, pre_w2=False)
    oracle_pre = _reference_local_path_result(x, scores, expert_ids, weights, pre_w2=True)
    torch.testing.assert_close(baseline.output, oracle_post.output, rtol=1e-12, atol=1e-12)
    for left, right in zip(baseline.grads, oracle_post.grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(candidate.output, oracle_pre.output, rtol=1e-12, atol=1e-12)
    for left, right in zip(candidate.grads, oracle_pre.grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(candidate.output, baseline.output, rtol=1e-12, atol=1e-12)
    for left, right in zip(baseline.grads, candidate.grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def _install_cpu_npu_moe_fakes():
    def fake_permute(tokens, expert_ids):
        top_k = expert_ids.shape[1]
        order = torch.argsort(expert_ids.reshape(-1), stable=True)
        routed = tokens.repeat_interleave(top_k, dim=0)[order]
        return routed, torch.argsort(order)

    def fake_unpermute(permuted_tokens, sorted_indices, probs=None):
        restored = permuted_tokens[sorted_indices.to(torch.long)]
        if probs is None:
            return restored
        return (restored.view(probs.shape[0], probs.shape[1], -1) * probs.unsqueeze(-1)).sum(1)

    def fake_unpermute_grad(permuted_tokens, grad_output, sorted_indices, *, probs):
        sorted_indices = sorted_indices.to(torch.long)
        restored = permuted_tokens[sorted_indices]
        if probs is None:
            grad_restored = grad_output
            grad_probs = None
        else:
            restored = restored.view(probs.shape[0], probs.shape[1], -1)
            grad_restored = (grad_output[:, None, :] * probs[:, :, None]).reshape_as(permuted_tokens)
            grad_probs = (restored * grad_output[:, None, :]).sum(-1)
        grad_tokens = torch.empty_like(permuted_tokens)
        grad_tokens[sorted_indices] = grad_restored
        return grad_tokens, grad_probs

    def fake_re_routing(tokens, counts, *, per_token_scales, expert_token_num_type, idx_type):
        del expert_token_num_type, idx_type
        counts = counts.to(torch.long)
        chunks = []
        offset = 0
        for expert in range(counts.shape[1]):
            for rank in range(counts.shape[0]):
                count = counts[rank, expert]
                end = offset + int(count)
                chunks.append(torch.arange(offset, end, device=tokens.device))
                offset = end
        order = torch.cat(chunks)
        permuted_scales = tokens.new_empty(0) if per_token_scales is None else per_token_scales[order]
        return (
            tokens[order],
            permuted_scales,
            order.to(torch.int32),
            counts.sum(0).to(torch.int32),
        )

    npu_token_dispatcher.torch_npu.npu_moe_token_permute = fake_permute
    moe_token_unpermute.torch_npu.npu_moe_token_unpermute = fake_unpermute
    moe_token_unpermute.torch_npu.npu_moe_token_unpermute_grad = fake_unpermute_grad
    moe_re_routing.torch_npu.npu_moe_re_routing = fake_re_routing


def _alltoall_worker(rank: int, rendezvous: str, result_file: str, use_npu: bool = False):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        dispatcher_patch.apply()
        if use_npu:
            _install_cpu_npu_moe_fakes()
        mesh = torch.distributed.device_mesh.init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        # Gloo's list all-to-all requires equal peer split sizes, so every rank
        # sends two tokens to each expert-owner rank. The dispatcher path under
        # test (score absorption) does not depend on the split being uneven.
        x = torch.tensor(
            [
                [10.0 + rank, 0.2],
                [20.0 + rank, 1.1],
                [30.0 + rank, -0.6],
                [40.0 + rank, 0.4],
            ],
            dtype=_REFERENCE_DTYPE,
        )
        expert_ids = torch.tensor([[0], [1], [2], [3]])
        token_ids = x[:, 0]
        scores = 0.11 + token_ids * 0.013
        counts = torch.bincount(expert_ids.reshape(-1), minlength=4)
        base_weights = _weights(4, dim=2, hidden=3)

        def reference_oracle(x, expert_ids, scores, weights, pre_w2):
            out = torch.zeros_like(x)
            for i, expert in enumerate(expert_ids.reshape(-1).tolist()):
                hidden = F.silu(x[i] @ weights.w1[expert].T) * (x[i] @ weights.w3[expert].T)
                if pre_w2:
                    hidden = hidden * scores[i]
                    out[i] += hidden @ weights.w2[expert].T
                else:
                    out[i] += (hidden @ weights.w2[expert].T) * scores[i]
            return out

        def oracle_run(pre_w2, x, expert_ids, scores, weights, expert_offset, num_local_experts):
            xo = x.detach().clone().requires_grad_()
            so = scores.detach().clone().requires_grad_()
            wo = _clone_weights(weights)
            out = reference_oracle(xo, expert_ids, so, wo, pre_w2)
            out.square().sum().backward()
            end = expert_offset + num_local_experts
            return out.detach(), (
                xo.grad.detach().clone(),
                so.grad.detach().clone(),
                wo.w1.grad.detach().clone()[expert_offset:end],
                wo.w2.grad.detach().clone()[expert_offset:end],
                wo.w3.grad.detach().clone()[expert_offset:end],
            )

        def run(pre_w2: bool, weights: _ExpertWeights):
            dispatcher_cls = npu_token_dispatcher.AscAllToAllTokenDispatcher if use_npu else AllToAllTokenDispatcher
            dispatcher = dispatcher_cls(dispatcher_cls.Config(num_experts=4, top_k=1, absorb_router_scores=True))
            local_x = x.detach().clone().requires_grad_()
            local_scores = scores.detach().clone().requires_grad_()
            routed, local_counts, metadata = dispatcher.dispatch(local_x, local_scores[:, None], expert_ids, counts)
            aligned_scores = metadata.routed_scores_R
            assert aligned_scores is not None

            # Encoding the token id in column zero lets every rank verify that
            # scores remain attached after the real two-rank all-to-all.
            torch.testing.assert_close(
                aligned_scores,
                0.11 + routed[:, 0] * 0.013,
                rtol=1e-12,
                atol=1e-12,
            )
            routed_output = _expert_forward(
                routed,
                local_counts,
                weights,
                aligned_scores if pre_w2 else None,
                expert_offset=rank * (weights.w1.shape[0] // mesh.size()),
            )
            if not pre_w2:
                routed_output = routed_output * aligned_scores[:, None]
            output = dispatcher.combine(
                routed_output,
                metadata,
                local_x,
            )
            output.square().sum().backward()
            num_local_experts = weights.w1.shape[0] // mesh.size()
            start = rank * num_local_experts
            end = start + num_local_experts
            return output.detach(), (
                local_x.grad.detach().clone(),
                local_scores.grad.detach().clone(),
                weights.w1.grad.detach().clone()[start:end],
                weights.w2.grad.detach().clone()[start:end],
                weights.w3.grad.detach().clone()[start:end],
            )

        post = run(False, _clone_weights(base_weights))
        candidate = run(True, _clone_weights(base_weights))
        torch.testing.assert_close(candidate[0], post[0], rtol=1e-12, atol=1e-12)
        for left, right in zip(candidate[1], post[1], strict=True):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

        if not use_npu:
            num_local_experts = base_weights.w1.shape[0] // mesh.size()
            expert_offset = rank * num_local_experts
            oracle_post = oracle_run(False, x, expert_ids, scores, base_weights, expert_offset, num_local_experts)
            torch.testing.assert_close(post[0], oracle_post[0], rtol=1e-12, atol=1e-12)
            # x/score grads are local and directly comparable; expert-weight
            # grads in EP aggregate tokens from all ranks, so they are validated
            # by candidate-vs-baseline and the Local oracle test instead.
            for left, right in zip(post[1][:2], oracle_post[1][:2], strict=True):
                torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

            oracle_pre = oracle_run(True, x, expert_ids, scores, base_weights, expert_offset, num_local_experts)
            torch.testing.assert_close(candidate[0], oracle_pre[0], rtol=1e-12, atol=1e-12)
            for left, right in zip(candidate[1][:2], oracle_pre[1][:2], strict=True):
                torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

        torch.save(candidate[0], f"{result_file}.{rank}")
    finally:
        dist.destroy_process_group()


def test_alltoall_dispatch_pre_w2_matches_post_w2_in_float64():
    dispatcher_patch.apply()
    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = os.path.join(tmp, "rendezvous")
        result_file = os.path.join(tmp, "result")
        mp.spawn(
            _alltoall_worker,
            args=(rendezvous, result_file),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        assert os.path.exists(f"{result_file}.0")
        assert os.path.exists(f"{result_file}.1")


def test_npu_alltoall_dispatch_pre_w2_matches_post_w2_in_float64():
    dispatcher_patch.apply()
    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = os.path.join(tmp, "rendezvous")
        result_file = os.path.join(tmp, "result")
        mp.spawn(
            _alltoall_worker,
            args=(rendezvous, result_file, True),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        assert os.path.exists(f"{result_file}.0")
        assert os.path.exists(f"{result_file}.1")
