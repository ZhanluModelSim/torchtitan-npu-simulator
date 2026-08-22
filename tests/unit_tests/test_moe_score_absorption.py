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

    model_spec = model_registry("debugmodel", moe_comm_backend="standard")
    routed_configs = [layer.moe.routed_experts for layer in model_spec.model.layers if layer.moe is not None]

    assert routed_configs
    assert all(config.token_dispatcher.absorb_router_scores for config in routed_configs)


def test_deepseek_v3_2_enables_score_absorption():
    from torchtitan_npu.models.deepseek_v3_2 import model_registry

    model_spec = model_registry("debugmodel")
    routed_configs = [layer.moe.routed_experts for layer in model_spec.model.layers if layer.moe is not None]

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
        num_local_tokens_after_padding=2,
        local_seq_len_after_padding=2,
    )
    candidate = absorbed_dispatcher.combine(
        expert_output_RD * routed_scores_R.to(expert_output_RD.dtype).reshape(-1, 1),
        metadata,
        x_TD,
        num_local_tokens_after_padding=2,
        local_seq_len_after_padding=2,
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
    assert LocalTokenDispatcher._local_reorder is dispatcher_patch.TorchTitanLocalTokenDispatcher._local_reorder
    assert (
        AllToAllTokenDispatcher._token_count_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._token_count_exchange
    )
    assert (
        AllToAllTokenDispatcher._sync_token_count_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._sync_token_count_exchange
    )
    assert (
        AllToAllTokenDispatcher._dispatch_token_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._dispatch_token_exchange
    )
    assert (
        AllToAllTokenDispatcher._combine_token_exchange
        is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._combine_token_exchange
    )
    assert AllToAllTokenDispatcher._permute is not dispatcher_patch.TorchTitanAllToAllTokenDispatcher._permute
    assert AllToAllTokenDispatcher._unpermute is dispatcher_patch.TorchTitanAllToAllTokenDispatcher._unpermute


def test_ep_dispatch_transports_scores_in_router_dtype(monkeypatch):
    dispatcher = AllToAllTokenDispatcher(
        AllToAllTokenDispatcher.Config(num_experts=2, top_k=1, absorb_router_scores=True)
    )
    fake_ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    dispatcher.ep_mesh = fake_ep_mesh
    exchanged = []

    monkeypatch.setattr(dispatcher_patch, "get_spmd_backend", lambda: "full_dtensor")
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
            [1, 1],
            [1, 1],
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
        lambda routed_input, counts, routed_scores=None: (
            routed_input.shape,
            routed_input[permuted_indices],
            permuted_indices,
            counts,
            routed_scores[permuted_indices] if routed_scores is not None else None,
        ),
    )

    x_TD = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    scores_TK = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    result = dispatcher.dispatch(x_TD, scores_TK, torch.tensor([[0], [1]]), torch.tensor([1, 1]))

    assert len(exchanged) == 2
    assert exchanged[1].dtype == scores_TK.dtype
    expected_scores = scores_TK.flatten().flip(0)[permuted_indices]
    assert result[2].routed_scores_R is not None
    torch.testing.assert_close(result[2].routed_scores_R, expected_scores)


def test_grouped_experts_apply_scores_to_w2_input(monkeypatch):
    experts = moe_patch._ClampGroupedExperts(
        moe_patch._ClampGroupedExperts.Config(
            dim=2,
            hidden_dim=2,
            num_experts=1,
            swiglu_limit=0.0,
        )
    )
    gate = torch.ones(2, 2, dtype=torch.bfloat16)
    up = torch.full((2, 2), 2.0, dtype=torch.bfloat16)
    grouped_mm_inputs = []

    def fake_grouped_mm(A, B_t, *, offs):
        del B_t, offs
        grouped_mm_inputs.append(A)
        if len(grouped_mm_inputs) == 1:
            return gate
        if len(grouped_mm_inputs) == 2:
            return up
        return A

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm)
    scores = torch.tensor([0.25, 0.5])
    experts(
        torch.zeros(2, 2, dtype=torch.bfloat16),
        torch.tensor([2]),
        routed_scores_R=scores,
    )

    expected_w2_input = (F.silu(gate).float() * up.float() * scores.reshape(-1, 1)).to(torch.bfloat16)
    torch.testing.assert_close(grouped_mm_inputs[-1], expected_w2_input)


def _build_routed_experts(*, absorb: bool):
    cfg = moe_patch._RouterScoreAbsorbingRoutedExperts.Config(
        inner_experts=moe_patch._ClampGroupedExperts.Config(
            dim=4,
            hidden_dim=4,
            num_experts=2,
            swiglu_limit=0.0,
        ),
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1, absorb_router_scores=absorb),
    )
    return cfg.build()


def _fake_grouped_mm_factory(gate, up):
    inputs = []

    def fake_grouped_mm(A, B_t, *, offs):
        del B_t, offs
        inputs.append(A)
        if len(inputs) == 1:
            return gate
        if len(inputs) == 2:
            return up
        return A

    return fake_grouped_mm, inputs


def test_routed_experts_forward_absorbs_scores_through_local_dispatcher(monkeypatch):
    experts = _build_routed_experts(absorb=True)
    gate = torch.ones(4, 4, dtype=torch.bfloat16)
    up = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    fake_mm, mm_inputs = _fake_grouped_mm_factory(gate, up)
    monkeypatch.setattr(torch, "_grouped_mm", fake_mm)

    x_BLD = torch.randn(1, 4, 4, dtype=torch.bfloat16)
    scores = torch.tensor(
        [
            [[0.25], [0.5], [0.75], [1.0]],
        ]
    )
    expert_ids = torch.tensor(
        [
            [[0], [1], [0], [1]],
        ]
    )
    counts_E = torch.tensor([2, 2])

    out_BLD = experts(
        x_BLD,
        scores,
        expert_ids,
        counts_E,
        num_local_tokens_after_seq_dim_padding=4,
    )

    sorted_scores = torch.tensor([0.25, 0.75, 0.5, 1.0]).bfloat16()
    expected_w2_input = (F.silu(gate).float() * up.float() * sorted_scores.float().reshape(-1, 1)).to(torch.bfloat16)
    torch.testing.assert_close(mm_inputs[-1], expected_w2_input)
    assert out_BLD.shape == x_BLD.shape


def test_routed_experts_forward_default_path_without_absorption(monkeypatch):
    experts = _build_routed_experts(absorb=False)
    gate = torch.ones(4, 4, dtype=torch.bfloat16)
    up = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    fake_mm, mm_inputs = _fake_grouped_mm_factory(gate, up)
    monkeypatch.setattr(torch, "_grouped_mm", fake_mm)

    x_BLD = torch.randn(1, 4, 4, dtype=torch.bfloat16)
    scores = torch.tensor(
        [
            [[0.25], [0.5], [0.75], [1.0]],
        ]
    )
    expert_ids = torch.tensor(
        [
            [[0], [1], [0], [1]],
        ]
    )
    counts_E = torch.tensor([2, 2])

    experts(
        x_BLD,
        scores,
        expert_ids,
        counts_E,
        num_local_tokens_after_seq_dim_padding=4,
    )

    expected_w2_input = F.silu(gate) * up
    torch.testing.assert_close(mm_inputs[-1], expected_w2_input)


class _UnsupportedDispatcher(LocalTokenDispatcher):
    pass


def test_routed_experts_forward_falls_back_for_unsupported_dispatcher(monkeypatch):
    experts = _build_routed_experts(absorb=True)
    experts.token_dispatcher = _UnsupportedDispatcher(LocalTokenDispatcher.Config(num_experts=2, top_k=1))
    gate = torch.ones(4, 4, dtype=torch.bfloat16)
    up = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    fake_mm, mm_inputs = _fake_grouped_mm_factory(gate, up)
    monkeypatch.setattr(torch, "_grouped_mm", fake_mm)

    x_BLD = torch.randn(1, 4, 4, dtype=torch.bfloat16)
    scores = torch.tensor(
        [
            [[0.25], [0.5], [0.75], [1.0]],
        ]
    )
    expert_ids = torch.tensor(
        [
            [[0], [1], [0], [1]],
        ]
    )
    counts_E = torch.tensor([2, 2])

    experts(
        x_BLD,
        scores,
        expert_ids,
        counts_E,
        num_local_tokens_after_seq_dim_padding=4,
    )

    expected_w2_input = F.silu(gate) * up
    torch.testing.assert_close(mm_inputs[-1], expected_w2_input)


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
) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, count in enumerate(tokens_per_expert.tolist()):
        end = start + count
        x = routed_input[start:end]
        gate = x @ weights.w1[expert].transpose(-2, -1)
        up = x @ weights.w3[expert].transpose(-2, -1)
        hidden = F.silu(gate) * up
        if routed_scores is not None:
            hidden = hidden * routed_scores[start:end, None]
        outputs.append(hidden @ weights.w2[expert].transpose(-2, -1))
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
        num_local_tokens_after_padding=x.shape[0],
        local_seq_len_after_padding=x.shape[0],
    )
    output.square().sum().backward()
    grads = (
        x.grad.detach().clone(),
        scores.grad.detach().clone(),
        *(p.grad.detach().clone() for p in (weights.w1, weights.w2, weights.w3)),
    )
    return _PathResult(output.detach(), grads)


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
    counts = torch.tensor([3, 3, 2])
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
    torch.testing.assert_close(candidate.output, baseline.output, rtol=1e-12, atol=1e-12)
    for left, right in zip(baseline.grads, candidate.grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def _alltoall_worker(rank: int, rendezvous: str, result_file: str):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        dispatcher_patch.apply()
        mesh = torch.distributed.device_mesh.init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        x = torch.tensor(
            [[10.0 + rank, 0.2], [20.0 + rank, 1.1], [30.0 + rank, -0.6]],
            dtype=_REFERENCE_DTYPE,
        )
        expert_ids = torch.tensor([[0], [2], [3]]) if rank == 0 else torch.tensor([[1], [2], [0]])
        token_ids = x[:, 0]
        scores = 0.11 + token_ids * 0.013
        counts = torch.bincount(expert_ids.reshape(-1), minlength=4)
        base_weights = _weights(4, dim=2, hidden=3)

        def run(pre_w2: bool, weights: _ExpertWeights):
            dispatcher = AllToAllTokenDispatcher(
                AllToAllTokenDispatcher.Config(num_experts=4, top_k=1, absorb_router_scores=True)
            )
            dispatcher.wire_meshes(ep_mesh=mesh, tp_mesh=None)
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
            )
            if not pre_w2:
                routed_output = routed_output * aligned_scores[:, None]
            output = dispatcher.combine(
                routed_output,
                metadata,
                local_x,
                num_local_tokens_after_padding=local_x.shape[0],
                local_seq_len_after_padding=local_x.shape[0],
            )
            output.square().sum().backward()
            return output.detach(), (
                local_x.grad.detach(),
                local_scores.grad.detach(),
                weights.w1.grad.detach(),
                weights.w2.grad.detach(),
                weights.w3.grad.detach(),
            )

        post = run(False, _clone_weights(base_weights))
        candidate = run(True, _clone_weights(base_weights))
        torch.testing.assert_close(candidate[0], post[0], rtol=1e-12, atol=1e-12)
        for left, right in zip(candidate[1], post[1], strict=True):
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
        )
        assert os.path.exists(f"{result_file}.0")
        assert os.path.exists(f"{result_file}.1")
