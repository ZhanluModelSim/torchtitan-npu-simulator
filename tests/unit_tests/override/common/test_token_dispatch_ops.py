# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU-bound contract tests for the AscendC MoE operator wrappers."""

import torch
import torch.nn.functional as F

from torchtitan_npu.ops.ascendc import moe_re_routing, moe_token_unpermute
from torchtitan_npu.override.common import token_dispatcher


def test_token_permute_unpermute_autograd(monkeypatch):
    def fake_permute(tokens, expert_ids):
        forward = torch.argsort(expert_ids.reshape(-1), stable=True)
        routed = tokens.repeat_interleave(expert_ids.shape[1], dim=0)[forward]
        return routed, torch.argsort(forward)

    def fake_unpermute(permuted_tokens, sorted_indices, probs):
        restored = permuted_tokens[sorted_indices.to(torch.long)]
        if probs is None:
            return restored
        return (restored.view(probs.shape[0], probs.shape[1], -1) * probs.unsqueeze(-1)).sum(1)

    def fake_unpermute_grad(permuted_tokens, grad_output, sorted_indices, *, probs):
        del permuted_tokens, probs
        return grad_output.repeat_interleave(2, dim=0)[sorted_indices.to(torch.long)], None

    monkeypatch.setattr(token_dispatcher.torch_npu, "npu_moe_token_permute", fake_permute)
    monkeypatch.setattr(moe_token_unpermute.torch_npu, "npu_moe_token_unpermute", fake_unpermute)
    monkeypatch.setattr(moe_token_unpermute.torch_npu, "npu_moe_token_unpermute_grad", fake_unpermute_grad)

    tokens = torch.arange(12, dtype=torch.float32).view(4, 3).requires_grad_()
    expert_ids = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    probs = torch.full((4, 2), 0.5)
    routed, sorted_indices = token_dispatcher.torch_npu.npu_moe_token_permute(tokens, expert_ids)
    output = moe_token_unpermute.npu_moe_token_unpermute(routed, sorted_indices, probs)
    output.sum().backward()

    assert output.shape == tokens.shape
    assert tokens.grad is not None
    assert tokens.grad.shape == tokens.shape


def test_re_routing_autograd_uses_restore_indices(monkeypatch):
    order = torch.tensor([2, 0, 3, 1])

    def fake_re_routing(tokens, counts, **kwargs):
        assert kwargs == {
            "per_token_scales": None,
            "expert_token_num_type": 1,
            "idx_type": 0,
        }
        return (
            tokens[order],
            tokens.new_empty(0),
            order.to(torch.int32),
            counts.sum(0).to(torch.int32),
        )

    def fake_unpermute(tokens, sorted_indices, probs):
        assert probs is None
        return tokens[sorted_indices.to(torch.long)]

    monkeypatch.setattr(moe_re_routing.torch_npu, "npu_moe_re_routing", fake_re_routing)
    monkeypatch.setattr(moe_token_unpermute.torch_npu, "npu_moe_token_unpermute", fake_unpermute)

    tokens = torch.arange(12, dtype=torch.float32).view(4, 3).requires_grad_()
    counts = torch.tensor([[1, 1], [1, 1]])
    routed, permuted_scales, restore_indices, num_tokens = moe_re_routing.npu_moe_re_routing(tokens, counts)
    routed.sum().backward()

    assert permuted_scales.numel() == 0
    assert torch.equal(restore_indices, torch.argsort(order))
    assert torch.equal(num_tokens, torch.tensor([2, 2]))
    assert tokens.grad is not None
    assert tokens.grad.shape == tokens.shape


def test_re_routing_transports_scales_and_backward(monkeypatch):
    order = torch.tensor([2, 0, 3, 1])

    def fake_re_routing(tokens, counts, **kwargs):
        assert kwargs["expert_token_num_type"] == 1
        assert kwargs["idx_type"] == 0
        scales = kwargs["per_token_scales"]
        return tokens[order], scales[order], order.to(torch.int32), counts.sum(0).to(torch.int32)

    def fake_unpermute(tokens, sorted_indices, probs):
        assert probs is None
        return tokens[sorted_indices.to(torch.long)]

    monkeypatch.setattr(moe_re_routing.torch_npu, "npu_moe_re_routing", fake_re_routing)
    monkeypatch.setattr(moe_token_unpermute.torch_npu, "npu_moe_token_unpermute", fake_unpermute)

    tokens = torch.arange(12, dtype=torch.float32).view(4, 3).requires_grad_()
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    counts = torch.tensor([[1, 1], [1, 1]])
    routed, permuted_scales, restore_indices, num_tokens = moe_re_routing.npu_moe_re_routing(
        tokens,
        counts,
        per_token_scales=scales,
    )
    (routed * permuted_scales.reshape(-1, 1)).sum().backward()

    assert torch.equal(permuted_scales, scales[order])
    assert torch.equal(restore_indices, torch.argsort(order))
    assert torch.equal(num_tokens, torch.tensor([2, 2]))
    assert tokens.grad is not None
    assert scales.grad is not None
    assert tokens.grad.shape == tokens.shape
    assert scales.grad.shape == scales.shape
    expected_tokens_grad = torch.zeros_like(tokens)
    expected_tokens_grad[order] = scales[order].reshape(-1, 1)
    expected_scales_grad = torch.zeros_like(scales)
    expected_scales_grad[order] = tokens[order].sum(dim=1)
    torch.testing.assert_close(tokens.grad, expected_tokens_grad)
    torch.testing.assert_close(scales.grad, expected_scales_grad)


def test_dispatcher_absorption_local_path_matches_post_w2_reference(monkeypatch):
    def fake_permute(tokens, expert_ids):
        top_k = expert_ids.shape[1]
        order = torch.argsort(expert_ids.reshape(-1), stable=True)
        routed = tokens.repeat_interleave(top_k, dim=0)[order]
        return routed, torch.argsort(order)

    monkeypatch.setattr(token_dispatcher.torch_npu, "npu_moe_token_permute", fake_permute)

    def dense_experts(routed, counts, weights, routed_scores=None):
        outputs = []
        start = 0
        for expert, count in enumerate(counts.tolist()):
            expert_input = routed[start : start + count]
            gate = expert_input @ weights[0][expert].transpose(-2, -1)
            up = expert_input @ weights[2][expert].transpose(-2, -1)
            hidden = F.silu(gate) * up
            if routed_scores is not None:
                hidden = hidden * routed_scores[start : start + count, None]
            outputs.append(hidden @ weights[1][expert].transpose(-2, -1))
            start += count
        return torch.cat(outputs, dim=0)

    def reference_oracle(x, scores, expert_ids, weights, pre_w2):
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(scores.shape[1]):
                e = int(expert_ids[t, k])
                hidden = F.silu(x[t] @ weights[0][e].transpose(-2, -1)) * (x[t] @ weights[2][e].transpose(-2, -1))
                if pre_w2:
                    hidden = hidden * scores[t, k]
                    out[t] += hidden @ weights[1][e].transpose(-2, -1)
                else:
                    out[t] += (hidden @ weights[1][e].transpose(-2, -1)) * scores[t, k]
        return out

    def oracle_run(pre_w2, x, scores, expert_ids, weights):
        xo = x.detach().clone().requires_grad_()
        so = scores.detach().clone().requires_grad_()
        wo = tuple(weight.detach().clone().requires_grad_() for weight in weights)
        out = reference_oracle(xo, so, expert_ids, wo, pre_w2)
        out.square().sum().backward()
        return (
            out.detach(),
            (
                xo.grad.detach().clone(),
                so.grad.detach().clone(),
                *(weight.grad.detach().clone() for weight in wo),
            ),
        )

    def run(absorb_scores, x, scores, weights, seen_probs):
        dispatcher = object.__new__(token_dispatcher.NPUAllToAllTokenDispatcher)
        dispatcher.ep_mesh = None
        dispatcher.absorb_router_scores = absorb_scores
        routed, _, metadata = dispatcher.dispatch(x, scores, expert_ids, counts)
        aligned_scores = metadata.routed_scores_R
        if absorb_scores:
            assert aligned_scores is not None

        routed_output = dense_experts(
            routed,
            counts,
            weights,
            aligned_scores if absorb_scores else None,
        )

        def fake_unpermute(tokens, sorted_indices, probs):
            restored = tokens[sorted_indices.to(torch.long)]
            seen_probs.append(probs)
            if probs is None:
                return restored
            return (restored.view(probs.shape[0], probs.shape[1], -1) * probs.unsqueeze(-1)).sum(1)

        monkeypatch.setattr(token_dispatcher, "npu_moe_token_unpermute", fake_unpermute)
        output = dispatcher.combine(
            routed_output,
            metadata,
            x,
        )
        output.square().sum().backward()
        return (
            output.detach(),
            (
                x.grad.detach().clone(),
                scores.grad.detach().clone(),
                *(weight.grad.detach().clone() for weight in weights),
            ),
            metadata,
        )

    x = torch.tensor(
        [[0.2, -0.4, 0.7], [1.1, 0.3, -0.8], [-0.6, 0.9, 0.5], [0.4, -1.2, 0.2]],
        dtype=torch.float64,
    )
    scores = torch.tensor(
        [[0.1, 0.9], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.float64,
    )
    expert_ids = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    counts = torch.tensor([4, 4])
    generator = torch.Generator().manual_seed(17)
    weights = tuple(
        torch.randn(shape, dtype=torch.float64, generator=generator, requires_grad=True)
        for shape in ((2, 3, 3), (2, 3, 3), (2, 3, 3))
    )

    baseline = run(
        False,
        x.detach().clone().requires_grad_(),
        scores.detach().clone().requires_grad_(),
        tuple(weight.detach().clone().requires_grad_() for weight in weights),
        [],
    )
    seen_probs = []
    candidate = run(
        True,
        x.detach().clone().requires_grad_(),
        scores.detach().clone().requires_grad_(),
        tuple(weight.detach().clone().requires_grad_() for weight in weights),
        seen_probs,
    )

    oracle_post = oracle_run(False, x, scores, expert_ids, weights)
    torch.testing.assert_close(baseline[0], oracle_post[0], rtol=1e-12, atol=1e-12)
    for baseline_grad, oracle_grad in zip(baseline[1], oracle_post[1], strict=True):
        torch.testing.assert_close(baseline_grad, oracle_grad, rtol=1e-12, atol=1e-12)

    oracle_pre = oracle_run(True, x, scores, expert_ids, weights)
    torch.testing.assert_close(candidate[0], oracle_pre[0], rtol=1e-12, atol=1e-12)
    for candidate_grad, oracle_grad in zip(candidate[1], oracle_pre[1], strict=True):
        torch.testing.assert_close(candidate_grad, oracle_grad, rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(candidate[0], baseline[0], rtol=1e-12, atol=1e-12)
    for candidate_grad, baseline_grad in zip(candidate[1], baseline[1], strict=True):
        torch.testing.assert_close(candidate_grad, baseline_grad, rtol=1e-12, atol=1e-12)

    expected_order = torch.argsort(expert_ids.reshape(-1), stable=True)
    torch.testing.assert_close(
        candidate[2].routed_scores_R,
        scores.reshape(-1)[expected_order],
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(seen_probs) == 1
    torch.testing.assert_close(seen_probs[0], torch.ones_like(scores))
