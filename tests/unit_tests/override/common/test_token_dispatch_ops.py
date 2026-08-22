# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU-bound contract tests for the AscendC MoE operator wrappers."""

import torch

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
        assert kwargs == {"expert_token_num_type": 1, "idx_type": 0}
        return tokens[order], None, order.to(torch.int32), counts.sum(0).to(torch.int32)

    def fake_unpermute(tokens, sorted_indices, probs):
        assert probs is None
        return tokens[sorted_indices.to(torch.long)]

    monkeypatch.setattr(moe_re_routing.torch_npu, "npu_moe_re_routing", fake_re_routing)
    monkeypatch.setattr(moe_token_unpermute.torch_npu, "npu_moe_token_unpermute", fake_unpermute)

    tokens = torch.arange(12, dtype=torch.float32).view(4, 3).requires_grad_()
    counts = torch.tensor([[1, 1], [1, 1]])
    routed, restore_indices, num_tokens = moe_re_routing.npu_moe_re_routing(tokens, counts)
    routed.sum().backward()

    assert torch.equal(restore_indices, torch.argsort(order))
    assert torch.equal(num_tokens, torch.tensor([2, 2]))
    assert tokens.grad is not None
    assert tokens.grad.shape == tokens.shape
