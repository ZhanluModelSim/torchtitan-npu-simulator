# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required"),
]


def test_merged_gdn_forward_backward():
    from torchtitan_npu.ops.triton.gdn import gated_delta_rule

    torch.npu.set_device(0)
    q = torch.randn(1, 64, 1, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    g = torch.randn(1, 64, 1, device="npu", dtype=torch.float32, requires_grad=True)
    beta_logits = torch.randn(1, 64, 1, device="npu", dtype=torch.bfloat16, requires_grad=True)
    beta = torch.sigmoid(beta_logits)

    output = gated_delta_rule(q, k, v, g, beta)
    output.float().square().mean().backward()
    torch.npu.synchronize()

    assert output.shape == v.shape
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v, g))
    assert beta_logits.grad is not None and torch.isfinite(beta_logits.grad).all()


def test_qwen3_5_learned_position_embedding_has_npu_backward():
    __import__("torchtitan_npu.models.qwen3_5")

    from torchtitan.models.qwen3_5.vision_encoder import _compute_learned_pos_embeds

    torch.npu.set_device(0)
    pos_embed = torch.randn(
        16 * 16,
        8,
        device="npu",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = _compute_learned_pos_embeds(
        pos_embed,
        [[1, 4, 4]],
        max_num_patch=16,
        num_grid_per_side=16,
        spatial_merge_size=2,
        dim=8,
    )
    output.float().sum().backward()
    torch.npu.synchronize()

    assert output.shape == (1, 16, 8)
    assert pos_embed.grad is not None
    assert torch.isfinite(pos_embed.grad).all()
    assert bool(pos_embed.grad.abs().sum() > 0)
