# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Indexer distillation gradients and the SMLAG teacher scaling contract."""

import pytest
import torch
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss

from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.models.deepseek_v4.indexer_loss import scale_fused_teacher
from torchtitan_npu_recipes.models.deepseek_v4 import _enable_indexer_aux_loss


@pytest.mark.parametrize("enable_aux_loss", [False, True])
def test_v4_recipe_indexer_loss_uses_unit_coefficient(enable_aux_loss):
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    if enable_aux_loss:
        _enable_indexer_aux_loss(config)

    losses = [
        layer.attention.inner_attention.aux_loss for layer in config.layers if layer.attention.indexer is not None
    ]
    assert losses
    for loss in losses:
        assert loss is not None
        assert loss.coeff == 1.0
        assert loss.mass_weighted
        assert loss.reduce_mesh == "loss"


@pytest.mark.parametrize(("heads", "coeff", "denominator"), [(1, 0.25, 8.0), (2, 1.0, 8.0), (2, 1.0, 32.0)])
def test_fused_teacher_matches_upstream_gradients_and_metric(monkeypatch, heads, coeff, denominator):
    monkeypatch.setattr(AuxLoss, "_group_counts", AuxLoss._group_counts.copy())
    config = SparseIndexerLoss.Config(
        softmax_scale=0.5, num_heads=heads, compress_ratio=1, coeff=coeff, mass_weighted=True, reduce_mesh="loss"
    )
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(4, heads, 2, generator=generator)
    k = torch.randn(4, 2, generator=generator)
    indices = torch.tensor([[0, -1], [1, 0], [-1, -1], [1, 0]])
    scores = torch.randn(4, 2, generator=generator).masked_fill(indices < 0, -torch.inf).requires_grad_()
    padding = torch.tensor([False, False, False, True])
    selected = torch.einsum("thd,tkd->thk", q, k[indices.clamp_min(0)]) * config.softmax_scale
    selected = selected.masked_fill(indices[:, None, :] < 0, -torch.inf)
    # Include a window and sink so that the selected keys' marginal need not sum to one.
    full = torch.cat((selected, torch.zeros(4, heads, 2)), dim=-1)
    lse = full.logsumexp(dim=-1)
    raw = full.softmax(-1)[..., :2].mean(dim=1)
    count = torch.tensor(denominator)

    reference, fused = config.build(), config.build()
    reference(
        q,
        k,
        indices,
        scores,
        lse,
        torch.tensor([0, 4], dtype=torch.int32),
        carrier=q,
        denominator=count,
        padding_mask_T=padding,
    ).sum().backward()
    # SMLAG can report residual mass even on padded slots.
    teacher = scale_fused_teacher(
        fused, raw.masked_fill(indices < 0, 37.0), scores, indices, denominator=count, padding_mask_T=padding
    )
    expected_teacher = raw.masked_fill(padding[:, None], 0.0) * (coeff / denominator)
    torch.testing.assert_close(teacher, expected_teacher)
    torch.testing.assert_close(fused.instance_acc, reference.instance_acc)
    student = scores.detach().masked_fill(indices < 0, -1e9).softmax(dim=-1)
    torch.testing.assert_close(scores.grad, teacher.sum(dim=-1, keepdim=True) * student - teacher)
