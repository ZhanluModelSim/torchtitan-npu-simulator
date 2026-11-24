# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt the teacher returned by SMLAG to TorchTitan's mass-weighted SparseIndexerLoss."""

import spmd_types as spmd
import torch
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss


@torch.no_grad()
def scale_fused_teacher(
    aux_loss: SparseIndexerLoss,
    teacher: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    *,
    denominator: torch.Tensor,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Record the same KL as the reference and pass its scaled mass to SLIKG."""
    if not aux_loss.mass_weighted:
        raise ValueError("Fused SparseIndexerLoss requires mass_weighted=True.")
    logits = scores.reshape(-1, scores.shape[-1]).float()
    valid = (indices.reshape_as(logits) >= 0) & torch.isfinite(logits)
    if padding_mask_T is not None:
        valid = valid & ~padding_mask_T.reshape(-1, 1)
    p = teacher.reshape_as(logits).float().masked_fill(~valid, 0.0)
    target = p / p.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    with spmd.no_typecheck():
        scale = 1.0 / denominator.clamp_min(1)
        aux_loss.instance_acc.add_(aux_loss._logged_kl(p, target, logits, valid).sum() * scale)
        # SLIKG differentiates sum(p) * softmax(logits) - p internally.
        return (p * (aux_loss.coeff * scale)).reshape_as(teacher)
