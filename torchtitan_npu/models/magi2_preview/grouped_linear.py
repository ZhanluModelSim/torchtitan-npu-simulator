# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview grouped (per-modality expert) linear projection.

Fork reason: MAGI-2-preview uses per-modality expert weights for its
attention/MLP projections; upstream torchtitan has no equivalent.
Reference: inference/model/magi2_preview.py::GroupedLinearBase
"""

import torch
import torch.nn.functional as F
from torch import nn


class GroupedLinear(nn.Module):
    """Bias-less linear with one expert weight per modality.

    The flat weight layout ``(num_experts * out_features, in_features)``
    matches the official MAGI-2-preview checkpoint. Rows arrive sorted by
    modality and are split with ``m_splits`` (row count per expert); each
    chunk is projected by its own expert slice of the weight. With a single
    expert this is a plain ``F.linear`` and ``m_splits`` is ignored.

    Args:
        in_features: input feature dim.
        out_features: output feature dim of each expert.
        num_experts: number of modality experts.
    """

    __constants__ = ["in_features", "out_features", "num_experts"]

    in_features: int
    out_features: int
    num_experts: int

    def __init__(self, in_features: int, out_features: int, num_experts: int = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.empty(num_experts * out_features, in_features)
        )

    def forward(
        self, x: torch.Tensor, m_splits: list[int] | None = None
    ) -> torch.Tensor:
        if self.num_experts == 1:
            return F.linear(x, self.weight)
        if m_splits is None:
            raise ValueError("m_splits is required when num_experts > 1")
        splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else m_splits
        splits = [int(s) for s in splits]
        if len(splits) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} m_splits entries, got {len(splits)}"
            )

        weight = self.weight.view(self.num_experts, self.out_features, self.in_features)
        outs = [
            F.linear(chunk, weight[i])
            for i, chunk in enumerate(torch.split(x, splits, dim=0))
        ]
        if not outs:
            return x.new_empty((0, self.out_features))
        return torch.cat(outs, dim=0)
