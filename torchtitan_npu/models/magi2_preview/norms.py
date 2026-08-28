# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview multi-modality RMSNorm.

Fork reason: MAGI-2-preview applies per-modality gains on top of RMSNorm,
which upstream torchtitan's norms cannot express.
Reference: inference/model/magi2_preview.py::MultiModalityRMSNorm
"""

import torch
from torch import nn


class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with per-modality gains over modality-sorted rows.

    Rows arrive sorted by modality and ``m_splits`` gives the row count of
    each modality chunk. The mean-square is computed over the last dim in
    fp32, then each modality chunk is scaled by its own gain
    ``weight_chunk + weight_bias``. Weights are filled with zeros by the
    model-level ``init_weights``, so the initial gain is the identity.

    For ``num_patterns > 1`` (q/k norms) the input is ``(T, H, dim)`` and the
    weight is viewed as ``(H, dim)``; splitting still happens along the token
    dim, which is equivalent to reshaping to ``(T * H, dim)`` and scaling
    ``m_splits`` by ``H`` since rows stay token-major modality-contiguous.

    Args:
        dim: normalized feature dim (per-head dim for q/k norms).
        eps: epsilon added to the mean square.
        num_modality: number of modality experts; 1 disables the split logic
            and ``m_splits`` is ignored.
        num_patterns: head patterns folded into the weight (attention heads
            for q/k norms).
        out_dtype: output dtype cast; defaults to the input dtype. q/k norms
            use ``torch.float32``.
    """

    __constants__ = ["dim", "eps", "num_modality", "num_patterns", "weight_bias"]

    dim: int
    eps: float
    num_modality: int
    num_patterns: int
    weight_bias: float

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        num_modality: int = 1,
        num_patterns: int = 1,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.num_patterns = num_patterns
        self.out_dtype = out_dtype
        self.weight_bias = 1.0
        self.weight = nn.Parameter(
            torch.empty(num_patterns * dim * num_modality, dtype=torch.float32)
        )

    def forward(
        self, x: torch.Tensor, m_splits: list[int] | None = None
    ) -> torch.Tensor:
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)

        out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
        if self.num_modality == 1:
            gain = self.weight.view(self.num_patterns, self.dim) + self.weight_bias
            return (t * gain).to(out_dtype)

        if m_splits is None:
            raise ValueError("m_splits is required when num_modality > 1")
        splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else m_splits
        splits = [int(s) for s in splits]
        if len(splits) != self.num_modality:
            raise ValueError(
                f"Expected {self.num_modality} m_splits entries, got {len(splits)}"
            )

        weight_chunks = self.weight.chunk(self.num_modality)
        scaled = [
            chunk * (w.view(self.num_patterns, self.dim) + self.weight_bias)
            for chunk, w in zip(torch.split(t, splits, dim=0), weight_chunks, strict=True)
        ]
        return torch.cat(scaled, dim=0).to(out_dtype)
