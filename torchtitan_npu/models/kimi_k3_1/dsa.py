# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 indexer and differentiable selected-key attention reference.

Geometry: moonshotai/Kimi-K3.1-Pro@d9e99b0792a5e93b78ea9bea62a68167e36ddd9e.
Indexer reference: NVIDIA-NeMo/Automodel@c581829ae0709eabafa333bd0c5f33486ae439e9.
The indexer follows KimiDSAIndexer: normalized query
latent, biased key LayerNorm, weighted ReLU scores, causal hard top-k. This
is an eager reference, not a fused sparse NPU kernel or indexer distillation.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class KimiIndexerLayerNorm(Module):
    """Biased FP32 LayerNorm with output cast back to the projection dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))
        self.bias = nn.Parameter(torch.empty(config.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps).to(x.dtype)


class KimiDSAIndexer(Module):
    """Frozen per-layer selector; language-model loss does not train hard top-k."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        wq_b: Linear.Config
        wk: Linear.Config
        weights_proj: Linear.Config
        k_norm: KimiIndexerLayerNorm.Config
        num_heads: int = 32
        head_dim: int = 128
        topk: int = 2048

    def __init__(self, config: Config):
        super().__init__()
        if min(config.num_heads, config.head_dim, config.topk) < 1:
            raise ValueError("Kimi DSA index heads, head dimension and top-k must be positive")
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.topk = config.topk
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.weights_proj = config.weights_proj.build()
        self.k_norm = config.k_norm.build()
        self.requires_grad_(False)

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, query_latent: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        queries = self.wq_b(query_latent).reshape(batch, seq_len, self.num_heads, self.head_dim)
        keys = self.k_norm(self.wk(hidden))
        weights = self.weights_proj(hidden).float() * (self.head_dim * self.num_heads) ** -0.5
        return self._select(queries, keys, weights)

    def _select(self, queries: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Select keys; the simulator may replace only this computational boundary."""
        seq_len = queries.shape[1]
        positions = torch.arange(seq_len, device=queries.device)
        indices = []
        # Chunk only query rows; every query still scores its causal key history.
        # Explicit FP32 scores must stay FP32 even under external autocast.
        # Meta has no autocast backend; capture itself never enables autocast.
        device_type = "cpu" if queries.device.type == "meta" else queries.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            for start in range(0, seq_len, 128):
                end = min(start + 128, seq_len)
                q = queries[:, start:end].float().transpose(1, 2)
                logits = torch.matmul(q, keys[:, :end].float().unsqueeze(1).transpose(-1, -2))
                scores = (logits.relu() * weights[:, start:end].transpose(1, 2).unsqueeze(-1)).sum(1)
                scores = scores.masked_fill(
                    positions[:end].unsqueeze(0) > positions[start:end].unsqueeze(1), -torch.inf
                )
                selected = scores.topk(min(self.topk, end), dim=-1)
                selected_indices = selected.indices.masked_fill(~selected.values.isfinite(), -1)
                indices.append(
                    F.pad(selected_indices, (0, min(self.topk, seq_len) - selected_indices.shape[-1]), value=-1)
                )
        return torch.cat(indices, dim=1)


def selected_key_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend only to selected keys, in B/H/S/D layout, retaining native autograd.

    Materialized gathers are bounded to 32 query rows. Invalid early causal
    slots use -1 and are masked before softmax; they never read a future key.
    """
    batch, heads, seq_len, width = query.shape
    output = []
    for start in range(0, seq_len, 32):
        end = min(start + 32, seq_len)
        selected = indices[:, start:end]
        count = end - start
        offsets = torch.arange(batch, device=indices.device)[:, None, None] * key.shape[2]
        gather_indices = (selected.clamp_min(0) + offsets).reshape(-1)
        # Flatten batch/sequence before indexing so backward scatters into the
        # original KV storage, not a query-expanded B/H/Q/S/D temporary.
        selected_keys = key.transpose(1, 2).reshape(batch * key.shape[2], heads, width).index_select(0, gather_indices)
        selected_keys = selected_keys.reshape(batch, count, selected.shape[-1], heads, width).permute(0, 3, 1, 2, 4)
        selected_values = value.transpose(1, 2).reshape(batch * value.shape[2], heads, value.shape[-1])
        selected_values = selected_values.index_select(0, gather_indices)
        selected_values = selected_values.reshape(batch, count, selected.shape[-1], heads, value.shape[-1])
        selected_values = selected_values.permute(0, 3, 1, 2, 4)
        scores = torch.matmul(query[:, :, start:end].unsqueeze(-2), selected_keys.transpose(-1, -2)).squeeze(-2)
        scores = (scores.float() * scale).masked_fill(selected[:, None] < 0, -torch.inf)
        probabilities = scores.softmax(-1).to(value.dtype)
        output.append(torch.matmul(probabilities.unsqueeze(-2), selected_values).squeeze(-2))
    return torch.cat(output, dim=2)
