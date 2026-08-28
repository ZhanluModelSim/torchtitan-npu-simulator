# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview grouped (per-modality expert) linear projection.

Fork reason: MAGI-2-preview uses per-modality expert weights for its
attention/MLP projections; upstream torchtitan has no equivalent.
Reference: inference/model/magi2_preview.py::GroupedLinearBase

Tensor parallelism: ``parallelize._apply_tensor_parallel`` shards these
weights across the TP mesh. Row (input-dim) splits become ``Shard(1)``
DTensors and single-expert column splits ``Shard(0)`` DTensors; the
multi-expert column splits (per-modality expert out-dim slices) are plain
rank-local slices because no single DTensor placement expresses them on
the fused expert-major layout. The rank-local weights keep the exact
layout ``forward`` expects, so the math below is unchanged: slicing at
head/pair granularity plus rank-local ``in_features``/``out_features``
bookkeeping is all the parallelize step needs (see the ``slice_*``
helpers).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor


def _maybe_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard of a DTensor, or the tensor itself.

    Tensor parallelism and head-parallel MoE (see ``expert_parallel.py``)
    replace sharded weights with DTensors over the parallel mesh; the
    grouped-linear/routing math always runs on the plain local shard.
    """
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def slice_grouped_linear_by_heads(
    weight: torch.Tensor,
    num_experts: int,
    num_heads: int,
    head_dim: int,
    num_sections: int,
    head_range: tuple[int, int],
) -> torch.Tensor:
    """Head-granular out-dim slice of a GroupedLinear weight for TP.

    The fused weight ``(num_experts * num_sections * num_heads * head_dim,
    in_features)`` holds, per expert, ``num_sections`` consecutive sections
    of ``num_heads * head_dim`` rows each, head-major inside a section
    (head ``h`` owns rows ``[h * head_dim, (h + 1) * head_dim)``). The
    slice keeps heads ``[head_start, head_end)`` of every expert and
    section, concatenated per expert in section order — the exact layout a
    GroupedLinear with ``out_features = num_sections * local_heads *
    head_dim`` consumes unchanged (e.g. attention ``linear_g`` uses
    ``num_sections=1, head_dim=1`` and ``linear_qkv`` ``num_sections=3``
    with the q/k/v section layout).

    Args:
        weight: full (unsharded) fused weight.
        num_experts: number of modality experts.
        num_heads: global head count of each section.
        head_dim: rows per head.
        num_sections: consecutive head-indexed sections per expert.
        head_range: (head_start, head_end) owned by this rank.

    Returns:
        ``(num_experts * num_sections * local_heads * head_dim,
        in_features)`` contiguous slice.
    """
    head_start, head_end = head_range
    if not 0 <= head_start < head_end <= num_heads:
        raise ValueError(
            f"head_range {head_range} must satisfy "
            f"0 <= start < end <= num_heads={num_heads}"
        )
    out_features = num_sections * num_heads * head_dim
    expected = num_experts * out_features
    if weight.shape[0] != expected:
        raise ValueError(
            f"GroupedLinear weight leading dim {weight.shape[0]} does not "
            f"match num_experts={num_experts} * num_sections={num_sections}"
            f" * num_heads={num_heads} * head_dim={head_dim} = {expected}"
        )
    w = weight.view(num_experts, num_sections, num_heads, head_dim, -1)
    sliced = w[:, :, head_start:head_end]
    return sliced.reshape(
        num_experts * num_sections * (head_end - head_start) * head_dim,
        weight.shape[1],
    ).contiguous()


def slice_grouped_linear_by_pairs(
    weight: torch.Tensor,
    num_experts: int,
    num_pairs: int,
    pair_range: tuple[int, int],
) -> torch.Tensor:
    """Pair-granular out-dim slice of a GroupedLinear weight for TP.

    For swiglu7-fused projections (``up_gate_proj``, shared expert
    ``fc1``), the per-expert out dim interleaves gate/up pairs: rows
    ``2 * i`` (gate) and ``2 * i + 1`` (up) of pair ``i``. The slice keeps
    pairs ``[pair_start, pair_end)`` of every expert — an even-offset,
    even-length row range per expert, so the swiglu7 ``0::2``/``1::2``
    pairing of the local output is preserved exactly.

    Args:
        weight: full (unsharded) fused weight
            ``(num_experts * 2 * num_pairs, in_features)``.
        num_experts: number of modality experts.
        num_pairs: global gate/up pair count of each expert.
        pair_range: (pair_start, pair_end) owned by this rank.

    Returns:
        ``(num_experts * 2 * local_pairs, in_features)`` contiguous slice.
    """
    pair_start, pair_end = pair_range
    if not 0 <= pair_start < pair_end <= num_pairs:
        raise ValueError(
            f"pair_range {pair_range} must satisfy "
            f"0 <= start < end <= num_pairs={num_pairs}"
        )
    expected = num_experts * 2 * num_pairs
    if weight.shape[0] != expected:
        raise ValueError(
            f"GroupedLinear weight leading dim {weight.shape[0]} does not "
            f"match num_experts={num_experts} * 2 * num_pairs={num_pairs}"
            f" = {expected}"
        )
    w = weight.view(num_experts, num_pairs, 2, weight.shape[1])
    return w[:, pair_start:pair_end].reshape(
        num_experts * 2 * (pair_end - pair_start), weight.shape[1]
    ).contiguous()


class GroupedLinear(nn.Module):
    """Bias-less linear with one expert weight per modality.

    The flat weight layout ``(num_experts * out_features, in_features)``
    matches the official MAGI-2-preview checkpoint. Rows arrive sorted by
    modality and are split with ``m_splits`` (row count per expert); each
    chunk is projected by its own expert slice of the weight. With a single
    expert this is a plain ``F.linear`` and ``m_splits`` is ignored.

    Under TP the weight becomes a rank-local slice (plain or DTensor, see
    the module docstring) and ``in_features``/``out_features`` hold the
    rank-local dims; the forward math is identical.

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
        weight = _maybe_local(self.weight)
        if self.num_experts == 1:
            return F.linear(x, weight)
        if m_splits is None:
            raise ValueError("m_splits is required when num_experts > 1")
        splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else m_splits
        splits = [int(s) for s in splits]
        if len(splits) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} m_splits entries, got {len(splits)}"
            )

        weight = weight.view(self.num_experts, self.out_features, self.in_features)
        outs = [
            F.linear(chunk, weight[i])
            for i, chunk in enumerate(torch.split(x, splits, dim=0))
        ]
        if not outs:
            return x.new_empty((0, self.out_features))
        return torch.cat(outs, dim=0)
