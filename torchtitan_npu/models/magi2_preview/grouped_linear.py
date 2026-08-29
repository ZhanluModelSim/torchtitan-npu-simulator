# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview grouped (per-modality expert) linear projection.

Fork reason: MAGI-2-preview uses per-modality expert weights for its
attention/MLP projections; upstream torchtitan has no equivalent.
Reference: inference/model/magi2_preview.py::GroupedLinearBase

Internal weight layout (checkpoint contract):
- ``num_experts == 1``: the weight is the plain 2D ``(out_features,
  in_features)`` matrix (a regular bias-less linear).
- ``num_experts > 1``: the weight is ALWAYS stored with a per-expert
  leading dim, shape ``(num_experts, out_features, in_features)``, with or
  without tensor parallelism. This is the single internal layout the
  checkpoint code path relies on: the official MAGI-2 checkpoint keeps the
  fused expert-major 2D shape ``(num_experts * out_features, in_features)``,
  and ``state_dict_adapter.Magi2PreviewStateDictAdapter`` converts bijectively
  between that 2D form and this 3D form (its only layout rule).

Tensor parallelism: ``parallelize._apply_tensor_parallel`` shards these
weights across the TP mesh, and the per-expert leading dim makes every split
an honest, single DTensor placement (no plain local slices): column splits
shard the out dim (``Shard(1)`` on the 3D layout, ``Shard(0)`` for a
single-expert 2D weight) and row splits shard the in dim (``Shard(2)`` /
``Shard(1)``). The out dim is ordered so each rank-local shard is a
contiguous range: one row per head for head-indexed weights, contiguous
gate/up pairs for the swiglu7 weights (see the ``slice_*`` helpers), so the
forward math below is unchanged by parallelism.
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


def _grouped_weight_rows(weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    """View a GroupedLinear weight as ``(leading, out_features, in_features)``.

    ``leading`` is ``num_experts`` for the multi-expert 3D layout and 1 for
    the single-expert 2D layout; the last dim is always ``in_features``.
    """
    leading = num_experts if num_experts > 1 else 1
    return weight.reshape(leading, -1, weight.shape[-1])


def slice_grouped_linear_by_heads(
    weight: torch.Tensor,
    num_experts: int,
    num_heads: int,
    head_dim: int,
    num_sections: int,
    head_range: tuple[int, int],
) -> torch.Tensor:
    """Head-granular out-dim slice of a GroupedLinear weight for TP.

    Operates on the internal layout: 2D ``(out_features, in_features)`` for a
    single expert, 3D ``(num_experts, out_features, in_features)`` otherwise.
    The out dim is head-major with ``num_sections`` consecutive sections of
    ``head_dim`` rows per head, so head ``h`` owns the contiguous rows
    ``[h * num_sections * head_dim, (h + 1) * num_sections * head_dim)``.
    The slice keeps heads ``[head_start, head_end)`` of every expert — a
    contiguous out-dim range, hence expressible as one honest ``Shard`` on the
    out dim. Attention ``linear_g`` uses ``num_sections=1, head_dim=1`` and
    ``linear_qkv`` ``num_sections=3`` (head-major q/k/v per head).

    Args:
        weight: full (unsharded) weight in the internal layout.
        num_experts: number of modality experts.
        num_heads: global head count.
        head_dim: rows per head per section.
        num_sections: consecutive sections per head.
        head_range: (head_start, head_end) owned by this rank.

    Returns:
        The rank-local slice, out dim narrowed to
        ``(head_end - head_start) * num_sections * head_dim`` rows (3D for a
        multi-expert weight, 2D for a single expert).
    """
    head_start, head_end = head_range
    if not 0 <= head_start < head_end <= num_heads:
        raise ValueError(
            f"head_range {head_range} must satisfy "
            f"0 <= start < end <= num_heads={num_heads}"
        )
    rows_per_head = num_sections * head_dim
    out_features = num_heads * rows_per_head
    flat = _grouped_weight_rows(weight, num_experts)
    if flat.shape[1] != out_features:
        raise ValueError(
            f"GroupedLinear weight out dim {flat.shape[1]} does not match "
            f"num_heads={num_heads} * num_sections={num_sections}"
            f" * head_dim={head_dim} = {out_features}"
        )
    sliced = flat[:, head_start * rows_per_head : head_end * rows_per_head]
    if num_experts > 1:
        return sliced.contiguous()
    return sliced.reshape(
        (head_end - head_start) * rows_per_head, weight.shape[-1]
    ).contiguous()


def slice_grouped_linear_by_pairs(
    weight: torch.Tensor,
    num_experts: int,
    num_pairs: int,
    pair_range: tuple[int, int],
) -> torch.Tensor:
    """Pair-granular out-dim slice of a GroupedLinear weight for TP.

    For swiglu7-fused projections (``up_gate_proj``, shared expert ``fc1``)
    the out dim interleaves gate/up pairs: rows ``2 * i`` (gate) and
    ``2 * i + 1`` (up) of pair ``i``. Operates on the internal layout and
    keeps pairs ``[pair_start, pair_end)`` of every expert — a contiguous
    out-dim range, so the swiglu7 ``0::2``/``1::2`` pairing of the local
    output is preserved exactly and the split is one honest out-dim ``Shard``.

    Args:
        weight: full (unsharded) weight in the internal layout.
        num_experts: number of modality experts.
        num_pairs: global gate/up pair count of each expert.
        pair_range: (pair_start, pair_end) owned by this rank.

    Returns:
        The rank-local slice, out dim narrowed to ``2 * (pair_end -
        pair_start)`` rows (3D for a multi-expert weight, 2D otherwise).
    """
    pair_start, pair_end = pair_range
    if not 0 <= pair_start < pair_end <= num_pairs:
        raise ValueError(
            f"pair_range {pair_range} must satisfy "
            f"0 <= start < end <= num_pairs={num_pairs}"
        )
    out_features = 2 * num_pairs
    flat = _grouped_weight_rows(weight, num_experts)
    if flat.shape[1] != out_features:
        raise ValueError(
            f"GroupedLinear weight out dim {flat.shape[1]} does not match "
            f"2 * num_pairs={num_pairs} = {out_features}"
        )
    sliced = flat[:, 2 * pair_start : 2 * pair_end]
    if num_experts > 1:
        return sliced.contiguous()
    return sliced.reshape(
        2 * (pair_end - pair_start), weight.shape[-1]
    ).contiguous()


class GroupedLinear(nn.Module):
    """Bias-less linear with one expert weight per modality.

    The weight layout is documented in the module docstring: a single expert
    keeps the plain 2D ``(out_features, in_features)`` matrix, while multiple
    experts always use the per-expert leading dim ``(num_experts,
    out_features, in_features)``. Rows arrive sorted by modality and are split
    with ``m_splits`` (row count per expert); each chunk is projected by its
    own expert slice of the weight. With a single expert this is a plain
    ``F.linear`` and ``m_splits`` is ignored.

    Under TP the weight becomes a rank-local DTensor shard (see the module
    docstring) and ``in_features``/``out_features`` hold the rank-local dims;
    the forward math is identical.

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
        if num_experts > 1:
            weight = torch.empty(num_experts, out_features, in_features)
        else:
            weight = torch.empty(out_features, in_features)
        self.weight = nn.Parameter(weight)

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

        weight = weight.reshape(self.num_experts, self.out_features, self.in_features)
        outs = [
            F.linear(chunk, weight[i])
            for i, chunk in enumerate(torch.split(x, splits, dim=0))
        ]
        if not outs:
            return x.new_empty((0, self.out_features))
        return torch.cat(outs, dim=0)
