# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve opaque framework ProcessGroup names to parallel-dimension names."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from torchtitan_npu.simulator.rank_table import RankTable

if TYPE_CHECKING:
    from torchtitan_npu.simulator.capture.comm_events import CommEvent


_DIMENSION_ALIASES = {
    "dp_shard": "fsdp",
    "efsdp": "fsdp",
}

_MODEL_COMMUNICATION_DIMENSIONS = {
    "pp",
    "dp_replicate",
    "dp_shard",
    "fsdp",
    "cp",
    "tp",
    "ep",
    "efsdp",
    "etp",
}

_DIMENSION_PRIMITIVES = {
    "pp": {"p2p_send", "p2p_recv"},
    "dp_replicate": {"allreduce"},
    "fsdp": {"allgather", "reduce_scatter"},
    "cp": {"allgather", "all_to_all", "p2p_send", "p2p_recv"},
    "tp": {"allgather", "allreduce", "reduce_scatter", "all_to_all"},
    "ep": {"all_to_all"},
    "etp": {"allgather", "allreduce", "reduce_scatter", "all_to_all"},
}


def _semantic_dimension(dimension: str) -> str:
    return _DIMENSION_ALIASES.get(dimension, dimension)


def _normalized_group(group: Iterable[int]) -> frozenset[int]:
    return frozenset(int(rank) for rank in group)


class CommGroupResolver:
    """Resolve a captured communication group without guessing on ambiguity."""

    def __init__(self, rank_table: RankTable) -> None:
        self._dimension_by_raw_name = {
            str(raw_name): _semantic_dimension(dimension)
            for raw_name, dimension in rank_table.dim_by_group_name.items()
        }
        dimensions_by_group: dict[frozenset[int], set[str]] = {}
        for dimension, groups in rank_table.process_groups.items():
            if dimension not in _MODEL_COMMUNICATION_DIMENSIONS:
                continue
            if rank_table.dim_degrees.get(dimension, 1) <= 1:
                continue
            for group in groups:
                dimensions_by_group.setdefault(
                    _normalized_group(group), set()
                ).add(_semantic_dimension(dimension))
        self._dimension_by_group = {
            groups: next(iter(dimensions))
            for groups, dimensions in dimensions_by_group.items()
            if len(dimensions) == 1
        }

    def resolve(
        self,
        raw_group_name: str,
        comm_ranks: Iterable[Iterable[int]],
        comm_primitive: str = "",
    ) -> str:
        """Return a semantic dimension, or the original name when unresolved."""
        raw_group_name = str(raw_group_name)
        dimension = self._dimension_by_raw_name.get(raw_group_name)
        if dimension:
            return dimension

        groups = [_normalized_group(group) for group in comm_ranks if group]
        resolved_dimensions = [
            self._dimension_by_group.get(group) for group in groups
        ]
        if (
            resolved_dimensions
            and all(dimension is not None for dimension in resolved_dimensions)
            and len(set(resolved_dimensions)) == 1
        ):
            dimension = resolved_dimensions[0]
            if (
                dimension
                and (
                    not comm_primitive
                    or comm_primitive
                    in _DIMENSION_PRIMITIVES.get(dimension, set())
                )
            ):
                return dimension
        return raw_group_name


def resolve_comm_event_groups(
    comm_events: Iterable[CommEvent],
    rank_table: RankTable,
) -> dict[int, tuple[str, str]]:
    """Enrich CommEvents and return L0 op names keyed by op ID.

    ``group_name`` is the public semantic field. ``raw_group_name`` preserves
    the framework-generated ProcessGroup identifier. ``comm_dim`` remains a
    compatibility alias for existing memory and visualization consumers.
    """
    resolver = CommGroupResolver(rank_table)
    group_by_op_id: dict[int, tuple[str, str]] = {}

    for event in comm_events:
        raw_group_name = event.raw_group_name or event.group_name
        group_name = resolver.resolve(
            raw_group_name,
            event.comm_ranks,
            event.comm_primitive,
        )
        event.raw_group_name = raw_group_name
        event.group_name = group_name
        event.comm_dim = group_name
        if event.op_id:
            group_by_op_id[event.op_id] = (group_name, raw_group_name)

    return group_by_op_id
