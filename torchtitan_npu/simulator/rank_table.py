# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expands ParallelDims/DeviceMesh into a communication-domain RankTable:
for every named mesh axis (pp, dp_replicate, fsdp/dp_shard, cp, tp, ep,
efsdp, and torchtitan_npu's own "etp" once
`_patch_for_parallel_dims_build_mesh` has run), which global ranks belong
to each communication group, and each rank's coordinate along every axis.
See design doc §5.6."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankTable:
    world_size: int
    dim_degrees: dict[str, int] = field(default_factory=dict)
    rank_coordinates: dict[int, dict[str, int]] = field(default_factory=dict)
    process_groups: dict[str, list[list[int]]] = field(default_factory=dict)
    dim_by_group_name: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "dim_degrees": dict(self.dim_degrees),
            "rank_coordinates": {str(rank): dict(coords) for rank, coords in self.rank_coordinates.items()},
            "process_groups": {dim: [list(group) for group in groups] for dim, groups in self.process_groups.items()},
        }


def _groups_along_axis(full_tensor: Any, axis_pos: int) -> list[list[int]]:
    """Every group of ranks that varies along `axis_pos`, with every other
    axis held fixed -- i.e. every "row" of the mesh along that axis."""
    other_dims = [d for d in range(full_tensor.dim()) if d != axis_pos]
    ranges = [range(full_tensor.shape[d]) for d in other_dims]
    groups: list[list[int]] = []
    for combo in itertools.product(*ranges) if ranges else [()]:
        index: list[Any] = [slice(None)] * full_tensor.dim()
        for dim, value in zip(other_dims, combo):
            index[dim] = value
        groups.append([int(r) for r in full_tensor[tuple(index)].flatten().tolist()])
    return groups


def build_rank_table(parallel_dims: Any) -> RankTable:
    """Expand `parallel_dims` (after `.build_mesh()`) into a RankTable."""
    world_size = int(parallel_dims.world_size)
    dim_degrees: dict[str, int] = {
        "pp": int(parallel_dims.pp),
        "dp_replicate": int(parallel_dims.dp_replicate),
        "dp_shard": int(parallel_dims.dp_shard),
        "cp": int(parallel_dims.cp),
        "tp": int(parallel_dims.tp),
        "ep": int(parallel_dims.ep),
    }

    process_groups: dict[str, list[list[int]]] = {}
    dim_by_group_name: dict[str, str] = {}

    composite_meshes = getattr(parallel_dims, "_global_meshes", {}) or {}
    for composite in composite_meshes.values():
        mesh_dim_names = getattr(composite, "mesh_dim_names", None)
        if not mesh_dim_names:
            continue
        full_tensor = composite.mesh
        for axis_pos, axis_name in enumerate(mesh_dim_names):
            if axis_name in process_groups:
                continue  # already captured from an earlier composite mesh
            groups = _groups_along_axis(full_tensor, axis_pos)
            process_groups[axis_name] = groups
            dim_degrees.setdefault(axis_name, int(full_tensor.shape[axis_pos]))
            try:
                group_name = str(composite[axis_name].get_group().group_name)
                dim_by_group_name[group_name] = axis_name
            except (ValueError, RuntimeError, AttributeError):
                pass  # single-axis view unavailable (e.g. degree-1 dim) -- harmless

    # Any dimension never discovered via a composite mesh (e.g. tp/cp
    # disabled, degree 1) still gets a trivial per-rank singleton group.
    for name, degree in list(dim_degrees.items()):
        if name not in process_groups:
            process_groups[name] = [[rank] for rank in range(world_size)]

    rank_coordinates: dict[int, dict[str, int]] = {rank: {} for rank in range(world_size)}
    for name, groups in process_groups.items():
        for group in groups:
            for idx, rank in enumerate(group):
                if 0 <= rank < world_size:
                    rank_coordinates[rank][name] = idx

    return RankTable(
        world_size=world_size,
        dim_degrees=dim_degrees,
        rank_coordinates=rank_coordinates,
        process_groups=process_groups,
        dim_by_group_name=dim_by_group_name,
    )
