# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped Meta admission for native DistMuon, without replacing its execution."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torchtitan.distributed.flex_shard import BucketConfig

from torchtitan_simulator.capture.optimizer import capture_parameter_update

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchtitan.distributed.flex_shard.dist_muon import _ParameterComputeLayout


def _bucket_configs_for_count(layouts: Sequence[_ParameterComputeLayout], count: int) -> tuple[BucketConfig, ...]:
    if count > len(layouts):
        raise ValueError(f"muon_bucket_count={count} exceeds the {len(layouts)} DistMuon parameters")

    groups: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    local: list[str] = []
    for layout in layouts:
        if layout.storage_is_compute_ready:
            local.append(layout.fqn)
            continue
        mesh = layout.param.device_mesh
        axis = layout.redistribution_storage_mesh_axis
        if axis is None or mesh.mesh_dim_names is None:
            raise RuntimeError(f"Redistributed Muon parameter {layout.fqn!r} has no transport axis")
        axis_name = mesh.mesh_dim_names[axis]
        key = (axis_name, tuple(mesh[axis_name].mesh.tolist()))
        groups.setdefault(key, []).append(layout.fqn)

    if count < len(groups):
        raise ValueError(f"muon_bucket_count={count} is too small for {len(groups)} distinct transport groups")
    if count == 1:
        return (BucketConfig(name="all", patterns=("*",)),)
    pools = list(groups.values()) or [[]]
    pools[0].extend(local)

    # Reserve one bucket per transport group, then split the largest share
    # of parameters until all requested buckets can be filled.
    splits = [1] * len(pools)
    for _ in range(count - len(pools)):
        index = max(range(len(pools)), key=lambda i: len(pools[i]) / splits[i])
        splits[index] += 1

    buckets = []
    for names, sizes in zip(pools, splits, strict=True):
        size, remainder = divmod(len(names), sizes)
        start = 0
        for index in range(sizes):
            end = start + size + (index < remainder)
            buckets.append(BucketConfig(name=f"bucket_{len(buckets) + 1}", patterns=tuple(names[start:end])))
            start = end
    return tuple(buckets)


def install_meta_muon(scope, *, bucket_count: int | None = None) -> dict[str, str]:
    from torchtitan.distributed.flex_shard import dist_muon

    DistMuon = dist_muon.DistMuon
    original_storage = DistMuon._validate_parameter_storage
    original_plan = DistMuon._validate_plan_across_ranks
    original_update = DistMuon._apply_update

    def meta_parameters(optimizer):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        return bool(parameters) and all(
            isinstance(parameter, DTensor) and parameter.to_local().device.type == "meta" for parameter in parameters
        )

    @wraps(original_storage)
    def validate_storage(optimizer):
        if meta_parameters(optimizer):
            return torch.device("meta")
        return original_storage(optimizer)

    @wraps(original_plan)
    def validate_plan(optimizer):
        if meta_parameters(optimizer):
            # Native layout/bucket construction still validates the local plan.
            # The hash collective compares values across real peers; a single
            # logical-rank Meta capture cannot perform or certify that comparison.
            return
        return original_plan(optimizer)

    @wraps(original_update)
    def apply_update(optimizer, compute_layout, direction):
        with capture_parameter_update(compute_layout.param, compute_layout.fqn):
            return original_update(optimizer, compute_layout, direction)

    scope.replace(DistMuon, "_validate_parameter_storage", validate_storage)
    scope.replace(DistMuon, "_validate_plan_across_ranks", validate_plan)
    scope.replace(DistMuon, "_apply_update", apply_update)
    if bucket_count is not None:
        original_bind = dist_muon._bind_bucket_configs

        def bind_bucket_configs(configs, layouts, **kwargs):
            configs = _bucket_configs_for_count(layouts, bucket_count)
            return original_bind(configs, layouts, **kwargs)

        scope.replace(dist_muon, "_bind_bucket_configs", bind_bucket_configs)
    return {
        "muon_execution": "native DistMuon compute, redistribution, streams and events on Meta",
        "muon_peer_plan_validation": "not_performed_single_logical_rank; native local plan checks retained",
    }
