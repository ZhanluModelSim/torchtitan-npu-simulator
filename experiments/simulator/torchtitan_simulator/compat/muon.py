# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped Meta admission for native DistMuon, without replacing its execution."""

import re
from collections.abc import Iterable
from functools import wraps
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Shard

from torchtitan_simulator.capture.optimizer import capture_parameter_update


def validate_muon_layouts(
    optimizer: Any, named_parameters: Iterable[tuple[str, torch.Tensor]], *, expert_parallel_degree: int
) -> None:
    """Reject the unsupported EP=1 transition before building any optimizer state."""
    if expert_parallel_degree != 1:
        return
    layouts = optimizer.optimizer_factory_kwargs_by_name.get("DistMuon", {}).get("compute_sharding_by_fqn", {})
    if not layouts:
        return
    for name, parameter in named_parameters:
        if not parameter.requires_grad or not isinstance(parameter, DTensor):
            continue
        owner = next((group for group in optimizer.param_groups if re.search(group.pattern, name)), None)
        if owner is None or owner.optimizer_name != "DistMuon" or name not in layouts:
            continue
        mesh = parameter.device_mesh
        assert mesh.mesh_dim_names is not None
        for index, (axis, source) in enumerate(zip(mesh.mesh_dim_names, parameter.placements, strict=True)):
            target = layouts[name].shardings_by_mesh_axis.get(axis)
            if (
                mesh.size(index) > 1
                and isinstance(source, Shard)
                and source.dim == 1
                and type(target) is Shard
                and target.dim == 0
            ):
                raise ValueError(
                    f"Muon with EP=1 does not support Shard(1) -> Shard(0): parameter {name!r}, "
                    f"shape={tuple(parameter.shape)}, mesh axis={axis!r}, size={mesh.size(index)}. "
                    "The native FSDP storage layout splits each expert matrix, but Muon requires "
                    "complete matrices sharded along the expert axis. Use AdamW or a supported "
                    "EP/FSDP configuration; EP=1 layouts already using Shard(0) remain supported."
                )


def install_meta_muon(scope) -> dict[str, str]:
    from torchtitan.distributed.flex_shard.dist_muon import DistMuon

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
    return {
        "muon_execution": "native DistMuon compute, redistribution, streams and events on Meta",
        "muon_peer_plan_validation": "not_performed_single_logical_rank; native local plan checks retained",
    }
