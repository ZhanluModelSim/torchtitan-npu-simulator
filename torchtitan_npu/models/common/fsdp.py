# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Translate explicit per-model dtype selections and expert mesh dims for FSDP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch.distributed.fsdp import DataParallelMeshDims
from torchtitan.models.common.moe import MoE

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh


def make_param_dtype_override_fn(
    model: nn.Module,
    parameter_dtypes: Mapping[str, torch.dtype],
) -> Callable[[nn.Parameter], torch.dtype | None]:
    """Resolve exact parameter names before FSDP adds aliases or replaces parameters.

    The native callback receives the parameter itself, so tied aliases and the
    temporary MTP layer names resolve to the same policy. The selected logical
    parameters must be identical on every rank, as required by FSDP.
    """
    parameters = dict(model.named_parameters(remove_duplicate=False))
    unknown = parameter_dtypes.keys() - parameters.keys()
    if unknown:
        raise ValueError(f"Parameter dtype overrides name unknown parameters: {sorted(unknown)}")
    dtypes_by_id = {}
    for name, dtype in parameter_dtypes.items():
        param_id = id(parameters[name])
        if param_id in dtypes_by_id and dtypes_by_id[param_id] != dtype:
            raise ValueError(f"Conflicting dtype overrides for tied parameter {name}")
        dtypes_by_id[param_id] = dtype

    def override(param: nn.Parameter) -> torch.dtype | None:
        return dtypes_by_id.get(id(param))

    return override


def disable_expert_replication_dims(
    model: nn.Module,
    edp_mesh: DeviceMesh | None,
    edp_mesh_dims: DataParallelMeshDims | None,
    *,
    ep_degree: int,
) -> DataParallelMeshDims:
    """Fold dense replica axes into routed-expert FSDP shard axes."""
    if edp_mesh is None or edp_mesh_dims is None:
        raise ValueError("disable_expert_replication requires an explicit expert mesh")
    replicate_names = edp_mesh_dims.replicate_names
    if not replicate_names:
        raise ValueError("disable_expert_replication requires an expert data-parallel replicate axis")
    shard_names = tuple(dict.fromkeys((*replicate_names, *edp_mesh_dims.shard_names)))
    efsdp_ep_size = math.prod(edp_mesh[axis].size() for axis in shard_names) * ep_degree
    for module in model.modules():
        if isinstance(module, MoE) and efsdp_ep_size > module.num_experts:
            raise ValueError(
                f"disable_expert_replication requires effective expert DP ({efsdp_ep_size}) "
                f"<= num_experts ({module.num_experts})"
            )
    return DataParallelMeshDims(shard=shard_names, replicate=None)
