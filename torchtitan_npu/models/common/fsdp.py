# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native per-parameter precision for decoder models with MTP layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.distributed.fsdp import apply_fsdp_to_decoder

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import DataParallelMeshDims
    from torchtitan.config.parallelism import FSDPSymmMemScope
    from torchtitan.models.common.decoder import Decoder


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


def apply_fsdp_to_mtp_decoder(
    model: Decoder,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    edp_mesh_dims: DataParallelMeshDims | None = None,
    symm_mem_scope: FSDPSymmMemScope = None,
    *,
    param_dtype_override_fn: Callable[[nn.Parameter], torch.dtype | None] | None = None,
) -> None:
    """Forward native precision overrides through the released MTP sharding layout.

    TorchTitan v0.4.0's MTP helper does not forward the decoder callback. Keep
    its temporary layer aliases while delegating all FSDP behavior upstream.
    """
    mtp_layer_keys = []
    try:
        mtp_layers = getattr(model, "mtp_layers", None)
        if mtp_layers is not None:
            first_mtp_layer_id = len(model.layers)
            for index, layer in enumerate(mtp_layers):
                key = str(first_mtp_layer_id + index)
                model.layers[key] = layer
                mtp_layer_keys.append(key)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=pp_enabled,
            cpu_offload=cpu_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=symm_mem_scope,
            param_dtype_override_fn=param_dtype_override_fn,
        )
    finally:
        for key in mtp_layer_keys:
            del model.layers[key]
