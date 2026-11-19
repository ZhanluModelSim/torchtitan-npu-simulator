# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the system3 decoder.

DP/FSDP/HSDP and EP follow the shared decoder recipe: native AC policy first,
then ``apply_fsdp_to_decoder`` (routed experts land on the expert FSDP mesh
when EP > 1). Both the default DTensor backend and the ``spmd_types``
sharding-config backend are supported, mirroring the multi-head-moe
parallelizer.

TP and CP fail fast: the ISA/GDN2 head axes and the mHC branch mixing would
need head-sharded layouts, and the compressed KV grid, GDN2 state and Engram
hash all span the full sequence (``System3Model.Config.update_from_config``
rejects both at the config level as well).
"""

import torch
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.models.common.decoder import Decoder

from torchtitan_npu.extensions.torchtitan.distributed.fsdp import apply_fsdp_to_decoder


def _shard_system3_engram_tables(
    model,
    *,
    edp_mesh,
    edp_mesh_dims,
) -> set[torch.nn.Parameter]:
    """Exclude CPU hash tables from FSDP and wire their sparse-grad replicas.

    A table FSDP does not manage still has replicas along the sparse
    data-parallel axes, and their owner-local sparse gradients have to be
    summed before the optimizer step.
    """
    ignored_params: set[torch.nn.Parameter] = set()
    for module in model.modules():
        if getattr(module, "uses_host_offload", False):
            ignored_params.add(module.weight)
            wire_replicas = getattr(module, "wire_sparse_grad_replicas", None)
            if wire_replicas is not None:
                wire_replicas(edp_mesh=edp_mesh, edp_mesh_dims=edp_mesh_dims)
    return ignored_params


def parallelize_system3(
    model: Decoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig | None,
    dump_folder: str,
):
    """Apply AC, compile and data/expert parallelism to the model."""
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported for system3.")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "system3 does not support context parallel: the compressed KV grid, "
            "GDN2 state and Engram hash span the full sequence"
        )

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        # Module protocol: wires the EP mesh on the latent token dispatcher via
        # LatentRoutedExperts.parallelize; dense modules carry no sharding configs.
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)

    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    ignored_params = _shard_system3_engram_tables(
        model,
        edp_mesh=edp_mesh if parallel_dims.ep_enabled else dp_mesh,
        edp_mesh_dims=edp_mesh_dims if parallel_dims.ep_enabled else dp_mesh_dims,
    )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        ignored_params=ignored_params,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
    )

    return model
