# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the sys2mot VLA.

DP/FSDP follows the shared decoder recipe: native AC policy first, then
``apply_fsdp_to_decoder``. Every backbone block carries ``moe_enabled`` with a
``moe.routed_experts.inner_experts`` grouped-GEMM child, so the shared helper's
MoE branch applies; the vision encoder and the action stream ride the root
FSDP unit.

TP and CP fail fast (config level and here): the CSA2 head axis, the grouped
output low-rank factors and the mHC branch mixing would need sharded layouts,
and the compressed KV grid, Engram hash and the per-layer prefix/action joint
attention span the full sequence. EP follows the shared MoE decoder recipe:
the routed experts ride the expert-parallel FSDP mesh (efsdp) and the token
dispatcher runs the all-to-all exchange (local fallback at EP=1).
"""

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
from torchtitan.models.common.decoder import Decoder


def parallelize_sys2mot(
    model: Decoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig | None,
    dump_folder: str,
):
    """Apply AC, compile and data parallelism to the model."""
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported for sys2mot.")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "sys2mot does not support context parallel: the CSA2 compressed KV "
            "grid, Engram hash and the per-layer prefix/action joint attention "
            "span the full sequence"
        )

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        # Module protocol: sharding configs are replicated on TP (rejected
        # above); with EP this wires the dispatcher's EP meshes.
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

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
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
