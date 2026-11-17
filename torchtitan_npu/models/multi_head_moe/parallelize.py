# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the SLA2 + Multi-Head-MoE decoder.

DP/FSDP/HSDP and EP follow the shared decoder recipe: native AC policy first,
then ``apply_fsdp_to_decoder`` (routed experts land on the expert FSDP mesh
when EP > 1). Both the default DTensor backend and the ``spmd_types``
sharding-config backend are supported, mirroring the Qwen3.5 parallelizer.

TP and CP fail fast: the SLA2 head axis and the per-head gate would need
head-sharded layouts, and the block pooling plus per-block linear state span
the full sequence (``MultiHeadMoEModel.Config.update_from_config`` rejects
both at the config level as well).
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
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.models.common.decoder import Decoder


def parallelize_multi_head_moe(
    model: Decoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AC, compile and data/expert parallelism to the model."""
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported for multi_head_moe.")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "multi_head_moe does not support context parallel: the SLA2 block pooling "
            "and per-block linear state span the full sequence"
        )

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        # Module protocol: wires the EP mesh on the MoE token dispatchers via
        # RoutedExperts.parallelize; dense modules carry no sharding configs.
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
