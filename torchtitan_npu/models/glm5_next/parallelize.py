# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for GLM-5.3-Flash (torchtitan 0.3.0).

v1 support matrix (capture + real-run): single device, FSDP2 over the
dense data-parallel mesh, and expert parallelism (the routed experts ride
the sparse expert-FSDP mesh via the shared ``MoE``/``RoutedExperts``
assembly and the ``AllToAllTokenDispatcher``). TP, sequence parallelism,
CP, PP and ETP land with the parallelism matrix phase.
"""

import logging

import torch  # noqa: TC002  # runtime-evaluated parameter annotation below
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config

from torchtitan_npu.extensions.torchtitan.distributed.fsdp import apply_fsdp_to_decoder

from .attention import DSAIndexer
from .model import Glm5NextModel

logger = logging.getLogger(__name__)


def parallelize_glm5_next(
    model: Glm5NextModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AC and FSDP to the GLM-5.3-Flash model.

    NOTE: The passed-in model should be on meta device (capture builds it
    that way); init_states runs after this through the trainer.
    """
    if parallelism.spmd_backend != "spmd_types":
        raise ValueError(f"GLM-5.3-Flash requires the native spmd_types backend, got {parallelism.spmd_backend!r}")
    if getattr(parallel_dims, "expert_tensor_parallel_degree", 1) > 1:
        raise NotImplementedError(
            "GLM-5.3-Flash expert tensor parallelism is not in the v1 "
            "support matrix: routed experts are replicated across the TP "
            "mesh and sharded only along the EP dim."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError("GLM-5.3-Flash context parallelism is not implemented yet.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("GLM-5.3-Flash pipeline parallelism is not implemented yet.")
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "GLM-5.3-Flash tensor parallelism lands with the spmd_types "
            "sharding declarations; the v1 matrix is single + FSDP2 + EP."
        )

    validate_config(parallel_dims, model)
    model.parallelize(parallel_dims)
    # spmd distribution re-registers params with requires_grad=True
    # (torchtitan module.py re-wraps every param); restore the production
    # freeze: the indexer forward runs under no_grad and is never trained.
    for module in model.modules():
        if isinstance(module, DSAIndexer):
            module.requires_grad_(False)

    # Excluded from FSDP units so their gradients are not expected.
    ignored_params: set[torch.nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, DSAIndexer):
            ignored_params.update(module.parameters())

    if compile_config.enable:
        logger.warning(
            "GLM-5.3-Flash model compilation has not been validated; continuing without applying torch.compile"
        )

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    if model.vision_encoder is not None:
        from torchtitan.distributed.fsdp import apply_fsdp_to_vision_encoder

        apply_fsdp_to_vision_encoder(
            model.vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
            dp_mesh_dims=dp_mesh_dims,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        ignored_params=ignored_params,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        dp_mesh_dims=dp_mesh_dims,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        edp_mesh_dims=edp_mesh_dims,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    return model
