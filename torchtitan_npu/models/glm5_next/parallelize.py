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

import torch
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


class _LayersShim(torch.nn.Module):
    """Adapter for the upstream AC policy, which hardcodes
    ``model.get_submodule("layers")``: expose the vision tower's blocks under
    that name. The policy wraps each block and registers it back into the
    same ModuleList, so the tower ends up holding the checkpoint wrappers."""

    def __init__(self, layers: torch.nn.Module):
        super().__init__()
        self.layers = layers


def _distribute_vision_states(vision_encoder: torch.nn.Module, parallel_dims: ParallelDims) -> None:
    """Distribute the vision tower's plain ``nn`` states as replicated DTensors.

    The tower internals are plain ``nn.Module`` children (Conv3d etc. have no
    protocol counterpart), so the protocol ``Module.parallelize`` walk never
    covers their parameters/buffers. FSDP2 requires every parameter to be a
    DTensor on the full SPMD mesh when ``dp_mesh_dims`` is provided (EP > 1),
    so distribute them here with the dense replicated placement, mirroring
    ``Module._spmd_distribute_state``.
    """
    import spmd_types as spmd
    from torch.distributed.tensor import DTensor
    from torchtitan.distributed.spmd_types import set_current_spmd_mesh, spmd_distribute_tensor
    from torchtitan.models.common.decoder_sharding import dense_param_placement

    layout = dense_param_placement(tp=spmd.R)
    mesh = parallel_dims.get_optional_mesh([axis.value for axis in layout.axes()], include_singleton_axes=True)
    assert mesh is not None
    for child in vision_encoder.modules():
        for name, param in list(child.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                continue
            tensor = spmd_distribute_tensor(param, mesh, layout)
            child.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=param.requires_grad))
            with set_current_spmd_mesh(mesh):
                spmd.assert_type(child._parameters[name], layout.axis_types, layout.partition_spec)
        for name, buffer in list(child.named_buffers(recurse=False)):
            if isinstance(buffer, DTensor):
                continue
            tensor = spmd_distribute_tensor(buffer, mesh, layout)
            persistent = name not in child._non_persistent_buffers_set
            child.register_buffer(name, tensor, persistent=persistent)
            with set_current_spmd_mesh(mesh):
                spmd.assert_type(child._buffers[name], layout.axis_types, layout.partition_spec)


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
            ac_policy.apply(_LayersShim(model.vision_encoder.blocks))

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    if model.vision_encoder is not None:
        _distribute_vision_states(model.vision_encoder, parallel_dims)
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
