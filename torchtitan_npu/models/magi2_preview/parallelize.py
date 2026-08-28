# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for MAGI-2-preview."""

import logging

import torch
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .model import Magi2PreviewModel

logger = logging.getLogger(__name__)


def _apply_fsdp(
    model: Magi2PreviewModel,
    dp_mesh: DeviceMesh,
    *,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
) -> None:
    """Shard every transformer layer with FSDP2, then the root model.

    The upstream llama4 ``apply_fsdp`` hardcodes ``tok_embeddings``/``norm``/
    ``output``/``layers`` attributes that MAGI-2-preview does not have, so
    ``fully_shard`` is applied directly here instead.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # PP is rejected before this runs, so the policy always resolves with
    # pp_enabled=False (i.e. "default" means reshard after forward).
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, pp_enabled=False
    )

    for layer in model.block.layers.values():
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division; the training loop scales
    # the loss by the global valid-token count instead.
    disable_fsdp_gradient_division(model)


def _apply_replicate(
    model: Magi2PreviewModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    """Replicate the layers across dp_replicate ranks (no FSDP sharding)."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    for layer in model.block.layers.values():
        replicate(layer, **replicate_config)
    replicate(model, **replicate_config)
    disable_fsdp_gradient_division(model)


def parallelize_magi2_preview(
    model: Magi2PreviewModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply AC and FSDP to MAGI-2-preview; TP/CP/EP/PP are deferred."""
    del model_converters

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "MAGI-2-preview pipeline parallelism is intentionally deferred "
            "until FSDP/TP/EP/CP support is complete"
        )
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "MAGI-2-preview tensor parallelism is not implemented yet; the "
            "packed-token grouped projections have no TP sharding plan"
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "MAGI-2-preview context parallelism is not implemented yet; the "
            "varlen packed sequences have no CP sharding plan"
        )
    if parallel_dims.ep_enabled:
        raise NotImplementedError(
            "MAGI-2-preview expert parallelism is not implemented yet; the "
            "fused CoreMultiHeadMoE expert tensors have no EP sharding plan"
        )
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "MAGI-2-preview model compilation has not been validated with "
            "distributed parallelism; continuing without applying torch.compile"
        )

    # Model compilation is deliberately skipped above, so activation
    # checkpointing must not assume it is wrapping a compiled model.
    model_compile_enabled = False
    if ac_config.mode != "none":
        # Upstream apply_ac looks up model.get_submodule("layers"); MAGI-2
        # layers live at model.block.layers, so checkpoint the block instead.
        apply_moe_ac(
            model.block,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if parallel_dims.fsdp_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        _apply_fsdp(
            model,
            dp_mesh,
            training=training,
            parallelism=parallelism,
        )
        logger.info("Applied MAGI-2-preview FSDP")
    elif parallel_dims.dp_replicate_enabled:
        _apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        logger.info("Applied MAGI-2-preview replicate")

    return model
