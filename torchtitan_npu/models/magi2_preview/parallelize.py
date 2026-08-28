# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for MAGI-2-preview."""

import logging

import torch
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import (
    Shard,
    distribute_module,
    distribute_tensor,
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

from .cp_ulysses import apply_magi2_ulysses_cp
from .expert_parallel import (
    EXPERT_PARAM_NAMES,
    ROUTER_BUFFER_NAMES,
    all_reduce_head_parallel_input_grad,
    all_reduce_head_parallel_output,
    head_range_for_rank,
)
from .feed_forward import CoreMultiHeadMoE, MultiHeadMoELayer
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


def _partition_head_parallel_moe(
    name: str, module: nn.Module, device_mesh: DeviceMesh
) -> None:
    """Shard the fused expert tensors as DTensors along the leading dim.

    Expert rows are head-major (global expert id ``h * E + e``), so
    ``Shard(0)`` over a mesh whose degree divides ``moe_num_heads`` splits
    exactly at head boundaries (asserted in :func:`_apply_moe_parallel`);
    each rank's local shard holds whole heads. The router buffers share
    the same leading ``H * E`` dim and are sharded identically.
    """
    del name
    # distribute_module visits every submodule; only the MoE core carries
    # the fused expert tensors.
    if not isinstance(module, CoreMultiHeadMoE):
        return
    moe_mlp = module
    for param_name in EXPERT_PARAM_NAMES:
        param = getattr(moe_mlp, param_name)
        moe_mlp.register_parameter(
            param_name,
            nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
        )
    for buffer_name in ROUTER_BUFFER_NAMES:
        buffer = getattr(moe_mlp.router, buffer_name)
        moe_mlp.router.register_buffer(
            buffer_name, distribute_tensor(buffer, device_mesh, [Shard(0)])
        )


def _prepare_head_parallel_input(
    moe_mlp: CoreMultiHeadMoE,
    inputs: tuple,
    device_mesh: DeviceMesh,
) -> tuple:
    """Regime-(a) input handling for the replicated token stream.

    The MoE input is replicated across the expert mesh and every rank only
    back-propagates its own heads, so the input gradient must be summed
    across the mesh before it reaches the (replicated) modules upstream of
    the MoE. The forward passes the input through untouched.
    """
    del moe_mlp
    x = all_reduce_head_parallel_input_grad(inputs[0], device_mesh.get_group())
    return (x,) + tuple(inputs[1:])


def _combine_head_parallel_output(
    moe_mlp: CoreMultiHeadMoE, output: torch.Tensor, device_mesh: DeviceMesh
) -> torch.Tensor:
    """Regime-(a) assembly of the local-head partial MoE output."""
    return all_reduce_head_parallel_output(
        output, moe_mlp.head_range, moe_mlp.num_heads, device_mesh.get_group()
    )


def _apply_moe_parallel(
    model: Magi2PreviewModel,
    *,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
) -> None:
    """Shard the routed MoE across the head axis on the EP (or ETP) mesh.

    MAGI-2 routes per head, so expert parallelism is head-parallel: every
    rank owns ``moe_num_heads / degree`` whole heads of each MoE layer's
    fused expert tensors and computes them over all (replicated) tokens;
    the zero-padded partial outputs are all-reduced over the mesh
    (regime (a), see ``expert_parallel.py``). The Ulysses seq<->head
    all-to-all regime (b) needs sequence-sharded tokens and is wired once
    context parallelism lands.

    State-dict keys never change: the expert tensors become ``Shard(0)``
    DTensors with unchanged global shapes, so loading a full checkpoint
    still distributes through DTensor.
    """
    if ep_mesh is None and etp_mesh is None:
        return
    if ep_mesh is not None and etp_mesh is not None:
        raise NotImplementedError(
            "MAGI-2-preview combined EP+ETP would shard the head axis "
            "across two meshes and needs tensor parallelism, which is not "
            "implemented yet"
        )
    mesh = ep_mesh if ep_mesh is not None else etp_mesh
    if mesh.ndim != 1:
        raise ValueError(
            f"MAGI-2 head-parallel MoE expects a 1D mesh, got {mesh.ndim}D"
        )
    degree = mesh.size()
    for layer in model.block.layers.values():
        if not isinstance(layer.mlp, MultiHeadMoELayer):
            continue
        moe_mlp = layer.mlp.moe_mlp
        if moe_mlp.num_heads % degree != 0:
            raise ValueError(
                f"moe_num_heads={moe_mlp.num_heads} must be divisible by "
                f"the head-parallel degree={degree}"
            )
        moe_mlp.set_head_range(
            head_range_for_rank(
                mesh.get_local_rank(), degree, moe_mlp.num_heads
            )
        )
        distribute_module(
            moe_mlp,
            mesh,
            partition_fn=_partition_head_parallel_moe,
            input_fn=_prepare_head_parallel_input,
            output_fn=_combine_head_parallel_output,
        )
    logger.info("Applied MAGI-2-preview head-parallel MoE (regime a)")


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
    """Apply Ulysses CP, head-parallel MoE (EP/ETP), AC and FSDP.

    TP/PP remain deferred. CP shards the packed sequence in original token
    order across the cp mesh (Ulysses head-split all-to-all inside
    attention, autograd exit gather with gradient compensation; see
    ``cp_ulysses.py``). EP shards the routed MoE along the head axis
    (regime (a): replicated tokens + all-reduce); the Ulysses all-to-all
    MoE regime needs CP, so combining CP with EP raises until that
    integration lands.
    """
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
        # Ulysses CP before MoE/AC/FSDP: installs the model cp_context and
        # the attention hooks; torchtitan's "fsdp" mesh spans dp_shard x cp,
        # so CP-replicated parameter grads reduce there during FSDP.
        # Combining CP with EP (regime (b) all-to-all MoE) raises inside.
        apply_magi2_ulysses_cp(
            model,
            cp_mesh=parallel_dims.get_mesh("cp"),
            ep_degree=parallel_dims.ep,
        )
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "MAGI-2-preview model compilation has not been validated with "
            "distributed parallelism; continuing without applying torch.compile"
        )

    # Head-parallel MoE before AC/FSDP, mirroring kimi_k3's ordering of
    # expert parallelism ahead of activation checkpointing and FSDP.
    _apply_moe_parallel(
        model,
        ep_mesh=parallel_dims.get_optional_mesh("ep"),
        etp_mesh=parallel_dims.get_optional_mesh("etp"),
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
