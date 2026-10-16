# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for Qwen3-Next.

Initial support matrix (v1):
- FSDP / eFSDP data parallel sharding (via upstream ``apply_fsdp``)
- Expert parallelism for the routed MoE experts
- DP replicate
- Full / selective activation checkpointing

TP / CP / PP raise ``NotImplementedError`` until the fused qkvz grouped
sharding boundary is implemented.
"""

import logging

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .model import Qwen3NextModel

logger = logging.getLogger(__name__)


def _apply_expert_parallel(model: Qwen3NextModel, ep_mesh) -> None:
    """Shard routed experts across the expert-parallel mesh."""
    for layer in model.layers.values():
        if not layer.moe_enabled:
            continue
        if layer.moe.num_experts % ep_mesh.size() != 0:
            raise ValueError(
                f"num_experts={layer.moe.num_experts} must be divisible by "
                f"EP degree={ep_mesh.size()}"
            )
        parallelize_module(
            layer.moe.experts,
            ep_mesh,
            ExpertParallel(),
        )
    logger.info("Applied Qwen3-Next expert parallelism")


def _apply_expert_fsdp(model: Qwen3NextModel, edp_mesh, mp_policy) -> None:
    """Mirror MindSpeed ``expert_fully_shard_modules``.

    Each routed-experts module becomes its own FSDP unit over the
    EP-internal data-parallel mesh, sharded on dim 1
    (``efsdp_shard_placement_fn='shard_by_dim_1'`` default), so expert
    weights all-gather when the experts module is entered (forward and
    activation-checkpoint recompute), matching the measured stream.
    """
    for layer in model.layers.values():
        if not layer.moe_enabled:
            continue
        fully_shard(
            layer.moe.experts,
            mesh=edp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            shard_placement_fn=lambda p: Shard(1),
        )
    logger.info("Applied Qwen3-Next expert FSDP (eFSDP) over edp mesh")


def _apply_fsdp(
    model: Qwen3NextModel,
    dp_mesh,
    edp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    ep_enabled: bool = False,
) -> None:
    """Mirror the MindSpeed fsdp2 default plan.

    ``fsdp_modules = ['model.layers.{*}', 'model.embed_tokens', 'lm_head']``
    with ``reshard_after_forward=True``: the embedding, the (untied)
    lm_head, and every decoder layer are separate FSDP units; the final
    norm is left unwrapped (no all-gather of its own). Routed-expert
    weights live on the edp mesh as their own units (see
    ``_apply_expert_fsdp``), so the layer units only manage the
    non-expert parameters.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    # MindSpeed applies expert FSDP only when EP > 1
    # (``expert_parallel_size > 1`` gates both EP and eFSDP in their engine).
    if edp_mesh is not None and ep_enabled:
        _apply_expert_fsdp(model, edp_mesh, mp_policy)
    # First/last units keep their parameters unsharded through backward
    # (reshard_after_forward=False -> a single forward all-gather each),
    # matching the measured all-gather count under the MindSpeed plan.
    fully_shard(
        model.tok_embeddings,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False,
    )
    fully_shard(
        [model.norm, model.output],
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False,
    )
    for layer in model.layers.values():
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )
    # Root wrap owns no parameters (everything is managed above); FSDP
    # requires it for the multi-module [norm, output] unit to be valid.
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    disable_fsdp_gradient_division(model)
    logger.info("Applied Qwen3-Next FSDP (MindSpeed fsdp_modules plan)")


def parallelize_qwen3_next(
    model: Qwen3NextModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply EP, AC, and FSDP/eFSDP to the Qwen3-Next model."""
    del model_converters

    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Qwen3-Next tensor parallelism is not implemented yet; "
            "the fused in_proj_qkvz projection requires a grouped-shard "
            "boundary. Use TP=1 with FSDP/EP for now."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Qwen3-Next context parallelism is not implemented yet."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Qwen3-Next pipeline parallelism is not implemented yet."
        )

    ep_mesh = parallel_dims.get_optional_mesh("ep")
    if ep_mesh is not None:
        _apply_expert_parallel(model, ep_mesh)

    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "Qwen3-Next model compilation has not been validated; "
            "continuing without applying torch.compile"
        )

    if ac_config.mode != "none":
        apply_moe_ac(
            model,
            ac_config,
            model_compile_enabled=False,
            base_folder=dump_folder,
        )

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        _apply_fsdp(
            model,
            dp_mesh,
            edp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            ep_enabled=parallel_dims.ep_enabled,
        )
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
