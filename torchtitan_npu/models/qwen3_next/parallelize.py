# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for Qwen3-Next.

Support matrix (v1):
- Tensor parallelism with sequence parallelism for the non-MoE path
  (attention / GDN projections + norms + embedding + lm_head)
- Expert parallelism for the routed MoE experts (routed experts stay
  replicated across the TP mesh; ETP is rejected)
- FSDP / eFSDP data parallel sharding (via upstream ``apply_fsdp``)
- DP replicate
- Full / selective activation checkpointing

CP / PP raise ``NotImplementedError`` (out of the declared v1 support
matrix).
"""

import logging
from typing import Any

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .gated_deltanet import GatedDeltaNet
from .model import Qwen3NextModel

logger = logging.getLogger(__name__)


def _check_tp_divisibility(model: Qwen3NextModel, tp_size: int) -> None:
    """The fused qkvz / q+gate layouts are contiguous per-head blocks, so a
    uniform dim-0 column shard aligns exactly iff every head count divides."""
    for layer in model.layers.values():
        mixer = layer.token_mixer
        if isinstance(mixer, GatedDeltaNet):
            for name in ("num_k_heads", "num_v_heads"):
                if getattr(mixer, name) % tp_size != 0:
                    raise ValueError(
                        f"GDN {name}={getattr(mixer, name)} must be divisible "
                        f"by TP degree={tp_size}"
                    )
        else:
            for name in ("n_heads", "n_kv_heads"):
                if getattr(mixer, name) % tp_size != 0:
                    raise ValueError(
                        f"Attention {name}={getattr(mixer, name)} must be "
                        f"divisible by TP degree={tp_size}"
                    )


def _apply_tensor_parallel(model: Qwen3NextModel, tp_mesh, tp_size: int) -> None:
    """Apply TP + sequence parallelism to the non-MoE path.

    Mirrors the llama3-style plan: the residual stream travels as
    ``Shard(1)`` (sequence parallel), the row-parallel projections
    reduce-scatter their partial outputs back to ``Shard(1)``, and the
    token mixers gather to ``Replicate`` before their column-parallel
    projections.
    The fused ``q_proj`` (q|gate) and GDN ``in_proj_qkvz`` / ``in_proj_ba``
    outputs are contiguous per-head blocks, so a uniform dim-0 column
    shard keeps whole heads per rank; the per-rank head attributes are
    updated accordingly. The depthwise ``conv1d`` / ``dt_bias`` / ``A_log``
    are sharded manually along their head/channel dim. The routed experts
    stay replicated across the TP mesh (pure EP); the router gate is
    replicated. The lm_head is column-parallel with a replicated output
    (all-gather modeled), matching the MindSpeed vocab-parallel head
    semantics; the loss stays on the full-vocab local logits.
    """
    from torch.distributed.device_mesh import DeviceMesh

    assert isinstance(tp_mesh, DeviceMesh)
    _check_tp_divisibility(model, tp_size)

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            # The lm_head stays unwrapped: MindSpeed vocab-parallel head has
            # no measured tp>1 baseline, and the chunked loss needs
            # full-vocab logits. ``_lm_head_forward`` replicate-wraps the
            # weight for DTensor inputs (documented in the model contract).
        },
    )

    for layer in model.layers.values():
        mixer = layer.token_mixer
        layer_plan: dict[str, Any] = {
            "attention_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
            "token_mixer": (
                PrepareModuleInput(
                    input_layouts=(Shard(1), None, None),
                    desired_input_layouts=(Replicate(), None, None),
                )
                if not isinstance(mixer, GatedDeltaNet)
                else PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                )
            ),
        }
        if isinstance(mixer, GatedDeltaNet):
            layer_plan.update(
                {
                    "token_mixer.in_proj_qkvz": ColwiseParallel(),
                    "token_mixer.in_proj_ba": ColwiseParallel(),
                    "token_mixer.out_proj": RowwiseParallel(
                        output_layouts=Shard(1)
                    ),
                }
            )
        else:
            layer_plan.update(
                {
                    "token_mixer.q_proj": ColwiseParallel(),
                    "token_mixer.k_proj": ColwiseParallel(),
                    "token_mixer.v_proj": ColwiseParallel(),
                    "token_mixer.o_proj": RowwiseParallel(
                        output_layouts=Shard(1)
                    ),
                }
            )
        parallelize_module(layer, tp_mesh, layer_plan)

        # Per-rank head counts: the projections produce local heads, and the
        # module views read these attributes for the local shapes.
        if isinstance(mixer, GatedDeltaNet):
            mixer.num_k_heads //= tp_size
            mixer.num_v_heads //= tp_size
            mixer.key_dim = mixer.head_k_dim * mixer.num_k_heads
            mixer.value_dim = mixer.head_v_dim * mixer.num_v_heads
            mixer.conv1d.weight.data = mixer.conv1d.weight.data.chunk(tp_size, dim=0)[
                tp_mesh.get_local_rank()
            ]
            mixer.conv1d.in_channels = mixer.conv1d.out_channels = (
                mixer.conv1d.weight.data.shape[0]
            )
            mixer.conv1d.groups = mixer.conv1d.weight.data.shape[0]
            mixer.dt_bias.data = mixer.dt_bias.data.chunk(tp_size, dim=0)[
                tp_mesh.get_local_rank()
            ]
            mixer.A_log.data = mixer.A_log.data.chunk(tp_size, dim=0)[
                tp_mesh.get_local_rank()
            ]
        else:
            mixer.n_heads //= tp_size
            mixer.n_kv_heads //= tp_size

    logger.info("Applied Qwen3-Next TP + sequence parallelism")


def _apply_expert_parallel(model: Qwen3NextModel, ep_mesh) -> None:
    """Mirror MindSpeed ``expert_parallelize_modules`` (eager dispatcher).

    Routed-expert weights are split along dim 0 over the EP mesh
    (``distribute_tensor`` Shard(0) semantics: contiguous rows = whole
    experts per rank) and the experts forward is replaced with the eager
    dispatcher stream (count all-gather + a2a-v dispatch/combine +
    per-expert matmul loop).
    """
    from types import MethodType

    from torchtitan_npu.models.qwen3_next.feed_forward import _experts_ep_forward

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)
    for layer in model.layers.values():
        if not layer.moe_enabled:
            continue
        experts = layer.moe.experts
        num_global = experts.num_experts
        if num_global % ep_size != 0:
            raise ValueError(
                f"num_experts={num_global} must be divisible by "
                f"EP degree={ep_size}"
            )
        num_local = num_global // ep_size
        experts.num_global_experts = num_global
        experts.ep_group = ep_group
        experts.expert_ids_per_ep_rank = torch.tensor(
            [i % num_local for i in range(num_global)],
            dtype=torch.int32,
            device=experts.gate_up_proj.device,
        )
        experts.gate_up_proj.data = experts.gate_up_proj.data.chunk(ep_size, dim=0)[
            ep_rank
        ]
        experts.down_proj.data = experts.down_proj.data.chunk(ep_size, dim=0)[ep_rank]
        experts.forward = MethodType(_experts_ep_forward, experts)
        experts._ep_dispatcher = True
    logger.info("Applied Qwen3-Next expert parallelism (eager dispatcher)")


def _apply_expert_fsdp(model: Qwen3NextModel, edp_mesh, param_dtype) -> None:
    """Mirror MindSpeed ``expert_fully_shard_modules``.

    Each routed-experts module becomes its own FSDP unit over the
    EP-internal data-parallel mesh (``ep_fsdp_modules``), sharded on dim 1
    (``efsdp_shard_placement_fn`` default Shard(1)). MindSpeed passes no
    ``reshard_after_forward`` there, so torch's default True applies:
    every experts unit re-unshards once before backward. The eFSDP
    precision policy is hardcoded in MindSpeed to param bf16 / reduce
    fp32 (independent of the main FSDP ``reduce_dtype``).
    """
    efsdp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=torch.float32
    )
    for layer in model.layers.values():
        if not layer.moe_enabled:
            continue
        fully_shard(
            layer.moe.experts,
            mesh=edp_mesh,
            mp_policy=efsdp_policy,
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
    all share ``parallel_args.reshard_after_forward`` whose default is
    True: every unit (embedding, lm_head, decoder layers) re-unshards
    once before backward. The final norm is in none of the groups and
    stays unwrapped (no all-gather of its own). Routed-expert weights
    live on the edp mesh as their own units (see ``_apply_expert_fsdp``),
    so the layer units only manage the non-expert parameters.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    # MindSpeed applies expert FSDP only when EP > 1
    # (``expert_parallel_size > 1`` gates both EP and eFSDP in their engine).
    if edp_mesh is not None and ep_enabled:
        _apply_expert_fsdp(model, edp_mesh, param_dtype)
    fully_shard(
        model.tok_embeddings,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    fully_shard(
        model.output,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    for layer in model.layers.values():
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )
    # Root wrap owns no parameters (everything is managed above); FSDP
    # requires it for the child-unit layout to be valid.
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
    """Apply TP, EP, AC, and FSDP/eFSDP to the Qwen3-Next model."""
    del model_converters

    if parallel_dims.etp_enabled:
        raise NotImplementedError(
            "Qwen3-Next expert tensor parallelism is not in the v1 support "
            "matrix: routed experts are replicated across the TP mesh and "
            "sharded only along the EP dim (use EP, optionally combined "
            "with TP)."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Qwen3-Next context parallelism is not implemented yet."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Qwen3-Next pipeline parallelism is not implemented yet."
        )

    # Without a data-parallel wrapper (FSDP/replicate) there is no
    # MixedPrecisionPolicy to cast the meta-built fp32 parameters to the
    # deployment param dtype; the NPU op contracts require bf16. Mirror
    # the deployment dtype directly (e.g. TP-only, TP+EP, or
    # single-rank no-sharding layouts).
    if not (
        parallel_dims.fsdp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.dp_replicate_enabled
    ):
        model.to(TORCH_DTYPE_MAP[training.mixed_precision_param])

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    if tp_mesh is not None:
        _apply_tensor_parallel(model, tp_mesh, tp_mesh.size())

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
