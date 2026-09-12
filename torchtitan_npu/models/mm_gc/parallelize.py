# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for mm_gc.

Parallel layout (MODEL_CONTRACT.md §5):

- TP shards the attention head dimension (QKV colwise / output rowwise with
  all-reduce); the SLA2 core is per-head, so only ``n_heads`` must divide by
  the TP degree. Norms and the SLA router/alpha weights stay replicated.
- The MoE is sharded along its own head dimension ("head parallel"): on the
  TP mesh when EP is disabled, on the EP mesh when EP is enabled, and on the
  combined ["tp", "ep"] mesh for TP+EP. ``proj_in`` is colwise over
  ``heads * head_hidden``, the per-head router/expert bank is sharded over
  global head-major rows, and ``proj_out`` is rowwise with an all-reduce.
  Cross-rank communication per MoE layer is a single all-reduce proportional
  to N * hidden (head-parallel contract: independent of top-k).
- CP uses the conservative full-sequence strategy: the attention input is
  all-gathered along the sequence dimension and the output is sliced back,
  so SLA2's global block routing and linear branch see the whole sequence.
- FSDP/eFSDP shards every parameter on the DP mesh; the head-parallel expert
  parameters are additionally routed to the eFSDP mesh so they stay sharded
  at compute time.
- PP and ETP are not implemented and fail fast (onboarding guide order).
"""

import logging
from functools import partial

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.protocols import ModelConvertersContainer

from .attention import SLA2Attention
from .feed_forward import MultiHeadMoE
from .model import MMGcModel

logger = logging.getLogger(__name__)


def _replace_sharded(module: nn.Module, name: str, mesh: DeviceMesh, dim: int) -> None:
    param = getattr(module, name)
    placements = [Shard(dim)] * mesh.ndim
    sharded = nn.Parameter(
        distribute_tensor(param, mesh, placements), requires_grad=param.requires_grad
    )
    module.register_parameter(name, sharded)


def _apply_attention_tp(model: MMGcModel, tp_mesh: DeviceMesh) -> None:
    tp_size = tp_mesh.size()
    tp_group = tp_mesh.get_group()
    for layer in model.layers.values():
        attention = layer.attention
        if attention.n_heads % tp_size != 0:
            raise ValueError(
                f"n_heads={attention.n_heads} must be divisible by "
                f"TP degree={tp_size}"
            )
        for proj in (attention.to_q, attention.to_k, attention.to_v):
            _replace_sharded(proj, "weight", tp_mesh, 0)
        _replace_sharded(attention.to_out, "weight", tp_mesh, 1)
        attention.local_n_heads = attention.n_heads // tp_size
        attention.tp_group = tp_group
    logger.info("Applied mm_gc attention head-dim TP")


def _head_parallel_mesh(parallel_dims: ParallelDims) -> DeviceMesh | None:
    """Sharding mesh for head-parallel MoE params.

    EP-only shards along the EP mesh. With TP enabled, the framework's world
    mesh places TP and EP in separate sub-meshes, so the combined head
    sharding uses the ["ep", "etp"] mesh with ETP=TP (the torchtitan MoE+TP
    convention). TP-only shards along the TP mesh.
    """
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    if ep_mesh is not None:
        etp = parallel_dims.etp
        if etp != 1 and etp != parallel_dims.tp:
            raise NotImplementedError(
                "mm_gc only supports ETP=1 or ETP=TP for head-parallel MoE"
            )
        if etp == parallel_dims.tp and parallel_dims.tp > 1:
            return parallel_dims.get_mesh(["ep", "etp"])
        return ep_mesh
    if tp_mesh is not None:
        return tp_mesh
    return None


def _apply_moe_head_parallel(model: MMGcModel, mesh: DeviceMesh) -> int:
    mesh_size = mesh.size()
    if mesh.ndim == 1:
        head_groups = (mesh.get_group(),)
        flat_rank = mesh.get_local_rank()
    else:
        head_groups = tuple(
            mesh.get_group(mesh_dim=i) for i in range(mesh.ndim)
        )
        coordinate = mesh.get_coordinate()
        flat_rank = 0
        for i, dim_size in enumerate(mesh.shape):
            flat_rank = flat_rank * dim_size + coordinate[i]
    local_heads_of_first_moe = 0
    for layer in model.layers.values():
        moe = layer.moe
        if moe is None:
            continue
        if moe.moe_num_heads % mesh_size != 0:
            raise ValueError(
                f"moe_num_heads={moe.moe_num_heads} must be divisible by "
                f"head-parallel degree={mesh_size}"
            )
        local_heads = moe.moe_num_heads // mesh_size
        head_start = flat_rank * local_heads

        _replace_sharded(moe.proj_in, "weight", mesh, 0)
        _replace_sharded(moe.proj_out, "weight", mesh, 1)
        _replace_sharded(moe.gate, "router", mesh, 0)
        for name in ("w1", "w2", "w3"):
            _replace_sharded(moe.experts, name, mesh, 0)

        moe._head_groups = head_groups
        moe._local_num_heads = local_heads
        moe._local_head_start = head_start
        moe._local_num_experts = local_heads * moe.experts_per_head
        moe.gate.local_head_start = head_start
        moe.gate.local_num_heads = local_heads
        moe.experts.local_num_experts = local_heads * moe.experts_per_head
        if local_heads_of_first_moe == 0:
            local_heads_of_first_moe = local_heads
    logger.info(
        "Applied mm_gc MoE head parallel (degree=%d, local heads=%d)",
        mesh_size,
        local_heads_of_first_moe,
    )
    return mesh_size


def _apply_context_parallel(model: MMGcModel, cp_mesh: DeviceMesh) -> None:
    def _pre_hook(module, args, kwargs, mesh=cp_mesh):  # noqa: ANN001
        hidden_states = args[0]
        gathered = funcol.all_gather_tensor_autograd(
            hidden_states.contiguous(),
            gather_dim=1,
            group=mesh.get_group(),
        )
        if isinstance(gathered, funcol.AsyncCollectiveTensor):
            gathered = torch.ops._c10d_functional.wait_tensor(gathered)
        return (gathered, *args[1:]), kwargs

    def _post_hook(module, args, output, mesh=cp_mesh):  # noqa: ANN001
        if output.shape[1] % mesh.size() != 0:
            raise ValueError(
                f"mm_gc attention sequence length={output.shape[1]} must be "
                f"divisible by CP degree={mesh.size()}"
            )
        return output.chunk(mesh.size(), dim=1)[mesh.get_local_rank()].contiguous()

    for layer in model.layers.values():
        attention = layer.attention
        attention.register_forward_pre_hook(partial(_pre_hook, mesh=cp_mesh), with_kwargs=True)
        attention.register_forward_hook(
            partial(_post_hook, mesh=cp_mesh), prepend=True
        )
    logger.info("Applied mm_gc full-sequence all-gather context parallelism")


def _apply_mm_gc_fsdp(
    model: MMGcModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
) -> None:
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        ShardPlacementResult,
    )
    from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=False
    )

    fully_shard(
        model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward
    )
    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )

    for transformer_block in model.layers.values():
        if ep_degree > 1:
            head_parallel_params: set[nn.Parameter] = set()
            if transformer_block.moe is not None:
                head_parallel_params.update(
                    transformer_block.moe.experts.parameters()
                )
                head_parallel_params.add(transformer_block.moe.gate.router)
                head_parallel_params.add(transformer_block.moe.proj_in.weight)
                head_parallel_params.add(transformer_block.moe.proj_out.weight)

            assert edp_mesh is not None
            edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
            dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

            def _shard_placement_fn(
                param: nn.Parameter,
                _head_params: set = head_parallel_params,
                _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
            ) -> ShardPlacementResult:
                if param in _head_params:
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=_edp_mesh_info
                    )
                return ShardPlacementResult(placement=Shard(0), mesh_info=_dp_mesh_info)

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_shard_placement_fn,
            )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)


def parallelize_mm_gc(
    model: MMGcModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    del model_converters

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "mm_gc pipeline parallelism is deferred until FSDP/TP/EP/CP "
            "support is complete (onboarding guide order)"
        )
    if parallel_dims.etp_enabled and not parallel_dims.ep_enabled:
        raise NotImplementedError(
            "mm_gc standalone expert-tensor parallelism is not declared yet; "
            "ETP is only used as the TP axis of head-parallel MoE (EP+ETP=TP)"
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("mm_gc torch.compile is not supported yet")

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    cp_mesh = parallel_dims.get_optional_mesh("cp")

    if training.seq_len % (parallel_dims.cp or 1) != 0:
        raise ValueError(
            f"seq_len={training.seq_len} must be divisible by "
            f"CP degree={parallel_dims.cp}"
        )

    if tp_mesh is not None:
        _apply_attention_tp(model, tp_mesh)

    ep_degree = ep_mesh.size() if ep_mesh is not None else 1
    head_mesh = _head_parallel_mesh(parallel_dims)
    if head_mesh is not None:
        _apply_moe_head_parallel(model, head_mesh)

    if cp_mesh is not None:
        _apply_context_parallel(model, cp_mesh)

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
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
        _apply_mm_gc_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
        )
        logger.info("Applied mm_gc FSDP/eFSDP")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        logger.info("Applied DP replicate to the model")

    return model
