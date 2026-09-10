# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for glm5_next.

Order follows the onboarding guide: non-MoE TP -> CP -> MoE EP/TP -> AC ->
FSDP/eFSDP -> DP replicate. Contract (MODEL_CONTRACT.md section 10):

- TP shards KDA/DSA attention along the head dim; the fused qkv short conv
  rebuilds its local weight slices per rank; the DSA indexer stays replicated
  so top-k selections are identical across TP ranks; mHC runs on sequence
  shards with replicated fp32 parameters.
- EP shards routed experts along the expert dim; without EP, experts shard
  along the intermediate dim (single all-reduce inside the expert).
- CP reuses the conservative all-gather strategy (kimi_k3/ar_llm) and only
  supports no-TP combinations. The vision tower follows FSDP only.
- PP/ETP fail fast with pointers to MODEL_CONTRACT.md.
"""

import logging
from functools import partial

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
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
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.distributed.tensor_parallel import NoParallel, maybe_enable_async_tp
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .attention import GlmDeltaAttention, GlmDsaAttention
from .feed_forward import GlmGroupedExperts, GlmMLP
from .model import Glm5NextModel

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_NAMES = ("w1", "w2", "w3")


class _GlmExpertParallel(ExpertParallel):
    """EP plan for the standard w1/w2/w3 grouped expert layout."""

    def _partition_fn(self, name, module, device_mesh):  # noqa: ANN001
        for weight_name in _EXPERT_WEIGHT_NAMES:
            weight = getattr(module, weight_name)
            module.register_parameter(
                weight_name, nn.Parameter(distribute_tensor(weight, device_mesh, [Shard(0)]))
            )


class _GlmExpertInterTP:
    """TP-only expert plan: shard w1/w3 along inter (dim 1), w2 along inter (dim 2)."""

    @staticmethod
    def _partition(name, module, mesh):  # noqa: ANN001
        if module.w1.shape[1] % mesh.size() != 0:
            raise ValueError(
                f"moe_intermediate_size={module.w1.shape[1]} must be divisible by TP={mesh.size()}"
            )
        module.w1 = nn.Parameter(distribute_tensor(module.w1, mesh, [Shard(1)]))
        module.w3 = nn.Parameter(distribute_tensor(module.w3, mesh, [Shard(1)]))
        module.w2 = nn.Parameter(distribute_tensor(module.w2, mesh, [Shard(2)]))


def _replicated_local_output():
    return NoParallel(local_output_grad_placements=(Partial(),))


def _apply_inter_shard_mlp_tp(mlp, tp_mesh: DeviceMesh) -> None:  # noqa: ANN001
    """Shard a dense MLP along the intermediate dim with an in-expert all-reduce.

    Used for the MoE shared expert, whose input is already a gathered full
    sequence inside the MoE's local computation; ``GlmMLP`` performs the
    all-reduce when ``_tp_group`` is set (ar_llm LatentExpertMLP pattern,
    keeps Partial DTensors out of the MoE's plain-tensor computation).
    """
    inter = mlp.gate_proj.weight.shape[0]
    if inter % tp_mesh.size() != 0:
        raise ValueError(
            f"moe_intermediate_size={inter} must be divisible by TP degree={tp_mesh.size()}"
        )
    for module, placements in (
        (mlp.gate_proj, [Shard(0)]),
        (mlp.up_proj, [Shard(0)]),
        (mlp.down_proj, [Shard(1)]),
    ):
        weight = module.weight
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(weight, tp_mesh, placements), requires_grad=weight.requires_grad
            ),
        )
    mlp._tp_group = tp_mesh.get_group()


def _apply_kda_tp(attention: GlmDeltaAttention, tp_mesh: DeviceMesh) -> None:
    tp_degree = tp_mesh.size()
    kda = attention
    if kda.num_heads % tp_degree != 0:
        raise ValueError(
            f"KDA num_heads={kda.num_heads} must be divisible by TP degree={tp_degree}"
        )
    parallelize_module(
        kda,
        tp_mesh,
        {
            "q_proj": ColwiseParallel(use_local_output=True),
            "k_proj": ColwiseParallel(use_local_output=True),
            "v_proj": ColwiseParallel(use_local_output=True),
            "b_proj": ColwiseParallel(use_local_output=True),
            # Low-rank gate bottlenecks keep the full head_dim (shared across
            # heads); only the up projections shard along heads.
            "forget_gate.f_a_proj": _replicated_local_output(),
            "forget_gate.f_b_proj": ColwiseParallel(use_local_output=True),
            "g_a_proj": _replicated_local_output(),
            "g_b_proj": ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
        },
    )
    local_heads = kda.num_heads // tp_degree
    local_qkv = local_heads * kda.head_dim
    local_rank = tp_mesh.get_local_rank()
    kda.conv1d.set_local_slice(
        channel_starts=(
            local_rank * local_qkv,
            kda.qkv_dim + local_rank * local_qkv,
            2 * kda.qkv_dim + local_rank * local_qkv,
        ),
        local_channels=local_qkv,
    )
    for module, name in (
        (kda.forget_gate, "A_log"),
        (kda.forget_gate, "dt_bias"),
    ):
        param = getattr(module, name)
        if param.shape[0] % tp_degree != 0:
            raise ValueError(
                f"forget_gate.{name} size={param.shape[0]} must be divisible by TP degree={tp_degree}"
            )
        module.register_parameter(
            name,
            nn.Parameter(distribute_tensor(param, tp_mesh, [Shard(0)], src_data_rank=0)),
        )
    kda.num_heads = local_heads
    kda.qkv_dim = local_qkv
    kda.forget_gate.num_heads = local_heads


def _apply_dsa_tp(attention: GlmDsaAttention, tp_mesh: DeviceMesh) -> None:
    tp_degree = tp_mesh.size()
    if attention.num_heads % tp_degree != 0:
        raise ValueError(
            f"DSA num_heads={attention.num_heads} must be divisible by TP degree={tp_degree}"
        )
    parallelize_module(
        attention,
        tp_mesh,
        {
            "q_a_proj": _replicated_local_output(),
            "q_a_norm": _replicated_local_output(),
            "q_b_proj": ColwiseParallel(use_local_output=True),
            "kv_a_proj_with_mqa": _replicated_local_output(),
            "kv_a_norm": _replicated_local_output(),
            "kv_b_proj": ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
        },
    )
    # The indexer scores full-sequence, all-head candidates; keeping it
    # replicated guarantees identical top-k selections on every TP rank
    # (same rationale as ar_llm HCA).
    parallelize_module(
        attention.indexer,
        tp_mesh,
        {
            "wq_b": _replicated_local_output(),
            "wk": _replicated_local_output(),
            "k_norm": _replicated_local_output(),
            "weights_proj": _replicated_local_output(),
        },
    )
    attention.num_heads //= tp_degree


def _apply_non_moe_tp(
    model: Glm5NextModel,
    tp_mesh: DeviceMesh,
    *,
    loss_parallel: bool,
) -> None:
    sequence_shard = Shard(1)
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sequence_shard,
                use_local_output=True,
            ),
            "norm": SequenceParallel(use_local_output=True),
            "output": ColwiseParallel(
                input_layouts=sequence_shard,
                output_layouts=(Shard(-1) if loss_parallel else Replicate()),
                use_local_output=not loss_parallel,
            ),
        },
    )

    all_blocks = list(model.layers.values())
    for block in all_blocks:
        parallelize_module(
            block,
            tp_mesh,
            {
                "input_layernorm": SequenceParallel(use_local_output=True),
                "post_attention_layernorm": SequenceParallel(use_local_output=True),
                "attention": PrepareModuleInput(
                    input_layouts=(sequence_shard, None, None),
                    desired_input_layouts=(Replicate(), None, None),
                    use_local_output=True,
                ),
            },
        )
        if isinstance(block.attention, GlmDeltaAttention):
            _apply_kda_tp(block.attention, tp_mesh)
        else:
            _apply_dsa_tp(block.attention, tp_mesh)
        if block.moe is not None:
            parallelize_module(
                block.moe,
                tp_mesh,
                PrepareModuleInputOutput(
                    input_layouts=(sequence_shard,),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(sequence_shard,),
                    use_local_output=True,
                ),
            )
            if block.moe.shared_experts is not None:
                # Shared experts run inside the MoE on the gathered (full)
                # sequence: shard inter dim manually and all-reduce inside the
                # expert (ar_llm LatentExpertMLP pattern, keeps Partial flow
                # out of the MoE's local computation).
                _apply_inter_shard_mlp_tp(block.moe.shared_experts, tp_mesh)
        else:
            assert block.mlp is not None
            if block.mlp.gate_proj.weight.shape[0] % tp_mesh.size() != 0:
                raise ValueError(
                    f"intermediate_size={block.mlp.gate_proj.weight.shape[0]} must be "
                    f"divisible by TP degree={tp_mesh.size()}"
                )
            # llama3 dense-MLP pattern: Partial flows from the colwise
            # projections into the rowwise down projection, which
            # reduce-scatters back to the sequence shard.
            parallelize_module(
                block.mlp,
                tp_mesh,
                {
                    "gate_proj": ColwiseParallel(input_layouts=sequence_shard),
                    "up_proj": ColwiseParallel(input_layouts=sequence_shard),
                    "down_proj": RowwiseParallel(output_layouts=sequence_shard, use_local_output=True),
                },
            )

    logger.info("Applied glm5_next tensor and sequence parallelism")


def _apply_moe_parallel(
    model: Glm5NextModel,
    *,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
) -> None:
    if tp_mesh is None and ep_mesh is None:
        return
    all_blocks = list(model.layers.values())
    for block in all_blocks:
        if block.moe is None:
            continue
        moe = block.moe
        if ep_mesh is not None:
            if moe.num_experts % ep_mesh.size() != 0:
                raise ValueError(
                    f"n_routed_experts={moe.num_experts} must be divisible by "
                    f"EP degree={ep_mesh.size()}"
                )
            parallelize_module(moe.experts, ep_mesh, _GlmExpertParallel())
        elif tp_mesh is not None:
            _GlmExpertInterTP._partition("experts", moe.experts, tp_mesh)
            for weight_name in _EXPERT_WEIGHT_NAMES:
                if not isinstance(getattr(moe.experts, weight_name), DTensor):
                    raise RuntimeError(
                        f"glm5_next expert weight {weight_name} was not distributed to a "
                        "DTensor; the expert tensor parallel plan did not apply"
                    )
        if tp_mesh is not None and moe.shared_experts is not None:
            parallelize_module(
                moe.shared_experts,
                tp_mesh,
                {
                    "gate_proj": ColwiseParallel(use_local_output=True),
                    "up_proj": ColwiseParallel(use_local_output=True),
                    "down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
                },
            )

    logger.info("Applied glm5_next expert parallelism")


class _GlmAttentionContextParallel:
    """All-gather the full sequence before attention, re-shard afterwards.

    KDA accumulates delta-rule state over the whole sequence and the DSA
    indexer top-k needs cross-rank context (conservative kimi_k3/ar_llm CP
    strategy). NoPE means no rope offset handling is required.
    """

    @staticmethod
    def _pre_hook(module, args, kwargs, mesh):  # noqa: ANN001
        if mesh.ndim != 1:
            raise ValueError(f"glm5_next CP expects a 1D mesh, got {mesh.ndim}D")
        hidden_states = args[0]
        gathered = funcol.all_gather_tensor_autograd(
            hidden_states.contiguous(),
            gather_dim=0,
            group=mesh.get_group(),
        )
        hidden_states = torch.cat(torch.chunk(gathered, mesh.size(), dim=0), dim=1)
        return (hidden_states, *args[1:]), kwargs

    @staticmethod
    def _post_hook(module, args, output, mesh):  # noqa: ANN001
        if output.shape[1] % mesh.size() != 0:
            raise ValueError(
                f"glm5_next attention sequence length={output.shape[1]} must be "
                f"divisible by CP degree={mesh.size()}"
            )
        return output.chunk(mesh.size(), dim=1)[mesh.get_local_rank()].contiguous()

    @classmethod
    def apply(cls, module, device_mesh: DeviceMesh):
        module.register_forward_pre_hook(
            partial(cls._pre_hook, mesh=device_mesh),
            with_kwargs=True,
        )
        module.register_forward_hook(
            partial(cls._post_hook, mesh=device_mesh),
            prepend=True,
        )
        return module


def _apply_context_parallel(
    model: Glm5NextModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
) -> None:
    if not parallel_dims.cp_enabled:
        return
    if parallelism.context_parallel_load_balancer is not None:
        raise ValueError(
            "glm5_next context parallelism requires contiguous sequence shards; "
            "set context_parallel_load_balancer=None"
        )
    cp_mesh = parallel_dims.get_mesh("cp")
    if cp_mesh.ndim != 1:
        raise ValueError(f"glm5_next CP expects a 1D mesh, got {cp_mesh.ndim}D")
    for block in model.layers.values():
        _GlmAttentionContextParallel.apply(block.attention, cp_mesh)

    logger.info("Applied glm5_next all-gather context parallelism")


def parallelize_glm5_next(
    model: Glm5NextModel,
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

    if training.seq_len % parallel_dims.seq_len_divisor != 0:
        raise ValueError(
            f"seq_len={training.seq_len} must be divisible by "
            f"seq_len_divisor={parallel_dims.seq_len_divisor}"
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "glm5_next pipeline parallelism is deferred until the loop-region and "
            "hc_head cross-stage contract is defined; see "
            "torchtitan_npu/models/glm5_next/MODEL_CONTRACT.md"
        )
    if parallel_dims.etp_enabled:
        raise NotImplementedError("glm5_next EP+ETP is not supported yet; see MODEL_CONTRACT.md")
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "glm5_next model compilation has not been validated; continuing without "
            "applying torch.compile"
        )

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    ep_mesh = parallel_dims.get_optional_mesh("ep")

    if parallel_dims.cp_enabled and tp_mesh is not None:
        raise NotImplementedError(
            "glm5_next supports the all-gather CP strategy only without tensor "
            "parallelism for now; see MODEL_CONTRACT.md"
        )

    if tp_mesh is not None:
        _apply_non_moe_tp(model, tp_mesh, loss_parallel=not parallelism.disable_loss_parallel)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    _apply_context_parallel(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
    )

    _apply_moe_parallel(model, tp_mesh=tp_mesh, ep_mesh=ep_mesh)

    if ac_config.mode != "none":
        apply_moe_ac(
            model,
            ac_config,
            model_compile_enabled=False,
            base_folder=dump_folder,
        )

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        edp_mesh_names = (
            ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )
        logger.info("Applied glm5_next FSDP/eFSDP")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
