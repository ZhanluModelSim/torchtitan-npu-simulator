# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for ar_llm.

Order follows the onboarding guide: non-MoE TP -> CP -> MoE EP/TP -> AC ->
FSDP/eFSDP -> DP replicate. CP uses kimi_k3's conservative all-gather strategy
(full sequence before attention/Engram) and is only supported without TP for
now. PP/ETP fail fast with pointers to the missing contracts in
MODEL_CONTRACT.md instead of silently degrading.
"""

import logging
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
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
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel
from torchtitan.distributed.tensor_parallel import NoParallel, maybe_enable_async_tp
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .attention import HeavilyCompressedAttention
from .feed_forward import LatentExpertMLP, LatentGroupedExperts
from .model import ArLlmModel

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_NAMES = ("w1", "w2", "w3", "w4", "w5")


def _apply_ar_llm_fsdp(
    model: ArLlmModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
) -> None:
    """FSDP/eFSDP application for ar_llm.

    Mirrors upstream ``apply_fsdp`` (llama4) with one extension required by
    the Engram table deployment (MODEL_CONTRACT.md §6): Engram hash-table
    parameters are routed to the edp/eFSDP mesh like routed-expert parameters
    so the tables stay sharded at compute time. All other parameters follow
    the standard dp-mesh Shard(0) policy.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        ShardPlacementResult,
    )
    from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
    from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=False
    )

    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward)
    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )

    for transformer_block in model.layers.values():
        expert_params = set(transformer_block.moe.experts.parameters())
        engram_table_params: set[nn.Parameter] = set()
        if transformer_block.has_engram:
            for table in transformer_block.engram.hash_tables:
                engram_table_params.update(table.parameters())

        if ep_degree > 1:
            assert edp_mesh is not None
            efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            expert_shard_placement = (
                Shard(1) if efsdp_ep_size > transformer_block.moe.experts.num_experts else Shard(0)
            )
            edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
            dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

            def _shard_placement_fn(
                param: nn.Parameter,
                _expert_params: set = expert_params,
                _engram_params: set = engram_table_params,
                _expert_placement: Shard = expert_shard_placement,
                _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
            ) -> ShardPlacementResult:
                if param in _engram_params:
                    # Hash tables are bucket-sharded and stay sharded at
                    # compute time (a2a lookup addresses the owning rank).
                    return ShardPlacementResult(placement=Shard(0), mesh_info=_edp_mesh_info)
                if param in _expert_params:
                    return ShardPlacementResult(placement=_expert_placement, mesh_info=_edp_mesh_info)
                return ShardPlacementResult(placement=Shard(0), mesh_info=_dp_mesh_info)

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_shard_placement_fn,
            )
        elif fsdp_config["mesh"].size() > transformer_block.moe.experts.num_experts:
            def _experts_shard_placement_fn(
                param: nn.Parameter,
                _expert_params: set = expert_params,
            ) -> Shard | None:
                if param in _expert_params:
                    return Shard(1)
                return None

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
    del gradient_divide_factor

    # Explicit prefetching under EP (mirrors upstream; D2H syncs in EP can
    # interfere with FSDP's implicit prefetching).
    if ep_degree == 1:
        return

    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])
    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        else:
            transformer_block.set_modules_to_forward_prefetch([model.norm, model.output])

    reversed_transformer_blocks = list(reversed(list(model.layers.values())))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])
    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        else:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def _replicated_local_output() -> ParallelStyle:
    return NoParallel(local_output_grad_placements=(Partial(),))


class _GroupedOProjTP(ParallelStyle):
    """Shard the grouped O projection along its group (= head-group) dim.

    Each group produces a disjoint slice of the output features, so the TP
    boundary all-gathers the feature dim and re-slices the sequence dim instead
    of all-reducing.
    """

    def _apply(self, module, device_mesh):  # noqa: ANN001
        def partition(name, mod, mesh):  # noqa: ANN001
            tp_degree = mesh.size()
            if mod.o_groups % tp_degree != 0:
                raise ValueError(
                    f"o_groups={mod.o_groups} must be divisible by TP degree={tp_degree}"
                )
            mod.o_down = nn.Parameter(distribute_tensor(mod.o_down, mesh, [Shard(0)]))
            mod.o_up = nn.Parameter(distribute_tensor(mod.o_up, mesh, [Shard(0)]))
            mod.o_groups //= tp_degree
            mod.hidden_size //= tp_degree

        distributed = distribute_module(module, device_mesh, partition_fn=partition)
        distributed.register_forward_hook(partial(self._post_hook, mesh=device_mesh))
        return distributed

    @staticmethod
    def _post_hook(module, args, output, mesh):  # noqa: ANN001
        gathered = funcol.all_gather_tensor_autograd(
            output.contiguous(), gather_dim=-1, group=mesh.get_group()
        )
        local_sequence = gathered.chunk(mesh.size(), dim=1)[mesh.get_local_rank()]
        return local_sequence.contiguous()


class _LatentInterTP(ParallelStyle):
    """Shard latent-expert weights along the intermediate (inter) dim.

    ``latent_to_inter`` becomes colwise and ``inter_to_latent`` rowwise, so the
    SwiGLU runs on an exact local inter slice and a single all-reduce inside
    the expert restores the full latent activations.
    """

    def _apply(self, module, device_mesh):  # noqa: ANN001
        def partition(name, mod, mesh):  # noqa: ANN001
            if isinstance(mod, LatentGroupedExperts):
                mod.w4 = nn.Parameter(distribute_tensor(mod.w4, mesh, [Shard(1)]))
                mod.w5 = nn.Parameter(distribute_tensor(mod.w5, mesh, [Shard(2)]))
            elif isinstance(mod, LatentExpertMLP):
                mod.latent_to_inter = nn.Parameter(
                    distribute_tensor(mod.latent_to_inter, mesh, [Shard(1)])
                )
                mod.inter_to_latent = nn.Parameter(
                    distribute_tensor(mod.inter_to_latent, mesh, [Shard(2)])
                )
            else:
                raise TypeError(f"Unsupported latent-expert module: {type(mod).__name__}")
            mod._tp_group = mesh.get_group()

        return distribute_module(module, device_mesh, partition_fn=partition)


class _ArLlmExpertParallel(ExpertParallel):
    """EP plan that also shards the extra latent-expert weights (w4/w5)."""

    def _partition_fn(self, name, module, device_mesh):  # noqa: ANN001
        for weight_name in _EXPERT_WEIGHT_NAMES:
            weight = getattr(module, weight_name)
            module.register_parameter(
                weight_name, nn.Parameter(distribute_tensor(weight, device_mesh, [Shard(0)]))
            )


class _ArLlmExpertTensorParallel(TensorParallel):
    """TP-only expert plan: shard the expert intermediate dim."""

    def _apply(self, module, device_mesh):  # noqa: ANN001
        def partition(name, mod, mesh):  # noqa: ANN001
            mod.w4 = nn.Parameter(distribute_tensor(mod.w4, mesh, [Shard(1)]))
            mod.w5 = nn.Parameter(distribute_tensor(mod.w5, mesh, [Shard(2)]))
            mod._tp_group = mesh.get_group()

        return distribute_module(module, device_mesh, partition_fn=partition)


class _ArLlmAttentionContextParallel(ParallelStyle):
    """All-gather the full sequence before attention, re-shard afterwards.

    Same conservative CP strategy as kimi_k3: KDA accumulates delta-rule state
    over the whole sequence, CSA windows and HCA indexer top-k need cross-rank
    context, and RoPE positions are global. RoPE buffers are re-sliced to the
    gathered length inside the pre-hook (the model slices them to the local
    sequence length before CP).
    """

    @staticmethod
    def _pre_hook(module, args, kwargs, mesh):  # noqa: ANN001
        if mesh.ndim != 1:
            raise ValueError(f"ar_llm CP expects a 1D mesh, got {mesh.ndim}D")
        hidden_states = args[0]
        gathered = funcol.all_gather_tensor_autograd(
            hidden_states.contiguous(),
            gather_dim=0,
            group=mesh.get_group(),
        )
        hidden_states = torch.cat(torch.chunk(gathered, mesh.size(), dim=0), dim=1)
        seq_len = hidden_states.shape[1]
        rope_cos = args[1][:seq_len] if args[1] is not None else None
        rope_sin = args[2][:seq_len] if args[2] is not None else None
        return (hidden_states, rope_cos, rope_sin), kwargs

    @staticmethod
    def _post_hook(module, args, output, mesh):  # noqa: ANN001
        if output.shape[1] % mesh.size() != 0:
            raise ValueError(
                f"ar_llm attention sequence length={output.shape[1]} must be "
                f"divisible by CP degree={mesh.size()}"
            )
        return output.chunk(mesh.size(), dim=1)[mesh.get_local_rank()].contiguous()

    def _apply(self, module, device_mesh):  # noqa: ANN001
        module.register_forward_pre_hook(
            partial(self._pre_hook, mesh=device_mesh),
            with_kwargs=True,
        )
        module.register_forward_hook(
            partial(self._post_hook, mesh=device_mesh),
            prepend=True,
        )
        return module


class _ArLlmEngramIdsContextParallel(ParallelStyle):
    """All-gather the token ids for the Engram module under CP.

    The hash windows need cross-rank token context, but ids are int64 and
    negligible in volume. Hidden states are NOT gathered: hash indices are
    sliced to the local sequence window and gate/conv run on the local hidden
    states (see the Engram deployment contract in MODEL_CONTRACT.md §6).
    """

    @staticmethod
    def _pre_hook(module, args, kwargs, mesh):  # noqa: ANN001
        if mesh.ndim != 1:
            raise ValueError(f"ar_llm CP expects a 1D mesh, got {mesh.ndim}D")
        hidden_states, input_ids = args[0], args[1]
        gathered = funcol.all_gather_tensor_autograd(
            input_ids.contiguous(),
            gather_dim=0,
            group=mesh.get_group(),
        )
        input_ids = torch.cat(torch.chunk(gathered, mesh.size(), dim=0), dim=1)
        return (hidden_states, input_ids), kwargs

    def _apply(self, module, device_mesh):  # noqa: ANN001
        module.register_forward_pre_hook(
            partial(self._pre_hook, mesh=device_mesh),
            with_kwargs=True,
        )
        return module


def _apply_engram_table_parallel(
    model: ArLlmModel,
    mesh: DeviceMesh | None,
    *,
    force_balance: bool,
) -> None:
    """Shard Engram hash tables by hash bucket across the EP/TP mesh.

    Owner = ``bucket // E_local`` (contiguous Shard(0)); the forward dispatches
    keys to owners via All-to-All and autograd returns row gradients via the
    reverse All-to-All (Engram paper's training deployment). Tables stay
    sharded at compute time, mirroring expert-parameter treatment.
    """
    if mesh is None or mesh.size() <= 1:
        return
    for layer in model.layers.values():
        if not layer.has_engram:
            continue
        for table in layer.engram.hash_tables:
            total = table.embedding.weight.shape[0]
            if total % mesh.size() != 0:
                raise ValueError(
                    f"Engram table size ({total}) must be divisible by the "
                    f"sharding world size ({mesh.size()})"
                )
            table.embedding.weight = nn.Parameter(
                distribute_tensor(table.embedding.weight, mesh, [Shard(0)])
            )
            table._table_group = mesh.get_group()
            table._table_world_size = mesh.size()
            table._table_rank = dist.get_rank(mesh.get_group())
            table._table_local_size = total // mesh.size()
            table._table_local_offset = table._table_rank * table._table_local_size
            table._force_balance = force_balance

    logger.info("Applied ar_llm Engram table sharding (world=%d)", mesh.size())


def _apply_context_parallel(
    model: ArLlmModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    training_seq_len: int,
) -> None:
    if not parallel_dims.cp_enabled:
        return
    if parallelism.context_parallel_load_balancer is not None:
        raise ValueError(
            "ar_llm context parallelism requires contiguous sequence shards; "
            "set context_parallel_load_balancer=None"
        )
    cp_mesh = parallel_dims.get_mesh("cp")
    if cp_mesh.ndim != 1:
        raise ValueError(f"ar_llm CP expects a 1D mesh, got {cp_mesh.ndim}D")
    cp_degree = cp_mesh.size()
    if training_seq_len % cp_degree != 0:
        raise ValueError(
            f"seq_len={training_seq_len} must be divisible by CP degree={cp_degree}"
        )
    for layer in model.layers.values():
        parallelize_module(
            layer.attention,
            cp_mesh,
            _ArLlmAttentionContextParallel(),
        )
        if layer.has_engram:
            cp_rank = cp_mesh.get_local_rank()
            local_seq = training_seq_len // cp_degree
            layer.engram._ids_window = (cp_rank * local_seq, (cp_rank + 1) * local_seq)
            parallelize_module(
                layer.engram,
                cp_mesh,
                _ArLlmEngramIdsContextParallel(),
            )

    logger.info("Applied ar_llm all-gather context parallelism")


def _apply_kda_tp(attention, tp_mesh: DeviceMesh) -> None:  # noqa: ANN001
    kda = attention.kda
    tp_degree = tp_mesh.size()
    if kda.n_heads % tp_degree != 0:
        raise ValueError(
            f"KDA num_heads={kda.n_heads} must be divisible by TP degree={tp_degree}"
        )
    parallelize_module(
        kda,
        tp_mesh,
        {
            "q_proj": ColwiseParallel(use_local_output=True),
            "k_proj": ColwiseParallel(use_local_output=True),
            "v_proj": ColwiseParallel(use_local_output=True),
            "alpha_gate": ColwiseParallel(use_local_output=True),
            "erase_gate": ColwiseParallel(use_local_output=True),
            "write_gate": ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
        },
    )
    kda.n_heads //= tp_degree
    attention.n_heads //= tp_degree


def _apply_mla_tp(attention, tp_mesh: DeviceMesh) -> None:  # noqa: ANN001
    tp_degree = tp_mesh.size()
    if attention.n_heads % tp_degree != 0:
        raise ValueError(
            f"attention num_heads={attention.n_heads} must be divisible by TP degree={tp_degree}"
        )
    parallelize_module(
        attention,
        tp_mesh,
        {
            "q_proj.q_down": _replicated_local_output(),
            "q_proj.q_norm": _replicated_local_output(),
            "q_proj.q_up_nope": ColwiseParallel(use_local_output=True),
            "q_proj.q_up_rope": ColwiseParallel(use_local_output=True),
            "q_proj.q_up_nope_norm": _replicated_local_output(),
            "kv_proj.kv_down": _replicated_local_output(),
            "kv_proj.kv_norm": _replicated_local_output(),
            "kv_proj.kv_latent_norm": _replicated_local_output(),
            "kv_proj.k_up_nope": ColwiseParallel(use_local_output=True),
            "kv_proj.v_up": ColwiseParallel(use_local_output=True),
        },
    )
    if isinstance(attention.core, HeavilyCompressedAttention):
        # The indexer scores a full-sequence replicated input; keeping it
        # replicated avoids a partial-sum all-reduce on the indexer scores.
        parallelize_module(
            attention.core,
            tp_mesh,
            {
                "indexer_q": _replicated_local_output(),
                "indexer_k": _replicated_local_output(),
                "indexer_q_norm": _replicated_local_output(),
                "indexer_k_norm": _replicated_local_output(),
            },
        )
    parallelize_module(attention.o_proj, tp_mesh, _GroupedOProjTP())
    attention.q_proj.n_heads //= tp_degree
    attention.kv_proj.n_heads //= tp_degree
    attention.n_heads //= tp_degree
    if attention.attn_sink is not None:
        attention.register_parameter(
            "attn_sink",
            nn.Parameter(
                distribute_tensor(attention.attn_sink, tp_mesh, [Shard(0)], src_data_rank=0)
            ),
        )


def _apply_non_moe_tp(
    model: ArLlmModel,
    tp_mesh: DeviceMesh,
    *,
    loss_parallel: bool,
    training_seq_len: int,
) -> None:
    if model.tok_embeddings.embedding_dim % tp_mesh.size() != 0:
        raise ValueError(
            f"hidden size={model.tok_embeddings.embedding_dim} must be divisible by "
            f"TP degree={tp_mesh.size()}"
        )
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

    for layer in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(use_local_output=True),
            "attn_post_norm": SequenceParallel(use_local_output=True),
            "moe_pre_norm": SequenceParallel(use_local_output=True),
            "moe_post_norm": SequenceParallel(use_local_output=True),
            "attention": PrepareModuleInput(
                input_layouts=(sequence_shard, None, None),
                desired_input_layouts=(Replicate(), None, None),
                use_local_output=True,
            ),
        }
        parallelize_module(layer, tp_mesh, layer_plan)
        if layer.attention.layer_type == "kda":
            _apply_kda_tp(layer.attention, tp_mesh)
        else:
            _apply_mla_tp(layer.attention, tp_mesh)
        if layer.has_engram:
            # Ids are global under TP: slice the hash windows to the local
            # sequence range. No hidden-state gather is needed -- the table
            # lookup dispatches keys over the table-sharding mesh instead.
            tp_rank = tp_mesh.get_local_rank()
            local_seq = training_seq_len // tp_mesh.size()
            layer.engram._ids_window = (tp_rank * local_seq, (tp_rank + 1) * local_seq)

        parallelize_module(
            layer.moe,
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

    logger.info("Applied ar_llm tensor and sequence parallelism")


def _apply_moe_parallel(
    model: ArLlmModel,
    *,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
) -> None:
    if tp_mesh is None and ep_mesh is None:
        return
    for layer in model.layers.values():
        moe = layer.moe
        if ep_mesh is not None:
            if moe.num_experts % ep_mesh.size() != 0:
                raise ValueError(
                    f"num_routed_experts={moe.num_experts} must be divisible by "
                    f"EP degree={ep_mesh.size()}"
                )
            parallelize_module(moe.experts, ep_mesh, _ArLlmExpertParallel())
        else:
            assert tp_mesh is not None
            if moe.experts.w4.shape[1] % tp_mesh.size() != 0:
                raise ValueError(
                    f"moe_intermediate_size={moe.experts.w4.shape[1]} must be divisible by "
                    f"TP degree={tp_mesh.size()}"
                )
            parallelize_module(moe.experts, tp_mesh, _ArLlmExpertTensorParallel())
        for weight_name in _EXPERT_WEIGHT_NAMES:
            if not isinstance(getattr(moe.experts, weight_name), DTensor):
                raise RuntimeError(
                    f"ar_llm expert weight {weight_name} was not distributed to a DTensor; "
                    "the expert parallel plan did not apply"
                )
        if tp_mesh is not None and moe.shared_experts is not None:
            if moe.shared_experts.latent_to_inter.shape[1] % tp_mesh.size() != 0:
                raise ValueError(
                    "moe_intermediate_size="
                    f"{moe.shared_experts.latent_to_inter.shape[1]} must be divisible by "
                    f"TP degree={tp_mesh.size()}"
                )
            parallelize_module(moe.shared_experts, tp_mesh, _LatentInterTP())

    logger.info("Applied ar_llm expert parallelism")


def parallelize_ar_llm(
    model: ArLlmModel,
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
            "ar_llm pipeline parallelism is deferred until the MoR cross-stage KV "
            "sharing contract is defined; see torchtitan_npu/models/ar_llm/MODEL_CONTRACT.md"
        )
    if parallel_dims.etp_enabled:
        raise NotImplementedError(
            "ar_llm EP+ETP is not supported yet; see MODEL_CONTRACT.md"
        )
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "ar_llm model compilation has not been validated; continuing without "
            "applying torch.compile"
        )

    if parallelism.expert_parallel_comm_backend == "deepep":
        raise NotImplementedError("ar_llm does not support the DeepEP dispatch backend yet")

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    ep_mesh = parallel_dims.get_optional_mesh("ep")

    if parallel_dims.cp_enabled and tp_mesh is not None:
        raise NotImplementedError(
            "ar_llm supports the all-gather CP strategy only without tensor "
            "parallelism for now; see MODEL_CONTRACT.md"
        )
    if parallel_dims.cp_enabled:
        sequence_parallel_degree = parallel_dims.cp
        if training.seq_len % sequence_parallel_degree != 0:
            raise ValueError(
                f"seq_len={training.seq_len} must be divisible by "
                f"CP degree={sequence_parallel_degree}"
            )

    if tp_mesh is not None:
        _apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            training_seq_len=training.seq_len,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    _apply_context_parallel(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        training_seq_len=training.seq_len,
    )

    _apply_moe_parallel(model, tp_mesh=tp_mesh, ep_mesh=ep_mesh)

    # Engram tables shard over a compute-time-stable mesh dim: EP (expert
    # treatment) when enabled, else TP; under pure DP/FSDP they stay FSDP
    # sharded and lookups are local.
    engram_mesh = ep_mesh if ep_mesh is not None else tp_mesh
    _apply_engram_table_parallel(
        model,
        engram_mesh,
        force_balance=model.model_args.debug_force_load_balance,
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
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        edp_mesh_names = (
            ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        _apply_ar_llm_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )
        logger.info("Applied ar_llm FSDP/eFSDP")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
