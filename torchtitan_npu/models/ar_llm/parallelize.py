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
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .attention import HeavilyCompressedAttention
from .feed_forward import LatentExpertMLP, LatentGroupedExperts
from .model import ArLlmModel

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_NAMES = ("w1", "w2", "w3", "w4", "w5")


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


class _ArLlmEngramContextParallel(ParallelStyle):
    """All-gather hidden states and token ids for the Engram module.

    The n-gram rolling hash needs left-halo tokens and the dilated short conv
    needs cross-rank sequence context, so both the block hidden states and the
    raw input ids are gathered before the module runs.
    """

    @staticmethod
    def _gather_sequence(tensor, mesh):  # noqa: ANN001
        gathered = funcol.all_gather_tensor_autograd(
            tensor.contiguous(),
            gather_dim=0,
            group=mesh.get_group(),
        )
        return torch.cat(torch.chunk(gathered, mesh.size(), dim=0), dim=1)

    @staticmethod
    def _pre_hook(module, args, kwargs, mesh):  # noqa: ANN001
        if mesh.ndim != 1:
            raise ValueError(f"ar_llm CP expects a 1D mesh, got {mesh.ndim}D")
        hidden_states = _ArLlmEngramContextParallel._gather_sequence(args[0], mesh)
        input_ids = _ArLlmEngramContextParallel._gather_sequence(args[1], mesh)
        return (hidden_states, input_ids), kwargs

    @staticmethod
    def _post_hook(module, args, output, mesh):  # noqa: ANN001
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


def _apply_context_parallel(
    model: ArLlmModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
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
    for layer in model.layers.values():
        parallelize_module(
            layer.attention,
            cp_mesh,
            _ArLlmAttentionContextParallel(),
        )
        if layer.has_engram:
            parallelize_module(
                layer.engram,
                cp_mesh,
                _ArLlmEngramContextParallel(),
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
            # The n-gram hash lookup and the dilated short conv need
            # cross-rank sequence context: gather the full sequence for the
            # engram module and re-shard its output.
            parallelize_module(
                layer.engram,
                tp_mesh,
                PrepareModuleInputOutput(
                    input_layouts=(sequence_shard, Replicate()),
                    desired_input_layouts=(Replicate(), Replicate()),
                    use_local_input=True,
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(sequence_shard,),
                    use_local_output=True,
                ),
            )

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
        )
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
        logger.info("Applied ar_llm FSDP/eFSDP")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
