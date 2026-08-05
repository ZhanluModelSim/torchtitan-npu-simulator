# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for Kimi K3."""

import logging
from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
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
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import NoParallel, maybe_enable_async_tp
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .attention import KimiDeltaAttention, KimiGatedMLA, ShortConvolution
from .model import KimiAttentionResidual, KimiK3Model

logger = logging.getLogger(__name__)


class _KimiTensorParallel(TensorParallel):
    """Expert tensor parallel with a replicated latent output."""

    @staticmethod
    def _reduce_output(
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return dist_nn.all_reduce(output, group=device_mesh.get_group())

    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            output_fn=self._reduce_output,
        )


class _KimiExpertTensorParallel(ExpertTensorParallel):
    """EP+ETP with reduction before LatentMoE normalization/decompression."""

    def _token_combine(
        self,
        module: nn.Module,
        routed_output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        routed_output = super()._token_combine(
            module,
            routed_output,
            device_mesh,
        )
        return dist_nn.all_reduce(
            routed_output,
            group=device_mesh["etp"].get_group(),
        )


class _KimiAttentionContextParallel(ParallelStyle):
    """All-gather sequence context before attention and shard it afterwards.

    KDA contains causal depthwise convolutions and recurrent delta state, so
    swapping sequence for heads after projection would lose convolution halos
    and cross-rank recurrent state. Gathering the complete sequence is the
    conservative correct CP strategy for both KDA and MLA.
    """

    @staticmethod
    def _pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if mesh.ndim != 1:
            raise ValueError(
                f"Kimi CP expects a 1D mesh, got {mesh.ndim}D"
            )
        hidden_states = args[0]
        gathered = funcol.all_gather_tensor_autograd(
            hidden_states.contiguous(),
            gather_dim=0,
            group=mesh.get_group(),
        )
        if isinstance(gathered, funcol.AsyncCollectiveTensor):
            gathered = torch.ops._c10d_functional.wait_tensor(gathered)
        hidden_states = torch.cat(
            torch.chunk(gathered, mesh.size(), dim=0),
            dim=1,
        )
        return (hidden_states, *args[1:]), kwargs

    @staticmethod
    def _post_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
        mesh: DeviceMesh,
    ) -> torch.Tensor:
        if output.shape[1] % mesh.size() != 0:
            raise ValueError(
                f"Kimi attention sequence length={output.shape[1]} must be "
                f"divisible by CP degree={mesh.size()}"
            )
        return output.chunk(mesh.size(), dim=1)[mesh.get_local_rank()].contiguous()

    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        module.register_forward_pre_hook(
            partial(self._pre_hook, mesh=device_mesh),
            with_kwargs=True,
        )
        module.register_forward_hook(
            partial(self._post_hook, mesh=device_mesh),
            prepend=True,
        )
        return module


class _KimiAttentionResidualParallel(ParallelStyle):
    """Sequence-shard both AttnRes inputs while replicating its parameters."""

    _sequence_placement = (Shard(0),)

    @classmethod
    def _prepare_input(
        cls,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[DTensor, DTensor]:
        prepared_inputs: list[DTensor] = []
        for input_tensor in inputs:
            if isinstance(input_tensor, DTensor):
                if input_tensor.placements != cls._sequence_placement:
                    input_tensor = input_tensor.redistribute(
                        placements=cls._sequence_placement,
                        async_op=True,
                    )
            else:
                input_tensor = DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    cls._sequence_placement,
                    run_check=False,
                )
            prepared_inputs.append(input_tensor)
        return prepared_inputs[0], prepared_inputs[1]

    @classmethod
    def _prepare_output(
        cls,
        module: nn.Module,
        output: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        if output.placements != cls._sequence_placement:
            raise RuntimeError(
                "Kimi AttnRes output must remain sequence-sharded; got "
                f"{output.placements}"
            )
        return output.to_local(grad_placements=cls._sequence_placement)

    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        if not isinstance(module, KimiAttentionResidual):
            raise TypeError(
                "Kimi AttnRes parallel style expects "
                f"KimiAttentionResidual, got {type(module).__name__}"
            )
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._prepare_input,
            output_fn=self._prepare_output,
        )


class _KimiHeadNormParallel(ParallelStyle):
    """Shard KDA head-local activations and reduce replicated norm gradients."""

    _head_placement = (Shard(2),)

    @classmethod
    def _prepare_input(
        cls,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[DTensor, DTensor]:
        prepared: list[DTensor] = []
        for input_tensor in inputs:
            prepared.append(
                DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    cls._head_placement,
                    run_check=False,
                )
            )
        return prepared[0], prepared[1]

    @classmethod
    def _prepare_output(
        cls,
        module: nn.Module,
        output: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return output.to_local(grad_placements=cls._head_placement)

    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._prepare_input,
            output_fn=self._prepare_output,
        )


class _KimiDepthwiseConvHeadParallel(ParallelStyle):
    """Shard a KDA depthwise convolution's channels across TP ranks."""

    @staticmethod
    def _partition(
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        del name
        if not isinstance(module, ShortConvolution):
            return
        conv = module.conv
        conv.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    conv.weight,
                    device_mesh,
                    [Shard(0)],
                    src_data_rank=0,
                )
            ),
        )
        tp_degree = device_mesh.size()
        module.hidden_size //= tp_degree
        conv.in_channels //= tp_degree
        conv.out_channels //= tp_degree
        conv.groups //= tp_degree

    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        if not isinstance(module, ShortConvolution):
            raise TypeError(
                "Kimi depthwise-convolution TP expects ShortConvolution, "
                f"got {type(module).__name__}"
            )
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition,
        )


def _replicated_local_output() -> NoParallel:
    return NoParallel(local_output_grad_placements=(Partial(),))


def _apply_kda_tensor_parallel(
    attention: KimiDeltaAttention,
    tp_mesh: DeviceMesh,
    *,
    output_layout: Shard,
) -> None:
    """Shard every KDA head-owned projection, state, and convolution."""

    tp_degree = tp_mesh.size()
    if attention.num_heads % tp_degree != 0:
        raise ValueError(
            f"Kimi KDA num_heads={attention.num_heads} must be divisible "
            f"by TP degree={tp_degree}"
        )
    parallelize_module(
        attention,
        tp_mesh,
        {
            "q_proj": ColwiseParallel(use_local_output=True),
            "k_proj": ColwiseParallel(use_local_output=True),
            "v_proj": ColwiseParallel(use_local_output=True),
            "q_conv1d": _KimiDepthwiseConvHeadParallel(),
            "k_conv1d": _KimiDepthwiseConvHeadParallel(),
            "v_conv1d": _KimiDepthwiseConvHeadParallel(),
            "f_a_proj": _replicated_local_output(),
            "f_b_proj": ColwiseParallel(use_local_output=True),
            "b_proj": ColwiseParallel(use_local_output=True),
            "o_norm": _KimiHeadNormParallel(),
            "o_proj": RowwiseParallel(
                output_layouts=output_layout,
                use_local_output=True,
            ),
        },
    )
    if attention.use_full_rank_gate:
        parallelize_module(
            attention.g_proj,
            tp_mesh,
            ColwiseParallel(use_local_output=True),
        )
    else:
        parallelize_module(
            attention,
            tp_mesh,
            {
                "g_a_proj": _replicated_local_output(),
                "g_b_proj": ColwiseParallel(use_local_output=True),
            },
        )
    for parameter_name in ("A_log", "dt_bias"):
        parameter = getattr(attention, parameter_name)
        attention.register_parameter(
            parameter_name,
            nn.Parameter(
                distribute_tensor(
                    parameter,
                    tp_mesh,
                    [Shard(0)],
                    src_data_rank=0,
                )
            ),
        )
    attention.num_heads //= tp_degree


def _apply_mla_tensor_parallel(
    attention: KimiGatedMLA,
    tp_mesh: DeviceMesh,
    *,
    output_layout: Shard,
) -> None:
    tp_degree = tp_mesh.size()
    if attention.num_heads % tp_degree != 0:
        raise ValueError(
            f"Kimi MLA num_heads={attention.num_heads} must be divisible "
            f"by TP degree={tp_degree}"
        )

    parallelize_module(
        attention,
        tp_mesh,
        {
            "q_a_proj": _replicated_local_output(),
            "q_a_layernorm": _replicated_local_output(),
            "q_b_proj": ColwiseParallel(use_local_output=True),
            "kv_a_proj_with_mqa": _replicated_local_output(),
            "kv_a_layernorm": _replicated_local_output(),
            "kv_b_proj": ColwiseParallel(use_local_output=True),
            "g_proj": ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(
                output_layouts=output_layout,
                use_local_output=True,
            ),
        },
    )
    attention.num_heads //= tp_degree


def _apply_non_moe_tp(
    model: KimiK3Model,
    tp_mesh: DeviceMesh,
    *,
    loss_parallel: bool,
) -> None:
    if tp_mesh.ndim != 1:
        raise ValueError(f"Kimi TP expects a 1D mesh, got {tp_mesh.ndim}D")
    if model.tok_embeddings.embedding_dim % tp_mesh.size() != 0:
        raise ValueError(
            f"Kimi hidden size={model.tok_embeddings.embedding_dim} must be "
            f"divisible by TP degree={tp_mesh.size()}"
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
                output_layouts=(
                    Shard(-1) if loss_parallel else Replicate()
                ),
                use_local_output=not loss_parallel,
            ),
        },
    )
    if model.attn_res_block_size is not None:
        parallelize_module(
            model.output_attn_res,
            tp_mesh,
            _KimiAttentionResidualParallel(),
        )

    for layer in model.layers.values():
        layer_plan: dict[str, ParallelStyle] = {
            "attention_norm": SequenceParallel(use_local_output=True),
            "ffn_norm": SequenceParallel(use_local_output=True),
        }
        if not layer.moe_enabled:
            if (
                layer.feed_forward.gate_proj.out_features
                % tp_mesh.size()
                != 0
            ):
                raise ValueError(
                    "Kimi dense intermediate size="
                    f"{layer.feed_forward.gate_proj.out_features} must be "
                    f"divisible by TP degree={tp_mesh.size()}"
                )
            layer_plan["feed_forward"] = PrepareModuleInput(
                input_layouts=(sequence_shard,),
                desired_input_layouts=(Replicate(),),
                use_local_output=True,
            )
        if layer.attn_res_block_size is not None:
            parallelize_module(
                layer.self_attention_res,
                tp_mesh,
                _KimiAttentionResidualParallel(),
            )
            parallelize_module(
                layer.mlp_res,
                tp_mesh,
                _KimiAttentionResidualParallel(),
            )

        if isinstance(layer.attention, KimiDeltaAttention):
            layer_plan["attention"] = PrepareModuleInput(
                input_layouts=(sequence_shard, None, None),
                desired_input_layouts=(
                    Replicate(),
                    None,
                    None,
                ),
                use_local_output=True,
            )
            parallelize_module(layer, tp_mesh, layer_plan)
            _apply_kda_tensor_parallel(
                layer.attention,
                tp_mesh,
                output_layout=sequence_shard,
            )
        else:
            layer_plan["attention"] = PrepareModuleInput(
                input_layouts=(sequence_shard, None, None),
                desired_input_layouts=(
                    Replicate(),
                    None,
                    None,
                ),
                use_local_output=True,
            )
            parallelize_module(layer, tp_mesh, layer_plan)
            _apply_mla_tensor_parallel(
                layer.attention,
                tp_mesh,
                output_layout=sequence_shard,
            )

        if not layer.moe_enabled:
            parallelize_module(
                layer.feed_forward,
                tp_mesh,
                {
                    "gate_proj": ColwiseParallel(use_local_output=True),
                    "up_proj": ColwiseParallel(use_local_output=True),
                    "down_proj": RowwiseParallel(
                        output_layouts=sequence_shard,
                        use_local_output=True,
                    ),
                },
            )

    logger.info("Applied Kimi K3 tensor and sequence parallelism")


def _apply_moe_tp_boundary(
    layer: nn.Module,
    tp_mesh: DeviceMesh,
) -> None:
    sequence_shard = Shard(1)
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

    replicated_modules = [
        layer.moe.gate.gate,
        layer.moe.routed_expert_down_proj,
        layer.moe.routed_expert_norm,
        layer.moe.routed_expert_up_proj,
        layer.moe.shared_experts,
    ]
    for module in replicated_modules:
        if module is not None:
            parallelize_module(
                module,
                tp_mesh,
                _replicated_local_output(),
            )


def _apply_moe_parallel(
    model: KimiK3Model,
    *,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
) -> None:
    if tp_mesh is None and ep_mesh is None:
        return
    for mesh_name, mesh in (
        ("TP", tp_mesh),
        ("EP", ep_mesh),
        ("ETP", etp_mesh),
    ):
        if mesh is not None and mesh.ndim != 1:
            raise ValueError(
                f"Kimi {mesh_name} expects a 1D mesh, got {mesh.ndim}D"
            )

    for layer in model.layers.values():
        if not layer.moe_enabled:
            continue

        if layer.moe.num_experts % (ep_mesh.size() if ep_mesh else 1) != 0:
            raise ValueError(
                f"num_experts={layer.moe.num_experts} must be divisible "
                f"by EP degree={ep_mesh.size() if ep_mesh else 1}"
            )
        expert_tensor_degree = (
            etp_mesh.size()
            if etp_mesh is not None
            else (
                tp_mesh.size()
                if ep_mesh is None and tp_mesh is not None
                else 1
            )
        )
        if (
            layer.moe.experts.w1.shape[1] % expert_tensor_degree
            != 0
        ):
            raise ValueError(
                "Kimi expert intermediate size="
                f"{layer.moe.experts.w1.shape[1]} must be divisible by "
                f"expert tensor degree={expert_tensor_degree}"
            )

        if tp_mesh is not None:
            _apply_moe_tp_boundary(layer, tp_mesh)

        if ep_mesh is None:
            assert tp_mesh is not None
            experts_mesh = tp_mesh
            experts_plan: ParallelStyle = _KimiTensorParallel()
        elif etp_mesh is None:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
        else:
            if ep_etp_mesh is None:
                raise ValueError("EP+ETP requires the combined ep_etp mesh")
            if ep_etp_mesh.ndim != 2:
                raise ValueError(
                    "Kimi EP+ETP expects a 2D mesh, got "
                    f"{ep_etp_mesh.ndim}D"
                )
            experts_mesh = ep_etp_mesh
            experts_plan = _KimiExpertTensorParallel()

        parallelize_module(
            layer.moe.experts,
            experts_mesh,
            experts_plan,
        )

    logger.info("Applied Kimi K3 expert parallelism")


def _apply_context_parallel(
    model: KimiK3Model,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
) -> None:
    if not parallel_dims.cp_enabled:
        return
    if parallelism.context_parallel_load_balancer is not None:
        raise ValueError(
            "Kimi KDA context parallelism requires contiguous sequence shards; "
            "set context_parallel_load_balancer=None"
        )

    cp_mesh = parallel_dims.get_mesh("cp")
    if cp_mesh.ndim != 1:
        raise ValueError(f"Kimi CP expects a 1D mesh, got {cp_mesh.ndim}D")
    for layer in model.layers.values():
        parallelize_module(
            layer.attention,
            cp_mesh,
            _KimiAttentionContextParallel(),
        )

    logger.info("Applied Kimi K3 all-gather context parallelism")


def parallelize_kimi_k3(
    model: KimiK3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP, EP/ETP, CP, AC, and FSDP/eFSDP in DeepSeek order."""
    del model_converters

    sequence_parallel_degree = (
        getattr(parallel_dims, "tp", 1)
        * getattr(parallel_dims, "cp", 1)
    )
    if training.seq_len % sequence_parallel_degree != 0:
        raise ValueError(
            f"seq_len={training.seq_len} must be divisible by "
            f"TP * CP={sequence_parallel_degree}"
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Kimi K3 pipeline parallelism is intentionally deferred until "
            "FSDP/TP/EP/CP support is complete"
        )
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "Kimi K3 model compilation has not been validated with distributed "
            "parallelism; continuing without applying torch.compile"
        )

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    etp_mesh = parallel_dims.get_optional_mesh("etp")
    ep_etp_mesh = parallel_dims.get_optional_mesh(["ep", "etp"])

    if tp_mesh is not None:
        _apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    _apply_moe_parallel(
        model,
        tp_mesh=tp_mesh,
        ep_mesh=ep_mesh,
        etp_mesh=etp_mesh,
        ep_etp_mesh=ep_etp_mesh,
    )
    _apply_context_parallel(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
    )

    # Model compilation is deliberately skipped above, so activation
    # checkpointing must not assume it is wrapping a compiled model.
    model_compile_enabled = False
    if ac_config.mode != "none":
        apply_moe_ac(
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
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=(
                parallelism.fsdp_reshard_after_forward
            ),
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=(
                parallel_dims.fsdp_gradient_divide_factor
            ),
        )
        logger.info("Applied Kimi K3 FSDP/eFSDP")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
