# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/deepseek_v3/infra/parallelize.py
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/llama4/infra/parallelize.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import (
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
from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import (
    DeepEPExpertParallel,
    ExpertParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import NoParallel, maybe_enable_async_tp
from torchtitan.models.common import FlexAttention, VarlenAttention
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import apply_fsdp

from torchtitan_npu.converters.kernels.rms_norm import NPURMSNorm
from torchtitan_npu.converters.registry import has_npu_converter
from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLossLoggingHelper
from torchtitan_npu.models.deepseek_v4.model import Attention
from torchtitan_npu.models.deepseek_v4.model import MoE as DeepSeekV4MoE

logger = logging.getLogger(__name__)

FSDP_MP_POLICY_ATTR = "_mp_policy"


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}


class AwaitRowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # wait for async redistribution to complete
        real_tensor = outputs._local_tensor
        torch.ops._c10d_functional.wait_tensor(real_tensor)

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs


class PrepareModuleInputOutputWithBwdAllReduce(PrepareModuleInputOutput):
    """
    Extension of PrepareModuleInputOutput that registers backward hooks on specified inputs
    to perform allreduce on their gradients during backpropagation.

    This is useful when certain inputs participate in computations that require
    gradient synchronization across devices (e.g., in tensor parallelism scenarios).
    """

    def __init__(self, *, bwd_allreduce_inputs: tuple[bool, ...], **kwargs):
        super().__init__(**kwargs)
        self.bwd_allreduce_inputs = bwd_allreduce_inputs

        if self.prepare_module_input.input_layouts is not None:
            assert len(self.bwd_allreduce_inputs) == len(self.prepare_module_input.input_layouts), (
                f"bwd_allreduce_inputs must have the same length as input_layouts! "
                f"Got {len(self.bwd_allreduce_inputs)} vs {len(self.prepare_module_input.input_layouts)}"
            )

    def _attach_bwd_hook_fn(self, module: nn.Module, inputs: tuple) -> None:
        """
        Register backward hooks on specified inputs to perform allreduce on gradients.

        Args:
            module: The module to register hooks on
            inputs: Tuple of input tensors to the module
        """
        for _, (inp, needs_allreduce) in enumerate(zip(inputs, self.bwd_allreduce_inputs, strict=True)):
            if not needs_allreduce:
                continue

            if not isinstance(inp, torch.Tensor) or not inp.requires_grad:
                continue

            def _allreduce_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                # Ensure gradient is contiguous for efficient communication
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=self.group)
                return grad

            inp.register_hook(_allreduce_grad_hook)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        super()._apply(module, device_mesh)

        self.group = device_mesh.get_group()
        if self.prepare_module_input.use_local_output:
            module.register_forward_pre_hook(self._attach_bwd_hook_fn)

        return module


def _register_distributed_parameter(
    module: nn.Module,
    name: str,
    device_mesh: DeviceMesh,
    placements: list,
):
    dt = nn.Parameter(
        distribute_tensor(
            getattr(module, name),
            device_mesh=device_mesh,
            placements=placements,
            src_data_rank=0,
        )
    )
    module.register_parameter(name, dt)


def _model_uses_attention_masks(model_args: Any) -> bool:
    for layer_cfg in getattr(model_args, "layers", ()):
        inner_attention = getattr(getattr(layer_cfg, "attention", None), "inner_attention", None)
        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            return True
    return False


class HcHeadParallelStyle(ParallelStyle):
    _param_names = ("hc_head_fn", "hc_head_base", "hc_head_scale")

    def __init__(self) -> None:
        self.io_plan = PrepareModuleInputOutput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_input=True,
            output_layouts=(Replicate()),
            desired_output_layouts=(Shard(1)),
            use_local_output=False,
        )

    def partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        del name
        for param_name in self._param_names:
            param = getattr(module, param_name, None)
            if param is None:
                continue
            module.register_parameter(
                param_name,
                nn.Parameter(
                    distribute_tensor(
                        param,
                        device_mesh=device_mesh,
                        placements=[Replicate()],
                        src_data_rank=0,
                    )
                ),
            )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module = distribute_module(module, device_mesh, self.partition_fn)
        return parallelize_module(module, device_mesh, self.io_plan)


def _disable_lm_head_cast_forward_inputs(model: nn.Module) -> None:
    """Keep LM head inputs in fp32 after data parallel wrapping.

    DeepSeek V4 casts hidden states with ``self.output(h.float())`` before the
    vocab projection. That cast is intentional because the large projection
    with vocab_size=129280 is numerically fragile with bf16 inputs.

    FSDP can cast forward inputs back to the mixed-precision parameter dtype.
    On the MTP head, that changes the fp32 hidden states back to bf16, while
    the main head still receives fp32. Disable only that input cast on the
    ``[norm, output]`` group, so both heads run the LM projection from fp32
    inputs.

    This changes only ``FSDPState._mp_policy``. It leaves
    ``FSDPParamGroup.mp_policy`` untouched, so the LM head parameters remain
    bf16-sharded. The same state exists for both ``apply_fsdp`` and
    ``apply_replicate`` because ``ReplicateModule`` inherits ``FSDPModule``.
    """
    # TODO(upstream sync): remove this once the pinned torchtitan baseline
    # includes pytorch/torchtitan#2881 (merge 21c4913), which sets
    # cast_forward_inputs=False natively. Until then, this workaround reaches
    # into the FSDP2 internal FSDPState._mp_policy field from torch 2.12,
    # accessed via getattr/setattr as a deliberate protected-member access;
    # drop the indirection once upstream provides native support.
    output = getattr(model, "output", None)
    if output is None:
        # No LM head on this (pipeline) stage, or data parallelism not applied.
        return
    fsdp_state = fully_shard.state(output)  # pyrefly: ignore [missing-attribute]
    if fsdp_state is None:
        return
    mp_policy = getattr(fsdp_state, FSDP_MP_POLICY_ATTR)
    setattr(
        fsdp_state,
        FSDP_MP_POLICY_ATTR,
        dataclasses.replace(mp_policy, cast_forward_inputs=False),
    )


def parallelize_deepseek_v4(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # patch the indexer loss tracking with distributed version to get the synchronized indexer loss metric
    model_args = cast("Any", model).model_args
    apply_distributed_indexer_loss_tracking(parallel_dims, model_args.n_layers, model_args.compress_ratios)

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            raise NotImplementedError("Currently, float8 tensorwise TP is not tested for deepseekv4")

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            parallelism=parallelism,
            model_converters=model_converters,
            ac_config=ac_config,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.cp_enabled:
        from torchtitan_npu.distributed.context_parallel.registry import (
            apply_cp_to_attention_module,
        )

        if not has_npu_converter(model_converters.converters, "npu_smla"):
            raise NotImplementedError(
                "Context parallel is supported only when DSv4 fusion kernels "
                "are enabled, please add `npu_smla` to converters."
            )

        apply_cp_to_attention_module(
            # pyrefly: ignore[missing-attribute, not-callable]
            [block.attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # Check if using DeepEP for MoE communication
    if parallelism.expert_parallel_comm_backend == "deepep":
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # Import deepep module to register custom ops before accessing them
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            use_deepep=use_deepep,
        )

    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )
        _disable_lm_head_cast_forward_inputs(model)

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_replicate(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        _disable_lm_head_cast_forward_inputs(model)

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    parallelism,
    model_converters,
    ac_config,
):
    """Apply tensor parallelism."""

    enable_activation_checkpoint = ac_config.mode in [
        "full",
        "selective",
    ]

    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    (
        rowwise_parallel,
        await_rowwise_parallel,
        colwise_parallel,
        prepare_module_input,
        prepare_module_input_output,
    ) = (
        RowwiseParallel,
        AwaitRowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleInputOutput,
    )
    hc_head_plan = HcHeadParallelStyle()
    tok_embeddings = getattr(model, "tok_embeddings", None)
    norm = getattr(model, "norm", None)
    output = getattr(model, "output", None)

    root_parallelize_plan: dict[str, Any] = {}
    if tok_embeddings is not None:
        root_parallelize_plan["tok_embeddings"] = rowwise_parallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        )
    if norm is not None:
        root_parallelize_plan["norm"] = SequenceParallel()
    if output is not None:
        root_parallelize_plan["output"] = colwise_parallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )

    if root_parallelize_plan:
        parallelize_module(model, tp_mesh, root_parallelize_plan)

    # NOTE: hc_head now lives inside the last main transformer layer
    # (``transformer_block.hc_head`` when ``is_last_layer``); its TP plan is
    # applied in the per-layer loop below.

    attention_kernel_plan_ratio1 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, False, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio4 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, True, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), Replicate()),
        desired_input_layouts=(
            Shard(2),
            Replicate(),
            Shard(0),
            Replicate(),
            Replicate(),
        ),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio128 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, True, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    indexer_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
        ),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate(), Replicate()),
        use_local_output=False,
    )

    li_compute_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate()),
        use_local_output=False,
    )

    li_compute_smla_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
            None,
        ),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate()),
        use_local_output=False,
    )

    compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=False,
    )

    indexer_compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    hc_pre_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=False,
    )

    hc_post_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate()),
        desired_output_layouts=(Shard(1)),
    )

    hc_pre_sinkhon_plan = prepare_module_input(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        use_local_output=True,
    )

    get_window_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(None, None, None),
        desired_input_layouts=(None, None, None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    get_compress_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(Replicate(), None),
        desired_input_layouts=(Replicate(), None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    li_loss_plan = prepare_module_input(
        input_layouts=(
            Shard(1),
            Replicate(),
            Replicate(),
            Shard(0),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        use_local_output=True,
    )

    model_args = cast("Any", model).model_args
    use_attention_masks = _model_uses_attention_masks(model_args)

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        _register_distributed_parameter(
            # pyrefly: ignore [missing-attribute]
            transformer_block.attention.inner_attention,
            "attn_sink",
            tp_mesh,
            [Shard(0)],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_fn",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_fn",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_base",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_base",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_scale",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_scale",
            tp_mesh,
            [Replicate()],
        )
        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.compress_ratio == 1:
            attention_kernel_plan = attention_kernel_plan_ratio1
        # pyrefly: ignore [missing-attribute]
        elif transformer_block.attention.compress_ratio == 4:
            attention_kernel_plan = attention_kernel_plan_ratio4
        else:
            attention_kernel_plan = attention_kernel_plan_ratio128

        layer_plan: dict[str, ParallelStyle] = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.inner_attention.sparse_attn.get_window_topk_idxs": get_window_topk_idxs_plan,
            "attention.inner_attention.sparse_attn.get_compress_topk_idxs": get_compress_topk_idxs_plan,
            # NOTE: use_local_output=False make the output to be a DTensor instead of a plain Tensor
            # so that the intermedidate results k is generated as a DTensor and its gradient is
            # correctly handled by the autograd engine.
            "attention.pre_attention.wq_a": NoParallel(),
            "attention.pre_attention.q_norm": NoParallel(),
            "attention.pre_attention.wq_b": colwise_parallel(use_local_output=False),
            "attention.pre_attention.wkv": NoParallel(),
            "attention.pre_attention.kv_norm": NoParallel(),
            "attention.post_attention.wo_a": colwise_parallel(use_local_output=False),
            "attention.post_attention.wo_b": rowwise_parallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "attention.inner_attention.sparse_attn": attention_kernel_plan,
            "attention.inner_attention.li_loss": li_loss_plan,
            "hc_post": hc_post_plan,
            "hc_pre": hc_pre_plan,
            "hc_pre.torch_hc_split_sinkhorn": hc_pre_sinkhon_plan,
            "ffn_norm": SequenceParallel(),
        }
        if getattr(transformer_block, "is_last_layer", False):
            # hc_head runs at the end of this block's forward (input is the
            # post-hc_post Shard(1) activation, same layout it had when hc_head
            # was at the root), so the same plan applies.
            layer_plan.update({"hc_head": hc_head_plan})
        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.compress_ratio > 1:
            # pyrefly: ignore [missing-attribute]
            compress_ratio = transformer_block.attention.compress_ratio
            compressor_attr = "compressor" if compress_ratio == 4 else "compressor_128"
            compressor_module = getattr(
                # pyrefly: ignore [missing-attribute]
                transformer_block.attention.pre_attention,
                compressor_attr,
            )
            compressor_key = f"attention.pre_attention.{compressor_attr}"
            _register_distributed_parameter(compressor_module, "ape", tp_mesh, [Replicate()])
            layer_plan.update(
                {
                    compressor_key: compressor_plan,
                    f"{compressor_key}.wkv": NoParallel(),
                    f"{compressor_key}.wgate": NoParallel(),
                    f"{compressor_key}.norm": NoParallel(),
                }
            )
            if compress_ratio == 4:
                li_compute_parallel_plan = li_compute_smla_plan if use_attention_masks else li_compute_plan
                _register_distributed_parameter(
                    # pyrefly: ignore [missing-attribute]
                    transformer_block.attention.pre_attention.indexer.compressor,
                    "ape",
                    tp_mesh,
                    [Replicate()],
                )
                layer_plan.update(
                    {
                        "attention.inner_attention.li_compute": li_compute_parallel_plan,
                        "attention.pre_attention.indexer": indexer_plan,
                        "attention.pre_attention.indexer.compressor": indexer_compressor_plan,
                        "attention.pre_attention.indexer.wq_b": NoParallel(local_output_grad_placements=(Replicate(),)),
                        "attention.pre_attention.indexer.weights_proj": NoParallel(
                            local_output_grad_placements=(Replicate(),)
                        ),
                        # Upstream NoParallel dropped ``use_local_output``;
                        # output stays a DTensor by default.
                        "attention.pre_attention.indexer.compressor.wkv": NoParallel(),
                        "attention.pre_attention.indexer.compressor.wgate": NoParallel(),
                        "attention.pre_attention.indexer.compressor.norm": NoParallel(),
                    }
                )

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            # Select the appropriate parallel strategy:
            # Use AwaitRowwiseParallel when activation checkpoint is enabled to handle
            # async redistribution. The custom implementation ensures wait_tensor() is called
            # on _local_tensor to prevent memory leaks caused by incomplete async operations.
            safe_rowwise_parallel = await_rowwise_parallel if enable_activation_checkpoint else rowwise_parallel
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": safe_rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        # pyrefly: ignore [missing-attribute]
        if transformer_block.layer_id >= model.model_args.n_layers:
            layer_plan.update(
                {
                    "enorm": SequenceParallel(),
                    "hnorm": SequenceParallel(),
                    "e_proj": SequenceParallel(use_local_output=True),
                    "h_proj": SequenceParallel(use_local_output=True),
                    "mtp_norm": SequenceParallel(),
                    "mtp_hc_head": hc_head_plan,
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}Tensor Parallelism to the model")


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    use_deepep: bool = False,
):
    assert tp_mesh is not None or ep_mesh is not None, f"""
        At least one of Tensor Parallel mesh (tp_mesh) or Expert Parallel mesh (ep_mesh) must be provided.
        Current status: tp_mesh={tp_mesh}, ep_mesh={ep_mesh}
        """

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1), Replicate()),
                    desired_input_layouts=(Shard(1), Shard(1)),
                    use_local_input=True,
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": SequenceParallel(sequence_dim=0, use_local_output=True),
            }
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # input: sharded on fused batch-seq dimension (dim=0)
                # all-gather for input, reduce-scatter for output
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts": PrepareModuleInput(
                            input_layouts=(Shard(0),),
                            desired_input_layouts=(Replicate(),),
                        ),
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(output_layouts=Shard(0)),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        # Currently only TP and TP extend EP are supported
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            experts_mesh = ep_mesh
            if use_deepep:
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts
                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                experts_plan = ExpertParallel()
        else:
            raise NotImplementedError("ETP is not supported currently")

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def _compile_moe_transformer_block(
    transformer_block: nn.Module,
    compile_config,
) -> nn.Module:
    # MoE layers contain FSDP(GroupedExperts) hooks. Compile around those hooks so
    # activation checkpointing does not fall the whole graph back to eager.
    block = (
        transformer_block._checkpoint_wrapped_module
        if isinstance(transformer_block, CheckpointWrapper)
        else transformer_block
    )
    for attr_name, submod in block.named_children():
        if getattr(block, attr_name) != getattr(transformer_block, attr_name):
            raise RuntimeError(f"Checkpoint-wrapped block child {attr_name!r} is not exposed on wrapper")
        if attr_name in {"hc_pre"}:
            continue
        if isinstance(submod, Attention):
            _compile_children_except(submod, {"inner_attention"}, compile_config)
        elif isinstance(submod, DeepSeekV4MoE):
            _compile_children_except(submod, {"experts"}, compile_config)
        elif isinstance(submod, NPURMSNorm):
            continue
        else:
            setattr(
                block,
                attr_name,
                torch.compile(submod, backend=compile_config.backend, fullgraph=True),
            )
    return transformer_block


def _compile_children_except(
    module: nn.Module,
    skip_names: set[str],
    compile_config,
) -> None:
    for child_name, child_module in module.named_children():
        if child_name in skip_names:
            continue
        setattr(
            module,
            child_name,
            torch.compile(
                child_module,
                backend=compile_config.backend,
                fullgraph=True,
            ),
        )


_NPU_GMM_COMPILED_ATTR = "_npu_gmm_compiled"
_NPU_GMM_FN_ATTR = "_run_experts_grouped_mm"
_TORCH_DYNAMO_ATTR = "_dynamo"


def _patch_grouped_mm_compile(compile_config, ep_enabled: bool) -> None:
    from torchtitan_npu.converters.kernels import gmm as gmm_module

    grouped_mm = getattr(gmm_module, _NPU_GMM_FN_ATTR)
    if getattr(grouped_mm, _NPU_GMM_COMPILED_ATTR, False):
        return

    compiled_fn = gmm_module.compile_grouped_mm(grouped_mm, backend=compile_config.backend)
    maybe_mark_dynamic = getattr(torch, _TORCH_DYNAMO_ATTR).maybe_mark_dynamic

    def _run_experts_grouped_mm_compiled(
        w13: torch.Tensor | None,
        w2: torch.Tensor,
        _w3: torch.Tensor | None,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        swiglu_limit: float | None = None,
        routed_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ep_enabled:
            maybe_mark_dynamic(x, 0)
        return compiled_fn(
            w13,
            w2,
            _w3,
            x,
            num_tokens_per_expert,
            swiglu_limit,
            routed_scores,
        )

    setattr(_run_experts_grouped_mm_compiled, _NPU_GMM_COMPILED_ATTR, True)
    setattr(gmm_module, _NPU_GMM_FN_ATTR, _run_experts_grouped_mm_compiled)


def apply_compile(model: nn.Module, compile_config, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # Required for torch.compile to avoid graph breaking on dynamic shapes in
    # token-choice MoE.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    for (
        layer_id,
        transformer_block,
    ) in model.layers.named_children():  # pyrefly: ignore [missing-attribute]
        if transformer_block.moe_enabled:
            transformer_block = _compile_moe_transformer_block(transformer_block, compile_config)
        else:
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )
        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    _patch_grouped_mm_compile(compile_config, ep_enabled)
    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_distributed_indexer_loss_tracking(
    parallel_dims: ParallelDims,
    num_layers: int,
    compress_ratios: tuple[int, ...],
):
    """
    Dynamically patch track_dsa_indexer_metrics to support 3D/4D parallelism
    synchronization efficiently using a single global communication step.

    Before synchronization, the indexer loss on each GPU is merely an average
    over its local [B, S] (Batch, Sequence) shape. In a distributed scenario,
    this local loss must be synchronized (averaged) across all parallel domains,
    including Pipeline Parallel (PP), Tensor Parallel (TP), Data Parallel (DP),
    and Context Parallel (CP) groups, to obtain the globally accurate metric.
    """

    # Pre-compute which layer indices have an indexer (compress_ratio == 4).
    # This is static model structure info — same on every rank — so it can be
    # used as a safe early-return guard and as an index into the values tensor
    # without introducing any cross-rank divergence.
    valid_indices = [i for i in range(num_layers) if i < len(compress_ratios) and compress_ratios[i] == 4]

    # Normalization factor: each valid layer is computed by every rank in its
    # PP stage.  world_size / pp == dp * tp * cp (ranks per PP stage).
    norm_factor = dist.get_world_size() // max(parallel_dims.pp, 1)

    def _new_empty_tracker_tensor() -> torch.Tensor:
        device = torch.device("npu", cast("Any", torch).npu.current_device())
        return torch.zeros(num_layers, device=device)

    def distributed_track_dsa_indexer_metrics(total_acc_steps: int):
        # valid_indices is derived from model config — identical on all ranks,
        # so this early return fires uniformly and cannot cause a hang.
        if not valid_indices:
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        values = tracker.get("values")
        dsa_indexer_losses = _new_empty_tracker_tensor() if values is None else values.clone()

        # all_reduce is unconditional so every rank participates, even those
        # on PP stages that have no indexer layers (their values are zeros).
        if dist.is_initialized():
            dist.all_reduce(dsa_indexer_losses, op=dist.ReduceOp.SUM)

        loss = dsa_indexer_losses[valid_indices].mean() / (norm_factor * total_acc_steps)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")

    # Apply the monkey patch
    DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics = distributed_track_dsa_indexer_metrics
