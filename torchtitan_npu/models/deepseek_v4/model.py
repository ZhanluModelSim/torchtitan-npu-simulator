# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.deepseek_v4.model import DeepSeekV4TransformerBlock as DeepSeekV4TransformerBlock

from .mtp import DeepSeekV4MTPDecoder

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

    from .compressor import Indexer
    from .mtp import DeepSeekV4MTPTransformerBlock


# Decoder block arithmetic is owned by upstream TorchTitan.


class DeepSeekV4Model(DeepSeekV4MTPDecoder):
    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        from .peft_adapter import is_lora_parameter

        if not any(is_lora_parameter(name) for model in model_parts for name, _ in model.named_parameters()):
            register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh

        from torchtitan_npu.models.common.fsdp import apply_fsdp_to_mtp_decoder, make_param_dtype_override_fn

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)

        from .parallelize import _dsv4_param_dtypes

        apply_fsdp_to_mtp_decoder(
            # The helper requires the decoder/layers/MTP structure, shared by
            # this implementation and the upstream DeepSeek V4 model.
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
            param_dtype_override_fn=make_param_dtype_override_fn(self, _dsv4_param_dtypes(self, training)),
        )

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MTPDecoder.Config):
        vocab_size: int
        compress_ratios: tuple[int, ...]
        n_layers: int

        def update_from_config(self, *, config, **kwargs) -> None:
            from torchtitan.models.common.linear import Linear

            for _, linear, _, _ in self.traverse(Linear.Config):
                linear.inplace_wgrad_accum = False
            self.mtp_isolate_gradients = getattr(config.loss, "mtp_isolate_gradients", False)
            parallelism = config.parallelism
            if (
                parallelism.tensor_parallel_degree != 1
                or parallelism.context_parallel_degree != 1
                or parallelism.pipeline_parallel_degree != 1
            ):
                raise NotImplementedError("The eager upgrade supports TP=CP=PP=1 only.")
            from torchtitan.models.common.moe.sharding import routed_expert_linear_sharding_config

            for layer in (*self.layers, *(self.mtp_layers or ())):
                experts = cast("MoE.Config", layer.feed_forward).routed_experts
                experts.w13.sharding_config = routed_expert_linear_sharding_config(
                    enable_ep=parallelism.expert_parallel_degree > 1,
                )
                experts.w2.sharding_config = routed_expert_linear_sharding_config(
                    enable_ep=parallelism.expert_parallel_degree > 1,
                )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            deepseek_v4_model = cast("DeepSeekV4Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(deepseek_v4_model)
            num_flops_per_token = 6 * active_nparams

            for layers in (self.layers, self.mtp_layers or ()):
                for layer in layers:
                    attention = layer.attention
                    inner_attention = attention.inner_attention
                    num_flops_per_token += (
                        6 * attention.n_heads * (2 * attention.head_dim) * min(seq_len, inner_attention.window_size)
                    )

                    if attention.compress_ratio > 1:
                        compressed_seq_len = seq_len // attention.compress_ratio
                        if attention.compress_ratio == 4:
                            indexer = cast("Indexer.Config", attention.indexer)
                            num_flops_per_token += (
                                6 * indexer.num_index_heads * indexer.index_head_dim * compressed_seq_len
                            )
                            compressed_seq_len = min(compressed_seq_len, inner_attention.index_topk)
                        num_flops_per_token += 6 * attention.n_heads * (2 * attention.head_dim) * compressed_seq_len

            model_mtp_layers = deepseek_v4_model.mtp_layers
            if model_mtp_layers is not None:
                lm_head = cast("nn.Module", deepseek_v4_model.lm_head)
                num_flops_per_token += 6 * len(model_mtp_layers) * sum(param.numel() for param in lm_head.parameters())
                num_flops_per_token += (
                    6
                    * (self.hc_mult - 1)
                    * sum(
                        param.numel()
                        for mtp_layer in model_mtp_layers
                        for param in cast("DeepSeekV4MTPTransformerBlock", mtp_layer).h_proj.parameters()
                    )
                )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)


class GraphTrainerDeepSeekV4Model(DeepSeekV4Model):
    def parallelize(self, **kwargs):
        from .parallelize import parallelize_graph_trainer_deepseek_v4

        kwargs.pop("skip_dp", None)
        return parallelize_graph_trainer_deepseek_v4(self, **kwargs)

    """DeepSeek V4 model variant for the GraphTrainer compilation path.

    Wraps ``init_states`` with ``disable_active_parametrization`` so that
    lazy-init parametrizations (e.g., RoPE freq buffers) are materialized
    before the FX tracer records the graph.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        pass

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            disable_active_parametrization,
        )

        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
