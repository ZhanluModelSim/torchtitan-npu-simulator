# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torchtitan.models.base.module import Module
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.deepseek_v4.model import DeepSeekV4TransformerBlock as DeepSeekV4TransformerBlock

from .mtp import DeepSeekV4MTPDecoder

if TYPE_CHECKING:
    from torchtitan.config import TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.parallelism_context import ParallelismContext

    from .compressor import Indexer
    from .mtp import DeepSeekV4MTPTransformerBlock


# Decoder block arithmetic is owned by upstream TorchTitan.


class DeepSeekV4Model(DeepSeekV4MTPDecoder):
    supports_pipeline_parallel = False

    def parallelize(self, **kwargs):
        from torchtitan_npu.models.common.moe import validate_dispatcher_capacity

        parallelism = kwargs["parallelism"]
        if (
            parallelism.tensor_parallel_degree != 1
            or parallelism.context_parallel_degree != 1
            or parallelism.pipeline_parallel_degree != 1
        ):
            raise NotImplementedError("The eager upgrade supports TP=CP=PP=1 only.")
        validate_dispatcher_capacity(self.config, training=kwargs["training"], parallelism=parallelism)
        return super().parallelize(**kwargs)

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
        parallelism_context: "ParallelismContext",
        training: "TrainingConfig",
        parallelism: "ParallelismConfig",
    ) -> None:
        # The released FP32 override rejected BF16 storage with a BF16 policy.
        # Preserving that rejection avoids silently enabling a different policy.
        if training.mixed_precision_param == "bfloat16":
            for name, parameter in Module.named_preserve_dtype_params(self):  # pyrefly: ignore [missing-attribute]
                if parameter.dtype is not torch.float32:
                    raise ValueError(
                        f"DeepSeek-V4 requires FP32 storage for protected parameter {name!r} "
                        f"with mixed_precision_param='bfloat16', got {parameter.dtype}. "
                        "Use FP32 master parameters or mixed_precision_param='float32'."
                    )
        super()._apply_fsdp(
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
        )

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MTPDecoder.Config):
        vocab_size: int
        compress_ratios: tuple[int, ...]
        n_layers: int

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
        mtp_layers = cast("nn.ModuleList | None", self.mtp_layers)
        for layer in (*self.layers.values(), *(mtp_layers if mtp_layers is not None else ())):
            # The training router policy also covers adapter/quantization children.
            router = cast("nn.Module | None", getattr(layer.feed_forward, "router", None))
            if router is not None:
                for module in router.modules():
                    names = tuple(name for name, parameter in module._parameters.items() if parameter is not None)
                    setattr(module, "_preserve_dtype_params", names)  # noqa: B010 -- API backported on existing modules
            attention = cast("nn.Module", layer.attention)
            sink = cast("nn.Module | nn.Parameter", attention.attn_sink)
            if isinstance(sink, nn.Parameter):
                setattr(attention, "_preserve_dtype_params", ("attn_sink",))  # noqa: B010
            else:
                setattr(sink, "_preserve_dtype_params", ("weight",))  # noqa: B010


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

    def _get_attention_metadata(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ):
        from torchtitan_npu.override.deepseek_v4.sparse_attn.native import NativeAscV4Metadata

        metadata = super()._get_attention_metadata(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )
        if max_num_documents is None:
            for backend, value in metadata.items():
                if isinstance(value, NativeAscV4Metadata) and value.ratio > 1:
                    metadata[backend] = value.with_fixed_native_offsets(positions.numel())
        return metadata

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
