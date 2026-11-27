# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import spmd_types as spmd
import torch
from torch import nn
from torchtitan.config import derive
from torchtitan.models.base.module import Module
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model as TorchTitanV4Model
from torchtitan.models.deepseek_v4.model import DeepSeekV4TransformerBlock as DeepSeekV4TransformerBlock

from torchtitan_npu.models.common.attention_metadata import AttentionMetadataMap, GeometryAttentionMetadataMixin
from torchtitan_npu.models.common.linear import SharedLinear

from .mtp import _mtp_head_context

if TYPE_CHECKING:
    from torchtitan.config import TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.parallelism_context import ParallelismContext

    from .compressor import Indexer


def _require_attention_metadata(metadata: AttentionMetadataMap, layer: Any) -> Any:
    key = getattr(layer.attention, "attention_metadata_key", type(layer.attention.inner_attention))
    entry = metadata.get(key)
    if entry is None:
        raise ValueError(f"no attention metadata was built for {key!r}")
    return entry


class DeepSeekV4Model(GeometryAttentionMetadataMixin, TorchTitanV4Model):
    supports_pipeline_parallel = False

    def parallelize(self, **kwargs):
        from torchtitan_npu.models.common.moe import validate_dispatcher_capacity

        parallelism = kwargs["parallelism"]
        if parallelism.tensor_parallel_degree != 1 or parallelism.pipeline_parallel_degree != 1:
            raise NotImplementedError("DeepSeek-V4 eager training requires TP=PP=1.")
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
        if not getattr(parallelism, "disable_expert_replication", False):
            super()._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
            return

        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
        from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder

        from torchtitan_npu.models.common.fsdp import disable_expert_replication_dims

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        edp_mesh_dims = disable_expert_replication_dims(self, edp_mesh, edp_mesh_dims, ep_degree=parallelism_context.ep)
        apply_fsdp_to_mtp_decoder(
            self,  # pyrefly: ignore [bad-argument-type]
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
        )

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanV4Model.Config):
        _supports_disable_expert_replication: ClassVar[bool] = True
        mtp_isolate_gradients: bool = False
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
                    * sum(param.numel() for mtp_layer in model_mtp_layers for param in mtp_layer.h_proj.parameters())
                )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        if config.mtp_isolate_gradients:
            if not config.mtp_layers or type(config.lm_head) is not Linear.Config:
                raise ValueError("MTP gradient isolation requires MTP layers and an unmodified Linear LM head.")
            config = derive(config, type(config), lm_head=derive(config.lm_head, SharedLinear.Config))
        super().__init__(config)
        self.mtp_isolate_gradients = config.mtp_isolate_gradients
        if not config.mtp_layers:
            self.mtp_layers = None  # pyrefly: ignore [bad-assignment]

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

    # Prepare aligned tokens, labels and validity masks with the upstream producer.
    preprocess_inputs = MTPDecoder.preprocess_inputs

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.mtp_layers is not None:
            if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
                raise ValueError("MTP requires one token tensor per objective.")
            if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(self.mtp_layers):
                raise ValueError("MTP requires one validity mask per depth.")
            main_tokens = tokens[0]
        else:
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("Decoder without MTP requires one token tensor.")
            main_tokens = tokens
        if aux_loss_denominators is not None and aux_loss_denominators.numel() != (
            1 + (0 if self.mtp_layers is None else len(self.mtp_layers))
        ):
            raise ValueError("Expected one routing denominator per objective.")

        input_ids = main_tokens.detach().long()
        hidden = self.tok_embeddings(main_tokens) if self.tok_embeddings is not None else main_tokens
        hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        with spmd.no_typecheck():
            main_aux_loss_denominator = None if aux_loss_denominators is None else aux_loss_denominators[0]
        for layer in self.layers.values():
            hidden = layer(
                hidden,
                input_ids,
                _require_attention_metadata(attention_metadata, layer),
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=main_aux_loss_denominator,
            )
        main_hidden = self.hc_head(hidden)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        if self.mtp_layers is None:
            if self._skip_lm_head or self.lm_head is None:
                return main_hidden
            return self.lm_head(main_hidden)
        if self.tok_embeddings is None:
            raise ValueError("DeepSeek-V4 MTP forward requires token embeddings.")

        outputs = [
            main_hidden,
            *self._forward_mtp_layers(
                hidden,
                cast("tuple[torch.Tensor, ...]", tokens),
                cast("tuple[torch.Tensor, ...]", mtp_input_valid_masks),
                attention_metadata=attention_metadata,
                positions=positions,
                padding_mask=padding_mask,
                aux_loss_denominators=aux_loss_denominators,
            ),
        ]
        if not self._skip_lm_head and self.lm_head is not None:
            outputs[0] = self.lm_head(outputs[0])
            with _mtp_head_context(self.lm_head, isolate_gradients=self.mtp_isolate_gradients):
                outputs[1:] = [self.lm_head(item) for item in outputs[1:]]
        return tuple(outputs)

    def _forward_mtp_layers(
        self,
        prev_hc_hidden: torch.Tensor,
        tokens: tuple[torch.Tensor, ...],
        mtp_input_valid_masks: tuple[torch.Tensor, ...],
        *,
        attention_metadata: AttentionMetadataMap,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        mtp_outputs = []
        prev_hc_hidden = prev_hc_hidden.detach() if self.mtp_isolate_gradients else prev_hc_hidden
        for depth, layer in enumerate(self.mtp_layers, 1):
            mtp_tokens = tokens[depth]
            with torch.no_grad() if self.mtp_isolate_gradients else nullcontext():
                mtp_embed = self.tok_embeddings(mtp_tokens)
            # The upstream block runs the configured NPU components. Select its
            # geometry metadata here before passing the native block arguments.
            with spmd.no_typecheck():
                aux_loss_denominator = None if aux_loss_denominators is None else aux_loss_denominators[depth]
            prev_hc_hidden, prediction_hidden = layer(
                mtp_embed,
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                mtp_input_valid_masks[depth - 1],
                _require_attention_metadata(attention_metadata, layer),
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs


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
        _supports_disable_expert_replication: ClassVar[bool] = False

    def _get_attention_metadata(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ):
        from .metadata import DeepSeekV4Metadata

        metadata = super()._get_attention_metadata(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )
        if max_num_documents is None:
            for backend, value in metadata.items():
                if isinstance(value, DeepSeekV4Metadata) and value.documents is not None and value.ratio > 1:
                    # Fix the projection offsets for graph replay. Kernel frames keep only nonempty documents.
                    cu = value.documents.cu_seq_q
                    with spmd.local():
                        pad = positions.numel() // value.ratio + 1 - cu.numel()
                        torch._check(pad >= 0, lambda: "Document offsets exceed the compressed row capacity")
                        padded = torch.nn.functional.pad(cu, (0, pad), value=positions.numel())
                        if spmd.is_type_checking():
                            spmd.assert_type(padded, spmd.get_local_type(cu), spmd.get_partition_spec(cu))
                    metadata[backend] = replace(
                        value, documents=value.documents._replace(cu_seq_q=padded, cu_seq_k=padded)
                    )
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
