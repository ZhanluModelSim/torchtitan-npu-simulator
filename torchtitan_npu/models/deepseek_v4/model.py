# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Hashable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig, derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence
from torchtitan.models.utils import get_nparams_and_active_nparams
from torchtitan.protocols.module import ModuleList

from torchtitan_npu.models.common.linear import SharedLinear
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    AttentionMetadataMap,
)

from .metadata import DeepSeekV4Metadata
from .mhc import HcHead, HcPost, HcPre
from .mtp import MTPBatch, MTPModelOutput, _annotate_mtp_batch, _mtp_head_context, prepare_mtp_batch

if TYPE_CHECKING:
    from torchtitan.models.deepseek_v4.mtp import MTPBlock

    from .attention import Attention
    from .compressor import Indexer


def _require_attention_metadata(metadata: AttentionMetadataMap, key: Hashable) -> Any:
    entry = metadata.get(key)
    if entry is None:
        raise ValueError(f"no attention metadata was built for {key!r}")
    return entry


class DeepSeekV4TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # DeepSeek-V4 has no non-MoE layer; ``moe`` is required (overrides the
        # inherited ``MoE.Config | None = None``).
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.moe_enabled = True

        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.moe = cfg.moe.build()

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_metadata: DeepSeekV4Metadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_metadata, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        # The upstream MoE contract treats padding as a routing input.  Keep
        # the hash IDs for DSV4's first layers, while ensuring structural
        # padding does not update expert counters or AuxLoss.
        x = self.moe(
            self.ffn_norm(x),
            input_ids_T=input_ids,
            padding_mask_T=padding_mask,
        )
        x = self.hc_post(x, residual, post, comb)
        return x


class DeepSeekV4Model(Decoder):
    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        from .lora import is_lora_parameter

        if not any(is_lora_parameter(name) for model in model_parts for name, _ in model.named_parameters()):
            register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    def parallelize(self, **kwargs):
        from .parallelize import parallelize_deepseek_v4

        return parallelize_deepseek_v4(self, **kwargs)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
        from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_mtp_decoder(
            # The helper requires the decoder/layers/MTP structure, shared by
            # this implementation and the upstream DeepSeek V4 model.
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
    class Config(Decoder.Config):
        vocab_size: int
        compress_ratios: tuple[int, ...]
        n_layers: int
        hc_mult: int = 4
        hc_head: HcHead.Config
        mtp_layers: list["MTPBlock.Config"] = field(default_factory=list)
        mtp_isolate_gradients: bool = False

        def update_from_config(self, *, config, **kwargs):
            self.mtp_isolate_gradients = getattr(getattr(config, "loss", None), "mtp_isolate_gradients", False)
            # Apply runtime MoE/debug settings to auxiliary depths as well as main layers.
            num_main_layers = len(self.layers)
            self.layers.extend(self.mtp_layers or ())
            try:
                Decoder.Config.update_from_config(self, config=config, **kwargs)
            finally:
                del self.layers[num_main_layers:]
            if self.mtp_layers and config.parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError("DeepSeek V4 MTP does not support pipeline parallelism yet.")
            parallelism = config.parallelism
            if parallelism.context_parallel_degree > 1:
                from torchtitan_npu.models.common.alignment import require_dataloader_alignment

                require_dataloader_alignment(
                    config.dataloader, self.compress_ratios, parallelism.context_parallel_degree
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(f"n_heads ({n_heads}) must be divisible by tp ({tp})")
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(f"n_groups ({n_groups}) must be divisible by tp ({tp})")

            # Context parallel is supported on the AscendC fused path only: the
            # core's own metadata (its records depend on the frames) derives the
            # rank-local frames when the model's preprocessing hands the CP mesh
            # to the seam, and only the fused kernels take those frames through
            # their own metadata.  The compressor is CP-agnostic either way: its
            # carry is applied by ``CarryExchange`` outside it.
            from .sharding import set_deepseek_v4_sharding_config

            set_deepseek_v4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            deepseek_v4_model = cast("DeepSeekV4Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(deepseek_v4_model)
            num_flops_per_token = 6 * active_nparams

            for layers in (self.layers, self.mtp_layers):
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
            if model_mtp_layers:
                lm_head = cast("nn.Module", deepseek_v4_model.lm_head)
                num_flops_per_token += 6 * len(model_mtp_layers) * sum(param.numel() for param in lm_head.parameters())
                num_flops_per_token += (
                    6
                    * (self.hc_mult - 1)
                    * sum(
                        param.numel()
                        for mtp_layer in model_mtp_layers
                        for param in cast("MTPBlock", mtp_layer).h_proj.parameters()
                    )
                )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        if config.mtp_isolate_gradients:
            if not config.mtp_layers or type(config.lm_head) is not Linear.Config:
                raise ValueError("MTP gradient isolation requires MTP layers and an unmodified Linear LM head.")
            config = derive(config, type(config), lm_head=derive(config.lm_head, SharedLinear.Config))
        super().__init__(config)
        self.hc_mult = config.hc_mult
        self.hc_head = config.hc_head.build()
        self.mtp_layers = ModuleList(layer.build() for layer in config.mtp_layers)
        self.mtp_isolate_gradients = config.mtp_isolate_gradients

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        mtp_batch: MTPBatch | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if len(self.mtp_layers) > 0 and self.tok_embeddings is None:
            raise ValueError("DeepSeek V4 MTP forward requires token embeddings.")
        input_ids_T = tokens.detach().long()
        hidden = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            hidden = layer(
                hidden,
                input_ids_T,
                _require_attention_metadata(
                    attention_metadata, cast("Attention", layer.attention).attention_metadata_key
                ),
                positions,
                padding_mask=padding_mask,
            )
        prev_hc_hidden = hidden
        main_hidden = self.hc_head(hidden)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        if len(self.mtp_layers) == 0:
            if self._skip_lm_head or self.lm_head is None:
                return main_hidden
            return self.lm_head(main_hidden)

        outputs = [
            main_hidden,
            *self.mtp_forward(
                prev_hc_hidden,
                tokens,
                positions,
                attention_metadata=attention_metadata,
                mtp_batch=mtp_batch,
                padding_mask=padding_mask,
            ),
        ]
        if not self._skip_lm_head and self.lm_head is not None:
            outputs[0] = self.lm_head(outputs[0])
            with _mtp_head_context(self.lm_head, isolate_gradients=self.mtp_isolate_gradients):
                outputs[1:] = [self.lm_head(item) for item in outputs[1:]]
        return MTPModelOutput(outputs, mtp_labels=mtp_batch.labels if mtp_batch is not None else None)

    def mtp_forward(
        self,
        prev_hc_hidden: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        mtp_batch: MTPBatch | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run auxiliary MTP depths, using globally prepared inputs under CP."""
        if mtp_batch is not None:
            if mtp_batch.input_tokens.shape[-1] != len(self.mtp_layers):
                raise ValueError("Prepared MTP input depth does not match model layers.")
            _annotate_mtp_batch(mtp_batch)
        mtp_outputs = []
        prev_hc_hidden = prev_hc_hidden.detach() if self.mtp_isolate_gradients else prev_hc_hidden
        for depth, mtp_block in enumerate(self.mtp_layers, 1):
            if mtp_batch is None:
                mtp_tokens, valid_mask = roll_mtp_sequence(
                    tokens,
                    shift=depth,
                    fill_value=0,
                    positions=positions,
                    padding_mask=padding_mask,
                    return_valid_mask=True,
                )
            else:
                mtp_tokens = mtp_batch.input_tokens[..., depth - 1]
                valid_mask = mtp_batch.input_valid_mask[..., depth - 1]
            with torch.no_grad() if self.mtp_isolate_gradients else nullcontext():
                mtp_embed = self.tok_embeddings(mtp_tokens)
            # MTPBlock calls the upstream V4 TransformerBlock.forward. Changes to
            # the NPU main-stack block above also need review for this path.
            prev_hc_hidden, prediction_hidden = mtp_block(
                mtp_embed,
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                _require_attention_metadata(
                    attention_metadata, cast("Attention", mtp_block.attention).attention_metadata_key
                ),
                positions,
                padding_mask=padding_mask,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: Any,
        parallelism: Any,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...], dict[str, Any]]:
        """Prepare the MTP batch on the global row, then shard both for context parallel.

        The base seam (upstream #4116) builds each attention backend's metadata
        once per microbatch and shards the inputs for context parallel.  The MTP
        depths' rolled tokens have to be derived from the *global* row — a shift
        across a shard boundary is not a shift — so they are prepared here,
        before the base call, and cut by the same ``cp_shard`` afterwards: the
        layout is the permutation that shard shards by, so the depths land on the
        rows that predict them (a local slice, no communication).

        The layout itself -- **chunk ``r`` of every document**, i.e.
        ``--parallelism.context-parallel-load-balancer doc_chunk`` -- is the seam's:
        the shard builds its balancer from the row and permutes the replicated row
        before slicing, so no rank moves anything.  The depths ride the same shard,
        handed the same row's positions, so the two agree row for row by
        construction.  With CP disabled, the prepared batch already matches the input row.
        """
        labels = input_dict["labels"]
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        mtp_batch = None
        # MTP rolls must always be derived from the global row.  Context
        # parallelism adds a second step that shards the prepared tensors, but
        # the non-CP path must retain the same padding/document-boundary
        # validity masks as well.
        if self.mtp_layers:
            mtp_batch = prepare_mtp_batch(
                input_dict["input"], labels, positions, len(self.mtp_layers), padding_mask=padding_mask
            )

        inputs, labels, extra_kwargs = super().preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
            **kwargs,
        )
        if mtp_batch is not None:
            # With CP disabled the global row is already the model row.  Under
            # CP, cut every MTP tensor with the exact decoder input layout so
            # shifted tokens stay aligned with the main stream.
            if not parallelism_context.cp_enabled:
                extra_kwargs["mtp_batch"] = mtp_batch
                return inputs, labels, extra_kwargs
            from torchtitan.models.common.decoder_sharding import decoder_input_sharding

            from torchtitan_npu.patches.torchtitan.distributed.context_parallel import resolve_cp_load_balancer
            from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_layout_of

            mtp_inputs = {f"mtp_{index}": tensor for index, tensor in enumerate(mtp_batch)}
            if positions is not None:
                mtp_inputs["positions"] = positions
            sharding = decoder_input_sharding()["input"]
            # The main metadata selects doc_chunk implicitly. Reuse its layout
            # without sending rank-local attention metadata through CP again.
            balancer_config = resolve_cp_load_balancer(
                parallelism.context_parallel_load_balancer,
                cp_layout_of(extra_kwargs.get("attention_metadata")),
            )
            if balancer_config is not parallelism.context_parallel_load_balancer:
                parallelism = replace(parallelism, context_parallel_load_balancer=balancer_config)
            sharded = self._cp_shard(
                mtp_inputs,
                input_shardings=dict.fromkeys(mtp_inputs, sharding),
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
            extra_kwargs["mtp_batch"] = MTPBatch(*(sharded[f"mtp_{index}"] for index in range(len(mtp_batch))))
        return inputs, labels, extra_kwargs


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
