# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.utils import get_nparams_and_active_nparams

from .metadata import DeepSeekV4Metadata
from .mhc import HcPost, HcPre
from .mtp import DeepSeekV4MTPDecoder, MTPBatch, prepare_mtp_batch

if TYPE_CHECKING:
    from .compressor import Indexer
    from .mtp import DeepSeekV4MTPTransformerBlock


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


class DeepSeekV4Model(DeepSeekV4MTPDecoder):
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
        )

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MTPDecoder.Config):
        vocab_size: int
        compress_ratios: tuple[int, ...]
        n_layers: int

        def update_from_config(self, *, config, **kwargs):
            DeepSeekV4MTPDecoder.Config.update_from_config(self, config=config, **kwargs)
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
        construction.  Without context parallel or without MTP depths nothing changes.
        """
        labels = input_dict["labels"]
        cp_mesh = parallelism_context.get_mesh("cp") if parallelism_context.cp_enabled else None
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        mtp_batch = None
        # MTP rolls must always be derived from the global row.  Context
        # parallelism adds a second step that shards the prepared tensors, but
        # the non-CP path must retain the same padding/document-boundary
        # validity masks as well.
        if self.mtp_layers is not None:
            mtp_kwargs = {} if padding_mask is None else {"padding_mask": padding_mask}
            mtp_batch = prepare_mtp_batch(input_dict["input"], labels, positions, len(self.mtp_layers), **mtp_kwargs)

        # ``MTPDecoder.preprocess_inputs`` returns the community tuple protocol
        # (main tokens, shifted tokens, ...).  DSV4 owns a separate MTP block
        # and its global-row/CP preparation, so call the Decoder seam directly
        # and keep the model-facing input as one token tensor.
        inputs, labels, extra_kwargs = Decoder.preprocess_inputs(
            self,
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
            if cp_mesh is None:
                extra_kwargs["mtp_batch"] = mtp_batch
                return inputs, labels, extra_kwargs
            # The layout the batch was sharded by, resolved exactly as the decoder resolves it:
            # an unset flag means "the layout this stack declares", and reading the raw config
            # value here would cut the depths contiguously while the inputs are permuted.
            from torchtitan.models.common.decoder_sharding import decoder_input_sharding

            mtp_inputs = {f"mtp_{index}": tensor for index, tensor in enumerate(mtp_batch)}
            if positions is not None:
                mtp_inputs["positions"] = positions
            sharding = decoder_input_sharding()["input"]
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
