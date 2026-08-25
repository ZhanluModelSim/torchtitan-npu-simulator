# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE

from torchtitan_npu.models.common.metadata_extension import MetadataExtension

from .metadata import CompressedVarlenMetadata, build_compressed_varlen_metadata
from .mhc import HcHead, HcPost, HcPre
from .token_dispatcher import build_cp_plan


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
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        x = self.moe(self.ffn_norm(x), input_ids=input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class DeepSeekV4Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vocab_size: int
        hc_mult: int = 4
        compress_ratios: tuple[int, ...]
        n_layers: int
        window_size: int
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        metadata_extension: MetadataExtension.Config = field(default_factory=MetadataExtension.Config)
        hc_head: HcHead.Config

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

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
            # model's build_attention_masks derives the per-rank dispatch
            # plan when the trainer passes the CP mesh; the model-dir
            # reference tier and the golden stay no-CP-only and raise there.
            from .sharding import set_deepseek_v4_sharding_config

            set_deepseek_v4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model, seq_len):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_embed_params = sum(
                p.numel()
                for n, p in model.named_parameters()
                if p.requires_grad and "tok_embeddings" not in n and "lm_head" not in n
            )
            n_layers = self.n_layers
            head_dim = self.layers[0].attention.head_dim
            n_heads = self.layers[0].attention.n_heads
            flops_per_token = 6 * non_embed_params + 12 * n_layers * n_heads * head_dim * seq_len
            return total_params, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__(config)
        # TorchTitan 2807d3f invokes the MTP parallelization hook for all
        # DeepSeek-V3-compatible models; DeepSeek-V4 does not use MTP.
        self.mtp_layers = None
        cfg = config

        self.hc_mult = cfg.hc_mult
        self.compress_ratios = tuple(cfg.compress_ratios)
        self.window_size = cfg.window_size
        self.block_size = cfg.block_size

        self._metadata_extension = cfg.metadata_extension.build()

        self.hc_head = cfg.hc_head.build()

    def build_attention_masks(
        self,
        inputs,
        labels,
        extra_kwargs,
        *,
        cp_mesh: DeviceMesh | None = None,
        load_balancer_type: str | None = None,
    ):
        """The model-owned per-batch metadata construction (the single
        overridable mask-handling seam).

        One entry for both modes: the common contract
        (``build_compressed_varlen_metadata``) is always built; under CP
        ``_build_cp_metadata`` shards the inputs and derives the rank-local
        plan from the global context in-frame (no plan-time communication);
        the ``metadata_extension`` (e.g. the reference tier or the AscendC
        kernel metadata) runs last.
        """
        positions = extra_kwargs.get("positions")
        masks = self.get_attention_masks(positions=positions)
        if not isinstance(masks, VarlenMetadata):
            raise TypeError(
                "DeepSeek-V4 compression requires a varlen stream (the "
                "inner attention is varlen-typed), got "
                f"{type(masks)}."
            )
        common = build_compressed_varlen_metadata(masks, self.compress_ratios)
        if cp_mesh is not None:
            inputs, labels, positions, common = self._build_cp_metadata(
                inputs, labels, positions, common, cp_mesh, load_balancer_type
            )
            extra_kwargs["positions"] = positions
        if self._metadata_extension is not None:
            common = self._metadata_extension(common)
        extra_kwargs["attention_masks"] = common
        return inputs, labels, extra_kwargs

    def _build_cp_metadata(self, inputs, labels, positions, common, cp_mesh, load_balancer_type):
        """The context-parallel metadata: shard the tensors via the generic
        path and derive the rank-local plan from the global context (the
        common metadata's varlen + the load-balancer permutation).

        Returns ``(inputs, labels, positions, metadata)``."""
        seq_len = common.seq_len
        cp_size = cp_mesh.size(0)
        if seq_len % cp_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size}).")
        shard_len = seq_len // cp_size
        lb = _HeadTailLoadBalancer(seq_len, cp_size, cp_mesh.device_type) if load_balancer_type == "headtail" else None
        (inputs, labels, positions), _ = cp_shard(
            cp_mesh,
            (inputs, labels, positions),
            None,
            load_balancer_type,
            1,
        )
        rank = cp_mesh.get_local_rank()
        cp_meta, plans, window = build_cp_plan(
            common.varlen,
            lb,
            rank=rank,
            cp_size=cp_size,
            shard_len=shard_len,
            window_size=self.window_size,
            ratios=sorted(set(self.compress_ratios)),
        )
        return (
            inputs,
            labels,
            positions,
            CompressedVarlenMetadata(varlen=cp_meta, plans=plans, window=window),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        input_ids = tokens.detach().long()
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for layer in self.layers.values():
            h = layer(h, input_ids, attention_masks, positions)

        h = self.hc_head(h)
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        if self.lm_head is None:
            return h
        # Follow the dsv3/common-decoder convention: lm_head stays in BF16
        # (checkpoint dtype) and is applied as a plain module call.
        return self.lm_head(h)
