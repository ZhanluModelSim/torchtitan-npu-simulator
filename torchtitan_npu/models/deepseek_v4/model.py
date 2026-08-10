# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE

from torchtitan_npu.patches.torchtitan.models.common.mask_handler import BaseMaskHandler

from .mhc import HcHead, HcPost, HcPre


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
        hc_head: HcHead.Config
        mask_handler: BaseMaskHandler.Config

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(
                            f"n_heads ({n_heads}) must be divisible by tp ({tp})"
                        )
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(
                            f"n_groups ({n_groups}) must be divisible by tp ({tp})"
                        )

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not yet supported for DeepSeek V4 sparse attention."
                )

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
            flops_per_token = (
                6 * non_embed_params + 12 * n_layers * n_heads * head_dim * seq_len
            )
            return total_params, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.hc_mult = cfg.hc_mult

        self._mask_handler = cfg.mask_handler.build()

        self.hc_head = cfg.hc_head.build()

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
