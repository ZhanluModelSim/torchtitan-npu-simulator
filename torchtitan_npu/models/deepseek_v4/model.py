# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, replace
from functools import partial

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from torchtitan_npu.patches.torchtitan.models.common.mask_handler import (
    BaseMaskHandler,
)

from .mhc import HcHead, HcPost, HcPre
from .packed import build_dsv4_packed_metadata


class DeepSeekV4TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        dim: int
        hc_mult: int = 4
        norm_eps: float = 1e-6
        sinkhorn_iters: int = 20
        hc_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = (
            cfg.attention_norm.build() if cfg.attention_norm is not None else None
        )
        self.ffn_norm = (
            cfg.ffn_norm.build() if cfg.ffn_norm is not None else None
        )
        if cfg.moe is not None:
            self.moe = cfg.moe.build()
            self.feed_forward = None
            self.moe_enabled = True
        else:
            self.moe = None
            self.feed_forward = (
                cfg.feed_forward.build() if cfg.feed_forward is not None else None
            )
            self.moe_enabled = False

        self.hc_mult = cfg.hc_mult
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.dim

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        self.hc_pre = HcPre.Config(
            hc_mult=cfg.hc_mult,
            dim=cfg.dim,
            sinkhorn_iters=cfg.sinkhorn_iters,
            eps=cfg.hc_eps,
            norm_eps=cfg.norm_eps,
        ).build()
        self.hc_post = HcPost.Config().build()

        if self._param_init is None:
            self._param_init = {}
        self._param_init.update({
            "hc_attn_fn": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_fn": partial(_init_trunc_normal, std=0.02),
            "hc_attn_base": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_base": partial(_init_trunc_normal, std=0.02),
            "hc_attn_scale": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_scale": partial(_init_trunc_normal, std=0.02),
        })

    def _mhc_step(self, x, residual, hc_fn, hc_scale, hc_base, norm, fn, *a, **kw):
        x, post, comb = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        if norm is not None:
            x = norm(x)
        x = fn(x)
        x = self.hc_post(x, residual, post, comb)
        return x

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        if self.moe_enabled:
            x = self.moe(self.ffn_norm(x), input_ids)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x


class DSV4PackedMetadataHandler(BaseMaskHandler):
    """Derive the DSV4 compression plan from a rank-local varlen stream.

    The handler runs after ``cp_shard``, so all token and block mappings refer
    to the sequence shard held by the current rank.
    """

    # VarlenMetadata carries neither the [B, S] shape nor per-token positions.
    wants_positions = True

    @dataclass(kw_only=True, slots=True)
    class Config(BaseMaskHandler.Config):
        compress_ratios: tuple[int, ...] = ()

    def __init__(self, config: Config):
        super().__init__(config)
        self.compress_ratios = tuple(config.compress_ratios)

    def post_process(self, masks, *, positions=None):
        if positions is None:
            raise ValueError(
                "DSV4PackedMetadataHandler requires positions; the caller must "
                "route through run_mask_handler, which honours wants_positions."
            )
        return build_dsv4_packed_metadata(
            masks, self.compress_ratios, positions=positions
        )


def _require_packed_mask_handler(attention_config, mask_handler_config) -> None:
    """Validate the mask handler required by a varlen attention core.

    A varlen core makes ``get_attention_masks`` return ``VarlenMetadata``.
    Without the packed handler, compression and selection would not receive
    document-aware token and block mappings.
    """

    inner_attention = getattr(attention_config, "inner_attention", None)
    if not isinstance(inner_attention, VarlenAttention.Config):
        return
    if isinstance(mask_handler_config, DSV4PackedMetadataHandler.Config):
        return
    raise ValueError(
        f"{type(inner_attention).__name__} needs the DSV4 packed compression "
        f"plan, but mask_handler is {type(mask_handler_config).__name__}. Add "
        "torchtitan_npu.override.deepseek_v4.varlen_dsa."
        "npu_dsv4_packed_mask_handler_override to override.imports."
    )


class DeepSeekV4Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int
        vocab_size: int
        hc_mult: int = 4
        compress_ratios: tuple[int, ...] = (1, 1, 4, 4)
        n_layers: int = 4
        norm_eps: float = 1e-6
        # Flex attention consumes ``BlockMask`` through this pass-through
        # handler. Varlen recipes replace it with the packed metadata handler.
        mask_handler: BaseMaskHandler.Config = field(
            default_factory=BaseMaskHandler.Config
        )

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
            flops_per_token = 6 * non_embed_params + 12 * n_layers * n_heads * head_dim * seq_len
            return total_params, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        if self.lm_head is not None and self.lm_head.weight.dtype != torch.float32:
            self.lm_head.weight.data = self.lm_head.weight.data.float()

        self.hc_mult = cfg.hc_mult
        self.compress_ratios = list(cfg.compress_ratios)[: cfg.n_layers]
        self.n_main_layers = cfg.n_layers

        _require_packed_mask_handler(cfg.first_attention, cfg.mask_handler)
        # Numerical entry points build configs without ``Trainer``, so inject
        # compression ratios here. Copy the handler config rather than mutating
        # a config that may be inspected or built again.
        handler_config = cfg.mask_handler
        if isinstance(handler_config, DSV4PackedMetadataHandler.Config):
            handler_config = replace(
                handler_config,
                compress_ratios=tuple(self.compress_ratios),
            )
        self._mask_handler = handler_config.build()

        hc_dim = cfg.hc_mult * cfg.dim
        self.hc_head_fn = nn.Parameter(torch.empty(cfg.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(cfg.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.hc_head = HcHead.Config(
            hc_mult=cfg.hc_mult,
            dim=cfg.dim,
            norm_eps=cfg.norm_eps,
            eps=1e-6,
        ).build()

        if self._param_init is None:
            self._param_init = {}
        self._param_init.update({
            "hc_head_fn": partial(_init_trunc_normal, std=0.02),
            "hc_head_base": partial(_init_trunc_normal, std=0.02),
            "hc_head_scale": partial(_init_trunc_normal, std=0.02),
        })

        self._dsa_loss_tracker = {}

    def get_dsa_losses(self):
        losses = dict(self._dsa_loss_tracker)
        self._dsa_loss_tracker.clear()
        return losses

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        input_ids = tokens.detach().long()
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for i in range(self.n_main_layers):
            layer = self.layers[str(i)]
            h = layer(h, input_ids, attention_masks, positions)
        
        h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        if self.lm_head is None:
            return h
        output = torch.nn.functional.linear(
            h.float(), self.lm_head.weight.float()
        )
        return output


def _init_trunc_normal(x, std=0.02):
    nn.init.trunc_normal_(x, mean=0.0, std=std)
