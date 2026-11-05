# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash (glm5_next) decoder: mHC residual streams + hybrid
KDA / DSA (NoPE MLA + lightning indexer) + fine-grained MoE.

Reference: MindSpeed-MM
mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py
(Glm5NextTextModel path; the vision tower lives in .vision).

The model extends the torchtitan ``Decoder`` base: ``tok_embeddings`` /
``layers`` / ``norm`` / ``lm_head`` come from the shared contract, and the
chunked loss consumes the hidden states via ``_skip_lm_head``.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from torchtitan_npu.models.glm5_next.attention import DSAAttention
from torchtitan_npu.models.glm5_next.hyper import HyperConnection
from torchtitan_npu.models.glm5_next.kda import KDALinearAttention
from torchtitan_npu.models.glm5_next.vision import GlmVisionTower

logger = logging.getLogger(__name__)


class Glm5NextBlock(TransformerBlock):
    """Decoder layer with two mHC sites (attention + ffn).

    The residual travels as ``hc_mult`` parallel streams ``[B, S, H, D]``:
    each site collapses them with learned ``pre`` weights into the sublayer
    input, then re-expands the sublayer output with ``post`` weights and
    mixes the streams with the Sinkhorn-projected ``comb`` matrix.

    ``attention`` holds the token-mixer config (KDA or DSA). It intentionally
    does NOT follow the upstream ``BaseAttention`` contract: GLM-5.3-Flash is
    NoPE with maskless mixers, so ``Decoder.first_attention``-driven RoPE
    cache sizing and Flex/Varlen mask construction do not apply (the model
    config overrides ``max_seq_len`` and ``update_from_config``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: KDALinearAttention.Config | DSAAttention.Config | None = None
        layer_id: int = 0
        hc_mult: int = 4
        hc_sinkhorn_iters: int = 20
        hc_eps: float = 1e-6
        attn_hc: HyperConnection.Config | None = None
        ffn_hc: HyperConnection.Config | None = None

        def __post_init__(self) -> None:
            if self.attention is None:
                raise ValueError("Glm5NextBlock.Config requires a token-mixer config")
            if self.attn_hc is None:
                self.attn_hc = HyperConnection.Config(
                    dim=self.attention_norm.normalized_shape,
                    hc_mult=self.hc_mult,
                    sinkhorn_iters=self.hc_sinkhorn_iters,
                    eps=self.hc_eps,
                )
            if self.ffn_hc is None:
                self.ffn_hc = HyperConnection.Config(
                    dim=self.attention_norm.normalized_shape,
                    hc_mult=self.hc_mult,
                    sinkhorn_iters=self.hc_sinkhorn_iters,
                    eps=self.hc_eps,
                )

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.is_linear_attention = isinstance(config.attention, KDALinearAttention.Config)
        self.hc_mult = config.hc_mult

        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        if config.moe is not None:
            self.moe = config.moe.build()
            self.moe_enabled = True
        elif config.feed_forward is not None:
            self.feed_forward = config.feed_forward.build()
            self.moe_enabled = False
        else:
            raise ValueError("Either feed_forward or moe must be specified")

        self.attn_hc = config.attn_hc.build()
        self.ffn_hc = config.ffn_hc.build()

    def forward(
        self,
        hidden_streams: torch.Tensor,
        attention_masks=None,
        positions=None,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = hidden_streams.dtype

        residual = hidden_streams
        post, comb, hidden_states = self.attn_hc(hidden_streams)
        hidden_states = self.attention_norm(hidden_states)
        topk_indices = None
        if self.is_linear_attention:
            hidden_states = self.attention(hidden_states, attention_masks, positions)
        else:
            hidden_states, topk_indices = self.attention(hidden_states, attention_masks, positions, prev_topk_indices)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_streams = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_streams)
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.moe(hidden_states) if self.moe_enabled else self.feed_forward(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
        return hidden_states, topk_indices


class Glm5NextModel(Decoder):
    """GLM-5.3-Flash causal LM (NoPE; mHC streams; optional vision tower)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: GlmVisionTower.Config | None = None
        image_token_id: int = 1998
        # GLM-5.3-Flash has no RoPE on any text layer (DSA is NoPE, KDA is
        # linear attention): no RoPE cache exists to size against seq_len.
        accepts_context_parallel: ClassVar[bool] = False

        @property
        def max_seq_len(self) -> int:
            return 2**63 - 1

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """GLM-specific runtime sync: declare the spmd_types placements
            (EP included) and propagate the debug routing override. The
            upstream RoPE-cache sync does not apply (NoPE)."""
            if hasattr(config, "parallelism"):
                parallelism = config.parallelism
                from .sharding import set_glm5_next_sharding_config

                set_glm5_next_sharding_config(
                    self,
                    enable_sp=parallelism.enable_sequence_parallel,
                    enable_ep=parallelism.expert_parallel_degree > 1,
                )
                if parallelism.context_parallel_degree > 1:
                    raise NotImplementedError("GLM-5.3-Flash context parallelism is not implemented yet.")
                if hasattr(config, "debug"):
                    for layer_cfg in self.layers:
                        moe = getattr(layer_cfg, "moe", None)
                        if moe is None:
                            continue
                        moe.router.debug_force_load_balance = config.debug.moe_force_load_balance

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = None
        if config.vision_encoder is not None:
            self.vision_encoder = config.vision_encoder.build()

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, float]:
        nparams = sum(p.numel() for p in model.parameters())
        flops_per_token = 6.0 * nparams
        return nparams, flops_per_token

    def _image_embeds(self, pixel_values, grid_thw):
        """Vision tower forward. Under meta capture the grid values are
        never read: the uniform square grid (config-known gh/gw) drives all
        shapes, cu_seqlens are built with arange, and the vision RoPE uses
        coordinate ranges instead of gathered grid values.

        The trainer may hand the mm inputs over as DTensors (input
        sharding); the vision tower is a dense unit, so materialize them."""
        from torch.distributed.tensor import DTensor

        if isinstance(pixel_values, DTensor):
            pixel_values = pixel_values.full_tensor()
        if isinstance(grid_thw, DTensor):
            grid_thw = grid_thw.full_tensor()
        return self.vision_encoder(pixel_values, grid_thw)

    def forward(
        self,
        tokens: torch.Tensor,
        positions=None,
        attention_masks=None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        special_tokens=None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        if pixel_values is not None and self.vision_encoder is not None:
            image_embeds = self._image_embeds(pixel_values, grid_thw)
            image_mask = tokens == self.config.image_token_id
            # The reference validates n_image_tokens * dim == numel(image_embeds)
            # by reading mask sums; under meta capture that read is illegal, so
            # the count check is delegated to the dataloader contract (the
            # square-grid collator packs exactly one image token run per image).
            h = h.masked_scatter(
                image_mask.unsqueeze(-1),
                image_embeds.to(h.dtype),
            )
        if attention_masks is None:
            # Global per-token mask: all-valid for the text flavors (the
            # DSA indexer consumes it as the valid-channel signal).
            attention_masks = torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device)

        hidden_streams = h.unsqueeze(2).expand(-1, -1, self.layers["0"].hc_mult, -1).contiguous()
        topk_indices = None
        for layer in self.layers.values():
            hidden_streams, topk_indices = layer(
                hidden_streams,
                attention_masks,
                positions,
                prev_topk_indices=topk_indices,
            )
        h = self.norm(hidden_streams.mean(dim=2))
        # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
        # calls .requires_grad on all stage inputs, which fails on bool kwargs.
        if self._skip_lm_head:
            return h
        return self.lm_head(h)


class Glm5NextMultimodalModel(Glm5NextModel):
    """GLM-5.3-Flash with the vision tower active.

    Kept as a distinct class so config factories (and the simulator) can
    select the multimodal dataloader from the model-spec type alone.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5NextModel.Config):
        def __post_init__(self) -> None:
            if self.vision_encoder is None:
                raise ValueError("Glm5NextMultimodalModel.Config requires vision_encoder")
