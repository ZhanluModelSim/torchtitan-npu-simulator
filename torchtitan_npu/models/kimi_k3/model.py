# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model: hybrid KDA/Gated-MLA MoE decoder.

Fork reason: Upstream torchtitan has no K3 support. K3 introduces KDA linear
attention (69/93 layers), Gated MLA (24/93 layers), SiTU-GLU, and LatentMoE.
Reference: MindSpeed-MM mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py
"""

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.module import Module, ModuleDict

from .attention import KimiDeltaAttention, KimiGatedMLA
from .feed_forward import KimiMLP, KimiSparseMoeBlock

logger = logging.getLogger(__name__)


class KimiK3TransformerBlock(nn.Module):
    """Single decoder layer: attention (KDA or Gated MLA) + FFN (dense or MoE)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: KimiDeltaAttention.Config | KimiGatedMLA.Config = field(
            default_factory=KimiDeltaAttention.Config
        )
        attention_norm: nn.RMSNorm | None = None
        ffn_norm: nn.RMSNorm | None = None
        feed_forward: KimiMLP | None = None
        moe: KimiSparseMoeBlock.Config | None = None
        norm_eps: float = 1e-5
        dim: int = 7168

    def __init__(self, config: Config):
        super().__init__()
        # Build attention
        if isinstance(config.attention, KimiDeltaAttention.Config):
            self.attention = KimiDeltaAttention(config.attention)
        else:
            self.attention = KimiGatedMLA(config.attention)

        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # Build FFN or MoE
        if config.moe is not None:
            self.moe = KimiSparseMoeBlock(config.moe)
            self.feed_forward = None
        elif config.feed_forward is not None:
            self.feed_forward = config.feed_forward
            self.moe = None
        else:
            raise ValueError("Either feed_forward or moe must be specified")

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = self.attention_norm(x)
        h = self.attention(h, attention_masks, positions)
        x = x + h

        # Pre-norm FFN/MoE
        h = self.ffn_norm(x)
        if self.moe is not None:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        x = x + h

        return x


class KimiK3Model(nn.Module):
    """Kimi K3: 2.8T MoE model with hybrid KDA/Gated-MLA attention.

    Architecture:
    - 93 layers: 69 KDA + 24 Gated MLA (pattern: 3 KDA + 1 MLA per group of 4)
    - Layer 0 is dense FFN, layers 1-92 are MoE with LatentMoE
    - 896 routed experts (top-16) + 2 shared experts
    - SiTU-GLU activation throughout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int = 163840
        dim: int = 7168
        layers: list[KimiK3TransformerBlock.Config] = field(default_factory=list)
        norm_eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = ModuleDict(
            {str(i): KimiK3TransformerBlock(layer_cfg) for i, layer_cfg in enumerate(config.layers)}
        )
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            x = layer(x, attention_masks, positions)
        x = self.norm(x)
        return self.output(x)
