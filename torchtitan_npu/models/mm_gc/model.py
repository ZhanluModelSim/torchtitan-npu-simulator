# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""mm_gc model: SLA2 attention + Multi-Head MoE hybrid decoder.

Scale follows the enlarged sketch in
``torchtitan_npu/simulator/raw_model/mm_gc/multi-head_moe.py`` (dim=12288,
96 layers, 96 heads, MoE: 32 heads x 1024 experts, top-6 per head). Layer
layout: first 2 and last 2 layers use dense FFN, all middle layers use
Multi-Head MoE. Attention is bidirectional (no causal mask) SLA2.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module, ModuleDict

from .attention import SLA2Attention
from .core import SparseLinearAttention
from .feed_forward import MMGcMLP, MultiHeadMoE


class MMGcTransformerBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: SLA2Attention.Config = field(default_factory=SLA2Attention.Config)
        feed_forward: MMGcMLP.Config | None = None
        moe: MultiHeadMoE.Config | None = None
        norm_eps: float = 1e-6
        dim: int = 12288
        layer_id: int = 0

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.attention = SLA2Attention(config.attention)
        self.attention_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        self.ffn_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        if config.moe is not None and config.feed_forward is not None:
            raise ValueError("feed_forward and moe are mutually exclusive")
        if config.moe is None and config.feed_forward is None:
            raise ValueError("Either feed_forward or moe must be specified")
        if config.moe is not None:
            self.moe = MultiHeadMoE(config.moe)
            self.feed_forward = None
        else:
            self.feed_forward = config.feed_forward.build()
            self.moe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attention_norm(x)
        h = self.attention(h)
        x = x + h
        h = self.ffn_norm(x)
        if self.moe is not None:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        return x + h


class MMGcModel(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int = 102400
        dim: int = 12288
        seq_len: int = 4096
        rope_theta: float = 10000.0
        layers: list[MMGcTransformerBlock.Config] = field(default_factory=list)
        norm_eps: float = 1e-6

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            seq_len = trainer_config.training.seq_len
            self.seq_len = seq_len
            for layer in self.layers:
                layer.attention.seq_len = seq_len

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
            nparams = sum(p.numel() for p in model.parameters())
            flops_per_token = 6.0 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        if not config.layers:
            raise ValueError("MMGcModel requires at least one layer")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = ModuleDict(
            {str(i): MMGcTransformerBlock(layer_cfg) for i, layer_cfg in enumerate(config.layers)}
        )
        self.norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        self.output = Linear.Config(
            in_features=config.dim,
            out_features=config.vocab_size,
            bias=False,
        ).build()

        self._validate_layout()

    def _validate_layout(self) -> None:
        n_layers = len(self.layers)
        for layer in self.layers.values():
            is_dense = layer.feed_forward is not None
            layer_is_boundary = layer.layer_id < 2 or layer.layer_id >= n_layers - 2
            if is_dense != layer_is_boundary:
                raise ValueError(
                    f"layer {layer.layer_id} violates the first-2/last-2 dense "
                    "layer layout"
                )

    def verify_module_protocol(self) -> None:
        pass

    def init_weights(self, *, buffer_device=None) -> None:
        del buffer_device
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        for layer in self.layers.values():
            core = layer.attention.core
            if core.stage == 1:
                core.init_weights_1_()
            else:
                core.init_weights_2_()
            moe = layer.moe
            if moe is not None:
                nn.init.normal_(moe.gate.router, mean=0.0, std=init_std)
                nn.init.zeros_(moe.gate.expert_bias)
                nn.init.normal_(moe.experts.w1, mean=0.0, std=init_std)
                nn.init.normal_(moe.experts.w2, mean=0.0, std=init_std)
                nn.init.normal_(moe.experts.w3, mean=0.0, std=init_std)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks=None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_masks, positions
        x = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            x = layer(x)
        x = self.norm(x)
        return self.output(x)
