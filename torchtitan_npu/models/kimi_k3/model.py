# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model: hybrid KDA/Gated-MLA MoE decoder.

Fork reason: Upstream torchtitan has no K3 support. K3 introduces KDA linear
attention (69/93 layers), Gated MLA (24/93 layers), attention residuals,
SiTU-GLU, and LatentMoE.
Reference: MindSpeed-MM mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module, ModuleDict

from .attention import KimiDeltaAttention, KimiGatedMLA
from .feed_forward import KimiMLP, KimiSparseMoeBlock

logger = logging.getLogger(__name__)


class KimiAttentionResidual(nn.Module):
    """Token-local attention-residual mixing used at K3 block boundaries."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = RMSNorm.Config(
            normalized_shape=dim,
            eps=eps,
        ).build()
        self.proj = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat(
            (block_residual, prefix_sum.unsqueeze(1)),
            dim=1,
        )
        normalized = self.norm(values)
        scores = F.linear(
            normalized.float(),
            self.proj.weight.float(),
        ).squeeze(-1)
        probabilities = scores.softmax(-1).unsqueeze(1)
        return torch.matmul(
            probabilities,
            values.float(),
        ).squeeze(1).to(values.dtype)


class KimiK3TransformerBlock(Module):
    """Single decoder layer: attention (KDA or Gated MLA) + FFN (dense or MoE)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: KimiDeltaAttention.Config | KimiGatedMLA.Config = field(
            default_factory=KimiDeltaAttention.Config
        )
        feed_forward: KimiMLP.Config | None = None
        moe: KimiSparseMoeBlock.Config | None = None
        norm_eps: float = 1e-5
        dim: int = 7168
        layer_id: int = 0
        attn_res_block_size: int | None = 12

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None and self.attn_res_block_size <= 0:
            raise ValueError(
                "attn_res_block_size must be greater than zero when enabled, "
                f"got {self.attn_res_block_size}"
            )

        # Build attention
        if isinstance(config.attention, KimiDeltaAttention.Config):
            self.attention = KimiDeltaAttention(config.attention)
        else:
            self.attention = KimiGatedMLA(config.attention)

        self.attention_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        self.ffn_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        if self.attn_res_block_size is not None:
            self.self_attention_res = KimiAttentionResidual(
                config.dim,
                config.norm_eps,
            )
            self.mlp_res = KimiAttentionResidual(
                config.dim,
                config.norm_eps,
            )

        # Build FFN or MoE
        if config.moe is not None:
            self.moe = KimiSparseMoeBlock(config.moe)
            self.feed_forward = None
            self.moe_enabled = True
        elif config.feed_forward is not None:
            self.feed_forward = config.feed_forward.build()
            self.moe = None
            self.moe_enabled = False
        else:
            raise ValueError("Either feed_forward or moe must be specified")

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.attn_res_block_size is not None:
            return self._forward_attn_residual(
                x,
                attention_masks=attention_masks,
                positions=positions,
                block_residual=block_residual,
            )

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

    def _forward_attn_residual(
        self,
        x: torch.Tensor,
        *,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
        block_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.attn_res_block_size is not None
        batch_size, seq_len, hidden_size = x.shape
        if block_residual is None:
            block_residual = x.new_empty(batch_size * seq_len, 0, hidden_size)

        prefix_sum: torch.Tensor | None = x
        if block_residual.shape[1] > 0:
            x = self.self_attention_res(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_id % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    prefix_sum.reshape(-1, hidden_size).unsqueeze(1),
                ),
                dim=1,
            )
            prefix_sum = None

        h = self.attention_norm(x)
        h = self.attention(h, attention_masks, positions)
        prefix_sum = h if prefix_sum is None else prefix_sum + h

        assert prefix_sum is not None
        h = self.mlp_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
        ).view(batch_size, seq_len, hidden_size)
        h = self.ffn_norm(h)
        if self.moe is not None:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        prefix_sum = prefix_sum + h
        return prefix_sum, block_residual


class KimiK3Model(Module):
    """Kimi K3: 2.8T MoE model with hybrid KDA/Gated-MLA attention.

    Architecture:
    - 93 layers: 69 KDA + 24 Gated MLA (pattern: 3 KDA + 1 MLA per group of 4)
    - Layer 0 is dense FFN, layers 1-92 are MoE with LatentMoE
    - 896 routed experts (top-16) + 2 shared experts
    - Attention residual blocks collect one residual every 12 layers
    - SiTU-GLU activation throughout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int = 163840
        dim: int = 7168
        layers: list[KimiK3TransformerBlock.Config] = field(default_factory=list)
        norm_eps: float = 1e-5
        attn_res_block_size: int | None = 12

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            """Called by Trainer.__init__ to sync runtime params (seq_len, etc.)."""
            debug_force_load_balance = (
                trainer_config.debug.moe_force_load_balance
            )
            for layer in self.layers:
                if layer.moe is not None:
                    layer.moe.debug_force_load_balance = (
                        debug_force_load_balance
                    )

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
            """Return (total_params, flops_per_token) for MFU calculation."""
            nparams = sum(p.numel() for p in model.parameters())
            # Approximate: 6 * nparams * seq_len (2 for fwd + 4 for bwd)
            flops_per_token = 6.0 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        mismatched_layers = [
            layer.layer_id
            for layer in config.layers
            if layer.attn_res_block_size != config.attn_res_block_size
        ]
        if mismatched_layers:
            raise ValueError(
                "All Kimi K3 layers must use the model-level "
                "attn_res_block_size; mismatched layer ids: "
                f"{mismatched_layers}"
            )
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = ModuleDict(
            {str(i): KimiK3TransformerBlock(layer_cfg) for i, layer_cfg in enumerate(config.layers)}
        )
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None:
            self.output_attn_res = KimiAttentionResidual(
                config.dim,
                config.norm_eps,
            )
        self.norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def verify_module_protocol(self) -> None:
        """Verify model conforms to torchtitan's module protocol."""
        pass

    def init_weights(self, *, buffer_device=None) -> None:
        """Initialize model weights. Called by Trainer after parallelize."""
        del buffer_device
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=init_std,
                )
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        for layer in self.layers.values():
            if isinstance(layer.attention, KimiDeltaAttention):
                nn.init.uniform_(layer.attention.A_log, 1.0, 16.0)
                with torch.no_grad():
                    layer.attention.A_log.log_()
                nn.init.zeros_(layer.attention.dt_bias)
                nn.init.ones_(layer.attention.o_norm.weight)
            if layer.moe is not None:
                nn.init.normal_(
                    layer.moe.experts.w1,
                    mean=0.0,
                    std=init_std,
                )
                nn.init.normal_(
                    layer.moe.experts.w2,
                    mean=0.0,
                    std=init_std,
                )
                nn.init.normal_(
                    layer.moe.experts.w3,
                    mean=0.0,
                    std=init_std,
                )
                nn.init.zeros_(
                    layer.moe.gate.e_score_correction_bias
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        block_residual = None
        if self.attn_res_block_size is not None:
            block_residual = x.new_empty(x.shape[0] * x.shape[1], 0, x.shape[2])

        for layer in self.layers.values():
            if self.attn_res_block_size is None:
                x = layer(x, attention_masks, positions)
            else:
                x, block_residual = layer(
                    x,
                    attention_masks,
                    positions,
                    block_residual,
                )

        if self.attn_res_block_size is not None:
            assert block_residual is not None
            x = self.output_attn_res(
                x.reshape(-1, x.shape[-1]),
                block_residual,
            ).view_as(x)
        x = self.norm(x)
        return self.output(x)
