# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model: hybrid KDA/Gated-MLA MoE decoder.

Fork reason: Upstream torchtitan has no K3 support. K3 introduces KDA linear
attention (69/93 layers), Gated MLA (24/93 layers), attention residuals,
SiTU-GLU, and LatentMoE.
Reference: moonshotai/Kimi-K3, revision f831ab66814297da540d832a5235f8e904f29d06.
"""

import logging
from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

from .attention import KimiDeltaAttention, KimiGatedMLA
from .feed_forward import KimiLatentMoE, KimiMLP

logger = logging.getLogger(__name__)


class KimiResidualScoreNorm(Module):
    """FP32 normalization and folded score weight in official Attention Residual.

    Unlike an affine RMSNorm followed by a projection, the affine and
    projection weights are multiplied before the token-wise reduction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.dim))
        self.eps = config.eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, values: torch.Tensor, projection_weight: torch.Tensor) -> torch.Tensor:
        values = values.float()
        normalized = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.eps)
        score_weight = self.weight.float() * projection_weight.squeeze(0).float()
        return (normalized * score_weight).sum(-1)


class KimiAttentionResidual(Module):
    """Token-local attention-residual mixing used at K3 block boundaries."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: KimiResidualScoreNorm.Config
        proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm = config.norm.build()
        self.proj = config.proj.build()

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat(
            (block_residual, prefix_sum.unsqueeze(1)),
            dim=1,
        )
        values_float = values.float()
        scores = self.norm(values_float, self.proj.weight)
        probabilities = scores.softmax(-1).unsqueeze(1)
        return (
            torch.matmul(
                probabilities,
                values_float,
            )
            .squeeze(1)
            .to(values.dtype)
        )


class KimiK3TransformerBlock(Module):
    """Single decoder layer: attention (KDA or Gated MLA) + FFN (dense or MoE)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: KimiDeltaAttention.Config | KimiGatedMLA.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        self_attention_res: KimiAttentionResidual.Config | None = None
        mlp_res: KimiAttentionResidual.Config | None = None
        feed_forward: KimiMLP.Config | None = None
        moe: KimiLatentMoE.Config | None = None
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
                f"attn_res_block_size must be greater than zero when enabled, got {self.attn_res_block_size}"
            )

        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        if self.attn_res_block_size is not None:
            if config.self_attention_res is None or config.mlp_res is None:
                raise ValueError("Attention residuals require both self_attention_res and mlp_res configurations")
            self.self_attention_res = config.self_attention_res.build()
            if config.layer_id == 0:
                self.self_attention_res.requires_grad_(False)
            self.mlp_res = config.mlp_res.build()

        # Build FFN or MoE
        if config.moe is not None:
            self.moe = config.moe.build()
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
        h = self._feed_forward(h)
        x = x + h

        return x

    def _feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe is not None:
            return self.moe(hidden_states)
        assert self.feed_forward is not None
        return self.feed_forward(hidden_states)

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

        prefix_sum = x
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
        h = self._feed_forward(h)
        prefix_sum = prefix_sum + h
        return prefix_sum, block_residual


class KimiK3Model(BaseModel):
    """Kimi K3: 2.8T MoE model with hybrid KDA/Gated-MLA attention.

    Architecture:
    - 93 layers: 69 KDA + 24 Gated MLA (pattern: 3 KDA + 1 MLA per group of 4)
    - Layer 0 is dense FFN, layers 1-92 are MoE with LatentMoE
    - 896 routed experts (top-16) + 2 shared experts
    - Attention residual blocks collect one residual every 12 layers
    - SiTU-GLU activation throughout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        tok_embeddings: Embedding.Config
        norm: RMSNorm.Config
        lm_head: Linear.Config
        layers: list[KimiK3TransformerBlock.Config]
        output_attn_res: KimiAttentionResidual.Config | None = None
        vocab_size: int = 163840
        dim: int = 7168
        norm_eps: float = 1e-5
        attn_res_block_size: int | None = 12

        @property
        def n_layers(self) -> int:
            return len(self.layers)

        def update_from_config(self, *, config, **kwargs) -> None:
            """Called by Trainer.__init__ to sync runtime params (seq_len, etc.)."""
            from .sharding import set_kimi_sharding_config

            set_kimi_sharding_config(
                self,
                enable_sp=config.parallelism.enable_sequence_parallel,
                enable_ep=config.parallelism.expert_parallel_degree > 1,
            )
            debug_force_load_balance = config.debug.moe_force_load_balance
            for layer in self.layers:
                if layer.moe is not None:
                    layer.moe.router._debug_force_load_balance = debug_force_load_balance

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            """Return (total_params, flops_per_token) for MFU calculation."""
            nparams = sum(p.numel() for p in model.parameters())
            # Approximate: 6 * nparams * seq_len (2 for fwd + 4 for bwd)
            flops_per_token = 6 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        mismatched_layers = [
            layer.layer_id for layer in config.layers if layer.attn_res_block_size != config.attn_res_block_size
        ]
        if mismatched_layers:
            raise ValueError(
                "All Kimi K3 layers must use the model-level "
                "attn_res_block_size; mismatched layer ids: "
                f"{mismatched_layers}"
            )
        self.tok_embeddings = config.tok_embeddings.build()
        self.layers = ModuleDict({str(i): layer_cfg.build() for i, layer_cfg in enumerate(config.layers)})
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None:
            if config.output_attn_res is None:
                raise ValueError("Attention residuals require an output_attn_res configuration")
            self.output_attn_res = config.output_attn_res.build()
        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()

    _skip_lm_head: bool = False
    enable_weight_tying: bool = False

    def build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
        if cp_mesh is not None:
            if load_balancer_type is not None:
                raise ValueError(
                    "Kimi causal convolutions and KDA require contiguous CP intervals; disable load balancing"
                )
            degree, rank = cp_mesh.size(), cp_mesh.get_local_rank()
            if tokens.shape[1] % degree:
                raise ValueError("Kimi sequence length must be divisible by CP degree")
            tokens = tokens.chunk(degree, dim=1)[rank].contiguous()
            labels = labels.chunk(degree, dim=1)[rank].contiguous()
            extra_inputs = {key: value.chunk(degree, dim=1)[rank].contiguous() for key, value in extra_inputs.items()}
        return tokens, labels, extra_inputs

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
        return x if self._skip_lm_head else self.lm_head(x)
