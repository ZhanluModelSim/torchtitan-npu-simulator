# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next model: hybrid Gated DeltaNet / gated attention MoE decoder.

Architecture (Qwen3-Next-80B-A3B):
- 48 layers alternating 3 linear-attention (Gated DeltaNet) + 1 full
  attention (GQA with output gate) per group of 4.
- Every layer except the first is MoE (512 experts, top-10) with a gated
  shared expert.
- RMSNorm (1 + weight centered in HF layout, ones here) throughout.

Reference: MindSpeed-LLM
mindspeed_llm/fsdp2/models/qwen3_next/{qwen3_next,modeling_qwen3_next}.py
"""

import logging
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module, ModuleDict

from .attention import GatedAttention, RMSNormGated
from .feed_forward import Qwen3NextMoE
from .gated_deltanet import GatedDeltaNet

logger = logging.getLogger(__name__)


class Qwen3NextTransformerBlock(Module):
    """Single decoder layer: token mixer (GDN or gated attention) + FFN/MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 2048
        layer_id: int = 0
        token_mixer: GatedDeltaNet.Config | GatedAttention.Config = field(
            default_factory=GatedDeltaNet.Config
        )
        feed_forward: FeedForward.Config | None = None
        moe: Qwen3NextMoE.Config | None = None
        norm_eps: float = 1e-6

        @property
        def attention(self):
            """Alias so upstream code that pokes ``layer.attention`` (mask
            creation, SDPA rewrites) sees the token mixer config; its missing
            ``inner_attention`` attribute makes those paths a no-op."""
            return self.token_mixer

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.is_linear_attention = isinstance(config.token_mixer, GatedDeltaNet.Config)

        if isinstance(config.token_mixer, GatedDeltaNet.Config):
            self.token_mixer = GatedDeltaNet(config.token_mixer)
        else:
            self.token_mixer = GatedAttention(config.token_mixer)

        self.attention_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()
        self.ffn_norm = RMSNorm.Config(
            normalized_shape=config.dim,
            eps=config.norm_eps,
        ).build()

        if config.moe is not None:
            self.moe = Qwen3NextMoE(config.moe)
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
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm token mixer. Gated DeltaNet ignores RoPE; gated attention
        # applies the shared cos/sin rope cache.
        h = self.attention_norm(x)
        if self.is_linear_attention:
            h = self.token_mixer(h)
        else:
            h = self.token_mixer(h, freqs_cis, positions)
        x = x + h

        # Pre-norm FFN / MoE.
        h = self.ffn_norm(x)
        if self.moe_enabled:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        x = x + h
        return x


class Qwen3NextModel(Decoder):
    """Qwen3-Next decoder with hybrid linear/full attention and MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 151936

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug

            # Sync rope cache length with the training sequence length.
            if training.seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {training.seq_len} exceeds original "
                    f"maximum {self.rope.max_seq_len}."
                )
            self.rope = replace(self.rope, max_seq_len=training.seq_len)

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Qwen3-Next context parallelism is not implemented yet."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, float]:
            nparams = sum(p.numel() for p in model.parameters())
            flops_per_token = 6.0 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)

    def verify_module_protocol(self) -> None:
        """Custom modules (GDN/MoE) are plain nn.Module; skip strict checks."""
        pass

    def init_weights(self, *, buffer_device: torch.device | None = None) -> None:
        """Initialize weights. Called by Trainer after parallelize/to_empty.

        Skips ``init_states`` (which would trigger ``reset_parameters`` and
        its fill_-based kernels) and instead (1) rebuilds the RoPE cache on a
        real device — ``to_empty`` wipes the ``freqs_cis`` buffer the meta
        build created — and (2) initializes every parameter directly.
        """
        # Rebuild the rope cos/sin cache; without this the buffer stays
        # garbage after to_empty and full-attention layers apply wrong
        # rotations on real hardware.
        device = buffer_device if buffer_device is not None else self.freqs_cis.device
        with torch.device(device):
            rope = self.config.rope.build()
        self.freqs_cis = rope.cache

        del buffer_device
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RMSNorm, RMSNormGated)):
                nn.init.ones_(module.weight)

        for layer in self.layers.values():
            mixer = layer.token_mixer
            if isinstance(mixer, GatedDeltaNet):
                nn.init.ones_(mixer.dt_bias)
                with torch.no_grad():
                    mixer.A_log.uniform_(0, 16).log_()
            if layer.moe is not None:
                nn.init.normal_(
                    layer.moe.experts.gate_up_proj, mean=0.0, std=init_std
                )
                nn.init.normal_(layer.moe.experts.down_proj, mean=0.0, std=init_std)
