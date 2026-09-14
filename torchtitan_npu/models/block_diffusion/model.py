# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native TorchTitan model for block-diffusion meta training."""

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.protocols.module import Module


class BlockDiffusionTransformerBlock(TransformerBlock):
    """Pre-norm GQA block with either dense SwiGLU or token-choice MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
        if self.feed_forward is None and self.moe is None:
            raise ValueError("BlockDiffusion layer requires feed_forward or moe")
        if self.feed_forward is not None and self.moe is not None:
            raise ValueError("BlockDiffusion layer cannot contain both feed_forward and moe")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            attention_masks,
            positions,
        )
        ffn_input = self.ffn_norm(h)
        ffn_output = (
            self.moe(ffn_input)
            if self.moe is not None
            else self.feed_forward(ffn_input)
        )
        return h + ffn_output


class BlockDiffusionModel(Decoder):
    """Block-diffusion denoiser used by the training/meta-simulator path.

    One call performs a bidirectional denoising pass over the supplied block.
    The iterative inference scheduler from ``simulator/raw_model`` is not part
    of the training contract and is intentionally kept outside this model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        n_layers: int = 1
        n_heads: int = 1
        n_kv_heads: int = 1
        head_dim: int = 64
        intermediate_size: int = 256
        num_experts: int = 0
        moe_intermediate_size: int = 0
        num_experts_per_tok: int = 1
        block_size: int = 256
        mask_token_id: int = 100
        max_denoise_steps: int = 48
        enable_weight_tying: bool = False

        def __post_init__(self) -> None:
            if self.n_layers <= 0:
                raise ValueError("n_layers must be greater than zero")
            if self.dim <= 0 or self.head_dim <= 0:
                raise ValueError("dim and head_dim must be greater than zero")
            if self.n_heads <= 0 or self.n_kv_heads <= 0:
                raise ValueError("n_heads and n_kv_heads must be greater than zero")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_kv_heads must divide n_heads for GQA")
            if self.block_size <= 0 or self.max_denoise_steps <= 0:
                raise ValueError("block_size and max_denoise_steps must be positive")
            if not 0 <= self.mask_token_id < self.vocab_size:
                raise ValueError("mask_token_id must be inside the vocabulary")
            if self.num_experts < 0:
                raise ValueError("num_experts cannot be negative")
            if self.num_experts == 0:
                if self.intermediate_size <= 0:
                    raise ValueError("intermediate_size must be positive for dense FFN")
            else:
                if self.moe_intermediate_size <= 0:
                    raise ValueError("moe_intermediate_size must be positive for MoE")
                if not 1 <= self.num_experts_per_tok <= self.num_experts:
                    raise ValueError(
                        "num_experts_per_tok must be between 1 and num_experts"
                    )

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            del kwargs
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            if (
                training.seq_len < self.block_size
                or training.seq_len % self.block_size != 0
            ):
                raise ValueError(
                    "Block Diffusion expects a fixed prefix plus one final canvas: "
                    f"seq_len={training.seq_len} must be a positive multiple of "
                    f"block_size={self.block_size}"
                )

            tp = parallelism.tensor_parallel_degree
            cp = parallelism.context_parallel_degree
            ep = parallelism.expert_parallel_degree
            etp = parallelism.expert_tensor_parallel_degree
            for value, name in (
                (tp, "tensor_parallel_degree"),
                (cp, "context_parallel_degree"),
                (ep, "expert_parallel_degree"),
                (etp, "expert_tensor_parallel_degree"),
            ):
                if value < 1:
                    raise ValueError(f"{name} must be at least one, got {value}")

            if self.n_heads % tp != 0 or self.n_kv_heads % tp != 0:
                raise ValueError(
                    "tensor_parallel_degree must divide both n_heads and n_kv_heads"
                )
            if training.seq_len % cp != 0:
                raise ValueError("context_parallel_degree must divide seq_len")
            if self.n_heads % cp != 0 or self.n_kv_heads % cp != 0:
                raise ValueError(
                    "context_parallel_degree must divide both n_heads and "
                    "n_kv_heads for Ulysses CP"
                )
            if self.num_experts:
                if self.num_experts % ep != 0:
                    raise ValueError("expert_parallel_degree must divide num_experts")
                if self.moe_intermediate_size % etp != 0:
                    raise ValueError(
                        "expert_tensor_parallel_degree must divide moe_intermediate_size"
                    )
                if ep > 1 and etp > 1:
                    raise ValueError(
                        "simultaneous expert_parallel_degree > 1 and "
                        "expert_tensor_parallel_degree > 1 is not supported by "
                        "the current TorchTitan ExpertParallel plan"
                    )

        def get_nparams_and_flops(
            self,
            model: Module,
            seq_len: int,
        ) -> tuple[int, int]:
            nparams = sum(parameter.numel() for parameter in model.parameters())
            # Meta acceptance uses an operator ledger for exact MoE work.  This
            # conservative framework metric keeps the standard training API.
            return nparams, int(6 * nparams * seq_len)

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight
        super().init_states(buffer_device=buffer_device)

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if buffer_device is not None and buffer_device.type == "meta":
            # SimulationTrainer intentionally keeps every tensor on meta.  The
            # upstream Decoder rejects meta only because real training expects
            # RoPE buffers to be materialized after ``to_empty``; shape-only
            # simulation must retain the already-built meta cache instead.
            if self.rope is not None:
                self.freqs_cis = self.rope.cache
            else:
                with torch.device("meta"):
                    rope = self.config.rope.build()
                self.freqs_cis = rope.cache
            return
        super()._init_self_buffers(buffer_device=buffer_device)
