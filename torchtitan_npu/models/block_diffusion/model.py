# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native TorchTitan model for block-diffusion meta training."""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

RAW_CONTRACT_VERSION = 1


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
        self.feed_forward = config.feed_forward.build() if config.feed_forward is not None else None
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
        if self.moe is not None:
            ffn_output = self.moe(ffn_input)
        else:
            assert self.feed_forward is not None
            ffn_output = self.feed_forward(ffn_input)
        return h + ffn_output


class BlockDiffusionModel(Decoder):
    """Block-diffusion denoiser used by the training/meta-simulator path.

    Exact mode manually follows raw contract version 1, including split-half
    unscaled RoPE and exact block visibility. Legacy compute_proxy mode retains
    its full-sequence causal forward and simulator-only FLOP scaling. Neither
    mode imports or patches the raw reference at runtime.
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
        attention_compute_alpha: float = 1.0
        attention_mode: str = "compute_proxy"
        attention_policy: str = "block_causal"
        raw_contract_version: int = RAW_CONTRACT_VERSION
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
            if not math.isfinite(self.attention_compute_alpha) or self.attention_compute_alpha <= 0:
                raise ValueError("attention_compute_alpha must be finite and greater than zero")
            if self.attention_mode not in ("exact", "compute_proxy"):
                raise ValueError("attention_mode must be exact or compute_proxy")
            if self.attention_policy not in ("block_causal", "prefix_causal_canvas_bidirectional"):
                raise ValueError("unsupported attention_policy")
            if self.attention_mode == "exact":
                if self.attention_compute_alpha != 1.0:
                    raise ValueError("exact attention requires attention_compute_alpha=1")
                if type(self.raw_contract_version) is not int or self.raw_contract_version != RAW_CONTRACT_VERSION:
                    raise ValueError("unsupported exact raw_contract_version")
                if self.head_dim % 2:
                    raise ValueError("exact split-half RoPE requires an even head_dim")
                if (
                    self.rope.backend != "cos_sin"
                    or self.rope.scaling != "none"
                    or any(layer.attention.rope_backend != "cos_sin" for layer in self.layers)
                ):
                    raise ValueError(
                        "exact mode requires cos_sin RoPE with scaling=none; use an exact flavor or builder"
                    )
                for layer in self.layers:
                    if layer.moe is None:
                        continue
                    router = layer.moe.router
                    if (
                        router.score_func != "softmax"
                        or router.route_norm is not True
                        or router.route_scale != 1.0
                        or router.num_expert_groups is not None
                        or router.num_limited_groups is not None
                        or layer.moe.score_before_experts is not False
                    ):
                        raise ValueError(
                            "exact routing requires softmax, route_norm=True, route_scale=1, "
                            "no expert groups, and score_before_experts=False"
                        )
                    if layer.moe.load_balance_coeff is not None:
                        raise ValueError("exact routing does not support auxiliary expert-bias updates")
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
                    raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            del kwargs
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            if training.seq_len < self.block_size or training.seq_len % self.block_size != 0:
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

            if self.attention_mode == "exact" and cp > 1:
                raise ValueError("exact block attention does not yet support context parallelism")
            if self.n_heads % tp != 0 or self.n_kv_heads % tp != 0:
                raise ValueError("tensor_parallel_degree must divide both n_heads and n_kv_heads")
            if training.seq_len % cp != 0:
                raise ValueError("context_parallel_degree must divide seq_len")
            if self.n_heads % cp != 0 or self.n_kv_heads % cp != 0:
                raise ValueError("context_parallel_degree must divide both n_heads and n_kv_heads for Ulysses CP")
            if self.num_experts:
                if self.num_experts % ep != 0:
                    raise ValueError("expert_parallel_degree must divide num_experts")
                if self.moe_intermediate_size % etp != 0:
                    raise ValueError("expert_tensor_parallel_degree must divide moe_intermediate_size")
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
            # Trainer consumes FLOPs PER TOKEN for its conventional MFU metric.
            # Use active MoE parameters and the framework's dense-attention
            # estimate; the exact/proxy operator ledger remains authoritative
            # for phase-specific work and mask-aware kernels.
            _, per_token = get_moe_model_nparams_and_flops(self, model, self.n_heads, 2 * self.head_dim, seq_len)
            if self.enable_weight_tying:
                # named_parameters deduplicates the tied output weight under
                # tok_embeddings; the helper excludes embeddings but the output
                # projection still performs a full vocabulary matmul.
                per_token += 6 * self.vocab_size * self.dim
            nparams = sum(parameter.numel() for parameter in model.parameters())
            return nparams, int(per_token)

    def __init__(self, config: Config):
        config.__post_init__()
        # CLI overrides affect top-level semantics after nested flavor configs
        # have been built. Synchronize a deep copy, preserving caller-owned trees.
        config = deepcopy(config)
        exact = config.attention_mode == "exact"
        # Exact RoPE was validated above. Preserve explicit proxy overrides
        # rather than replacing caller choices with the flavor defaults.
        for layer in config.layers:
            layer.attention.rope_backend = config.rope.backend
            inner = layer.attention.inner_attention
            inner.compute_alpha = config.attention_compute_alpha
            inner.attention_mode = config.attention_mode
            inner.attention_policy = config.attention_policy
            inner.block_size = config.block_size
            if layer.moe is not None:
                layer.moe.exact_routing = exact
                if exact:
                    layer.moe.load_balance_coeff = None
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
