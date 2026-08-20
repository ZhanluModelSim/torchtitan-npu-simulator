# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible DeepSeek V3.2 model configuration overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torchtitan.models.common.attention import FlexAttention

from .model import DeepSeekV32ModelNpu

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class DeepSeekV32ModelOverrides:
    """Stable architecture inputs used to rebuild V3.2 per-layer configs."""

    vocab_size: int = 129280
    dim: int = 7168
    n_layers: int = 61
    n_dense_layers: int = 3
    dense_hidden_dim: int = 18432
    moe_hidden_dim: int = 2048

    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mscale: float = 1.0

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    enable_mla_absorb: bool = True

    num_experts: int = 256
    num_shared_experts: int = 1
    router_top_k: int = 8
    router_score_func: Literal["sigmoid", "softmax"] = "sigmoid"
    router_num_expert_groups: int | None = None
    router_num_limited_groups: int | None = None
    router_route_scale: float = 1.0
    router_route_norm: bool = True
    score_before_experts: bool = False

    norm_eps: float = 1e-6
    num_mtp_modules: int = 0
    mask_type: str = "causal"
    rope_max_seq_len: int = 16384
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_seq_len: int = 4096

    @classmethod
    def from_model_config(
        cls,
        model_config: DeepSeekV32ModelNpu.Config,
    ) -> DeepSeekV32ModelOverrides:
        layers = model_config.layers
        n_main_layers = len(layers) - model_config.num_mtp_modules
        if n_main_layers <= 0:
            raise ValueError("DeepSeek V3.2 model overrides require at least one main layer")

        main_layers = layers[:n_main_layers]
        dense_flags = [layer.feed_forward is not None for layer in main_layers]
        n_dense_layers = sum(dense_flags)
        if dense_flags != [True] * n_dense_layers + [False] * (n_main_layers - n_dense_layers):
            raise ValueError("DeepSeek V3.2 model overrides require dense layers to form a prefix")

        dense_layer = next(
            (layer for layer in main_layers if layer.feed_forward is not None),
            None,
        )
        moe_layer = next(
            (layer for layer in main_layers if layer.moe is not None),
            None,
        )
        if dense_layer is None or moe_layer is None:
            raise ValueError("DeepSeek V3.2 presets must contain both dense and MoE layers")

        attention = layers[0].attention
        _validate_uniform_architecture(layers)
        moe = moe_layer.moe
        assert moe is not None
        shared_hidden_dim = moe.shared_experts.w1.out_features if moe.shared_experts is not None else 0
        if shared_hidden_dim % moe.experts.hidden_dim != 0:
            raise ValueError("DeepSeek V3.2 shared expert width must be divisible by the routed expert width")

        inner_attention = attention.inner_attention
        if attention.mask_type == "block_causal" and not isinstance(inner_attention, FlexAttention.Config):
            raise ValueError("DeepSeek V3.2 block_causal presets require FlexAttention.Config")

        rope = model_config.rope
        return cls(
            vocab_size=model_config.vocab_size,
            dim=model_config.dim,
            n_layers=n_main_layers,
            n_dense_layers=n_dense_layers,
            dense_hidden_dim=dense_layer.feed_forward.w1.out_features,
            moe_hidden_dim=moe.experts.hidden_dim,
            n_heads=attention.n_heads,
            q_lora_rank=attention.q_lora_rank,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            mscale=attention.mscale,
            index_n_heads=attention.index_n_heads,
            index_head_dim=attention.index_head_dim,
            index_topk=attention.index_topk,
            enable_mla_absorb=attention.enable_mla_absorb,
            num_experts=moe.num_experts,
            num_shared_experts=shared_hidden_dim // moe.experts.hidden_dim,
            router_top_k=moe.router.top_k,
            router_score_func=moe.router.score_func,
            router_num_expert_groups=moe.router.num_expert_groups,
            router_num_limited_groups=moe.router.num_limited_groups,
            router_route_scale=moe.router.route_scale,
            router_route_norm=moe.router.route_norm,
            score_before_experts=moe.score_before_experts,
            norm_eps=attention.q_norm.eps,
            num_mtp_modules=model_config.num_mtp_modules,
            mask_type=attention.mask_type,
            rope_max_seq_len=rope.max_seq_len,
            rope_theta=rope.theta,
            rope_factor=rope.rope_factor,
            rope_beta_fast=rope.beta_fast,
            rope_beta_slow=rope.beta_slow,
            rope_original_seq_len=rope.original_seq_len,
        )

    def to_model_config(self) -> DeepSeekV32ModelNpu.Config:
        from . import _make_dsv32_model_config

        return _make_dsv32_model_config(
            vocab_size=self.vocab_size,
            dim=self.dim,
            inter_dim=self.dense_hidden_dim,
            moe_inter_dim=self.moe_hidden_dim,
            n_layers=self.n_layers,
            n_dense_layers=self.n_dense_layers,
            n_heads=self.n_heads,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            mscale=self.mscale,
            norm_eps=self.norm_eps,
            num_mtp_modules=self.num_mtp_modules,
            mask_type=self.mask_type,
            router_top_k=self.router_top_k,
            router_score_func=self.router_score_func,
            router_num_expert_groups=self.router_num_expert_groups,
            router_num_limited_groups=self.router_num_limited_groups,
            router_route_scale=self.router_route_scale,
            router_route_norm=self.router_route_norm,
            score_before_experts=self.score_before_experts,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            enable_mla_absorb=self.enable_mla_absorb,
            rope_max_seq_len=self.rope_max_seq_len,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            rope_beta_fast=self.rope_beta_fast,
            rope_beta_slow=self.rope_beta_slow,
            rope_original_seq_len=self.rope_original_seq_len,
        )


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, DeepSeekV32ModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, DeepSeekV32ModelNpu.Config):
        raise TypeError(
            f"DeepSeek V3.2 model overrides require DeepSeekV32ModelNpu.Config, got {type(model_config).__name__}"
        )
    return model_spec, DeepSeekV32ModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: DeepSeekV32ModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("DeepSeek V3.2 model overrides require model_spec")
    if not isinstance(model_spec.model, DeepSeekV32ModelNpu.Config):
        raise TypeError(
            f"DeepSeek V3.2 model overrides require DeepSeekV32ModelNpu.Config, got {type(model_spec.model).__name__}"
        )

    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: DeepSeekV32ModelOverrides) -> None:
    positive_fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "dense_hidden_dim",
        "moe_hidden_dim",
        "n_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mscale",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "num_experts",
        "router_top_k",
        "router_route_scale",
        "norm_eps",
        "rope_max_seq_len",
        "rope_theta",
        "rope_factor",
        "rope_beta_fast",
        "rope_original_seq_len",
    )
    for name in positive_fields:
        _require_positive(name, getattr(config, name))

    if not 0 <= config.n_dense_layers <= config.n_layers:
        raise ValueError(
            "model_overrides.n_dense_layers must be between 0 and n_layers, "
            f"got n_dense_layers={config.n_dense_layers}, "
            f"n_layers={config.n_layers}"
        )
    if config.num_mtp_modules < 0:
        raise ValueError(f"model_overrides.num_mtp_modules must be >= 0, got {config.num_mtp_modules}")
    if config.num_shared_experts < 0:
        raise ValueError(f"model_overrides.num_shared_experts must be >= 0, got {config.num_shared_experts}")
    if config.router_top_k > config.num_experts:
        raise ValueError(
            "model_overrides.router_top_k must be <= num_experts, "
            f"got router_top_k={config.router_top_k}, "
            f"num_experts={config.num_experts}"
        )
    if config.router_num_expert_groups is not None:
        _require_positive("router_num_expert_groups", config.router_num_expert_groups)
        if config.num_experts % config.router_num_expert_groups != 0:
            raise ValueError(
                "model_overrides.router_num_expert_groups must divide "
                f"num_experts, got {config.router_num_expert_groups} and "
                f"{config.num_experts}"
            )
    if config.router_num_limited_groups is not None:
        _require_positive("router_num_limited_groups", config.router_num_limited_groups)
        if (
            config.router_num_expert_groups is not None
            and config.router_num_limited_groups > config.router_num_expert_groups
        ):
            raise ValueError("model_overrides.router_num_limited_groups must be <= router_num_expert_groups")
    if config.rope_beta_slow < 0:
        raise ValueError(f"model_overrides.rope_beta_slow must be >= 0, got {config.rope_beta_slow}")
    if config.rope_beta_fast <= config.rope_beta_slow:
        raise ValueError("model_overrides.rope_beta_fast must be greater than rope_beta_slow")
    if config.qk_rope_head_dim > config.index_head_dim:
        raise ValueError("model_overrides.qk_rope_head_dim must be <= index_head_dim")
    if not _is_power_of_two(config.index_head_dim):
        raise ValueError(
            "model_overrides.index_head_dim must be a power of two for the "
            f"Hadamard transform, got {config.index_head_dim}"
        )


def _validate_uniform_architecture(layers: list) -> None:
    first = layers[0].attention
    fields = (
        "dim",
        "n_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mscale",
        "mask_type",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "enable_mla_absorb",
    )
    for layer in layers[1:]:
        attention = layer.attention
        if any(getattr(attention, name) != getattr(first, name) for name in fields):
            raise ValueError("DeepSeek V3.2 model overrides require uniform attention configs")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
