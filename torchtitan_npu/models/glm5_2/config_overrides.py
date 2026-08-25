# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-stable GLM-5.2 architecture overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from torchtitan_npu.models.deepseek_v32 import _make_dsv32_model_config

from .model import GLM5_2ModelNpu

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


def make_indexer_types(n_layers: int, *, freq: int = 4, offset: int = 3) -> list[str]:
    """Return the official GLM-5.2 full/shared IndexShare schedule."""
    return [
        "full" if (max(layer_id - offset + 1, 0) % freq) == 0 else "shared"
        for layer_id in range(n_layers)
    ]


@dataclass(kw_only=True, slots=True)
class GLM52ModelOverrides:
    vocab_size: int = 154880
    dim: int = 6144
    n_layers: int = 78
    n_dense_layers: int = 3
    dense_hidden_dim: int = 12288
    moe_hidden_dim: int = 2048

    n_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    mscale: float = 1.0

    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    indexer_rope_interleave: bool = True
    index_share_for_mtp_iteration: bool = True

    num_experts: int = 256
    num_shared_experts: int = 1
    router_top_k: int = 8
    router_score_func: Literal["sigmoid", "softmax"] = "sigmoid"
    router_num_expert_groups: int | None = None
    router_num_limited_groups: int | None = None
    router_route_scale: float = 2.5
    router_route_norm: bool = True
    score_before_experts: bool = False

    norm_eps: float = 1e-5
    num_mtp_modules: int = 1
    mask_type: str = "causal"
    rope_max_seq_len: int = 1048576
    rope_scaling: Literal["none", "llama", "yarn"] = "none"
    rope_theta: float = 8000000.0
    rope_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_seq_len: int = 1048576

    @classmethod
    def from_model_config(cls, model_config: GLM5_2ModelNpu.Config) -> GLM52ModelOverrides:
        layers = model_config.layers
        n_main_layers = len(layers) - model_config.num_mtp_modules
        if n_main_layers <= 0:
            raise ValueError("GLM-5.2 overrides require at least one main layer")

        main_layers = layers[:n_main_layers]
        dense_flags = [layer.feed_forward is not None for layer in main_layers]
        n_dense_layers = sum(dense_flags)
        if dense_flags != [True] * n_dense_layers + [False] * (n_main_layers - n_dense_layers):
            raise ValueError("GLM-5.2 dense layers must form a prefix")

        dense_layer = next((layer for layer in main_layers if layer.feed_forward is not None), None)
        moe_layer = next((layer for layer in main_layers if layer.moe is not None), None)
        if dense_layer is None or moe_layer is None:
            raise ValueError("GLM-5.2 presets must contain both dense and MoE layers")
        attention = main_layers[0].attention
        moe = moe_layer.moe
        assert moe is not None
        shared_hidden_dim = moe.shared_experts.w1.out_features if moe.shared_experts is not None else 0
        if shared_hidden_dim % moe.experts.hidden_dim != 0:
            raise ValueError("GLM-5.2 shared expert width must divide routed expert width")
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
            index_topk_freq=model_config.index_topk_freq,
            index_skip_topk_offset=model_config.index_skip_topk_offset,
            indexer_rope_interleave=model_config.indexer_rope_interleave,
            index_share_for_mtp_iteration=model_config.index_share_for_mtp_iteration,
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
            rope_scaling=getattr(attention, "rope_scaling", "none"),
            rope_theta=rope.theta,
            rope_factor=rope.rope_factor,
            rope_beta_fast=rope.beta_fast,
            rope_beta_slow=rope.beta_slow,
            rope_original_seq_len=rope.original_seq_len,
        )

    def to_model_config(self) -> GLM5_2ModelNpu.Config:
        validate_model_overrides(self)
        schedule = make_indexer_types(
            self.n_layers,
            freq=self.index_topk_freq,
            offset=self.index_skip_topk_offset,
        )
        base = _make_dsv32_model_config(
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
            indexer_types=schedule,
            indexer_rope_interleave=self.indexer_rope_interleave,
            index_share_for_mtp_iteration=self.index_share_for_mtp_iteration,
            rope_max_seq_len=self.rope_max_seq_len,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            rope_beta_fast=self.rope_beta_fast,
            rope_beta_slow=self.rope_beta_slow,
            rope_original_seq_len=self.rope_original_seq_len,
        )
        values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        return GLM5_2ModelNpu.Config(
            **values,
            index_topk_freq=self.index_topk_freq,
            index_skip_topk_offset=self.index_skip_topk_offset,
            indexer_types=schedule,
            indexer_rope_interleave=self.indexer_rope_interleave,
            index_share_for_mtp_iteration=self.index_share_for_mtp_iteration,
        )


def build_model_spec_with_overrides(model_spec: ModelSpec) -> tuple[ModelSpec, GLM52ModelOverrides]:
    if not isinstance(model_spec.model, GLM5_2ModelNpu.Config):
        raise TypeError(
            "GLM-5.2 model overrides require GLM5_2ModelNpu.Config, "
            f"got {type(model_spec.model).__name__}"
        )
    return model_spec, GLM52ModelOverrides.from_model_config(model_spec.model)


def apply_model_overrides(model_spec: ModelSpec | None, overrides: GLM52ModelOverrides) -> ModelSpec:
    if model_spec is None:
        raise ValueError("GLM-5.2 model overrides require model_spec")
    if not isinstance(model_spec.model, GLM5_2ModelNpu.Config):
        raise TypeError(
            "GLM-5.2 model overrides require GLM5_2ModelNpu.Config, "
            f"got {type(model_spec.model).__name__}"
        )
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: GLM52ModelOverrides) -> None:
    positive_fields = (
        "vocab_size", "dim", "n_layers", "dense_hidden_dim", "moe_hidden_dim",
        "n_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
        "qk_rope_head_dim", "v_head_dim", "mscale", "index_n_heads",
        "index_head_dim", "index_topk", "index_topk_freq", "num_experts",
        "router_top_k", "router_route_scale", "norm_eps", "rope_max_seq_len",
        "rope_theta", "rope_factor", "rope_original_seq_len",
    )
    for name in positive_fields:
        if getattr(config, name) <= 0:
            raise ValueError(f"model_overrides.{name} must be > 0")
    if not 0 <= config.n_dense_layers <= config.n_layers:
        raise ValueError("model_overrides.n_dense_layers must be between 0 and n_layers")
    if config.num_mtp_modules < 0:
        raise ValueError("model_overrides.num_mtp_modules must be >= 0")
    if config.router_top_k > config.num_experts:
        raise ValueError("model_overrides.router_top_k must be <= num_experts")
    if config.index_skip_topk_offset < 1:
        raise ValueError("model_overrides.index_skip_topk_offset must be >= 1")
    if config.qk_rope_head_dim > config.index_head_dim:
        raise ValueError("model_overrides.qk_rope_head_dim must be <= index_head_dim")
    if config.index_head_dim & (config.index_head_dim - 1):
        raise ValueError("model_overrides.index_head_dim must be a power of two")


__all__ = [
    "GLM52ModelOverrides",
    "apply_model_overrides",
    "build_model_spec_with_overrides",
    "make_indexer_types",
    "validate_model_overrides",
]
