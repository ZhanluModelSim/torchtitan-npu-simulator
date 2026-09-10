# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible glm5_next model configuration overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .model import Glm5NextModel, GlmVisionTower

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class Glm5NextTextOverrides:
    """Flat mirror of Glm5NextTextModel.Config for ``--model-overrides``."""

    vocab_size: int = 154880
    hidden_size: int = 24576
    num_hidden_layers: int = 96
    pre_layers: int = 16
    looped_layers: int = 56
    post_layers: int = 24
    loop_train_steps: int = 4
    share_loop_weights: bool = True

    num_attention_heads: int = 192
    kda_num_heads: int = 192
    kda_head_dim: int = 128
    kda_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0
    q_lora_rank: int = 6144
    kv_lora_rank: int = 2048
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    qk_rope_head_dim: int = 0
    indexer_n_heads: int = 64
    indexer_head_dim: int = 128
    index_topk: int = 8192
    index_kpool: int = 8
    index_kpool_always_select_tail: bool = True

    first_k_dense_replace: int = 4
    intermediate_size: int = 73728
    n_routed_experts: int = 2048
    num_experts_per_tok: int = 16
    n_shared_experts: int = 1
    moe_intermediate_size: int = 3072
    routed_scaling_factor: float = 2.5

    norm_eps: float = 1e-5
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    swiglu_limit: float = 10.0

    debug_force_load_balance: bool = False
    max_seq_len: int = 4096

    @classmethod
    def from_model_config(cls, model_config: "Glm5NextModel.Config") -> Glm5NextTextOverrides:
        values = {
            field.name: copy.deepcopy(getattr(model_config, field.name))
            for field in dataclasses.fields(cls)
            if hasattr(model_config, field.name)
        }
        missing = [
            field.name
            for field in dataclasses.fields(cls)
            if not hasattr(model_config, field.name)
        ]
        if missing:
            raise ValueError(f"Glm5NextTextModel.Config is missing override fields: {missing}")
        return cls(**values)

    def to_text_fields(self) -> dict:
        return {
            field.name: copy.deepcopy(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


@dataclass(kw_only=True, slots=True)
class Glm5NextVisionOverrides:
    """Flat mirror of GlmVisionTower.Config."""

    depth: int = 32
    hidden_size: int = 2048
    num_heads: int = 32
    intermediate_size: int = 8192
    out_hidden_size: int = 24576
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    in_channels: int = 3
    image_size: int = 672
    projection_intermediate_size: int = 49152
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_bias: bool = True

    @classmethod
    def from_model_config(cls, model_config: GlmVisionTower.Config) -> Glm5NextVisionOverrides:
        values = {
            field.name: copy.deepcopy(getattr(model_config, field.name))
            for field in dataclasses.fields(cls)
            if hasattr(model_config, field.name)
        }
        missing = [
            field.name
            for field in dataclasses.fields(cls)
            if not hasattr(model_config, field.name)
        ]
        if missing:
            raise ValueError(f"GlmVisionTower.Config is missing override fields: {missing}")
        return cls(**values)

    def to_model_config(self) -> GlmVisionTower.Config:
        values = {field.name: copy.deepcopy(getattr(self, field.name)) for field in dataclasses.fields(self)}
        return GlmVisionTower.Config(**values)


@dataclass(kw_only=True, slots=True)
class Glm5NextModelOverrides:
    """Top-level overrides: text + vision sub-configs and fusion token ids."""

    text_config: Glm5NextTextOverrides = field(default_factory=Glm5NextTextOverrides)
    vision_config: Glm5NextVisionOverrides = field(default_factory=Glm5NextVisionOverrides)
    image_token_id: int = 154854
    video_start_token_id: int = 154832

    @classmethod
    def from_model_config(cls, model_config: Glm5NextModel.Config) -> Glm5NextModelOverrides:
        return cls(
            text_config=Glm5NextTextOverrides.from_model_config(model_config),
            vision_config=Glm5NextVisionOverrides.from_model_config(model_config.vision_config),
            image_token_id=model_config.image_token_id,
            video_start_token_id=model_config.video_start_token_id,
        )

    def to_model_config(self) -> Glm5NextModel.Config:
        model_config = Glm5NextModel.Config(
            vision_config=self.vision_config.to_model_config(),
            image_token_id=self.image_token_id,
            video_start_token_id=self.video_start_token_id,
            **self.text_config.to_text_fields(),
        )
        model_config.validate()
        return model_config


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, Glm5NextModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, Glm5NextModel.Config):
        raise TypeError(
            "glm5_next model overrides require Glm5NextModel.Config, "
            f"got {type(model_config).__name__}"
        )
    return model_spec, Glm5NextModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: Glm5NextModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("glm5_next model overrides require model_spec")
    if not isinstance(model_spec.model, Glm5NextModel.Config):
        raise TypeError(
            "glm5_next model overrides require Glm5NextModel.Config, "
            f"got {type(model_spec.model).__name__}"
        )
    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: Glm5NextModelOverrides) -> None:
    # Architecture-level divisibility and consistency rules are enforced by
    # Glm5NextModel.Config.validate() during to_model_config().
    config.to_model_config()
    positive_int_fields = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "pre_layers",
        "looped_layers",
        "post_layers",
        "num_attention_heads",
        "kda_num_heads",
        "kda_head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "v_head_dim",
        "indexer_n_heads",
        "indexer_head_dim",
        "index_topk",
        "index_kpool",
        "first_k_dense_replace",
        "intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        "moe_intermediate_size",
        "hc_mult",
        "hc_sinkhorn_iters",
    )
    for name in positive_int_fields:
        _require_positive(f"text_config.{name}", getattr(config.text_config, name))

    vision_positive = (
        "depth",
        "hidden_size",
        "num_heads",
        "intermediate_size",
        "out_hidden_size",
        "patch_size",
        "spatial_merge_size",
        "image_size",
        "projection_intermediate_size",
    )
    for name in vision_positive:
        _require_positive(f"vision_config.{name}", getattr(config.vision_config, name))

    if not 1 <= config.text_config.loop_train_steps <= 4:
        raise ValueError(
            "model_overrides.text_config.loop_train_steps must be within [1, 4]; "
            "adaptive halting is not modeled"
        )
    if config.text_config.num_experts_per_tok > config.text_config.n_routed_experts:
        raise ValueError("num_experts_per_tok must be <= n_routed_experts")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")
