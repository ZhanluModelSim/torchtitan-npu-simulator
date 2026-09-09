# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible ar_llm model configuration overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .model import ArLlmModel

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class ArLlmModelOverrides:
    """Flat mirror of ArLlmModel.Config used for ``--model-overrides.*``."""

    vocab_size: int = 524288
    dim: int = 16384
    n_layers: int = 60
    n_heads: int = 128
    head_dim: int = 256
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 4096
    kv_lora_rank: int = 1024
    o_groups: int = 32
    o_lora_rank: int = 4096

    unit_size: int = 6
    csa_per_unit: int = 1
    hca_per_unit: int = 1

    max_seq_len: int = 4096
    rope_theta: float = 16384.0
    yarn_factor: float = 32.0
    yarn_original_max: int = 65536

    kda_d_state: int = 256
    kda_d_k: int = 128
    kda_d_v: int = 128

    csa_compress_ratio: int = 16
    csa_window_size: int = 1024

    hca_compress_ratio: int = 256
    indexer_n_heads: int = 64
    indexer_head_dim: int = 128
    indexer_topk: int = 4096

    num_routed_experts: int = 2048
    num_shared_experts: int = 2
    moe_intermediate_size: int = 4096
    moe_latent_dim: int = 7168
    num_experts_per_token: int = 16
    router_score_function: str = "sqrtsoftplus"
    route_scale: float = 2.5
    mor_expert_ratio: float = 0.05
    mor_expert_capacity: float = 1.0
    debug_force_load_balance: bool = False

    hc_mult: int = 4
    sinkhorn_iters: int = 20
    use_attn_sink: bool = True

    engram_layers: list[int] = field(default_factory=list)
    engram_ngram_orders: list[int] = field(default_factory=lambda: [2, 3])
    engram_num_hash_heads: int = 8
    engram_table_capacity: int = 8_388_608
    engram_memory_dim: int = 4096

    swiglu_clamp: float = 10.0
    attn_softmax_clamp: float = 50.0
    norm_eps: float = 1e-6

    @classmethod
    def from_model_config(cls, model_config: ArLlmModel.Config) -> ArLlmModelOverrides:
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
            raise ValueError(
                f"ArLlmModel.Config is missing override fields: {missing}"
            )
        return cls(**values)

    def to_model_config(self) -> ArLlmModel.Config:
        values = {
            field.name: copy.deepcopy(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        model_config = ArLlmModel.Config(**values)
        model_config.validate()
        return model_config


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, ArLlmModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, ArLlmModel.Config):
        raise TypeError(
            "ar_llm model overrides require ArLlmModel.Config, "
            f"got {type(model_config).__name__}"
        )
    return model_spec, ArLlmModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: ArLlmModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("ar_llm model overrides require model_spec")
    if not isinstance(model_spec.model, ArLlmModel.Config):
        raise TypeError(
            "ar_llm model overrides require ArLlmModel.Config, "
            f"got {type(model_spec.model).__name__}"
        )
    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: ArLlmModelOverrides) -> None:
    positive_int_fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "o_groups",
        "o_lora_rank",
        "unit_size",
        "csa_per_unit",
        "hca_per_unit",
        "max_seq_len",
        "yarn_original_max",
        "kda_d_state",
        "kda_d_k",
        "kda_d_v",
        "csa_compress_ratio",
        "csa_window_size",
        "hca_compress_ratio",
        "indexer_n_heads",
        "indexer_head_dim",
        "indexer_topk",
        "num_routed_experts",
        "num_shared_experts",
        "moe_intermediate_size",
        "moe_latent_dim",
        "num_experts_per_token",
        "hc_mult",
        "sinkhorn_iters",
        "engram_num_hash_heads",
        "engram_table_capacity",
        "engram_memory_dim",
    )
    for name in positive_int_fields:
        _require_positive(name, getattr(config, name))

    positive_float_fields = ("rope_theta", "yarn_factor", "route_scale", "norm_eps")
    for name in positive_float_fields:
        _require_positive(name, getattr(config, name))

    if not 0 < config.mor_expert_ratio < 1:
        raise ValueError(
            "model_overrides.mor_expert_ratio must be in (0, 1), "
            f"got {config.mor_expert_ratio}"
        )
    if not 0 < config.mor_expert_capacity:
        raise ValueError(
            f"model_overrides.mor_expert_capacity must be > 0, got {config.mor_expert_capacity}"
        )
    if config.num_experts_per_token > config.num_routed_experts:
        raise ValueError(
            "model_overrides.num_experts_per_token must be <= num_routed_experts, "
            f"got num_experts_per_token={config.num_experts_per_token}, "
            f"num_routed_experts={config.num_routed_experts}"
        )
    # Architecture-level divisibility and consistency rules are enforced by
    # ArLlmModel.Config.validate() during to_model_config().
    config.to_model_config()


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")
