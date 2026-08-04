# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible DeepSeek-V4 model configuration overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .model import DeepSeekV4Model

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class DeepSeekV4ModelOverrides(DeepSeekV4Model.Config):
    """A complete, editable copy of the selected DeepSeek-V4 model preset."""

    @classmethod
    def from_model_config(
        cls,
        model_config: DeepSeekV4Model.Config,
    ) -> DeepSeekV4ModelOverrides:
        values = {
            field.name: copy.deepcopy(getattr(model_config, field.name))
            for field in dataclasses.fields(DeepSeekV4Model.Config)
        }
        return cls(**values)

    def to_model_config(self) -> DeepSeekV4Model.Config:
        values = {
            field.name: copy.deepcopy(getattr(self, field.name))
            for field in dataclasses.fields(DeepSeekV4Model.Config)
        }
        return DeepSeekV4Model.Config(**values)


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, DeepSeekV4ModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, DeepSeekV4Model.Config):
        raise TypeError(
            "DeepSeekV4 model overrides require DeepSeekV4Model.Config, "
            f"got {type(model_config).__name__}"
        )
    return model_spec, DeepSeekV4ModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: DeepSeekV4ModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("DeepSeekV4 model overrides require model_spec")
    if not isinstance(model_spec.model, DeepSeekV4Model.Config):
        raise TypeError(
            "DeepSeekV4 model overrides require DeepSeekV4Model.Config, "
            f"got {type(model_spec.model).__name__}"
        )

    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: DeepSeekV4ModelOverrides) -> None:
    positive_int_fields = (
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "dim",
        "rope_head_dim",
        "q_lora_rank",
        "max_batch_size",
        "max_seq_len",
        "n_heads",
        "o_lora_rank",
        "head_dim",
        "o_groups",
        "window_size",
        "hc_sinkhorn_iters",
        "hc_mult",
        "vocab_size",
        "moe_inter_dim",
        "original_seq_len",
        "rope_theta",
        "rope_factor",
        "beta_fast",
        "n_layers",
    )
    for name in positive_int_fields:
        _require_positive(name, getattr(config, name))

    positive_float_fields = (
        "norm_eps",
        "hc_eps",
        "compress_rope_theta",
    )
    for name in positive_float_fields:
        _require_positive(name, getattr(config, name))

    if config.beta_slow < 0:
        raise ValueError(
            f"model_overrides.beta_slow must be >= 0, got {config.beta_slow}"
        )
    if config.beta_fast <= config.beta_slow:
        raise ValueError(
            "model_overrides.beta_fast must be greater than beta_slow, "
            f"got beta_fast={config.beta_fast}, beta_slow={config.beta_slow}"
        )
    if config.num_mtp_modules < 0:
        raise ValueError(
            "model_overrides.num_mtp_modules must be >= 0, "
            f"got {config.num_mtp_modules}"
        )
    if config.mtp_layer_compress_ratio < 0:
        raise ValueError(
            "model_overrides.mtp_layer_compress_ratio must be >= 0, "
            f"got {config.mtp_layer_compress_ratio}"
        )
    if config.load_balance_coeff <= 0:
        raise ValueError(
            "model_overrides.load_balance_coeff must be > 0, "
            f"got {config.load_balance_coeff}"
        )

    if len(config.compress_ratios) < config.n_layers:
        raise ValueError(
            "model_overrides.compress_ratios must contain at least one value "
            f"per main layer, got {len(config.compress_ratios)} values for "
            f"n_layers={config.n_layers}"
        )
    if any(ratio < 0 for ratio in config.compress_ratios):
        raise ValueError(
            "model_overrides.compress_ratios values must be >= 0, "
            f"got {config.compress_ratios}"
        )
    if config.rope_head_dim > config.head_dim:
        raise ValueError(
            "model_overrides.rope_head_dim must be <= head_dim, "
            f"got rope_head_dim={config.rope_head_dim}, "
            f"head_dim={config.head_dim}"
        )
    if config.rope_head_dim > config.index_head_dim:
        raise ValueError(
            "model_overrides.rope_head_dim must be <= index_head_dim, "
            f"got rope_head_dim={config.rope_head_dim}, "
            f"index_head_dim={config.index_head_dim}"
        )
    if config.n_heads % config.o_groups != 0:
        raise ValueError(
            "model_overrides.n_heads must be divisible by o_groups, "
            f"got n_heads={config.n_heads}, o_groups={config.o_groups}"
        )
    if not _is_power_of_two(config.index_head_dim):
        raise ValueError(
            "model_overrides.index_head_dim must be a power of two for the "
            f"Hadamard transform, got {config.index_head_dim}"
        )

    _validate_moe_args(config)


def _validate_moe_args(config: DeepSeekV4ModelOverrides) -> None:
    moe = config.moe_args
    _require_positive("moe_args.num_experts", moe.num_experts)
    _require_positive("moe_args.top_k", moe.top_k)
    _require_positive("moe_args.route_scale", moe.route_scale)
    _require_positive("moe_args.swiglu_limit", moe.swiglu_limit)

    if moe.num_shared_experts < 0:
        raise ValueError(
            "model_overrides.moe_args.num_shared_experts must be >= 0, "
            f"got {moe.num_shared_experts}"
        )
    if moe.top_k > moe.num_experts:
        raise ValueError(
            "model_overrides.moe_args.top_k must be <= num_experts, "
            f"got top_k={moe.top_k}, num_experts={moe.num_experts}"
        )
    if moe.n_hash_layers < 0:
        raise ValueError(
            "model_overrides.moe_args.n_hash_layers must be >= 0, "
            f"got {moe.n_hash_layers}"
        )
    if moe.load_balance_coeff is not None and moe.load_balance_coeff <= 0:
        raise ValueError(
            "model_overrides.moe_args.load_balance_coeff must be > 0 or None, "
            f"got {moe.load_balance_coeff}"
        )

    num_groups = moe.num_expert_groups
    if num_groups is not None:
        _require_positive("moe_args.num_expert_groups", num_groups)
        if moe.num_experts % num_groups != 0:
            raise ValueError(
                "model_overrides.moe_args.num_expert_groups must divide "
                f"num_experts, got num_expert_groups={num_groups}, "
                f"num_experts={moe.num_experts}"
            )

    limited_groups = moe.num_limited_groups
    if limited_groups is not None:
        _require_positive("moe_args.num_limited_groups", limited_groups)
        if num_groups is not None and limited_groups > num_groups:
            raise ValueError(
                "model_overrides.moe_args.num_limited_groups must be <= "
                f"num_expert_groups, got num_limited_groups={limited_groups}, "
                f"num_expert_groups={num_groups}"
            )


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
