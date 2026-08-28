# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible MAGI-2-preview model configuration overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .attention import ATTN_BACKENDS
from .model import Magi2PreviewModel

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class Magi2PreviewModelOverrides(Magi2PreviewModel.Config):
    """A complete, editable copy of the selected MAGI-2-preview model preset."""

    @classmethod
    def from_model_config(
        cls,
        model_config: Magi2PreviewModel.Config,
    ) -> Magi2PreviewModelOverrides:
        values = {
            field.name: copy.deepcopy(getattr(model_config, field.name))
            for field in dataclasses.fields(Magi2PreviewModel.Config)
        }
        return cls(**values)

    def to_model_config(self) -> Magi2PreviewModel.Config:
        values = {
            field.name: copy.deepcopy(getattr(self, field.name))
            for field in dataclasses.fields(Magi2PreviewModel.Config)
        }
        return Magi2PreviewModel.Config(**values)


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, Magi2PreviewModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, Magi2PreviewModel.Config):
        raise TypeError(
            "MAGI-2-preview model overrides require Magi2PreviewModel.Config, "
            f"got {type(model_config).__name__}"
        )
    return model_spec, Magi2PreviewModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: Magi2PreviewModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("MAGI-2-preview model overrides require model_spec")
    if not isinstance(model_spec.model, Magi2PreviewModel.Config):
        raise TypeError(
            "MAGI-2-preview model overrides require Magi2PreviewModel.Config, "
            f"got {type(model_spec.model).__name__}"
        )

    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: Magi2PreviewModelOverrides) -> None:
    if config.attn_backend not in ATTN_BACKENDS:
        raise ValueError(
            "model_overrides.attn_backend must be one of "
            f"{ATTN_BACKENDS}, got {config.attn_backend!r}"
        )

    positive_int_fields = (
        "num_layers",
        "hidden_size",
        "head_dim",
        "num_stream",
        "video_in_channels",
        "audio_in_channels",
        "text_in_channels",
        "time_channel_dim",
        "dense_intermediate_size",
        "moe_num_heads",
        "num_experts",
        "moe_top_k",
        "expert_intermediate_size",
        "shared_expert_intermediate_size",
        "sink_token_num",
    )
    for name in positive_int_fields:
        _require_positive(name, getattr(config, name))

    positive_float_fields = (
        "route_scale",
        "norm_eps",
        "alpha_init",
    )
    for name in positive_float_fields:
        _require_positive(name, getattr(config, name))

    if config.hidden_size % config.head_dim != 0:
        raise ValueError(
            "model_overrides.hidden_size must be divisible by head_dim, "
            f"got hidden_size={config.hidden_size}, head_dim={config.head_dim}"
        )
    if config.hidden_size % config.moe_num_heads != 0:
        raise ValueError(
            "model_overrides.hidden_size must be divisible by moe_num_heads, "
            f"got hidden_size={config.hidden_size}, "
            f"moe_num_heads={config.moe_num_heads}"
        )
    if config.moe_top_k > config.num_experts:
        raise ValueError(
            "model_overrides.moe_top_k must be <= num_experts, "
            f"got moe_top_k={config.moe_top_k}, num_experts={config.num_experts}"
        )

    for name, layer_ids in (
        ("mm_layers", config.mm_layers),
        ("moe_layers", config.moe_layers),
    ):
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(
                f"model_overrides.{name} must not contain duplicates, "
                f"got {layer_ids}"
            )
        out_of_range = [
            layer_id
            for layer_id in layer_ids
            if not 0 <= layer_id < config.num_layers
        ]
        if out_of_range:
            raise ValueError(
                f"model_overrides.{name} values must be in "
                f"[0, num_layers), got {out_of_range} for "
                f"num_layers={config.num_layers}"
            )

    overlap = sorted(set(config.mm_layers) & set(config.moe_layers))
    if overlap:
        raise ValueError(
            "model_overrides.mm_layers and moe_layers must be disjoint, "
            f"got overlap {overlap}"
        )


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")
