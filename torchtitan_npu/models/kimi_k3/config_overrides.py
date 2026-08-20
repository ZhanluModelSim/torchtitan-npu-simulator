# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI-visible Kimi K3 model configuration overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .attention import KimiDeltaAttention, KimiGatedMLA
from .feed_forward import KimiMLP, KimiSparseMoeBlock
from .model import KimiK3Model

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True, slots=True)
class KimiK3ModelOverrides:
    """Stable editable inputs used to generate Kimi K3 per-layer configs."""

    param_init: dict | None = None
    vocab_size: int = 163840
    dim: int = 7168
    n_layers: int = 93
    n_dense_layers: int = 1

    n_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    kda_head_dim: int = 128
    kda_layers: list[int] = field(
        default_factory=lambda: [
            i for i in range(93) if (i + 1) % 4 != 0 and i != 92
        ]
    )
    conv_kernel_size: int = 4
    gate_lower_bound: float | None = -5.0
    use_full_rank_gate: bool = True

    dense_hidden_dim: int = 33792
    moe_inter_dim: int = 3072
    num_experts: int = 896
    num_shared_experts: int = 2
    router_top_k: int = 16
    router_score_func: str = "sigmoid"
    num_expert_groups: int = 1
    topk_group: int = 1
    routed_expert_hidden_size: int | None = 3584
    latent_moe_use_norm: bool = True
    routed_scaling_factor: float = 1.0
    renormalize: bool = True

    situ_beta: float = 4.0
    situ_linear_beta: float | None = 25.0
    norm_eps: float = 1e-5
    attn_res_block_size: int | None = 12

    @classmethod
    def from_model_config(
        cls,
        model_config: KimiK3Model.Config,
    ) -> KimiK3ModelOverrides:
        layers = model_config.layers
        if not layers:
            raise ValueError("Kimi K3 model overrides require at least one layer")

        dense_flags = [layer.feed_forward is not None for layer in layers]
        n_dense_layers = sum(dense_flags)
        if dense_flags != [True] * n_dense_layers + [False] * (
            len(layers) - n_dense_layers
        ):
            raise ValueError(
                "Kimi K3 model overrides require dense layers to form a prefix"
            )

        kda_configs = [
            layer.attention
            for layer in layers
            if isinstance(layer.attention, KimiDeltaAttention.Config)
        ]
        mla_configs = [
            layer.attention
            for layer in layers
            if isinstance(layer.attention, KimiGatedMLA.Config)
        ]
        dense_configs = [
            layer.feed_forward
            for layer in layers
            if layer.feed_forward is not None
        ]
        moe_configs = [layer.moe for layer in layers if layer.moe is not None]
        if not kda_configs or not mla_configs or not dense_configs or not moe_configs:
            raise ValueError(
                "Kimi K3 presets must contain KDA, MLA, dense FFN, and MoE layers"
            )

        kda = _require_uniform("KDA", kda_configs)
        mla = _require_uniform("MLA", mla_configs)
        dense = _require_uniform("dense FFN", dense_configs)
        moe = _require_uniform(
            "MoE",
            moe_configs,
            ignored_fields={"debug_force_load_balance"},
        )
        _validate_derived_layer_fields(model_config)
        if (
            kda.dim != model_config.dim
            or mla.dim != model_config.dim
            or dense.hidden_size != model_config.dim
            or moe.hidden_size != model_config.dim
            or kda.num_heads != mla.n_heads
            or kda.norm_eps != model_config.norm_eps
            or mla.norm_eps != model_config.norm_eps
            or moe.norm_eps != model_config.norm_eps
            or dense.beta != moe.situ_beta
            or dense.linear_beta != moe.situ_linear_beta
        ):
            raise ValueError(
                "Kimi K3 model overrides require consistent shared layer parameters"
            )

        return cls(
            param_init=copy.deepcopy(model_config.param_init),
            vocab_size=model_config.vocab_size,
            dim=model_config.dim,
            n_layers=len(layers),
            n_dense_layers=n_dense_layers,
            n_heads=mla.n_heads,
            q_lora_rank=mla.q_lora_rank,
            kv_lora_rank=mla.kv_lora_rank,
            qk_nope_head_dim=mla.qk_nope_head_dim,
            qk_rope_head_dim=mla.qk_rope_head_dim,
            v_head_dim=mla.v_head_dim,
            kda_head_dim=kda.head_dim,
            kda_layers=[
                index
                for index, layer in enumerate(layers)
                if isinstance(layer.attention, KimiDeltaAttention.Config)
            ],
            conv_kernel_size=kda.conv_kernel_size,
            gate_lower_bound=kda.gate_lower_bound,
            use_full_rank_gate=kda.use_full_rank_gate,
            dense_hidden_dim=dense.intermediate_size,
            moe_inter_dim=moe.moe_intermediate_size,
            num_experts=moe.num_experts,
            num_shared_experts=moe.num_shared_experts,
            router_top_k=moe.top_k,
            router_score_func=moe.score_func,
            num_expert_groups=moe.num_expert_groups,
            topk_group=moe.topk_group,
            routed_expert_hidden_size=moe.routed_expert_hidden_size,
            latent_moe_use_norm=moe.latent_moe_use_norm,
            routed_scaling_factor=moe.routed_scaling_factor,
            renormalize=moe.renormalize,
            situ_beta=moe.situ_beta,
            situ_linear_beta=moe.situ_linear_beta,
            norm_eps=model_config.norm_eps,
            attn_res_block_size=model_config.attn_res_block_size,
        )

    def to_model_config(self) -> KimiK3Model.Config:
        # Local import avoids coupling package registration to this CLI layer.
        from . import _make_kimi_k3_model_config

        values = {
            field.name: copy.deepcopy(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        return _make_kimi_k3_model_config(**values)


def build_model_spec_with_overrides(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, KimiK3ModelOverrides]:
    model_config = model_spec.model
    if not isinstance(model_config, KimiK3Model.Config):
        raise TypeError(
            "Kimi K3 model overrides require KimiK3Model.Config, "
            f"got {type(model_config).__name__}"
        )
    return model_spec, KimiK3ModelOverrides.from_model_config(model_config)


def apply_model_overrides(
    model_spec: ModelSpec | None,
    overrides: KimiK3ModelOverrides,
) -> ModelSpec:
    if model_spec is None:
        raise ValueError("Kimi K3 model overrides require model_spec")
    if not isinstance(model_spec.model, KimiK3Model.Config):
        raise TypeError(
            "Kimi K3 model overrides require KimiK3Model.Config, "
            f"got {type(model_spec.model).__name__}"
        )

    validate_model_overrides(overrides)
    return dataclasses.replace(model_spec, model=overrides.to_model_config())


def validate_model_overrides(config: KimiK3ModelOverrides) -> None:
    positive_int_fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "kda_head_dim",
        "conv_kernel_size",
        "dense_hidden_dim",
        "moe_inter_dim",
        "num_experts",
        "router_top_k",
        "num_expert_groups",
        "topk_group",
    )
    for name in positive_int_fields:
        _require_positive(name, getattr(config, name))

    positive_float_fields = (
        "routed_scaling_factor",
        "situ_beta",
        "norm_eps",
    )
    for name in positive_float_fields:
        _require_positive(name, getattr(config, name))

    if not 0 <= config.n_dense_layers <= config.n_layers:
        raise ValueError(
            "model_overrides.n_dense_layers must be between 0 and n_layers, "
            f"got n_dense_layers={config.n_dense_layers}, "
            f"n_layers={config.n_layers}"
        )
    if config.num_shared_experts < 0:
        raise ValueError(
            "model_overrides.num_shared_experts must be >= 0, "
            f"got {config.num_shared_experts}"
        )
    if config.router_top_k > config.num_experts:
        raise ValueError(
            "model_overrides.router_top_k must be <= num_experts, "
            f"got router_top_k={config.router_top_k}, "
            f"num_experts={config.num_experts}"
        )
    if config.router_score_func not in {"sigmoid", "softmax"}:
        raise ValueError(
            "model_overrides.router_score_func must be 'sigmoid' or 'softmax', "
            f"got {config.router_score_func!r}"
        )
    if config.num_experts % config.num_expert_groups != 0:
        raise ValueError(
            "model_overrides.num_expert_groups must divide num_experts, "
            f"got num_expert_groups={config.num_expert_groups}, "
            f"num_experts={config.num_experts}"
        )
    if config.num_expert_groups > 1:
        if config.topk_group >= config.num_expert_groups:
            raise ValueError(
                "model_overrides.topk_group must be smaller than "
                "num_expert_groups when expert grouping is enabled, "
                f"got topk_group={config.topk_group}, "
                f"num_expert_groups={config.num_expert_groups}"
            )
        if config.num_experts // config.num_expert_groups < 2:
            raise ValueError(
                "model_overrides expert groups must contain at least 2 experts"
            )
        candidate_experts = (
            config.topk_group
            * config.num_experts
            // config.num_expert_groups
        )
        if config.router_top_k > candidate_experts:
            raise ValueError(
                "model_overrides.router_top_k must not exceed the number of "
                "experts in selected groups, "
                f"got router_top_k={config.router_top_k}, "
                f"candidate_experts={candidate_experts}"
            )
    elif config.topk_group != 1:
        raise ValueError(
            "model_overrides.topk_group must be 1 when num_expert_groups is 1, "
            f"got {config.topk_group}"
        )

    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.v_head_dim > q_head_dim:
        raise ValueError(
            "model_overrides.v_head_dim must be <= "
            "qk_nope_head_dim + qk_rope_head_dim, "
            f"got v_head_dim={config.v_head_dim}, q_head_dim={q_head_dim}"
        )
    if config.routed_expert_hidden_size is not None:
        _require_positive(
            "routed_expert_hidden_size",
            config.routed_expert_hidden_size,
        )
    if config.situ_linear_beta is not None:
        _require_positive("situ_linear_beta", config.situ_linear_beta)
    if config.attn_res_block_size is not None:
        _require_positive("attn_res_block_size", config.attn_res_block_size)

    if len(set(config.kda_layers)) != len(config.kda_layers):
        raise ValueError("model_overrides.kda_layers must not contain duplicates")
    if any(layer_id < 0 for layer_id in config.kda_layers):
        raise ValueError(
            "model_overrides.kda_layers values must be >= 0, "
            f"got {config.kda_layers}"
        )


def _validate_derived_layer_fields(model_config: KimiK3Model.Config) -> None:
    for layer_id, layer in enumerate(model_config.layers):
        if layer.layer_id != layer_id:
            raise ValueError(
                "Kimi K3 model overrides require sequential layer_id values"
            )
        if layer.dim != model_config.dim or layer.norm_eps != model_config.norm_eps:
            raise ValueError(
                "Kimi K3 model overrides require uniform model dimensions and norm_eps"
            )
        if layer.attn_res_block_size != model_config.attn_res_block_size:
            raise ValueError(
                "Kimi K3 model overrides require uniform attn_res_block_size"
            )


def _require_uniform(
    label: str,
    configs: list[Any],
    *,
    ignored_fields: set[str] | None = None,
) -> Any:
    ignored_fields = ignored_fields or set()
    first = configs[0]
    fields = [
        field.name
        for field in dataclasses.fields(first)
        if field.name not in ignored_fields
    ]
    for config in configs[1:]:
        if type(config) is not type(first) or any(
            getattr(config, name) != getattr(first, name) for name in fields
        ):
            raise ValueError(
                f"Kimi K3 model overrides require uniform {label} configs"
            )
    return first


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"model_overrides.{name} must be > 0, got {value}")
