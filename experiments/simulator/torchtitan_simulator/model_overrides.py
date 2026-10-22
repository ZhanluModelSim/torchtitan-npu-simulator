# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rebuild coherent native config trees for typed DSV4 geometry overrides."""

from dataclasses import asdict, dataclass, field, replace


@dataclass(kw_only=True)
class MoEOverrides:
    num_experts: int | None = None
    num_shared_experts: int | None = None
    top_k: int | None = None
    n_hash_layers: int | None = None
    route_norm: bool | None = None
    route_scale: float | None = None


@dataclass(kw_only=True)
class ModelOverrides:
    dim: int | None = None
    n_layers: int | None = None
    vocab_size: int | None = None
    n_heads: int | None = None
    head_dim: int | None = None
    rope_head_dim: int | None = None
    q_lora_rank: int | None = None
    o_lora_rank: int | None = None
    o_groups: int | None = None
    window_size: int | None = None
    norm_eps: float | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_topk: int | None = None
    moe_inter_dim: int | None = None
    hc_mult: int | None = None
    hc_sinkhorn_iters: int | None = None
    hc_eps: float | None = None
    compress_rope_theta: float | None = None
    original_seq_len: int | None = None
    rope_theta: float | None = None
    rope_factor: float | None = None
    load_balance_coeff: float | None = None
    compress_ratios: tuple[int, ...] | None = None
    moe_args: MoEOverrides = field(default_factory=MoEOverrides)

    def validate(self):
        import math

        values = asdict(self)
        moe = values.pop("moe_args")
        for name, value in [*values.items(), *((f"moe_args.{key}", value) for key, value in moe.items())]:
            if value is None or name == "compress_ratios":
                continue
            if name == "moe_args.route_norm":
                if not isinstance(value, bool):
                    raise ValueError("model_overrides.moe_args.route_norm must be a boolean")
                continue
            integer = name not in {
                "norm_eps",
                "hc_eps",
                "compress_rope_theta",
                "rope_theta",
                "rope_factor",
                "load_balance_coeff",
                "moe_args.route_scale",
            }
            if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
                raise ValueError(f"model_overrides.{name} has an invalid type")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"model_overrides.{name} must be finite")
            minimum = 0 if name in {"moe_args.num_shared_experts", "moe_args.n_hash_layers"} else 1
            invalid = value < minimum if isinstance(value, int) else value <= 0
            if invalid:
                raise ValueError(f"model_overrides.{name} has an invalid value: {value}")
        if self.compress_ratios is not None and any(value not in {0, 1, 4, 128} for value in self.compress_ratios):
            raise ValueError(
                "model_overrides.compress_ratios supports native ratios 1, 4, 128 and legacy 0 (mapped to 1)"
            )


@dataclass(kw_only=True)
class _Geometry:
    dim: int
    n_layers: int
    vocab_size: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    n_groups: int
    compress_ratios: tuple[int, ...]
    window_size: int
    norm_eps: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    moe_inter_dim: int
    num_experts: int
    num_shared_experts: int
    top_k: int
    n_hash_layers: int
    route_norm: bool
    route_scale: float
    load_balance_coeff: float
    hc_mult: int
    sinkhorn_iters: int
    hc_eps: float
    max_seq_len: int
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    num_mtp_layers: int


def apply_model_overrides(trainer, overrides: ModelOverrides, *, num_mtp_modules: int):
    from inspect import signature

    from torchtitan_npu.models.deepseek_v4 import _make_v4_config
    from torchtitan_npu.models.deepseek_v4.config_registry import _dsv4_muon_profile
    from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model

    overrides.validate()
    model = trainer.model_spec.model
    if type(model) is not DeepSeekV4Model.Config:
        raise ValueError("Geometry overrides require an unconverted native DeepSeekV4Model.Config")
    layer = model.layers[0]
    attention = layer.attention
    compressed = next((item.attention for item in model.layers if item.attention.compress_ratio > 1), None)
    plain = next(
        (item.attention for item in [*model.layers, *model.mtp_layers] if item.attention.compress_ratio <= 1), None
    )
    indexer = next((item.attention.indexer for item in model.layers if item.attention.indexer is not None), None)
    if compressed is None or indexer is None:
        raise ValueError("Geometry override source must retain native compressed/indexer configuration defaults")
    moe = layer.moe
    router = moe.router
    inter_dim = moe.routed_experts.inner_experts.hidden_dim
    plain_theta = plain.rope.theta if plain is not None else signature(_make_v4_config).parameters["rope_theta"].default
    if not isinstance(plain_theta, (int, float)):
        raise RuntimeError("Native plain RoPE default requires review")
    geometry = _Geometry(
        dim=model.dim,
        n_layers=model.n_layers,
        vocab_size=model.vocab_size,
        n_heads=attention.n_heads,
        head_dim=attention.head_dim,
        rope_head_dim=attention.rope_head_dim,
        q_lora_rank=attention.q_lora_rank,
        o_lora_rank=attention.wo_a.out_features,
        n_groups=attention.n_groups,
        compress_ratios=model.compress_ratios,
        window_size=model.window_size,
        norm_eps=attention.norm_eps,
        index_n_heads=indexer.num_index_heads,
        index_head_dim=indexer.index_head_dim,
        index_topk=compressed.compressed_sparse_attention.inner_attention.index_topk,
        moe_inter_dim=inter_dim,
        num_experts=moe.num_experts,
        num_shared_experts=0 if moe.shared_experts is None else moe.shared_experts.w1.out_features // inter_dim,
        top_k=router.top_k,
        n_hash_layers=sum(bool(item.moe.router.hash) for item in model.layers),
        route_norm=router.route_norm,
        route_scale=router.route_scale,
        load_balance_coeff=moe.load_balance_coeff,
        hc_mult=model.hc_mult,
        sinkhorn_iters=layer.hc_attn_pre.sinkhorn_iters,
        hc_eps=layer.hc_attn_pre.eps,
        max_seq_len=attention.rope.max_seq_len,
        compress_rope_theta=compressed.rope.theta,
        original_seq_len=compressed.rope.original_seq_len,
        # Pro's main stack may contain only compressed layers, so its plain
        # RoPE can be absent when MTP=0. Use the native builder's own default.
        rope_theta=plain_theta,
        rope_factor=compressed.rope.rope_factor,
        num_mtp_layers=num_mtp_modules,
    )
    values = asdict(geometry)
    changes = asdict(overrides)
    moe_changes = changes.pop("moe_args")
    aliases = {"o_groups": "n_groups", "hc_sinkhorn_iters": "sinkhorn_iters"}
    values.update({aliases.get(key, key): value for key, value in changes.items() if value is not None})
    values.update({key: value for key, value in moe_changes.items() if value is not None})
    if len(values["compress_ratios"]) < values["n_layers"]:
        raise ValueError("model_overrides.compress_ratios must cover every main layer")
    values["compress_ratios"] = tuple(
        1 if ratio == 0 else ratio for ratio in values["compress_ratios"][: values["n_layers"]]
    )
    if values["n_heads"] % values["n_groups"]:
        raise ValueError("model_overrides.n_heads must be divisible by o_groups")
    if values["rope_head_dim"] > min(values["head_dim"], values["index_head_dim"]):
        raise ValueError("model_overrides.rope_head_dim exceeds head_dim or index_head_dim")
    index_dim = values["index_head_dim"]
    if index_dim & (index_dim - 1):
        raise ValueError("model_overrides.index_head_dim must be a power of two")
    if values["top_k"] > values["num_experts"]:
        raise ValueError("model_overrides.moe_args.top_k exceeds num_experts")
    rebuilt = _make_v4_config(**values)
    trainer.model_spec = replace(trainer.model_spec, model=rebuilt)
    trainer.optimizer._muon_profile = _dsv4_muon_profile(trainer.model_spec)
    trainer.loss.loss_fn.global_vocab_size = rebuilt.vocab_size


def add_cli_arguments(parser):
    import argparse
    from dataclasses import fields
    from typing import get_args, get_type_hints

    for prefix, cls in (("", ModelOverrides), ("moe_args.", MoEOverrides)):
        hints = get_type_hints(cls)
        for item in fields(cls):
            if item.name == "moe_args":
                continue
            name = prefix + item.name
            annotation = next(kind for kind in get_args(hints[item.name]) if kind is not type(None))
            kwargs = {"dest": "model_override__" + name, "default": None}
            option = "--model-overrides." + name.replace("_", "-")
            if annotation is bool:
                parser.add_argument(option, action=argparse.BooleanOptionalAction, **kwargs)
                parent, _, leaf = option.rpartition(".")
                parser.add_argument(
                    f"{parent}.no-{leaf}", action="store_false", dest=kwargs["dest"], default=argparse.SUPPRESS
                )
            elif item.name == "compress_ratios":
                parser.add_argument(option, type=int, nargs="+", **kwargs)
            else:
                parser.add_argument(option, type=annotation, **kwargs)


def take_cli_overrides(arguments):
    values = {
        name.removeprefix("model_override__"): arguments.pop(name)
        for name in list(arguments)
        if name.startswith("model_override__")
    }
    if all(value is None for value in values.values()):
        return None
    moe = {name.removeprefix("moe_args."): value for name, value in values.items() if name.startswith("moe_args.")}
    model = {name: value for name, value in values.items() if not name.startswith("moe_args.")}
    if model["compress_ratios"] is not None:
        model["compress_ratios"] = tuple(model["compress_ratios"])
    return ModelOverrides(**model, moe_args=MoEOverrides(**moe))
