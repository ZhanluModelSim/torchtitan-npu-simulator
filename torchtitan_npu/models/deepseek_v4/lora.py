# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

__all__ = ["DEEPSEEK_V4_LORA_TARGETS", "DeepSeekV4LoRAConverter", "is_lora_parameter", "peft_target_modules"]

import math
import re
from copy import copy
from dataclasses import dataclass, fields, replace
from functools import cache
from typing import TYPE_CHECKING, cast

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torchtitan.components.lora import LoRAConverter
from torchtitan.config import derive
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.torchtitan.components.lora import is_non_strict_tracing, upstream_adapter_sharding
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .model import DeepSeekV4Model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Protocol


_LORA_PARAMETER_PREFIXES = ("lora_", "w13_lora_", "w1_lora_", "w3_lora_", "w2_lora_")


def is_lora_parameter(name: str) -> bool:
    """Identify native adapter parameters by their qualified-name components."""
    return any(part.startswith(_LORA_PARAMETER_PREFIXES) for part in name.split("."))


@dataclass(kw_only=True)
class LoRAOptions:
    rank: int
    alpha: float
    chunk_rows: int


if TYPE_CHECKING:

    class _LoRAModule(Protocol):
        Config: type

    @dataclass(kw_only=True)
    class _LinearLoRAConfig(LoRAOptions, Linear.Config):
        pass

    @dataclass(kw_only=True)
    class _BatchedLoRAConfig(LoRAOptions, BatchedLinear.Config):
        pass

    @dataclass(kw_only=True)
    class _GroupedLoRAConfig(LoRAOptions, GroupedExperts.Config):
        pass


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    base_weight_sharding = base_sharding.state_shardings.get("weight") if base_sharding else None
    replicated_weight = dense_param_placement(tp=spmd.R)
    if base_weight_sharding == replicated_weight:
        replicated = ShardingConfig(state_shardings={"weight": replicated_weight})
        return replicated, replicated
    return upstream_adapter_sharding(base_sharding)


def _make_lora_adapter_config_cls(parent_config_cls: type) -> type:
    @dataclass(kw_only=True, slots=True)
    class Config(LoRAOptions, parent_config_cls):  # type: ignore[misc]
        def __post_init__(self):
            parent_post_init = getattr(parent_config_cls, "__post_init__", None)
            if parent_post_init is not None:
                parent_post_init(self)
            policy = getattr(self, "_torchao_npu_config", None)
            if policy is None:
                return

            from torchao_npu.configs import ParamSwapConfig

            if not isinstance(policy, ParamSwapConfig):
                raise ValueError(
                    f"{type(policy).__name__} module replacement is not supported by "
                    f"{parent_config_cls.__qualname__} LoRA; its adapter forward must be preserved"
                )
            original_filter = policy.params_filter_fn

            def base_filter(param, fqn):
                return not is_lora_parameter(fqn) and (original_filter is None or original_filter(param, fqn))

            self._torchao_npu_config = copy(policy)
            self._torchao_npu_config.params_filter_fn = base_filter

    return Config


@cache
def linear_lora_class(parent_cls: type[Module]) -> type[_LoRAModule]:
    batched = issubclass(parent_cls, BatchedLinear)

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        Config = _make_lora_adapter_config_cls(parent_cls.Config)

        def __init__(self, config: _LinearLoRAConfig | _BatchedLoRAConfig) -> None:
            super().__init__(config)
            self._lora_scaling = config.alpha / config.rank
            self._lora_chunk_rows = config.chunk_rows
            lora_a_sharding, lora_b_sharding = adapter_sharding(config.sharding_config)
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=config.rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={"weight": lambda weight: nn.init.kaiming_uniform_(weight, a=math.sqrt(5))},
            ).build()
            if batched:
                self.lora_b = BatchedLinear.Config(
                    n_heads=cast("_BatchedLoRAConfig", config).n_heads,
                    in_features=config.rank,
                    out_features=config.out_features,
                    sharding_config=lora_b_sharding,
                    param_init={"weight": nn.init.zeros_},
                ).build()
            else:
                self.lora_b = Linear.Config(
                    in_features=config.rank,
                    out_features=config.out_features,
                    bias=False,
                    sharding_config=lora_b_sharding,
                    param_init={"weight": nn.init.zeros_},
                ).build()

        def _lora_delta(self, rows: torch.Tensor) -> torch.Tensor:
            return self.lora_b(self.lora_a(rows) if batched else rows)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = super().forward(x).clone()
            if batched:
                adapter_rows = x.reshape(-1, x.shape[-2], x.shape[-1])
                output_rows = output.reshape(-1, output.shape[-2], output.shape[-1])
            else:
                hidden = self.lora_a(x)
                adapter_rows = hidden.reshape(-1, hidden.shape[-1])
                output_rows = output.reshape(-1, output.shape[-1])
            if torch.compiler.is_compiling() or is_non_strict_tracing():
                output_rows.add_(self._lora_delta(adapter_rows), alpha=self._lora_scaling)
                return output
            for start in range(0, output_rows.shape[0], self._lora_chunk_rows):
                end = min(start + self._lora_chunk_rows, output_rows.shape[0])
                output_rows[start:end].add_(self._lora_delta(adapter_rows[start:end]), alpha=self._lora_scaling)
            return output

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = LoRALinear.__name__
    return LoRALinear


@cache
def grouped_lora_class(parent_cls: type[Module]) -> type[_LoRAModule]:
    class GroupedLoRAExperts(parent_cls):  # type: ignore[valid-type, misc]
        Config = _make_lora_adapter_config_cls(parent_cls.Config)

        w13_lora_a: nn.Parameter
        w1_lora_b: nn.Parameter
        w3_lora_b: nn.Parameter
        w2_lora_a: nn.Parameter
        w2_lora_b: nn.Parameter
        _grouped_mm: Callable[..., torch.Tensor]

        def __init__(self, config: _GroupedLoRAConfig) -> None:
            super().__init__(config)
            self._lora_scaling = config.alpha / config.rank
            self._lora_chunk_rows = config.chunk_rows
            adapter_shapes = {
                "w13_lora_a": (config.num_experts, config.rank, config.dim),
                "w1_lora_b": (config.num_experts, config.hidden_dim, config.rank),
                "w3_lora_b": (config.num_experts, config.hidden_dim, config.rank),
                "w2_lora_a": (config.num_experts, config.rank, config.hidden_dim),
                "w2_lora_b": (config.num_experts, config.dim, config.rank),
            }
            for name, shape in adapter_shapes.items():
                self.register_parameter(name, nn.Parameter(torch.empty(shape)))

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            legacy_key = f"{prefix}w13_lora_b"
            if legacy_key in state_dict:
                value = state_dict.pop(legacy_key)
                if value.ndim != 4 or value.shape[2] != 2:
                    raise ValueError("Legacy w13 LoRA B must have shape [E, F, 2, R]")
                for index, projection in enumerate(("w1_lora_b", "w3_lora_b")):
                    key = f"{prefix}{projection}"
                    if key in state_dict:
                        raise ValueError("Checkpoint contains both split and interleaved LoRA B weights")
                    state_dict[key] = value[:, :, index, :].contiguous()
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

        def forward(
            self,
            input_tokens: torch.Tensor,
            token_counts: torch.Tensor,
            *,
            routed_scores_R: torch.Tensor | None = None,
        ) -> torch.Tensor:
            routed_scores = routed_scores_R
            w1 = _local_tensor(cast("torch.Tensor", self.w1_EFD))
            w2 = _local_tensor(cast("torch.Tensor", self.w2_EDF))
            w3 = _local_tensor(cast("torch.Tensor", self.w3_EFD))
            offsets = torch.cumsum(token_counts, dim=0, dtype=torch.int32)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
                for axis in ("dp", "cp"):
                    spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

            gate = self._grouped_mm(A=input_tokens.bfloat16(), B_t=w1.bfloat16().transpose(-2, -1), offs=offsets)
            up = self._grouped_mm(A=input_tokens.bfloat16(), B_t=w3.bfloat16().transpose(-2, -1), offs=offsets)
            gate, up = self._add_grouped_w13_delta(gate=gate, up=up, x=input_tokens, offsets=offsets)
            swiglu_limit = getattr(self, "swiglu_limit", 0.0)
            if swiglu_limit > 0:
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gate = torch.clamp(gate, max=swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if routed_scores is not None:
                hidden = (hidden.float() * routed_scores.float().reshape(-1, 1)).to(hidden.dtype)
            output = self._grouped_mm(A=hidden, B_t=w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(input_tokens)
            return self._add_grouped_w2_delta(output=output, x=hidden, offsets=offsets)

        def _iter_grouped_row_chunks(self, num_rows: int, offsets: torch.Tensor):
            if torch.compiler.is_compiling() or is_non_strict_tracing():
                yield 0, num_rows, offsets.to(torch.int32)
                return
            group_ends = offsets.to(torch.int64)
            group_starts = torch.cat((group_ends.new_zeros(1), group_ends[:-1]))
            for start in range(0, num_rows, self._lora_chunk_rows):
                end = min(start + self._lora_chunk_rows, num_rows)
                chunk_counts = group_ends.clamp(min=start, max=end) - group_starts.clamp(min=start, max=end)
                yield start, end, torch.cumsum(chunk_counts, dim=0, dtype=torch.int32)

        def _add_grouped_w13_delta(
            self, *, gate: torch.Tensor, up: torch.Tensor, x: torch.Tensor, offsets: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = _local_tensor(self.w13_lora_a)
            a_t = a.bfloat16().transpose(-2, -1)
            gate_b_t = _local_tensor(self.w1_lora_b).bfloat16().transpose(-2, -1)
            up_b_t = _local_tensor(self.w3_lora_b).bfloat16().transpose(-2, -1)
            for start, end, chunk_offsets in self._iter_grouped_row_chunks(x.shape[0], offsets):
                hidden = self._grouped_mm(A=x[start:end].bfloat16(), B_t=a_t, offs=chunk_offsets)
                gate_delta = self._grouped_mm(A=hidden, B_t=gate_b_t, offs=chunk_offsets)
                up_delta = self._grouped_mm(A=hidden, B_t=up_b_t, offs=chunk_offsets)
                gate[start:end].add_(gate_delta.to(gate.dtype), alpha=self._lora_scaling)
                up[start:end].add_(up_delta.to(up.dtype), alpha=self._lora_scaling)
            return gate, up

        def _add_grouped_w2_delta(
            self, *, output: torch.Tensor, x: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            a = _local_tensor(self.w2_lora_a)
            b = _local_tensor(self.w2_lora_b)
            a_t = a.bfloat16().transpose(-2, -1)
            b_t = b.bfloat16().transpose(-2, -1)
            for start, end, chunk_offsets in self._iter_grouped_row_chunks(x.shape[0], offsets):
                hidden = self._grouped_mm(A=x[start:end].bfloat16(), B_t=a_t, offs=chunk_offsets)
                delta = self._grouped_mm(A=hidden, B_t=b_t, offs=chunk_offsets)
                output[start:end].add_(delta.to(output.dtype), alpha=self._lora_scaling)
            return output

    GroupedLoRAExperts.__name__ = f"LoRA{parent_cls.__name__}"
    GroupedLoRAExperts.__qualname__ = GroupedLoRAExperts.__name__
    return GroupedLoRAExperts


MODULE_PATHS = (
    ("lm_head", "lm_head", "head"),
    ("attention.wq_a", "self_attn.q_a_proj", "attn.wq_a"),
    ("attention.wq_b", "self_attn.q_b_proj", "attn.wq_b"),
    ("attention.wkv", "self_attn.kv_proj", "attn.wkv"),
    ("attention.wo_a", "self_attn.o_a_proj", "attn.wo_a"),
    ("attention.wo_b", "self_attn.o_b_proj", "attn.wo_b"),
    ("attention.compressor.wkv", "self_attn.compressor.kv_proj", "attn.compressor.wkv"),
    ("attention.compressor.wgate", "self_attn.compressor.gate_proj", "attn.compressor.wgate"),
    ("moe.router.gate", "mlp.gate", "ffn.gate"),
    ("moe.shared_experts.w1", "mlp.shared_experts.gate_proj", "ffn.shared_experts.w1"),
    ("moe.shared_experts.w2", "mlp.shared_experts.down_proj", "ffn.shared_experts.w2"),
    ("moe.shared_experts.w3", "mlp.shared_experts.up_proj", "ffn.shared_experts.w3"),
    ("e_proj", "e_proj", "e_proj"),
    ("h_proj", "h_proj", "h_proj"),
)

_PEFT_MODULE_SUFFIXES = {local: hf for local, hf, _ in MODULE_PATHS}
OFFICIAL_MODULES = {hf: official for _, hf, official in MODULE_PATHS}
PEFT_PREFIX = "base_model.model."


def pack_expert_factor(value, factor):
    if factor.endswith("lora_a"):
        return value.reshape(-1, value.shape[-1])
    if factor == "w13_lora_b":
        value = torch.cat((value[:, :, 0, :], value[:, :, 1, :]), dim=1)
    return value.permute(1, 2, 0).reshape(value.shape[1], -1)


def unpack_expert_factors(a, b, rank, experts=None):
    if rank is None or rank <= 0:
        raise ValueError("Routed-expert adapters require a positive rank")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Routed-expert adapters require two-dimensional factors")
    if a.shape[0] % rank or b.shape[1] != a.shape[0] or not a.shape[0]:
        raise ValueError("Routed-expert A/B shapes do not match expert count and rank")
    if experts is not None and a.shape[0] != experts * rank:
        raise ValueError("Routed-expert factors do not match base expert count and rank")
    experts = a.shape[0] // rank
    return a.reshape(experts, rank, a.shape[1]), b.reshape(b.shape[0], rank, experts).permute(2, 0, 1)


_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"


def _official_base_key(module: str) -> str | None:
    if module == "lm_head":
        return "head.weight"
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", module)
    if match and match[2] in OFFICIAL_MODULES:
        return f"layers.{match[1]}.{OFFICIAL_MODULES[match[2]]}.weight"
    return None


@dataclass
class _MergeContext:
    weight_map: dict
    base_shapes: dict
    rank: int | None
    expected_experts: int | None


def _split_expert_layout(layer, gate_up, context):
    weight_map = context.weight_map
    prefix = f"model.layers.{layer}.mlp.experts"
    if f"{prefix}.0.w1.weight" not in weight_map and f"{prefix}.0.w2.weight" not in weight_map:
        prefix = f"layers.{layer}.ffn.experts"
    if f"model.{prefix}.0.w1.weight" in weight_map or f"model.{prefix}.0.w2.weight" in weight_map:
        prefix = f"model.{prefix}"
    expert_keys = {}
    for key in weight_map:
        found = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(w[123])\.weight", key)
        if found:
            expert_keys.setdefault(found[2], set()).add(int(found[1]))
    all_ids = set().union(*expert_keys.values()) if expert_keys else set()
    experts = context.expected_experts if context.expected_experts is not None else len(all_ids)
    required = ("w1", "w3") if gate_up else ("w2",)
    if not experts or any(expert_keys.get(proj) != set(range(experts)) for proj in required):
        raise ValueError("Split base expert keys must be complete and contiguous")
    if any(ids != set(range(experts)) for ids in expert_keys.values()):
        raise ValueError("Split base expert projections disagree on expert count")
    return prefix, experts


def _expert_targets(module, lora_a, lora_b, context):
    match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts(\.base_layer)?", module)
    if not match:
        return None
    gate_up = match[2] is not None
    parameter = "gate_up_proj" if gate_up else "down_proj"
    fused = f"model.layers.{match[1]}.mlp.experts.{parameter}"
    if fused in context.weight_map:
        shape = context.base_shapes.get(fused)
        if shape is None or len(shape) != 3:
            raise ValueError(f"Expected fused expert base shape [E, O, I]: {fused}")
        experts = shape[0]
        if context.expected_experts is not None and experts != context.expected_experts:
            raise ValueError("Base expert count does not match model metadata")
        a, b = unpack_expert_factors(lora_a, lora_b, context.rank, experts)
        return [(fused, a, b)]
    prefix, experts = _split_expert_layout(match[1], gate_up, context)
    a, b = unpack_expert_factors(lora_a, lora_b, context.rank, experts)
    if gate_up:
        if b.shape[1] % 2:
            raise ValueError("Routed gate/up adapter requires an even output dimension")
        gate, up = b.chunk(2, dim=1)
        return [
            (f"{prefix}.{expert}.{projection}.weight", a[expert], values[expert])
            for expert in range(experts)
            for projection, values in (("w1", gate), ("w3", up))
        ]
    return [(f"{prefix}.{expert}.w2.weight", a[expert], b[expert]) for expert in range(experts)]


def _dense_target(module, lora_a, lora_b, context):
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("Dense LoRA A/B must be two-dimensional")
    if context.rank is None or lora_a.shape[0] != context.rank or lora_b.shape[1] != context.rank:
        raise ValueError("Dense LoRA A/B rank must match adapter config rank")
    base_key = f"{module}.weight"
    if base_key not in context.weight_map:
        base_key = _official_base_key(module) or base_key
        if base_key not in context.weight_map and f"model.{base_key}" in context.weight_map:
            base_key = f"model.{base_key}"
    return [(base_key, lora_a, lora_b)]


def _add_merge_targets(targets_by_key, targets, context):
    for base_key, a, b in targets:
        if base_key not in context.weight_map:
            raise ValueError(f"Adapter target has no matching base weight: {base_key}")
        if base_key in targets_by_key:
            raise ValueError(f"Multiple adapter pairs map to the same base weight: {base_key}")
        shape = context.base_shapes.get(base_key)
        if shape is not None and (a.shape[-1] != shape[-1] or b.shape[-2] != shape[-2]):
            raise ValueError(f"LoRA factors do not match base weight shape: {base_key}")
        targets_by_key[base_key] = (a, b)


def build_lora_merge_plan(
    adapter_tensors: dict[str, torch.Tensor],
    base_weight_map: dict[str, str],
    *,
    rank: int | None = None,
    base_shapes: dict | None = None,
    expected_experts: int | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    context = _MergeContext(base_weight_map, base_shapes or {}, rank, expected_experts)
    targets_by_key = {}
    lora_a_keys = {key for key in adapter_tensors if key.endswith(_LORA_A_SUFFIX)}
    if not lora_a_keys:
        raise ValueError("Adapter contains no LoRA A/B pairs")
    expected_keys = lora_a_keys | {key.removesuffix(_LORA_A_SUFFIX) + _LORA_B_SUFFIX for key in lora_a_keys}
    unexpected_keys = set(adapter_tensors) - expected_keys
    if unexpected_keys:
        raise ValueError(f"Adapter contains unsupported or unpaired tensors: {sorted(unexpected_keys)[:5]}")

    for lora_a_key in sorted(lora_a_keys):
        module_key = lora_a_key[: -len(_LORA_A_SUFFIX)]
        lora_b_key = f"{module_key}{_LORA_B_SUFFIX}"
        if lora_b_key not in adapter_tensors:
            raise ValueError(f"Adapter has {lora_a_key} but no matching {lora_b_key}")
        if not module_key.startswith(PEFT_PREFIX):
            raise ValueError(f"Expected adapter module key to start with {PEFT_PREFIX!r}, got {module_key!r}")
        module = module_key.removeprefix(PEFT_PREFIX)
        lora_a, lora_b = adapter_tensors[lora_a_key], adapter_tensors[lora_b_key]
        targets = _expert_targets(module, lora_a, lora_b, context)
        if targets is None:
            targets = _dense_target(module, lora_a, lora_b, context)
        _add_merge_targets(targets_by_key, targets, context)
    return targets_by_key


def merge_lora_weight(base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    if base.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"LoRA merge requires a decoded floating-point base, got {base.dtype}")
    delta = scaling * (lora_b.to(torch.float32) @ lora_a.to(torch.float32))
    if delta.shape != base.shape:
        raise ValueError(f"LoRA delta shape {tuple(delta.shape)} does not match base weight shape {tuple(base.shape)}")
    return (base.to(torch.float32) + delta).to(base.dtype)


DEEPSEEK_V4_LORA_TARGETS = tuple(_PEFT_MODULE_SUFFIXES)


def peft_target_modules(targets: Iterable[str]) -> tuple[str, ...]:
    names = []
    for target in targets:
        suffix = next((name for name in _PEFT_MODULE_SUFFIXES if _matches_suffix(target, name)), None)
        if suffix is None:
            raise ValueError(f"Unsupported PEFT LoRA target: {target}")
        prefix = target.removesuffix(suffix)
        name = f"{prefix}{_PEFT_MODULE_SUFFIXES[suffix]}"
        names.append(f"model.{name}" if prefix else name)
    return tuple(dict.fromkeys(names))


_GROUPED_EXPERTS_SUFFIX = "moe.routed_experts.inner_experts"


def _matches_suffix(fqn: str, suffix: str) -> bool:
    return fqn == suffix or fqn.endswith(f".{suffix}")


def _replace_config_node(root, replacement, parent, attr, fqn):
    if parent is None:
        return replacement
    if isinstance(parent, list):
        if not isinstance(attr, int):
            raise TypeError(f"List parent at {fqn!r} requires an integer index, got {type(attr).__name__}")
        parent[attr] = replacement
    else:
        if not isinstance(attr, str):
            raise TypeError(f"Config parent at {fqn!r} requires a string attribute, got {type(attr).__name__}")
        setattr(parent, attr, replacement)
    return root


class DeepSeekV4LoRAConverter(LoRAConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(LoRAConverter.Config):
        rank: int = 64
        alpha: float = 128.0
        rank_experts: int = 64
        dense_chunk_rows: int = 1024
        grouped_chunk_rows: int = 4096
        adapt_routed_experts: bool = True
        include_mtp: bool = True
        strict: bool = True

    def __init__(self, config: Config, **kwargs) -> None:
        if config.rank <= 0 or config.rank_experts <= 0:
            raise ValueError("LoRA ranks must be positive")
        if not math.isfinite(config.alpha) or config.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if config.dense_chunk_rows <= 0 or config.grouped_chunk_rows <= 0:
            raise ValueError("LoRA chunk sizes must be positive")
        self.config = config
        self.targets = tuple(config.target_modules if config.target_modules is not None else DEEPSEEK_V4_LORA_TARGETS)

    def convert(self, model_config: Module.Config) -> Module.Config:
        config = cast("DeepSeekV4LoRAConverter.Config", self.config)
        converted_root = model_config
        targets = self.targets
        if config.target_modules is None and (not config.include_mtp or not getattr(model_config, "mtp_layers", None)):
            targets = tuple(target for target in targets if target not in ("e_proj", "h_proj"))
        matched = dict.fromkeys(targets, 0)
        grouped_matches = 0
        configs = list(model_config.traverse(Module.Config, recurse=True))
        for fqn, cfg, parent, attr in reversed(configs):
            if not isinstance(cfg, Module.Config):
                raise TypeError(f"Expected a Module.Config at {fqn!r}, got {type(cfg).__name__}")
            if not config.include_mtp and (fqn == "mtp_layers" or fqn.startswith("mtp_layers.")):
                continue
            dense_target = next((target for target in targets if _matches_suffix(fqn, target)), None)
            if dense_target is not None and isinstance(cfg, (BatchedLinear.Config, Linear.Config)):
                new_cfg = self._make_lora_config(cfg)
                matched[dense_target] += 1
            elif (
                config.adapt_routed_experts
                and _matches_suffix(fqn, _GROUPED_EXPERTS_SUFFIX)
                and isinstance(cfg, GroupedExperts.Config)
            ):
                new_cfg = self._make_lora_config(cfg)
                grouped_matches += 1
            else:
                continue

            converted_root = _replace_config_node(converted_root, new_cfg, parent, attr, fqn)

        self._validate_matches(matched, grouped_matches)
        logger.info(
            "DeepSeek-V4 LoRA: dense=%d grouped=%d rank=%d expert_rank=%d",
            sum(matched.values()),
            grouped_matches,
            config.rank,
            config.rank_experts,
        )
        return self.attach_model_config(converted_root, replace(config, target_modules=list(targets)))

    def attach_model_config(self, model_config: Module.Config, lora_config: Config) -> Module.Config:
        return derive(model_config, _LoRAModelConfig, lora=lora_config)

    def _validate_matches(self, matched, grouped_matches):
        config = cast("DeepSeekV4LoRAConverter.Config", self.config)
        missing = [target for target, count in matched.items() if not count]
        if missing:
            message = f"DeepSeek-V4 LoRA targets did not match: {missing}"
            if config.strict:
                raise RuntimeError(message)
            logger.warning(message)
        if config.strict and config.adapt_routed_experts and not grouped_matches:
            raise RuntimeError("DeepSeek-V4 LoRA found no routed GroupedExperts")

    def _make_lora_config(self, cfg: Module.Config):
        config = cast("DeepSeekV4LoRAConverter.Config", self.config)
        factory = linear_lora_class if isinstance(cfg, (BatchedLinear.Config, Linear.Config)) else grouped_lora_class
        owner = getattr(cfg, "_owner", None)
        if owner is None:
            raise ValueError(f"{type(cfg).__name__} has no owner class to wrap with LoRA")
        lora_cls = factory(owner)
        values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}
        grouped = isinstance(cfg, GroupedExperts.Config)
        if grouped:
            values["param_init"] = {
                **(cfg.param_init or {}),
                "w13_lora_a": lambda value: nn.init.kaiming_uniform_(value, a=math.sqrt(5)),
                "w1_lora_b": nn.init.zeros_,
                "w3_lora_b": nn.init.zeros_,
                "w2_lora_a": lambda value: nn.init.kaiming_uniform_(value, a=math.sqrt(5)),
                "w2_lora_b": nn.init.zeros_,
            }
        return lora_cls.Config(
            **values,
            rank=config.rank_experts if grouped else config.rank,
            alpha=config.alpha,
            chunk_rows=config.grouped_chunk_rows if grouped else config.dense_chunk_rows,
        )


@dataclass(kw_only=True, slots=True)
class _LoRAModelConfig(DeepSeekV4Model.Config):
    lora: DeepSeekV4LoRAConverter.Config
