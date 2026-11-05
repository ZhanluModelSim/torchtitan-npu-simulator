# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit conversion between the original raw schema and the reference model.

The native TorchTitan BlockDiffusionStateDictAdapter continues to consume the
original ``backbone.*`` schema with stacked experts. No checkpoint is silently
interpreted as the new module layout. Conversion preserves tensor values/dtypes;
model architecture compatibility must still be checked by strict load_state_dict.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

_ROOT_TO_LEGACY = {
    "embed_tokens.weight": "backbone.embed_tokens.weight",
    "norm.weight": "backbone.norm.weight",
    "lm_head.weight": "lm_head.weight",
}
_COMMON_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}
_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def _check_tensor(key: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} is not a tensor")


def to_legacy_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Export to the schema accepted by the existing native state-dict adapter.

    The original raw model supports at most one shared expert. More shared experts
    cannot be represented faithfully and are rejected rather than dropped or summed.
    """
    converted = {}
    experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for key, value in state.items():
        _check_tensor(key, value)
        if key in _ROOT_TO_LEGACY:
            converted[_ROOT_TO_LEGACY[key]] = value
            continue
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
        if match is None:
            raise ValueError(f"unsupported reference state key: {key}")
        layer, suffix = match.groups()
        prefix = f"backbone.layers.{layer}."
        if suffix in _COMMON_SUFFIXES:
            converted[prefix + suffix] = value
        elif suffix == "mlp.router.weight":
            converted[prefix + "mlp.gate.weight"] = value
        elif match := re.fullmatch(r"mlp.shared_experts\.(\d+)\.(\w+)\.weight", suffix):
            expert, projection = match.groups()
            if expert != "0" or projection not in _PROJECTIONS:
                raise ValueError("legacy schema supports only shared expert 0 with SwiGLU projections")
            converted[prefix + f"mlp.shared_expert.{projection}.weight"] = value
        elif match := re.fullmatch(r"mlp.experts\.(\d+)\.(\w+)\.weight", suffix):
            expert, projection = match.groups()
            if projection not in _PROJECTIONS:
                raise ValueError(f"unsupported expert projection: {key}")
            experts.setdefault(layer, {}).setdefault(int(expert), {})[projection] = value
        else:
            raise ValueError(f"unsupported reference state key: {key}")
    for layer, members in experts.items():
        if sorted(members) != list(range(len(members))):
            raise ValueError("expert IDs must be contiguous starting at zero")
        gate_up, down = [], []
        expected_shape: torch.Size | None = None
        expected_storage: tuple[torch.dtype, torch.device] | None = None
        for expert in range(len(members)):
            parts = members[expert]
            if set(parts) != _PROJECTIONS:
                raise ValueError("each expert requires gate_proj, up_proj and down_proj")
            gate, up, output = (parts[name] for name in ("gate_proj", "up_proj", "down_proj"))
            if gate.ndim != 2 or gate.shape != up.shape or tuple(output.shape) != tuple(reversed(gate.shape)):
                raise ValueError("inconsistent SwiGLU expert weight shapes")
            if expected_shape is not None and gate.shape != expected_shape:
                raise ValueError("legacy stacked experts require identical dimensions")
            if (
                gate.dtype != up.dtype
                or gate.dtype != output.dtype
                or gate.device != up.device
                or gate.device != output.device
            ):
                raise ValueError("expert projection dtypes and devices must match")
            if expected_storage is not None and (gate.dtype, gate.device) != expected_storage:
                raise ValueError("legacy stacked experts require identical dtypes and devices")
            expected_shape = gate.shape
            expected_storage = (gate.dtype, gate.device)
            gate_up.append(torch.cat((gate.T, up.T), dim=-1))
            down.append(output.T)
        converted[f"backbone.layers.{layer}.mlp.experts.w13"] = torch.stack(gate_up)
        converted[f"backbone.layers.{layer}.mlp.experts.w2"] = torch.stack(down)
    return converted


def from_legacy_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Import original raw weights; unknown fields and incomplete experts fail."""
    root_from_legacy = {value: key for key, value in _ROOT_TO_LEGACY.items()}
    converted = {}
    experts: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        _check_tensor(key, value)
        if key in root_from_legacy:
            converted[root_from_legacy[key]] = value
            continue
        match = re.fullmatch(r"backbone\.layers\.(\d+)\.(.+)", key)
        if match is None:
            raise ValueError(f"unsupported legacy state key: {key}")
        layer, suffix = match.groups()
        prefix = f"layers.{layer}."
        if suffix in _COMMON_SUFFIXES:
            converted[prefix + suffix] = value
        elif suffix == "mlp.gate.weight":
            converted[prefix + "mlp.router.weight"] = value
        elif match := re.fullmatch(r"mlp.shared_expert\.(\w+)\.weight", suffix):
            projection = match.group(1)
            if projection not in _PROJECTIONS:
                raise ValueError(f"unsupported shared projection: {key}")
            converted[prefix + f"mlp.shared_experts.0.{projection}.weight"] = value
        elif suffix in ("mlp.experts.w13", "mlp.experts.w2"):
            experts.setdefault(layer, {})[suffix.rsplit(".", 1)[1]] = value
        else:
            raise ValueError(f"unsupported legacy state key: {key}")
    for layer, parts in experts.items():
        if set(parts) != {"w13", "w2"}:
            raise ValueError("legacy experts require both w13 and w2")
        gate_up, down = parts["w13"], parts["w2"]
        if gate_up.ndim != 3 or gate_up.shape[-1] % 2:
            raise ValueError("legacy w13 must have shape [experts, hidden, 2 * intermediate]")
        num_experts, hidden, doubled_intermediate = gate_up.shape
        if not num_experts or not hidden or not doubled_intermediate:
            raise ValueError("legacy expert dimensions must be positive")
        if tuple(down.shape) != (num_experts, doubled_intermediate // 2, hidden):
            raise ValueError("legacy w2 shape disagrees with w13")
        if gate_up.dtype != down.dtype or gate_up.device != down.device:
            raise ValueError("legacy expert dtypes and devices must match")
        for expert in range(num_experts):
            gate, up = gate_up[expert].chunk(2, dim=-1)
            prefix = f"layers.{layer}.mlp.experts.{expert}."
            converted[prefix + "gate_proj.weight"] = gate.T
            converted[prefix + "up_proj.weight"] = up.T
            converted[prefix + "down_proj.weight"] = down[expert].T
    return converted
