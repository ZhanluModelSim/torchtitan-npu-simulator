# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""DeepSeek V4 LoRA exports and offline adapter merge helpers."""

import re
from dataclasses import dataclass

import torch

from torchtitan_npu.models.common.deepseek_lora import (
    DEEPSEEK_V4_LORA_TARGETS as DEEPSEEK_V4_LORA_TARGETS,
)
from torchtitan_npu.models.common.deepseek_lora import (
    BatchedLinearLoRA as BatchedLinearLoRA,
)
from torchtitan_npu.models.common.deepseek_lora import (
    BatchedLinearLoRAHandler as BatchedLinearLoRAHandler,
)
from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekGroupedLinearLoRAHandler as DeepSeekGroupedLinearLoRAHandler,
)
from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekLinearLoRAHandler as DeepSeekLinearLoRAHandler,
)
from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekLoRATransform as DeepSeekLoRATransform,
)
from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekPEFTAdapter as DeepSeekPEFTAdapter,
)
from torchtitan_npu.models.common.deepseek_lora import (
    deepseek_v4_lora_transform as deepseek_v4_lora_transform,
)
from torchtitan_npu.models.common.deepseek_lora import (
    is_lora_parameter as is_lora_parameter,
)

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
