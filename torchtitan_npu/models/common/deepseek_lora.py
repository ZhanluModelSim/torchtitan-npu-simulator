# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""DeepSeek projections adapted through TorchTitan's standard LoRA transform."""

import copy
import math
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, cast

import spmd_types as spmd
import torch.nn as nn
from torchtitan.config import derive
from torchtitan.config.transform import GroupedLinearLoRAHandler, LinearLoRAHandler, LoRATransform
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.lora import (
    _LoRAGroupedLinearMixin,
    _LoRALinearMixin,
    get_lora_grouped_linear,
    get_lora_linear,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

DEEPSEEK_V4_LORA_TARGETS = (
    "lm_head",
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
    "wgate",
    "gate",
    "w13",
    "w2",
    "e_proj",
    "h_proj",
)


def _adapter_sharding(base_sharding):
    placement = base_sharding.state_shardings.get("weight") if base_sharding else None
    if placement is not None and placement == dense_param_placement(tp=spmd.R):
        replicated = ShardingConfig(state_shardings={"weight": placement})
        return replicated, replicated
    return _LoRALinearMixin._adapter_sharding(base_sharding)


@cache
def _replicated_lora_linear(owner):
    standard = get_lora_linear(owner)

    class ReplicatedLoRALinear(standard):
        @dataclass(kw_only=True, slots=True)
        class Config(standard.Config):
            pass

        _adapter_sharding = staticmethod(_adapter_sharding)

    return ReplicatedLoRALinear


class DeepSeekLinearLoRAHandler(LinearLoRAHandler):
    def make_config(self, cfg, *, rank, alpha):
        cls = _replicated_lora_linear(cfg._owner)
        # get_lora_linear creates the Config subclass dynamically with these fields.
        return cast("Any", cls.Config)(
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=rank,
            alpha=alpha,
        )


@cache
def _replicated_lora_grouped_linear(owner):
    standard = get_lora_grouped_linear(owner)

    class ReplicatedLoRAGroupedLinear(standard):
        @dataclass(kw_only=True, slots=True)
        class Config(standard.Config):
            pass

        @staticmethod
        def _adapter_sharding(base_sharding):
            placement = base_sharding.state_shardings.get("weight") if base_sharding else None
            if placement is not None and placement == dense_param_placement(tp=spmd.R):
                replicated = ShardingConfig(state_shardings={"weight": placement})
                return replicated, replicated
            return _LoRAGroupedLinearMixin._adapter_sharding(base_sharding)

    return ReplicatedLoRAGroupedLinear


class DeepSeekGroupedLinearLoRAHandler(GroupedLinearLoRAHandler):
    def make_config(self, cfg, *, rank, alpha):
        if rank % 8:
            raise ValueError(f"Grouped LoRA rank must be divisible by 8, got {rank}")
        cls = _replicated_lora_grouped_linear(cfg._owner)
        return cast("Any", cls.Config)(
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=rank,
            alpha=alpha,
        )


class BatchedLinearLoRA(BatchedLinear):
    """Per-head projection with standard named Linear LoRA factors."""

    @dataclass(kw_only=True, slots=True)
    class Config(BatchedLinear.Config):
        rank: int
        alpha: float

    def __init__(self, config):
        super().__init__(config)
        self.weight.requires_grad_(False)
        self._lora_scaling = config.alpha / config.rank
        a_sharding, b_sharding = _adapter_sharding(config.sharding_config)
        self.lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=config.rank,
            sharding_config=a_sharding,
            param_init={"weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))},
        ).build()
        self.lora_b = BatchedLinear.Config(
            n_heads=config.n_heads,
            in_features=config.rank,
            out_features=config.out_features,
            sharding_config=b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()

    def forward(self, x):
        return super().forward(x) + self._lora_scaling * self.lora_b(self.lora_a(x))


class BatchedLinearLoRAHandler:
    config_type = BatchedLinear.Config

    def make_config(self, cfg, *, rank, alpha):
        return derive(cfg, BatchedLinearLoRA.Config, rank=rank, alpha=alpha)


@dataclass(kw_only=True, slots=True)
class DeepSeekLoRATransform(LoRATransform):
    """Preserve DeepSeek's frozen indexer policy with the upstream transform."""

    def transform(self, model):
        frozen_indexers = {}
        freeze = LoRATransform(handlers=(), target_modules=[])
        for fqn, cfg, _, attr in model.traverse(Module.Config, recurse=True):
            if attr == "indexer":
                frozen_indexers[fqn] = freeze.transform(copy.deepcopy(cfg))
        transformed = LoRATransform.transform(self, model)
        for fqn, _, parent, attr in transformed.traverse(Module.Config, recurse=True):
            if fqn in frozen_indexers:
                assert parent is not None and isinstance(attr, str)
                setattr(parent, attr, frozen_indexers[fqn])
        return transformed


def deepseek_v4_lora_transform(*, rank: int = 16, alpha: float = 32.0) -> LoRATransform:
    return DeepSeekLoRATransform(
        handlers=(DeepSeekGroupedLinearLoRAHandler(), BatchedLinearLoRAHandler(), DeepSeekLinearLoRAHandler()),
        rank=rank,
        alpha=alpha,
        target_modules=list(DEEPSEEK_V4_LORA_TARGETS),
    )


def is_lora_parameter(name: str) -> bool:
    return any(part in {"lora_A", "lora_B", "lora_a", "lora_b"} for part in name.split("."))


_PEFT_MODULES = {
    "attention.wq_a": "self_attn.q_a_proj",
    "attention.wq_b": "self_attn.q_b_proj",
    "attention.wkv": "self_attn.kv_proj",
    "attention.wo_a": "self_attn.o_a_proj",
    "moe.router.gate": "mlp.gate",
    "attention.wo_b": "self_attn.o_b_proj",
    "attention.compressor.wkv": "self_attn.compressor.kv_proj",
    "attention.compressor.wgate": "self_attn.compressor.gate_proj",
    "attention.indexer.wq_b": "self_attn.indexer.q_b_proj",
    "attention.indexer.weights_proj": "self_attn.indexer.weights_proj",
    "attention.indexer.compressor.wkv": "self_attn.indexer.compressor.kv_proj",
    "attention.indexer.compressor.wgate": "self_attn.indexer.compressor.gate_proj",
    "attention.indexer.wk": "self_attn.indexer.k_proj",
}


class DeepSeekPEFTAdapter:
    """Standard PEFT conversion for upstream Linear/GroupedLinear adapters."""

    hf_assets_path: str | None

    def _initialize_peft(self, model_config):
        from torchtitan.protocols.module import Module

        self._lora_configs = {
            fqn: cfg
            for fqn, cfg, _, _ in model_config.traverse(Module.Config, recurse=True)
            if hasattr(cfg, "rank") and hasattr(cfg, "alpha")
        }

    def to_peft(self, state_dict):
        result = {}
        for module, cfg in self._lora_configs.items():
            if module.startswith("mtp_layers."):
                raise NotImplementedError("PEFT DeepSeek export does not support MTP adapter modules")
            a_key, b_key = f"{module}.lora_a.weight", f"{module}.lora_b.weight"
            if a_key not in state_dict or b_key not in state_dict:
                raise ValueError(f"Missing standard LoRA factors for {module}")
            a, b = state_dict[a_key], state_dict[b_key]
            if module == "lm_head":
                target = "base_model.model.lm_head"
                result[f"{target}.lora_A.weight"] = a.contiguous()
                result[f"{target}.lora_B.weight"] = b.contiguous()
                continue
            prefix, suffix = module.split(".", 2)[:2], module.split(".", 2)[-1]
            layer = ".".join(prefix)
            if suffix.startswith("moe.routed_experts."):
                projection = suffix.rsplit(".", 1)[-1]
                hf_module = "mlp.experts"
                if projection == "w13" and f"{layer}.moe.routed_experts.w2" in self._lora_configs:
                    hf_module += ".base_layer"
                if projection == "w13":
                    b = b.flatten(1, 2)
                a = a.flatten(0, 1)
                b = b.permute(1, 2, 0).flatten(1)
                target = f"base_model.model.model.{layer}.{hf_module}"
                result[f"{target}.lora_A.weight"] = a.contiguous()
                result[f"{target}.lora_B.weight"] = b.contiguous()
            elif suffix == "moe.shared_experts.w13":
                for index, projection in enumerate(("gate_proj", "up_proj")):
                    target = f"base_model.model.model.{layer}.mlp.shared_experts.{projection}"
                    result[f"{target}.lora_A.weight"] = a.contiguous()
                    result[f"{target}.lora_B.weight"] = b[index].contiguous()
            elif suffix == "moe.shared_experts.w2":
                target = f"base_model.model.model.{layer}.mlp.shared_experts.down_proj"
                result[f"{target}.lora_A.weight"] = a.contiguous()
                result[f"{target}.lora_B.weight"] = b.contiguous()
            else:
                if suffix not in _PEFT_MODULES:
                    raise ValueError(f"No standard PEFT projection mapping for {module}")
                target = f"base_model.model.model.{layer}.{_PEFT_MODULES[suffix]}"
                result[f"{target}.lora_A.weight"] = a.contiguous()
                result[f"{target}.lora_B.weight"] = b.contiguous()
        if not result:
            raise ValueError("No standard LoRA adapter tensors were found")
        return result

    def peft_adapter_config(self, *, base_model_name_or_path=None):
        if not self._lora_configs:
            raise ValueError("The model has no standard LoRA adapters")
        configs = tuple(self._lora_configs.values())
        rank, alpha = configs[0].rank, configs[0].alpha
        if any(cfg.rank != rank or cfg.alpha != alpha for cfg in configs):
            raise ValueError("PEFT export requires a shared adapter rank and alpha")
        target_parameters = []
        expert_targets = []
        for module in self._lora_configs:
            if module.startswith("mtp_layers."):
                raise NotImplementedError("PEFT DeepSeek export does not support MTP adapter modules")
            if module == "lm_head":
                target_parameters.append("lm_head.weight")
                continue
            layer, suffix = module.rsplit(".", 1)
            if ".moe.routed_experts." in module:
                layer = module.split(".moe.", 1)[0]
                projection = "gate_up_proj" if suffix == "w13" else "down_proj"
                expert_targets.append((layer, 0 if suffix == "w13" else 1, f"model.{layer}.mlp.experts.{projection}"))
            elif ".moe.shared_experts." in module:
                layer = module.split(".moe.", 1)[0]
                projections = ("gate_proj", "up_proj") if suffix == "w13" else ("down_proj",)
                target_parameters.extend(
                    f"model.{layer}.mlp.shared_experts.{projection}.weight" for projection in projections
                )
            else:
                parts = module.split(".", 2)
                suffix = parts[-1]
                if suffix not in _PEFT_MODULES:
                    raise ValueError(f"No standard PEFT projection mapping for {module}")
                target_parameters.append(f"model.{'.'.join(parts[:2])}.{_PEFT_MODULES[suffix]}.weight")
        target_parameters = sorted(target_parameters) + [target for _, _, target in sorted(expert_targets)]
        return {
            "base_model_name_or_path": base_model_name_or_path or self.hf_assets_path or "",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": alpha,
            "bias": "none",
            "lora_dropout": 0.0,
            "inference_mode": True,
            "target_modules": [],
            "target_parameters": target_parameters,
            "use_dora": False,
            "use_rslora": False,
        }
