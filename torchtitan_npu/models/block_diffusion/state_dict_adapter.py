# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict bridge for the original Block Diffusion workload schema."""

import re
from typing import Any

import torch

from torchtitan.models.utils import StateDictAdapter


class BlockDiffusionStateDictAdapter(StateDictAdapter):
    """Map the ``raw_model/block_diff`` layout to the native model layout."""

    _ROOT_FROM_RAW = {
        "backbone.embed_tokens.weight": "tok_embeddings.weight",
        "backbone.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    _LAYER_FROM_RAW = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate.weight": "moe.router.gate.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.shared_expert.gate_proj.weight": "moe.shared_experts.w1.weight",
        "mlp.shared_expert.down_proj.weight": "moe.shared_experts.w2.weight",
        "mlp.shared_expert.up_proj.weight": "moe.shared_experts.w3.weight",
    }

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)

    def from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert the raw workload schema into TorchTitan's state dict.

        The method name follows TorchTitan's adapter protocol.  The source is
        the checked-in raw workload rather than a claimed public HF checkpoint.
        """

        converted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key in self._ROOT_FROM_RAW:
                converted[self._ROOT_FROM_RAW[key]] = value
                continue

            match = re.fullmatch(r"backbone\.layers\.(\d+)\.(.+)", key)
            if match is None:
                continue
            layer, suffix = match.groups()
            mapped_suffix = self._LAYER_FROM_RAW.get(suffix)
            if mapped_suffix is not None:
                converted[f"layers.{layer}.{mapped_suffix}"] = value
                continue

            if suffix == "mlp.experts.w13":
                gate, up = value.chunk(2, dim=-1)
                converted[f"layers.{layer}.moe.experts.w1"] = gate.transpose(-1, -2)
                converted[f"layers.{layer}.moe.experts.w3"] = up.transpose(-1, -2)
            elif suffix == "mlp.experts.w2":
                converted[f"layers.{layer}.moe.experts.w2"] = value.transpose(-1, -2)

        return converted

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan weights back to the raw workload schema."""

        converted: dict[str, Any] = {}
        root_to_raw = {value: key for key, value in self._ROOT_FROM_RAW.items()}
        layer_to_raw = {value: key for key, value in self._LAYER_FROM_RAW.items()}
        expert_parts: dict[str, dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            if key in root_to_raw:
                converted[root_to_raw[key]] = value
                continue

            match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
            if match is None:
                continue
            layer, suffix = match.groups()
            raw_suffix = layer_to_raw.get(suffix)
            if raw_suffix is not None:
                converted[f"backbone.layers.{layer}.{raw_suffix}"] = value
                continue

            expert_match = re.fullmatch(r"moe\.experts\.(w[123])", suffix)
            if expert_match is not None:
                expert_parts.setdefault(layer, {})[expert_match.group(1)] = value

        for layer, parts in expert_parts.items():
            if "w1" in parts and "w3" in parts:
                converted[f"backbone.layers.{layer}.mlp.experts.w13"] = torch.cat(
                    (parts["w1"].transpose(-1, -2), parts["w3"].transpose(-1, -2)),
                    dim=-1,
                )
            if "w2" in parts:
                converted[f"backbone.layers.{layer}.mlp.experts.w2"] = parts[
                    "w2"
                ].transpose(-1, -2)

        return converted
