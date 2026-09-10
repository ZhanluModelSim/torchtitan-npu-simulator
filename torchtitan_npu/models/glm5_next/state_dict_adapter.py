# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for glm5_next HF-style checkpoint mapping.

The HF naming follows the raw reference (``raw_model/unified_mm/model.py``)
module tree:

- ``model.layers.{i}.*`` nominal decoder layers (0..95)
- ``model.layers.{i}.attn_hc.*`` / ``ffn_hc.*``  hyper-connection parameters
- ``model.layers.{i}.self_attn.*``  KDA or DSA attention (by layer type)
- ``model.layers.{i}.self_attn.indexer.*``  DSA k-pool indexer
- ``model.layers.{i}.mlp.*``  dense clamp-swiglu MLP
- ``model.layers.{i}.block_sparse_moe.gate.*``  router
- ``model.layers.{i}.block_sparse_moe.experts.{j}.gate_up_proj/down_proj.weight``
  per-expert weights; ``gate_up_proj [2*inter, d]`` splits into framework
  ``w1`` (gate, first inter rows) and ``w3`` (up, second inter rows)
- ``model.layers.{i}.block_sparse_moe.shared_experts.*``  shared expert MLP
- ``model.visual.*``  vision tower

Framework mapping of the nominal layer range:
``[0, pre_layers) -> layers.{i}``, ``[pre, pre+looped) -> loop_block`` (the
weight-shared loop region: from_hf keeps the last slot, to_hf fans the shared
block out to every nominal slot), ``[pre+looped, 96) -> post_layers.{i}``.

Round-trip closure (from_hf -> to_hf) is the meta acceptance requirement;
real HF weight compatibility is out of scope for now.
"""

import logging
import re
from typing import Any

import torch

from torchtitan.models.utils import StateDictAdapter

logger = logging.getLogger(__name__)

_EXPERT_W13 = ("gate_up_proj",)  # split into framework w1 (gate) / w3 (up)
_HF_INTERMEDIATE_SPLIT = 2


def _layer_group(layer_idx: int, pre_layers: int) -> str:
    return f"layers.{layer_idx}"


def _group_to_hf_slots(group: str, pre_layers: int, looped_layers: int, num_layers: int) -> list[int]:
    if int(group.split(".")[1]) == pre_layers:
        # The weight-shared loop block fans out to every nominal loop slot.
        return list(range(pre_layers, pre_layers + looped_layers))
    return [int(group.split(".")[1])]


class Glm5NextStateDictAdapter(StateDictAdapter):
    """Adapts glm5_next HF-style checkpoints to the torchtitan model format."""

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        text = model_config  # text fields live on the top-level config
        self.pre_layers = text.pre_layers
        self.looped_layers = text.looped_layers
        self.num_layers = text.num_hidden_layers

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # HF layer-scope patterns -> framework group-scope patterns. ``{g}`` is
        # substituted with the framework group prefix for the nominal layer.
        self.from_hf_layer_map = {
            "model.layers.{}.input_layernorm.weight": "{g}.input_layernorm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "{g}.post_attention_layernorm.weight",
            # mHC hyper connections (raw stores fn/base/scale per site)
            "model.layers.{}.attn_hc.fn": "{g}.hc_attn_fn",
            "model.layers.{}.attn_hc.base": "{g}.hc_attn_base",
            "model.layers.{}.attn_hc.scale": "{g}.hc_attn_scale",
            "model.layers.{}.ffn_hc.fn": "{g}.hc_ffn_fn",
            "model.layers.{}.ffn_hc.base": "{g}.hc_ffn_base",
            "model.layers.{}.ffn_hc.scale": "{g}.hc_ffn_scale",
            # KDA (Glm5NextTextLinearAttention) / DSA (Glm5NextTextAttention)
            "model.layers.{}.self_attn.q_proj.weight": "{g}.attention.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "{g}.attention.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight": "{g}.attention.v_proj.weight",
            "model.layers.{}.self_attn.conv1d.weight": "{g}.attention.conv1d.conv.weight",
            "model.layers.{}.self_attn.forget_gate.f_a_proj.weight": "{g}.attention.forget_gate.f_a_proj.weight",
            "model.layers.{}.self_attn.forget_gate.f_b_proj.weight": "{g}.attention.forget_gate.f_b_proj.weight",
            "model.layers.{}.self_attn.forget_gate.dt_bias": "{g}.attention.forget_gate.dt_bias",
            "model.layers.{}.self_attn.forget_gate.A_log": "{g}.attention.forget_gate.A_log",
            "model.layers.{}.self_attn.b_proj.weight": "{g}.attention.b_proj.weight",
            "model.layers.{}.self_attn.g_a_proj.weight": "{g}.attention.g_a_proj.weight",
            "model.layers.{}.self_attn.g_b_proj.weight": "{g}.attention.g_b_proj.weight",
            "model.layers.{}.self_attn.o_norm.weight": "{g}.attention.o_norm.weight",
            "model.layers.{}.self_attn.o_proj.weight": "{g}.attention.o_proj.weight",
            # DSA MLA (NoPE)
            "model.layers.{}.self_attn.q_a_proj.weight": "{g}.attention.q_a_proj.weight",
            "model.layers.{}.self_attn.q_a_layernorm.weight": "{g}.attention.q_a_norm.weight",
            "model.layers.{}.self_attn.q_b_proj.weight": "{g}.attention.q_b_proj.weight",
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "{g}.attention.kv_a_proj_with_mqa.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "{g}.attention.kv_a_norm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "{g}.attention.kv_b_proj.weight",
            # DSA indexer (k-pool compressed)
            "model.layers.{}.self_attn.indexer.wq_b.weight": "{g}.attention.indexer.wq_b.weight",
            "model.layers.{}.self_attn.indexer.wk.weight": "{g}.attention.indexer.wk.weight",
            "model.layers.{}.self_attn.indexer.k_norm.weight": "{g}.attention.indexer.k_norm.weight",
            "model.layers.{}.self_attn.indexer.k_norm.bias": "{g}.attention.indexer.k_norm.bias",
            "model.layers.{}.self_attn.indexer.weights_proj.weight": "{g}.attention.indexer.weights_proj.weight",
            "model.layers.{}.self_attn.indexer.index_kpool_compress_ape": "{g}.attention.indexer.kpool_ape",
            "model.layers.{}.self_attn.indexer.index_kpool_compress_gate": "{g}.attention.indexer.kpool_gate",
            # dense MLP
            "model.layers.{}.mlp.gate_proj.weight": "{g}.mlp.gate_proj.weight",
            "model.layers.{}.mlp.up_proj.weight": "{g}.mlp.up_proj.weight",
            "model.layers.{}.mlp.down_proj.weight": "{g}.mlp.down_proj.weight",
            # MoE router / shared experts
            "model.layers.{}.block_sparse_moe.gate.weight": "{g}.moe.gate.gate.weight",
            "model.layers.{}.block_sparse_moe.gate.e_score_correction_bias": (
                "{g}.moe.gate.e_score_correction_bias"
            ),
            "model.layers.{}.block_sparse_moe.shared_experts.gate_proj.weight": (
                "{g}.moe.shared_experts.gate_proj.weight"
            ),
            "model.layers.{}.block_sparse_moe.shared_experts.up_proj.weight": (
                "{g}.moe.shared_experts.up_proj.weight"
            ),
            "model.layers.{}.block_sparse_moe.shared_experts.down_proj.weight": (
                "{g}.moe.shared_experts.down_proj.weight"
            ),
        }

        self.from_hf_vision_map = {
            "model.visual.patch_embed.proj.weight": "visual.patch_embed.weight",
            "model.visual.patch_embed.proj.bias": "visual.patch_embed.bias",
            "model.visual.post_layernorm.weight": "visual.post_layernorm.weight",
            "model.visual.downsample.weight": "visual.downsample.weight",
            "model.visual.downsample.bias": "visual.downsample.bias",
            "model.visual.merger.proj.weight": "visual.merger.proj.weight",
            "model.visual.merger.post_projection_norm.weight": "visual.merger.post_projection_norm.weight",
            "model.visual.merger.post_projection_norm.bias": "visual.merger.post_projection_norm.bias",
            "model.visual.merger.gate_proj.weight": "visual.merger.gate_proj.weight",
            "model.visual.merger.up_proj.weight": "visual.merger.up_proj.weight",
            "model.visual.merger.down_proj.weight": "visual.merger.down_proj.weight",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        routed_w1: dict[str, dict[int, Any]] = {}
        routed_w3: dict[str, dict[int, Any]] = {}
        routed_w2: dict[str, dict[int, Any]] = {}

        for hf_key, tensor in hf_state_dict.items():
            if hf_key in self.from_hf_map:
                state_dict[self.from_hf_map[hf_key]] = tensor
                continue

            if hf_key in self.from_hf_vision_map:
                state_dict[self.from_hf_vision_map[hf_key]] = tensor
                continue

            vision_block = re.match(
                r"model\.visual\.blocks\.(\d+)\.(norm1|norm2|attn|mlp)\.(.+)$", hf_key
            )
            if vision_block:
                block_idx, module, tail = vision_block.groups()
                state_dict[f"visual.blocks.{block_idx}.{module}.{tail}"] = tensor
                continue

            layer_match = re.match(r"model\.layers\.(\d+)\.(.+)$", hf_key)
            if layer_match:
                layer_idx, rest = layer_match.groups()
                group = _layer_group(int(layer_idx), self.pre_layers)

                routed_expert = re.match(
                    r"block_sparse_moe\.experts\.(\d+)\.(gate_up_proj|down_proj)\.weight$", rest
                )
                if routed_expert:
                    expert_idx, weight_name = routed_expert.groups()
                    if weight_name == "gate_up_proj":
                        gate, up = torch.chunk(tensor, _HF_INTERMEDIATE_SPLIT, dim=0)
                        routed_w1.setdefault(group, {})[int(expert_idx)] = gate
                        routed_w3.setdefault(group, {})[int(expert_idx)] = up
                    else:
                        routed_w2.setdefault(group, {})[int(expert_idx)] = tensor
                    continue

                mapped = False
                for hf_pattern, titan_pattern in self.from_hf_layer_map.items():
                    pattern = re.escape(hf_pattern).replace(re.escape("{}"), r"(\d+)")
                    match = re.match(pattern + "$", hf_key)
                    if match:
                        state_dict[titan_pattern.format(g=group)] = tensor
                        mapped = True
                        break
                if not mapped:
                    logger.debug("from_hf: skipping unmapped layer key: %s", hf_key)
                continue

            logger.debug("from_hf: skipping unmapped key: %s", hf_key)

        for group, weights_by_expert in routed_w1.items():
            state_dict[f"{group}.moe.experts.w1"] = torch.stack(
                [weights_by_expert[index] for index in sorted(weights_by_expert)], dim=0
            )
        for group, weights_by_expert in routed_w3.items():
            state_dict[f"{group}.moe.experts.w3"] = torch.stack(
                [weights_by_expert[index] for index in sorted(weights_by_expert)], dim=0
            )
        for group, weights_by_expert in routed_w2.items():
            state_dict[f"{group}.moe.experts.w2"] = torch.stack(
                [weights_by_expert[index] for index in sorted(weights_by_expert)], dim=0
            )
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        to_hf_map = {value: key for key, value in self.from_hf_map.items()}
        to_hf_layer_map = {value: key for key, value in self.from_hf_layer_map.items()}
        to_hf_vision_map = {value: key for key, value in self.from_hf_vision_map.items()}
        grouped_experts: dict[tuple[str, str], Any] = {}

        for key, value in state_dict.items():
            if key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value
                continue

            if key in to_hf_vision_map:
                hf_state_dict[to_hf_vision_map[key]] = value
                continue

            vision_block = re.match(r"visual\.blocks\.(\d+)\.(norm1|norm2|attn|mlp)\.(.+)$", key)
            if vision_block:
                block_idx, module, tail = vision_block.groups()
                hf_state_dict[f"model.visual.blocks.{block_idx}.{module}.{tail}"] = value
                continue

            group_match = re.match(r"(layers\.\d+)\.(.+)$", key)
            if not group_match:
                logger.debug("to_hf: skipping unmapped key: %s", key)
                continue
            group, rest = group_match.groups()

            expert_match = re.match(r"moe\.experts\.(w1|w3|w2)$", rest)
            if expert_match:
                grouped_experts[(group, expert_match.group(1))] = value
                continue

            mapped = False
            for titan_pattern, hf_pattern in to_hf_layer_map.items():
                pattern = re.escape(titan_pattern).replace(
                    re.escape("{g}"), r"layers\.\d+"
                )
                match = re.match(pattern + "$", key)
                if match:
                    for slot in _group_to_hf_slots(group, self.pre_layers, self.looped_layers, self.num_layers):
                        hf_state_dict[hf_pattern.format(slot)] = value
                    mapped = True
                    break
            if not mapped:
                logger.debug("to_hf: skipping unmapped group key: %s", key)

        for (group, weight_name), stacked in grouped_experts.items():
            for slot in _group_to_hf_slots(group, self.pre_layers, self.looped_layers, self.num_layers):
                for expert_idx, expert_weight in enumerate(stacked.unbind(0)):
                    if weight_name == "w1":
                        hf_key = f"model.layers.{slot}.block_sparse_moe.experts.{expert_idx}.gate_up_proj.weight"
                        w3 = grouped_experts[(group, "w3")][expert_idx]
                        hf_state_dict[hf_key] = torch.cat([expert_weight, w3], dim=0)
                    elif weight_name == "w3":
                        continue  # emitted together with w1
                    else:
                        hf_state_dict[
                            f"model.layers.{slot}.block_sparse_moe.experts.{expert_idx}.down_proj.weight"
                        ] = expert_weight

        return hf_state_dict
