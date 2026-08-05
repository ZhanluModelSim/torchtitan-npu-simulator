# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for Kimi K3 HF checkpoint mapping.

Maps between HuggingFace Kimi-K3 weight names and torchtitan internal names.
Reference: moonshotai/Kimi-K3 config.json (model_type="kimi_k3")
"""

import logging
import re
from typing import Any

import torch

from torchtitan.models.utils import StateDictAdapter

logger = logging.getLogger(__name__)


class KimiK3StateDictAdapter(StateDictAdapter):
    """Adapts HF Kimi-K3 checkpoints to torchtitan KimiK3Model format."""

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # The public Kimi-K3 checkpoint is wrapped by KimiK3ForConditionalGeneration:
        # text-model weights live below ``language_model`` rather than at the root.
        # HF → torchtitan key mapping (non-layer keys)
        self.from_hf_map = {
            "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
            "language_model.model.output_attn_res_norm.weight": "output_attn_res.norm.weight",
            "language_model.model.output_attn_res_proj.weight": "output_attn_res.proj.weight",
            "language_model.model.norm.weight": "norm.weight",
            "language_model.lm_head.weight": "output.weight",
        }

        # Layer-level abstract mappings ({} = layer index)
        self.from_hf_layer_map = {
            # Attention (shared prefix for both KDA and MLA)
            "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "language_model.model.layers.{}.self_attention_res_norm.weight": "layers.{}.self_attention_res.norm.weight",
            "language_model.model.layers.{}.mlp_res_norm.weight": "layers.{}.mlp_res.norm.weight",
            "language_model.model.layers.{}.self_attention_res_proj.weight": "layers.{}.self_attention_res.proj.weight",
            "language_model.model.layers.{}.mlp_res_proj.weight": "layers.{}.mlp_res.proj.weight",
            # MLA-specific
            "language_model.model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.q_a_proj.weight",
            "language_model.model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_a_layernorm.weight",
            "language_model.model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.q_b_proj.weight",
            "language_model.model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": (
                "layers.{}.attention.kv_a_proj_with_mqa.weight"
            ),
            "language_model.model.layers.{}.self_attn.kv_a_layernorm.weight": (
                "layers.{}.attention.kv_a_layernorm.weight"
            ),
            "language_model.model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.kv_b_proj.weight",
            "language_model.model.layers.{}.self_attn.g_proj.weight": "layers.{}.attention.g_proj.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.o_proj.weight",
            # KDA-specific
            "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.k_proj.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.v_proj.weight",
            "language_model.model.layers.{}.self_attn.q_conv1d.weight": "layers.{}.attention.q_conv1d.conv.weight",
            "language_model.model.layers.{}.self_attn.k_conv1d.weight": "layers.{}.attention.k_conv1d.conv.weight",
            "language_model.model.layers.{}.self_attn.v_conv1d.weight": "layers.{}.attention.v_conv1d.conv.weight",
            "language_model.model.layers.{}.self_attn.A_log": "layers.{}.attention.A_log",
            "language_model.model.layers.{}.self_attn.dt_bias": "layers.{}.attention.dt_bias",
            "language_model.model.layers.{}.self_attn.f_a_proj.weight": "layers.{}.attention.f_a_proj.weight",
            "language_model.model.layers.{}.self_attn.f_b_proj.weight": "layers.{}.attention.f_b_proj.weight",
            "language_model.model.layers.{}.self_attn.b_proj.weight": "layers.{}.attention.b_proj.weight",
            "language_model.model.layers.{}.self_attn.o_norm.weight": "layers.{}.attention.o_norm.weight",
            # Dense FFN
            "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.gate_proj.weight",
            "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.up_proj.weight",
            "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.down_proj.weight",
            # MoE
            "language_model.model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.gate.gate.weight",
            "language_model.model.layers.{}.block_sparse_moe.gate.e_score_correction_bias": (
                "layers.{}.moe.gate.e_score_correction_bias"
            ),
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_down_proj.weight": (
                "layers.{}.moe.routed_expert_down_proj.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_up_proj.weight": (
                "layers.{}.moe.routed_expert_up_proj.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_norm.weight": (
                "layers.{}.moe.routed_expert_norm.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.gate_proj.weight": (
                "layers.{}.moe.shared_experts.gate_proj.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.up_proj.weight": (
                "layers.{}.moe.shared_experts.up_proj.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.down_proj.weight": (
                "layers.{}.moe.shared_experts.down_proj.weight"
            ),
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        expert_weights: dict[tuple[str, str], dict[int, Any]] = {}
        for hf_key, tensor in hf_state_dict.items():
            mapped = False

            # Non-layer keys
            if hf_key in self.from_hf_map:
                state_dict[self.from_hf_map[hf_key]] = tensor
                continue

            # Layer keys
            for hf_pattern, titan_pattern in self.from_hf_layer_map.items():
                pattern = re.escape(hf_pattern).replace(re.escape("{}"), r"(\d+)")
                m = re.match(pattern + "$", hf_key)
                if m:
                    layer_idx = m.group(1)
                    titan_key = titan_pattern.format(layer_idx)
                    state_dict[titan_key] = tensor
                    mapped = True
                    break

            # Expert weights: ...block_sparse_moe.experts.{j}.w1.weight etc.
            if not mapped and "block_sparse_moe.experts" in hf_key:
                expert_match = re.match(
                    r"language_model\.model\.layers\.(\d+)\.block_sparse_moe"
                    r"\.experts\.(\d+)\.(w[123])\.weight$",
                    hf_key,
                )
                if expert_match:
                    layer_idx, expert_idx, weight_name = expert_match.groups()
                    group_key = (layer_idx, weight_name)
                    expert_weights.setdefault(group_key, {})[
                        int(expert_idx)
                    ] = tensor
                    mapped = True

            if not mapped:
                logger.debug("from_hf: skipping unmapped key: %s", hf_key)

        for (layer_idx, weight_name), weights_by_expert in expert_weights.items():
            expert_indices = sorted(weights_by_expert)
            if expert_indices != list(range(len(expert_indices))):
                raise ValueError(
                    "Kimi expert checkpoint shards must contain contiguous "
                    f"indices starting at zero; got {expert_indices}"
                )
            state_dict[
                f"layers.{layer_idx}.moe.experts.{weight_name}"
            ] = torch.stack(
                [weights_by_expert[index] for index in expert_indices],
                dim=0,
            )

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        to_hf_layer_map = {v: k for k, v in self.from_hf_layer_map.items()}

        hf_state_dict = {}
        for key, value in state_dict.items():
            if key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value
                continue

            # Layer keys
            mapped = False
            for titan_pattern, hf_pattern in to_hf_layer_map.items():
                pattern = re.escape(titan_pattern).replace(re.escape("{}"), r"(\d+)")
                m = re.match(pattern + "$", key)
                if m:
                    layer_idx = m.group(1)
                    hf_key = hf_pattern.format(layer_idx)
                    hf_state_dict[hf_key] = value
                    mapped = True
                    break

            # Grouped expert weights
            if not mapped:
                expert_match = re.match(
                    r"layers\.(\d+)\.moe\.experts\.(w[123])$",
                    key,
                )
                if expert_match:
                    layer_idx, weight_name = expert_match.groups()
                    for expert_idx, expert_weight in enumerate(value.unbind(0)):
                        hf_key = (
                            f"language_model.model.layers.{layer_idx}"
                            f".block_sparse_moe.experts.{expert_idx}"
                            f".{weight_name}.weight"
                        )
                        hf_state_dict[hf_key] = expert_weight
                    mapped = True

            if not mapped:
                logger.debug("to_hf: skipping unmapped key: %s", key)

        return hf_state_dict
