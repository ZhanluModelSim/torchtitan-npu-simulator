# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for ar_llm HF-style checkpoint mapping.

There is no public checkpoint for this architecture yet, so the HF naming
below *defines* the interchange schema (see MODEL_CONTRACT.md):

- ``model.layers.{i}.attn.*``            MLA low-rank projections (CSA/HCA)
- ``model.layers.{i}.attn.indexer.*``    HCA indexer
- ``model.layers.{i}.self_attn.*``       KDA projections and gates
- ``model.layers.{i}.attn.o_down/o_up``  grouped O weights ``[G, out, in]``
- ``model.layers.{i}.mhc_attn.*`` / ``mhc_ffn.*``  hyper-connections
- ``model.layers.{i}.block_sparse_moe.gate.weight``  router
- ``model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3,4,5}.weight``
  per-expert latent weights, stacked into ``[E, out, in]`` grouped tensors
- ``model.layers.{i}.block_sparse_moe.shared_experts.{j}.*``  shared experts
- ``model.layers.{i}.engram.*``          hash tables, projections, norms

Round-trip closure (from_hf -> to_hf) is the meta acceptance requirement;
real HF weight compatibility is out of scope for now.
"""

import logging
import re
from typing import Any

import torch

from torchtitan.models.utils import StateDictAdapter

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_NAMES = ("w1", "w2", "w3", "w4", "w5")
_SHARED_WEIGHT_NAMES = (
    "gate_down",
    "up_down",
    "latent_to_inter",
    "inter_to_latent",
    "latent_to_out",
)


class ArLlmStateDictAdapter(StateDictAdapter):
    """Adapts ar_llm HF-style checkpoints to torchtitan ArLlmModel format."""

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        self.from_hf_layer_map = {
            "model.layers.{}.attn_pre_norm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.attn_post_norm.weight": "layers.{}.attn_post_norm.weight",
            "model.layers.{}.moe_pre_norm.weight": "layers.{}.moe_pre_norm.weight",
            "model.layers.{}.moe_post_norm.weight": "layers.{}.moe_post_norm.weight",
            # MLA low-rank projections (CSA/HCA)
            "model.layers.{}.attn.q_a_proj.weight": "layers.{}.attention.q_proj.q_down.weight",
            "model.layers.{}.attn.q_a_layernorm.weight": "layers.{}.attention.q_proj.q_norm.weight",
            "model.layers.{}.attn.q_b_nope_proj.weight": "layers.{}.attention.q_proj.q_up_nope.weight",
            "model.layers.{}.attn.q_b_rope_proj.weight": "layers.{}.attention.q_proj.q_up_rope.weight",
            "model.layers.{}.attn.q_b_nope_norm.weight": "layers.{}.attention.q_proj.q_up_nope_norm.weight",
            "model.layers.{}.attn.kv_a_proj.weight": "layers.{}.attention.kv_proj.kv_down.weight",
            "model.layers.{}.attn.kv_a_layernorm.weight": "layers.{}.attention.kv_proj.kv_norm.weight",
            "model.layers.{}.attn.kv_latent_layernorm.weight": (
                "layers.{}.attention.kv_proj.kv_latent_norm.weight"
            ),
            "model.layers.{}.attn.kv_b_nope_proj.weight": "layers.{}.attention.kv_proj.k_up_nope.weight",
            "model.layers.{}.attn.kv_b_v_proj.weight": "layers.{}.attention.kv_proj.v_up.weight",
            "model.layers.{}.attn.o_down.weight": "layers.{}.attention.o_proj.o_down",
            "model.layers.{}.attn.o_up.weight": "layers.{}.attention.o_proj.o_up",
            "model.layers.{}.attn.attn_sink": "layers.{}.attention.attn_sink",
            # HCA indexer
            "model.layers.{}.attn.indexer.q_proj.weight": "layers.{}.attention.core.indexer_q.weight",
            "model.layers.{}.attn.indexer.k_proj.weight": "layers.{}.attention.core.indexer_k.weight",
            "model.layers.{}.attn.indexer.q_norm.weight": "layers.{}.attention.core.indexer_q_norm.weight",
            "model.layers.{}.attn.indexer.k_norm.weight": "layers.{}.attention.core.indexer_k_norm.weight",
            # KDA
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.kda.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.kda.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.kda.v_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.kda.o_proj.weight",
            "model.layers.{}.self_attn.alpha_gate.weight": "layers.{}.attention.kda.alpha_gate.weight",
            "model.layers.{}.self_attn.erase_gate.weight": "layers.{}.attention.kda.erase_gate.weight",
            "model.layers.{}.self_attn.write_gate.weight": "layers.{}.attention.kda.write_gate.weight",
            # mHC
            "model.layers.{}.mhc_attn.expand_proj.weight": "layers.{}.hc_pre_attn.expand.weight",
            "model.layers.{}.mhc_attn.mix_weights": "layers.{}.hc_pre_attn.mix_weights",
            "model.layers.{}.mhc_attn.contract_proj.weight": "layers.{}.hc_pre_attn.contract.weight",
            "model.layers.{}.mhc_attn.pre_norm.weight": "layers.{}.hc_pre_attn.pre_norm.weight",
            "model.layers.{}.mhc_attn.post_norm.weight": "layers.{}.hc_pre_attn.post_norm.weight",
            "model.layers.{}.mhc_ffn.expand_proj.weight": "layers.{}.hc_pre_ffn.expand.weight",
            "model.layers.{}.mhc_ffn.mix_weights": "layers.{}.hc_pre_ffn.mix_weights",
            "model.layers.{}.mhc_ffn.contract_proj.weight": "layers.{}.hc_pre_ffn.contract.weight",
            "model.layers.{}.mhc_ffn.pre_norm.weight": "layers.{}.hc_pre_ffn.pre_norm.weight",
            "model.layers.{}.mhc_ffn.post_norm.weight": "layers.{}.hc_pre_ffn.post_norm.weight",
            # MoE
            "model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.router.gate.weight",
            # Engram
            "model.layers.{}.engram.memory_proj.weight": "layers.{}.engram.memory_proj.weight",
            "model.layers.{}.engram.gate_query_proj.weight": "layers.{}.engram.gate.query_proj.weight",
            "model.layers.{}.engram.gate_query_norm.weight": "layers.{}.engram.gate.query_norm.weight",
            "model.layers.{}.engram.gate_key_norm.weight": "layers.{}.engram.gate.key_norm.weight",
            "model.layers.{}.engram.conv.weight": "layers.{}.engram.short_term.conv.weight",
            "model.layers.{}.engram.input_norm.weight": "layers.{}.engram.input_norm.weight",
            "model.layers.{}.engram.memory_norm.weight": "layers.{}.engram.memory_norm.weight",
            "model.layers.{}.engram.gate_bias": "layers.{}.engram.gate_bias",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        routed_experts: dict[tuple[str, str], dict[int, Any]] = {}
        shared_experts: dict[tuple[str, str], dict[int, Any]] = {}
        engram_buffers: dict[str, Any] = {}

        for hf_key, tensor in hf_state_dict.items():
            if hf_key in self.from_hf_map:
                state_dict[self.from_hf_map[hf_key]] = tensor
                continue

            mapped = False
            for hf_pattern, titan_pattern in self.from_hf_layer_map.items():
                pattern = re.escape(hf_pattern).replace(re.escape("{}"), r"(\d+)")
                match = re.match(pattern + "$", hf_key)
                if match:
                    state_dict[titan_pattern.format(match.group(1))] = tensor
                    mapped = True
                    break
            if mapped:
                continue

            routed_match = re.match(
                r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[12345])\.weight$",
                hf_key,
            )
            if routed_match:
                layer_idx, expert_idx, weight_name = routed_match.groups()
                routed_experts.setdefault((layer_idx, weight_name), {})[int(expert_idx)] = tensor
                continue

            shared_match = re.match(
                r"model\.layers\.(\d+)\.block_sparse_moe\.shared_experts\.(\d+)\."
                r"(gate_down|up_down|latent_to_inter|inter_to_latent|latent_to_out)\.weight$",
                hf_key,
            )
            if shared_match:
                layer_idx, expert_idx, weight_name = shared_match.groups()
                shared_experts.setdefault((layer_idx, weight_name), {})[int(expert_idx)] = tensor
                continue

            engram_match = re.match(
                r"model\.layers\.(\d+)\.engram\.(hash_fns|hash_tables)\.(\d+)\."
                r"(embedding\.weight|multipliers|base_powers|head_offsets)$",
                hf_key,
            )
            if engram_match:
                layer_idx, kind, table_idx, tail = engram_match.groups()
                engram_buffers[f"layers.{layer_idx}.engram.{kind}.{table_idx}.{tail}"] = tensor
                continue

            logger.debug("from_hf: skipping unmapped key: %s", hf_key)

        for (layer_idx, weight_name), weights_by_expert in routed_experts.items():
            state_dict[f"layers.{layer_idx}.moe.experts.{weight_name}"] = torch.stack(
                [weights_by_expert[index] for index in sorted(weights_by_expert)], dim=0
            )
        for (layer_idx, weight_name), weights_by_expert in shared_experts.items():
            state_dict[f"layers.{layer_idx}.moe.shared_experts.{weight_name}"] = torch.stack(
                [weights_by_expert[index] for index in sorted(weights_by_expert)], dim=0
            )
        state_dict.update(engram_buffers)
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {value: key for key, value in self.from_hf_map.items()}
        to_hf_layer_map = {value: key for key, value in self.from_hf_layer_map.items()}

        hf_state_dict: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value
                continue

            mapped = False
            for titan_pattern, hf_pattern in to_hf_layer_map.items():
                pattern = re.escape(titan_pattern).replace(re.escape("{}"), r"(\d+)")
                match = re.match(pattern + "$", key)
                if match:
                    hf_state_dict[hf_pattern.format(match.group(1))] = value
                    mapped = True
                    break
            if mapped:
                continue

            routed_match = re.match(r"layers\.(\d+)\.moe\.experts\.(w[12345])$", key)
            if routed_match:
                layer_idx, weight_name = routed_match.groups()
                for expert_idx, expert_weight in enumerate(value.unbind(0)):
                    hf_state_dict[
                        f"model.layers.{layer_idx}.block_sparse_moe.experts."
                        f"{expert_idx}.{weight_name}.weight"
                    ] = expert_weight
                continue

            shared_match = re.match(
                r"layers\.(\d+)\.moe\.shared_experts\."
                r"(gate_down|up_down|latent_to_inter|inter_to_latent|latent_to_out)$",
                key,
            )
            if shared_match:
                layer_idx, weight_name = shared_match.groups()
                for expert_idx, expert_weight in enumerate(value.unbind(0)):
                    hf_state_dict[
                        f"model.layers.{layer_idx}.block_sparse_moe.shared_experts."
                        f"{expert_idx}.{weight_name}.weight"
                    ] = expert_weight
                continue

            buffer_match = re.match(
                r"layers\.(\d+)\.engram\.(hash_fns|hash_tables)\.(\d+)\."
                r"(embedding\.weight|multipliers|base_powers|head_offsets)$",
                key,
            )
            if buffer_match:
                hf_state_dict[f"model.{key}"] = value
                continue

            logger.debug("to_hf: skipping unmapped key: %s", key)

        return hf_state_dict
