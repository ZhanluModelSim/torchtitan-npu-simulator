# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from .model import DeepSeekV4Model


class DeepSeekV4StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(
        self,
        model_config: DeepSeekV4Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(
            model_config,  # pyrefly: ignore [bad-argument-type]
            hf_assets_path,
        )
        self._num_mtp_layers = len(model_config.mtp_layers)

        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            "head.weight": "lm_head.weight",
            # Attention
            "layers.{}.attn.attn_sink": "layers.{}.attention.attn_sink",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "layers.{}.attn.wo_a.weight": "layers.{}.attention.wo_a.weight",
            "layers.{}.attn.wo_b.weight": "layers.{}.attention.wo_b.weight",
            "layers.{}.attn.wkv.weight": "layers.{}.attention.wkv.weight",
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.wq_a.weight",
            "layers.{}.attn.wq_b.weight": "layers.{}.attention.wq_b.weight",
            # Norms
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "layers.{}.ffn.gate.weight": "layers.{}.moe.router.gate.weight",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias_E",
            "layers.{}.ffn.shared_experts.w1.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.ffn.shared_experts.w3.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.ffn.shared_experts.w2.weight": "layers.{}.moe.shared_experts.w2.weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_pre.hc_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_pre.hc_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_pre.hc_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_pre.hc_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_pre.hc_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_pre.hc_scale",
            # MTP-only tensors. Native ``mtp.{depth}.*`` keys map directly to
            # the local ``mtp_layers.{depth}.*`` namespace.
            "layers.{}.enorm.weight": "layers.{}.enorm.weight",
            "layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
            "layers.{}.e_proj.weight": "layers.{}.e_proj.weight",
            "layers.{}.h_proj.weight": "layers.{}.h_proj.weight",
            "layers.{}.norm.weight": "layers.{}.mtp_norm.weight",
            "layers.{}.hc_head_base": "layers.{}.hc_head.hc_base",
            "layers.{}.hc_head_fn": "layers.{}.hc_head.hc_fn",
            "layers.{}.hc_head_scale": "layers.{}.hc_head.hc_scale",
            "hc_head_base": "hc_head.hc_base",
            "hc_head_fn": "hc_head.hc_fn",
            "hc_head_scale": "hc_head.hc_scale",
            "norm.weight": "norm.weight",
        }

        self.compress_ratios = model_config.compress_ratios
        for layer_id in range(model_config.n_layers):
            cr = self.compress_ratios[layer_id]
            if cr != 1:
                comp = "compressor"
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.compressor.ape": (f"layers.{layer_id}.attention.{comp}.ape"),
                        f"layers.{layer_id}.attn.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.{comp}.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.{comp}.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.{comp}.wkv.weight"
                        ),
                    }
                )
            if cr == 4:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.indexer.compressor.ape": (
                            f"layers.{layer_id}.attention.indexer.compressor.ape"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.wkv.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.wq_b.weight": (
                            f"layers.{layer_id}.attention.indexer.wq_b.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.weights_proj.weight": (
                            f"layers.{layer_id}.attention.indexer.weights_proj.weight"
                        ),
                    }
                )
            layer_cfg = model_config.layers[layer_id]
            if layer_cfg.moe.router.hash:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.ffn.gate.tid2eid": (f"layers.{layer_id}.moe.router.tid2eid"),
                    }
                )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if any(t in key for t in ("compressor", "indexer", "tid2eid")):
                new_key = to_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.float32)
                hf_state_dict[new_key] = value

            elif "moe.routed_experts.inner_experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                new_abstract, layer_num = self._map_to_hf_layer_key(
                    key,
                    to_hf_map,
                )

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[abstract_key] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    local_fqn = self._get_local_experts_weights(
                        new_abstract,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_fqn)
                else:
                    num_experts = self.model_config.layers[  # pyrefly: ignore [missing-attribute]
                        0
                    ].moe.num_experts
                    split_values = self._split_experts_weights(value, num_experts)
                    for e in range(num_experts):
                        hf_state_dict[new_abstract.format(layer_num, e)] = split_values[e].squeeze()

            elif "layers" in key:
                new_key, layer_num = self._map_to_hf_layer_key(key, to_hf_map)
                if (
                    key.startswith("layers.")
                    and key.endswith(".moe.expert_bias_E")
                    and self.model_config.layers[  # pyrefly: ignore [missing-attribute]
                        int(layer_num)
                    ].moe.router.hash
                ):
                    continue
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key in to_hf_map:
                    hf_state_dict[to_hf_map[key]] = value
                else:
                    hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        expert_weights = {}

        for key, value in hf_state_dict.items():
            if any(t in key for t in ("compressor", "indexer", "tid2eid")):
                new_key = self.from_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.int64)
                state_dict[new_key] = value

            elif "ffn.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num, _ = re.findall(r"\d+", key)
                titan_abstract, layer_num = self._map_from_hf_layer_key(
                    abstract_key,
                    layer_num,
                )
                new_key = titan_abstract.format(layer_num)

                if layer_num not in expert_weights:
                    expert_weights[layer_num] = {}
                if titan_abstract not in expert_weights[layer_num]:
                    expert_weights[layer_num][titan_abstract] = {}
                expert_weights[layer_num][titan_abstract][int(expert_num)] = value

                if titan_abstract in self.local_experts_indices:
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights,
                        titan_abstract,
                        layer_num,
                    )
                else:
                    num_experts = self.model_config.layers[  # pyrefly: ignore [missing-attribute]
                        0
                    ].moe.num_experts
                    stacked = self._concatenate_expert_weights(
                        expert_weights,
                        titan_abstract,
                        layer_num,
                        num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked

            elif key.startswith(("layers.", "mtp.")):
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(  # pyrefly: ignore [missing-attribute]
                    r"\d+", key
                ).group(0)
                if (
                    key.startswith("layers.")
                    and key.endswith("ffn.gate.bias")
                    and self.model_config.layers[  # pyrefly: ignore [missing-attribute]
                        int(layer_num)
                    ].moe.router.hash
                ):
                    continue
                new_key, layer_num = self._map_from_hf_layer_key(
                    abstract_key,
                    layer_num,
                )
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                if key in self.from_hf_map:
                    state_dict[self.from_hf_map[key]] = value
                else:
                    state_dict[key] = value

        return state_dict

    def _map_from_hf_layer_key(
        self,
        abstract_key: str,
        layer_num: str,
    ) -> tuple[str, str]:
        """Map a checkpoint layer key to the corresponding local layer key."""
        is_mtp = abstract_key.startswith("mtp.{}.")
        if is_mtp:
            num_mtp_layers = self._num_mtp_layers
            if int(layer_num) >= num_mtp_layers:
                raise ValueError(
                    f"Checkpoint MTP stage {layer_num} is not present in the "
                    f"model config, which owns {num_mtp_layers} stage(s)."
                )
            abstract_key = abstract_key.replace("mtp.{}.", "layers.{}.", 1)

        new_key = self.from_hf_map[abstract_key]
        if is_mtp:
            new_key = new_key.replace("layers.{}.", "mtp_layers.{}.", 1)
        return new_key, layer_num

    def _map_to_hf_layer_key(
        self,
        key: str,
        to_hf_map: dict[str, str],
    ) -> tuple[str, str]:
        """Map a local layer key to the corresponding checkpoint layer key."""
        abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
        layer_num = re.search(r"\d+", key).group(0)  # pyrefly: ignore [missing-attribute]

        if key.startswith("mtp_layers."):
            num_mtp_layers = self._num_mtp_layers
            if int(layer_num) >= num_mtp_layers:
                raise ValueError(
                    f"Local MTP stage {layer_num} is not present in the model "
                    f"config, which owns {num_mtp_layers} stage(s)."
                )
            abstract_key = abstract_key.replace(
                "mtp_layers.{}.",
                "layers.{}.",
                1,
            )
            new_key = to_hf_map[abstract_key].replace(
                "layers.{}.",
                "mtp.{}.",
                1,
            )
            return new_key, layer_num

        return to_hf_map[abstract_key], layer_num
