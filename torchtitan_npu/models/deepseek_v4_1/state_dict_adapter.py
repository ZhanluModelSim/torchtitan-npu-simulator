# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 state-dict adapter: the text mapping plus, when there is one, the vision half.

The text mapping is the V4.1-effective subset of the DSV4 mapping: no MTP depths, no hash
routing tables; the routed experts merge/split and the DTensor-aware transfers are
inherited from the common DeepSeek-V3 adapter (upstream torchtitan).  The vision tower,
the markers and the router's vision bias belong to
:class:`DeepSeekV41VisionStateDictAdapter`, which this adapter composes only for
:class:`DeepSeekV41MultimodalModel` -- a text-only config gets no vision rule at all.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from torchtitan_npu.extensions.mx_storage_reader.engram import EngramHuggingFaceStorageReader, dequantize_engram_weight

from .engram.checkpoint import checkpoint_shard
from .indexer import REUSE

if TYPE_CHECKING:
    from .vision.state_dict_adapter import DeepSeekV41VisionStateDictAdapter


class DeepSeekV41StateDictAdapter(DeepSeekV3StateDictAdapter):
    """V4.1 adapter: the local text mapping plus, when the model has one, the vision half.

    Which half is present is read off the model config's class rather than off a None
    field: the vision namespace belongs to :class:`DeepSeekV41MultimodalModel`, so a text
    config gets no vision rule at all -- including the ``bias_vl`` entry, whose parameter
    a text-only router never builds.
    """

    def __init__(self, model_config, hf_assets_path):
        self._vision_adapter: DeepSeekV41VisionStateDictAdapter | None = None
        self._engram_tables = {
            i: layer.engram.table for i, layer in enumerate(model_config.layers) if layer.engram is not None
        }
        super().__init__(model_config, hf_assets_path)

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
            # Engram
            "layers.{}.engram.embed.weight": "layers.{}.engram.table.weight",
            "layers.{}.engram.wkv.weight": "layers.{}.engram.gate.wkv",
            "layers.{}.engram.q_weight": "layers.{}.engram.gate.q_weight",
            "layers.{}.engram.k_weight": "layers.{}.engram.gate.k_weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_pre.hc_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_pre.hc_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_pre.hc_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_pre.hc_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_pre.hc_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_pre.hc_scale",
            "norm.weight": "norm.weight",
        }

        self.compress_ratios = model_config.compress_ratios
        for layer_id in range(model_config.n_layers):
            layer_cfg = model_config.layers[layer_id]
            compressor_cfg = layer_cfg.attention.compressor
            if compressor_cfg.is_source:
                compressor_map = {
                    f"layers.{layer_id}.attn.compressor.norm.weight": (
                        f"layers.{layer_id}.attention.compressor.norm.weight"
                    ),
                    f"layers.{layer_id}.attn.compressor.wkv.weight": (
                        f"layers.{layer_id}.attention.compressor.wkv.weight"
                    ),
                }
                if compressor_cfg.wgate is not None:
                    compressor_map[f"layers.{layer_id}.attn.compressor.wgate.weight"] = (
                        f"layers.{layer_id}.attention.compressor.wgate.weight"
                    )
                self.from_hf_map.update(compressor_map)
            indexer_cfg = layer_cfg.attention.indexer
            if indexer_cfg.mode is not REUSE:
                indexer_map = {
                    f"layers.{layer_id}.attn.indexer.wq_b.weight": (f"layers.{layer_id}.attention.indexer.wq_b.weight"),
                    f"layers.{layer_id}.attn.indexer.weights_proj.weight": (
                        f"layers.{layer_id}.attention.indexer.weights_proj.weight"
                    ),
                }
                if indexer_cfg.wk is not None:
                    indexer_map[f"layers.{layer_id}.attn.indexer.wk.weight"] = (
                        f"layers.{layer_id}.attention.indexer.wk.weight"
                    )
                    if indexer_cfg.k_norm is not None:
                        indexer_map[f"layers.{layer_id}.attn.indexer.k_norm.weight"] = (
                            f"layers.{layer_id}.attention.indexer.k_norm.weight"
                        )
                self.from_hf_map.update(indexer_map)

        # Imported here, not at module scope: the vision package pulls in the tower and,
        # through it, the multimodal dataset stack -- a text-only run must not need them.
        from .model import DeepSeekV41MultimodalModel

        if isinstance(model_config, DeepSeekV41MultimodalModel.Config):
            from .vision.state_dict_adapter import DeepSeekV41VisionStateDictAdapter

            self._vision_adapter = DeepSeekV41VisionStateDictAdapter()

    def _owns_vision_hf_key(self, key: str) -> bool:
        return self._vision_adapter is not None and self._vision_adapter.owns_hf_key(key)

    def _owns_vision_local_key(self, key: str) -> bool:
        return self._vision_adapter is not None and self._vision_adapter.owns_local_key(key)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        vision_hf = {k: v for k, v in hf_state_dict.items() if self._owns_vision_hf_key(k)}
        base_hf = {k: v for k, v in hf_state_dict.items() if not self._owns_vision_hf_key(k)}
        result = self._from_hf_text(base_hf)
        if self._vision_adapter is not None:
            result.update(self._vision_adapter.from_hf(vision_hf))
        return result

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        if self._engram_tables:
            shapes = {
                f"layers.{i}.engram.embed.weight": (sum(config.head_vocab_sizes), config.embedding_dim)
                for i, config in self._engram_tables.items()
            }
            return EngramHuggingFaceStorageReader(path, table_shapes=shapes, from_quantized=from_quantized)
        return super().get_hf_storage_reader(path, from_quantized)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        vision_local = {k: v for k, v in state_dict.items() if self._owns_vision_local_key(k)}
        base_local = {k: v for k, v in state_dict.items() if not self._owns_vision_local_key(k)}
        result = self._to_hf_text(base_local)
        if self._vision_adapter is not None:
            result.update(self._vision_adapter.to_hf(vision_local))
        return result

    # ---- the text-backbone halves (the V4.1-effective DSV4 mapping) ----

    def _from_hf_text(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        expert_weights = {}

        for key, value in hf_state_dict.items():
            if ".engram." in key:
                if key.endswith(".scale"):
                    continue
                layer_id = int(key.split(".")[1])
                if layer_id not in self._engram_tables:
                    raise ValueError(f"HF checkpoint contains Engram for unconfigured layer {layer_id}")
                abstract = re.sub(r"(\d+)", "{}", key, count=1)
                local_key = self.from_hf_map[abstract].format(layer_id)
                if hasattr(value, "_engram_native_key"):
                    value[value._engram_valid_rows :].zero_()
                    state_dict[value._engram_native_key] = value
                    continue
                scale = hf_state_dict.get(key.removesuffix("weight") + "scale") if key.endswith(".weight") else None
                if scale is not None:
                    value = dequantize_engram_weight(value, scale, row_block=1 if ".embed." in key else 32)
                elif value.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"Missing scale for quantized Engram weight {key}")
                if key.endswith(".embed.weight"):
                    config = self._engram_tables[layer_id]
                    logical_rows = sum(config.head_vocab_sizes)
                    if tuple(value.shape) != (logical_rows, config.embedding_dim):
                        raise ValueError(f"Unexpected HF Engram table shape for {key}: {value.shape}")
                    padded = torch.zeros(
                        (config.num_embeddings, config.embedding_dim), dtype=torch.float32, device="cpu"
                    )
                    padded[:logical_rows].copy_(value)
                    value = padded
                state_dict[local_key] = value

            elif any(t in key for t in ("compressor", "indexer")):
                state_dict[self.from_hf_map[key]] = value

            elif "ffn.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num, _ = re.findall(r"\d+", key)
                titan_abstract, layer_num = self._map_from_hf_layer_key(abstract_key, layer_num)
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
                    num_experts = self.model_config.layers[0].moe.num_experts  # pyrefly: ignore [missing-attribute]
                    stacked = self._concatenate_expert_weights(
                        expert_weights,
                        titan_abstract,
                        layer_num,
                        num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked

            elif key.startswith("layers."):
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)  # pyrefly: ignore [missing-attribute]
                new_key, layer_num = self._map_from_hf_layer_key(abstract_key, layer_num)
                state_dict[new_key.format(layer_num)] = value

            else:
                if key in self.from_hf_map:
                    state_dict[self.from_hf_map[key]] = value
                else:
                    state_dict[key] = value

        return state_dict

    def _to_hf_text(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if ".engram.table.weight" in key:
                match = re.fullmatch(
                    r"layers\.(\d+)\.engram\.table\.weight(?:\.(?:efsdp_)?ep_shard_(\d+)_of_(\d+))?", key
                )
                if match is None:
                    raise ValueError(f"Unexpected Engram table key: {key}")
                layer_id = int(match[1])
                config = self._engram_tables[layer_id]
                rank, size = (int(match[2]), int(match[3])) if match[2] else (0, 1)
                if not 0 <= rank < size or config.num_embeddings % size:
                    raise ValueError(f"Invalid Engram table shard: {key}")
                shard_rows = config.num_embeddings // size
                if tuple(value.shape) != (shard_rows, config.embedding_dim):
                    raise ValueError(f"Unexpected native Engram shard shape for {key}: {value.shape}")
                logical_rows = sum(config.head_vocab_sizes)
                hf_state_dict[f"layers.{layer_id}.engram.embed.weight"] = (
                    value[:logical_rows]
                    if size == 1
                    else checkpoint_shard(value, key=key, offset=rank * shard_rows, logical_rows=logical_rows)
                )

            elif any(t in key for t in ("compressor", "indexer")):
                hf_state_dict[to_hf_map[key]] = value

            elif "moe.routed_experts.inner_experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                new_abstract, layer_num = self._map_to_hf_layer_key(key, to_hf_map)

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
                    num_experts = self.model_config.layers[0].moe.num_experts  # pyrefly: ignore [missing-attribute]
                    split_values = self._split_experts_weights(value, num_experts)
                    for e in range(num_experts):
                        hf_state_dict[new_abstract.format(layer_num, e)] = split_values[e].squeeze()

            elif "layers" in key:
                new_key, layer_num = self._map_to_hf_layer_key(key, to_hf_map)
                hf_state_dict[new_key.format(layer_num)] = value

            else:
                if key in to_hf_map:
                    hf_state_dict[to_hf_map[key]] = value
                else:
                    hf_state_dict[key] = value

        return hf_state_dict

    def _map_from_hf_layer_key(self, abstract_key: str, layer_num: str) -> tuple[str, str]:
        return self.from_hf_map[abstract_key], layer_num

    def _map_to_hf_layer_key(self, key: str, to_hf_map: dict[str, str]) -> tuple[str, str]:
        abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
        layer_num = re.search(r"\d+", key).group(0)  # pyrefly: ignore [missing-attribute]
        return to_hf_map[abstract_key], layer_num
