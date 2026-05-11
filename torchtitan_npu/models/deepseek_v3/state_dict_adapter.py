# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import re
from typing import Any

import torch
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter
from torchtitan.models.utils import MoEStateDictAdapter

from torchtitan_npu.tools.weight_utils import (
    _split_w13_for_mapping,
    convert_expert_format,
)

logger = logging.getLogger(__name__)


class DeepSeekV3StateDictAdapterNpu(DeepSeekV3StateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

        self.use_gmm = any(
            layer_cfg.moe is not None and layer_cfg.moe.experts.use_grouped_mm
            for layer_cfg in model_config.layers
        )
        self._input_format = "hf"
        self._input_expert_format = "standard"

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self._input_format == "dcp":
            return state_dict
        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            working_state = _split_w13_for_mapping(state_dict)
            return super().to_hf(working_state)
        else:
            return super().to_hf(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered = {
            k: v
            for k, v in hf_state_dict.items()
            if not k.endswith(".weight_scale_inv")
        }

        if self._input_format == "hf":
            state_dict = super().from_hf(filtered)
        else:
            state_dict = filtered
        target = "gmm" if self.use_gmm else "standard"
        state_dict = convert_expert_format(state_dict, target)

        return state_dict


class DeepSeek16BStateDictAdapterNpu(MoEStateDictAdapter):
    """Adapts deepseek-moe-16b-base (MHA: q/k/v_proj) → DSv3 model (MLA: wkv_a/wkv_b).

    The 16B model uses standard MHA with RoPE on full head_dim=128.
    The DSv3 model uses MLA with split qk_nope/qk_rope and compressed KV.

    We bridge this by:
    1. Splitting each head's 128-dim key into [k_nope(64), k_pe(64)]
    2. Setting kv_lora_rank = n_heads*(qk_nope+v_head) = 16*192 = 3072
       so kv_compressed can hold interleaved [k_nope, v] per head
    3. wkv_a = concat(interleaved_k_nope_v, k_pe_proj)
    4. wkv_b = identity (input/output layout already matches)
    5. kv_norm = ones (no compression, identity-like)
    """

    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

        self.use_gmm = any(
            layer_cfg.moe is not None and layer_cfg.moe.experts.use_grouped_mm
            for layer_cfg in model_config.layers
        )
        self._input_format = "hf"
        self._input_expert_format = "standard"

        self.model_config = model_config
        attn_cfg = model_config.layers[0].attention
        self.n_heads = attn_cfg.n_heads
        self.qk_nope_head_dim = attn_cfg.qk_nope_head_dim
        self.qk_rope_head_dim = attn_cfg.qk_rope_head_dim
        self.v_head_dim = attn_cfg.v_head_dim
        self.kv_lora_rank = attn_cfg.kv_lora_rank
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self._mla_dtensor_meta = {}  # {titan_key: {placements, shape, device_mesh}}
        self._skip_wkv_b_fold = False  # True during DCP template creation

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered = {
            k: v
            for k, v in hf_state_dict.items()
            if not k.endswith(".weight_scale_inv")
        }

        state_dict = {}
        expert_weights_by_layer = {}

        for hf_key, tensor in filtered.items():
            if ".self_attn.k_proj." in hf_key or ".self_attn.v_proj." in hf_key:
                continue

            if "mlp.experts" in hf_key:
                abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=2)
                layer_num, expert_num = re.findall(r"\d+", hf_key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = tensor

                if titan_abstract_key in self.local_experts_indices:
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    stacked = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        next(
                            l for l in self.model_config.layers if l.moe is not None
                        ).moe.num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked
            else:
                self._map_key(hf_key, tensor, state_dict)

        self._convert_kv_weights(filtered, state_dict)

        target = "gmm" if self.use_gmm else "standard"
        state_dict = convert_expert_format(state_dict, target)

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            state_dict = _split_w13_for_mapping(state_dict)

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}
        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value, moe_layer.moe.num_experts
                    )
                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key in to_hf_map:
                    new_key = to_hf_map[abstract_key]
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
                elif "wkv_a" in key or "wkv_b" in key or "kv_norm" in key:
                    li = int(layer_num)
                    if isinstance(value, DTensor):
                        self._mla_dtensor_meta[key] = {
                            "placements": value.placements,
                            "shape": value.shape,
                            "device_mesh": value.device_mesh,
                        }
                else:
                    logger.debug("to_hf: skipping unmapped layer key: %s", key)

            else:
                if key in to_hf_map:
                    new_key = to_hf_map[key]
                    hf_state_dict[new_key] = value
                else:
                    logger.debug("to_hf: skipping unmapped key: %s", key)

        self._reverse_convert_kv_weights(state_dict, hf_state_dict)

        return hf_state_dict

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        from torch.distributed.checkpoint import HuggingFaceStorageReader

        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint is not supported for 16B model."
            )
        return HuggingFaceStorageReader(path)

    def _reverse_convert_kv_weights(self, state_dict: dict, hf_state_dict: dict):
        n_h = self.n_heads
        qn = self.qk_nope_head_dim
        qr = self.qk_rope_head_dim
        vh = self.v_head_dim
        hd = self.head_dim

        layer_indices = set()
        for k in state_dict.keys():
            m = re.search(r"layers\.(\d+)\.attention\.wkv_a", k)
            if m:
                layer_indices.add(int(m.group(1)))

        for li in sorted(layer_indices):
            wkv_a_key = f"layers.{li}.attention.wkv_a.weight"
            if wkv_a_key not in state_dict:
                continue

            wkv_a = state_dict[wkv_a_key]
            if isinstance(wkv_a, DTensor):
                wkv_a = wkv_a.full_tensor()
            kv_comp_raw = wkv_a[: n_h * (qn + vh)]
            kv_comp_end = n_h * (qn + vh)
            k_pe_shared = wkv_a[kv_comp_end:]

            # Apply trained wkv_b and kv_norm to kv_compressed.
            # MLA forward applies wkv_b over kv_norm.
            # Effective weight combines wkv_b, kv_norm, and wkv_a.
            # Skip during DCP template creation.
            wkv_b_key = f"layers.{li}.attention.wkv_b.weight"
            kv_norm_key = f"layers.{li}.attention.kv_norm.weight"
            if (
                not self._skip_wkv_b_fold
                and wkv_b_key in state_dict
                and kv_norm_key in state_dict
            ):
                wkv_b = state_dict[wkv_b_key]
                kv_norm_w = state_dict[kv_norm_key]
                if isinstance(wkv_b, DTensor):
                    wkv_b = wkv_b.full_tensor()
                if isinstance(kv_norm_w, DTensor):
                    kv_norm_w = kv_norm_w.full_tensor()
                kv_comp = wkv_b @ torch.diag(kv_norm_w) @ kv_comp_raw
            else:
                kv_comp = kv_comp_raw

            k_nope = torch.zeros(
                n_h * qn, wkv_a.shape[1], dtype=wkv_a.dtype, device=wkv_a.device
            )
            v_w = torch.zeros(
                n_h * vh, wkv_a.shape[1], dtype=wkv_a.dtype, device=wkv_a.device
            )
            for h in range(n_h):
                s = h * (qn + vh)
                k_nope_start = h * qn
                k_nope_end = (h + 1) * qn
                v_start = h * vh
                v_end = (h + 1) * vh
                kv_k_start = s
                kv_k_end = s + qn
                kv_v_start = s + qn
                kv_v_end = s + qn + vh
                k_nope[k_nope_start:k_nope_end] = kv_comp[kv_k_start:kv_k_end]
                v_w[v_start:v_end] = kv_comp[kv_v_start:kv_v_end]

            k_pe_replicated = (
                k_pe_shared.unsqueeze(0).expand(n_h, -1, -1).reshape(n_h * qr, -1)
            )
            k_w = torch.cat(
                [
                    k_nope.view(n_h, qn, -1),
                    k_pe_replicated.view(n_h, qr, -1),
                ],
                dim=1,
            ).reshape(n_h * hd, -1)

            hf_state_dict[f"model.layers.{li}.self_attn.k_proj.weight"] = k_w
            hf_state_dict[f"model.layers.{li}.self_attn.v_proj.weight"] = v_w

            logger.info(
                f"Layer {li}: MLA→MHA k_proj{list(k_w.shape)} v_proj{list(v_w.shape)}"
            )

    def _map_key(self, hf_key: str, tensor: torch.Tensor, state_dict: dict) -> bool:
        for hf_pattern, titan_pattern in self.from_hf_map.items():
            if "{}" not in hf_pattern:
                if hf_key == hf_pattern:
                    state_dict[titan_pattern] = tensor
                    return True
                continue

            n_placeholders = hf_pattern.count("{}")
            pattern = re.escape(hf_pattern)
            for _ in range(n_placeholders):
                pattern = pattern.replace(re.escape("{}"), r"(\d+)", 1)
            m = re.match(pattern + "$", hf_key)
            if m:
                indices = [int(g) for g in m.groups()]
                titan_key = titan_pattern.format(*indices)
                state_dict[titan_key] = tensor
                return True

        return False

    def _convert_kv_weights(self, hf_state_dict: dict, state_dict: dict):
        n_h = self.n_heads
        qn = self.qk_nope_head_dim
        qr = self.qk_rope_head_dim
        vh = self.v_head_dim
        hd = self.head_dim

        layer_indices = set()
        for k in hf_state_dict.keys():
            m = re.search(r"model\.layers\.(\d+)\.self_attn\.k_proj", k)
            if m:
                layer_indices.add(int(m.group(1)))

        for li in sorted(layer_indices):
            k_key = f"model.layers.{li}.self_attn.k_proj.weight"
            v_key = f"model.layers.{li}.self_attn.v_proj.weight"
            if k_key not in hf_state_dict or v_key not in hf_state_dict:
                continue

            k_w = hf_state_dict[k_key]
            v_w = hf_state_dict[v_key]

            # Split k per head: [n_heads*head_dim, dim] → [n_heads, head_dim, dim]
            # k_nope_part: first qk_nope dims of each head
            # k_pe_part: last qk_rope dims of each head
            k_reshaped = k_w.view(n_h, hd, -1)
            k_nope = k_reshaped[:, :qn, :].reshape(n_h * qn, -1)
            k_pe = k_reshaped[:, qn:, :].reshape(n_h * qr, -1)

            # kv_compressed layout is interleaved
            # Total size is n_heads times sum of qk_nope and v_head
            kv_comp = torch.zeros(
                n_h * (qn + vh),
                k_w.shape[1],
                dtype=k_w.dtype,
                device=k_w.device,
            )
            for h in range(n_h):
                s = h * (qn + vh)
                kv_k_start = s
                kv_k_end = s + qn
                kv_v_start = s + qn
                kv_v_end = s + qn + vh
                k_nope_start = h * qn
                k_nope_end = (h + 1) * qn
                v_start = h * vh
                v_end = (h + 1) * vh
                kv_comp[kv_k_start:kv_k_end] = k_nope[k_nope_start:k_nope_end]
                kv_comp[kv_v_start:kv_v_end] = v_w[v_start:v_end]

            # MLA k_pe is shared across heads (shape: [qk_rope_head_dim, dim])
            # Average all heads' k_pe parts to get a single shared projection
            k_pe_shared = k_pe.view(n_h, qr, -1).mean(dim=0)

            wkv_a = torch.cat([kv_comp, k_pe_shared], dim=0)

            # wkv_b is identity since layout already matches
            wkv_b = torch.eye(
                n_h * (qn + vh),
                dtype=k_w.dtype,
                device=k_w.device,
            )

            # kv_norm is ones like identity
            kv_norm_w = torch.ones(
                n_h * (qn + vh),
                dtype=k_w.dtype,
                device=k_w.device,
            )

            state_dict[f"layers.{li}.attention.wkv_a.weight"] = self._maybe_distribute(
                wkv_a, f"layers.{li}.attention.wkv_a.weight"
            )
            state_dict[f"layers.{li}.attention.wkv_b.weight"] = self._maybe_distribute(
                wkv_b, f"layers.{li}.attention.wkv_b.weight"
            )
            state_dict[
                f"layers.{li}.attention.kv_norm.weight"
            ] = self._maybe_distribute(
                kv_norm_w, f"layers.{li}.attention.kv_norm.weight"
            )

            logger.info(
                f"Layer {li}: MHA→MLA wkv_a{list(wkv_a.shape)} "
                f"wkv_b{list(wkv_b.shape)} kv_norm{list(kv_norm_w.shape)}"
            )

    def _maybe_distribute(
        self, tensor: torch.Tensor, titan_key: str
    ) -> torch.Tensor | DTensor:
        meta = self._mla_dtensor_meta.get(titan_key)
        if meta is None:
            return tensor
        return distribute_tensor(
            tensor,
            device_mesh=meta["device_mesh"],
            placements=meta["placements"],
        )
