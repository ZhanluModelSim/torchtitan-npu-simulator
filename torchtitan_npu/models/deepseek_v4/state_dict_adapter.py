# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import replace
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor, Shard
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from .lora import (
    _PEFT_MODULE_SUFFIXES,
    DEEPSEEK_V4_LORA_TARGETS,
    MODULE_PATHS,
    PEFT_PREFIX,
    DeepSeekV4LoRAConverter,
    LoRAOptions,
    is_lora_parameter,
    pack_expert_factor,
    peft_target_modules,
)
from .model import DeepSeekV4Model

OFFICIAL_WEIGHT_MAP = {
    (f"{official}.weight" if local == "lm_head" else f"layers.{{}}.{official}.weight"): (
        f"{local}.weight" if local == "lm_head" else f"layers.{{}}.{local}.weight"
    )
    for local, _, official in MODULE_PATHS
    if ".compressor." not in local
}

_ROUTED_EXPERT_PEFT_PARAMETERS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def _peft_key(hf_weight_key: str, factor: str) -> str:
    if not hf_weight_key.endswith(".weight"):
        raise ValueError(f"PEFT LoRA base key must end in .weight: {hf_weight_key}")
    return f"{PEFT_PREFIX}{hf_weight_key.removesuffix('.weight')}.lora_{factor}.weight"


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
        config = getattr(model_config, "lora", DeepSeekV4LoRAConverter.Config(adapt_routed_experts=False))
        self._lora_config = replace(config)
        targets = config.target_modules if config.target_modules is not None else DEEPSEEK_V4_LORA_TARGETS
        self._lora_module_names = tuple(fqn for fqn, _, _, _ in model_config.traverse(LoRAOptions, recurse=True))
        self._peft_targets = self._lora_module_names or tuple(targets)

        self.from_hf_map = {
            **OFFICIAL_WEIGHT_MAP,
            "embed.weight": "tok_embeddings.weight",
            # Attention
            "layers.{}.attn.attn_sink": "layers.{}.attention.attn_sink",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            # Norms
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias_E",
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

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader:
        if from_quantized:
            from torchtitan_npu.extensions.mx_storage_reader.hf_storage import MXHuggingFaceStorageReader

            reader = MXHuggingFaceStorageReader(path)
            reader.read_metadata()
            if reader._mx_tensors:
                return reader
        return super().get_hf_storage_reader(path, from_quantized)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if is_lora_parameter(key):
                continue
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

    def to_peft(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        peft_state_dict = {}
        for key, value in state_dict.items():
            if not is_lora_parameter(key):
                continue
            if key.startswith("mtp_layers."):
                raise NotImplementedError("Transformers DeepSeek-V4 PEFT export does not consume MTP adapters")
            if key.endswith((".lora_a.weight", ".lora_b.weight")):
                base_key, factor, _ = key.rsplit(".", 2)
                for local_suffix, hf_suffix in _PEFT_MODULE_SUFFIXES.items():
                    if base_key == local_suffix or base_key.endswith(f".{local_suffix}"):
                        prefix = base_key.removesuffix(local_suffix)
                        hf_key = f"{prefix}{hf_suffix}.weight"
                        if hf_suffix != "lm_head":
                            hf_key = f"model.{hf_key}"
                        peft_state_dict[_peft_key(hf_key, factor[-1].upper())] = value
                        break
                else:
                    raise ValueError(f"Unsupported PEFT LoRA tensor: {key}")
            elif ".moe.routed_experts.inner_experts." in key:
                prefix, factor = key.split(".moe.routed_experts.inner_experts.")
                if factor in {"w1_lora_b", "w3_lora_b"}:
                    stem = f"{prefix}.moe.routed_experts.inner_experts."
                    if f"{stem}w13_lora_b" in state_dict:
                        raise ValueError("Checkpoint contains both split and interleaved LoRA B weights")
                    if any(f"{stem}{name}" not in state_dict for name in ("w1_lora_b", "w3_lora_b")):
                        raise ValueError("PEFT export requires both gate and up LoRA B weights")
                    if factor == "w3_lora_b":
                        continue
                    value = torch.stack((value, state_dict[f"{stem}w3_lora_b"]), dim=2)
                    factor = "w13_lora_b"
                if factor not in {"w13_lora_a", "w13_lora_b", "w2_lora_a", "w2_lora_b"}:
                    raise ValueError(f"Unsupported PEFT LoRA tensor: {key}")
                prefix = f"model.{prefix}.mlp.experts"
                # PEFT nests gate_up_proj inside down_proj, in target_parameters order.
                if factor.startswith("w13_"):
                    prefix += ".base_layer"
                value = pack_expert_factor(value, factor)
                peft_factor = "A" if factor.endswith("lora_a") else "B"
                if isinstance(value, DTensor):
                    # DCP needs contiguous shards after flattening the expert/rank axes.
                    value = value.redistribute(placements=[Shard(0)] * value.device_mesh.ndim)
                peft_state_dict[_peft_key(f"{prefix}.weight", peft_factor)] = value.contiguous()
            else:
                raise ValueError(f"Unsupported PEFT LoRA tensor: {key}")
        if not peft_state_dict:
            raise ValueError("No LoRA adapter tensors were found for PEFT export")
        return peft_state_dict

    def peft_adapter_config(self, *, base_model_name_or_path: str | None = None) -> dict[str, Any]:
        if any(name.startswith("mtp_layers.") for name in self._lora_module_names):
            raise NotImplementedError("Transformers DeepSeek-V4 PEFT export does not consume MTP adapters")
        expert_suffix = ".moe.routed_experts.inner_experts"
        targets = tuple(name for name in self._peft_targets if not name.endswith(expert_suffix))
        expert_targets = [
            f"model.{name.removesuffix(expert_suffix)}.{parameter}"
            for name in self._lora_module_names
            if name.endswith(expert_suffix)
            for parameter in _ROUTED_EXPERT_PEFT_PARAMETERS
        ]
        peft_targets = peft_target_modules(targets)
        config = self._lora_config
        if config.adapt_routed_experts and config.rank_experts != config.rank:
            raise ValueError(
                f"PEFT export requires equal rank ({config.rank}) and rank_experts ({config.rank_experts}) "
                "when adapt_routed_experts=True."
            )
        return {
            "base_model_name_or_path": base_model_name_or_path or self.hf_assets_path or "",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": config.alpha,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": config.rank,
            "revision": None,
            "target_modules": [],
            "target_parameters": [f"{name}.weight" for name in peft_targets] + expert_targets,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }

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
