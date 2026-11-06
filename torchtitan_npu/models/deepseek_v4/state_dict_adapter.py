# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter as UpstreamDeepSeekV4StateDictAdapter,
)

from torchtitan_npu.models.deepseek_v4.lora import DeepSeekPEFTAdapter, is_lora_parameter

from .model import DeepSeekV4Model


class DeepSeekV4StateDictAdapter(DeepSeekPEFTAdapter, UpstreamDeepSeekV4StateDictAdapter):
    """Reuse upstream V4 conversion with NPU sink, compressor and reader layouts."""

    def __init__(self, model_config: DeepSeekV4Model.Config, hf_assets_path: str | None):
        super().__init__(
            model_config,  # pyrefly: ignore [bad-argument-type]
            hf_assets_path,
        )
        self._initialize_peft(model_config)
        self._num_mtp_layers = len(model_config.mtp_layers)
        self._hash_layers = {
            i
            for i, layer in enumerate(model_config.layers)
            if layer.moe.router.layer_id < layer.moe.router.n_hash_layers
        }
        # NPU uses one compressor name for every compression ratio, and a
        # bare sink parameter rather than upstream's Linear weight wrapper.
        self.from_hf_map = {
            key: value.replace(".compressor_128.", ".compressor.")
            for key, value in self.from_hf_map.items()
            if not key.endswith(".ffn.gate.tid2eid")
        }
        self.from_hf_map["layers.{}.attn.attn_sink"] = "layers.{}.attention.attn_sink"
        for i in self._hash_layers:
            self.from_hf_map[f"layers.{i}.ffn.gate.tid2eid"] = f"layers.{i}.moe.router.tid2eid"

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader:
        if from_quantized:
            # Probe AO protocol first; its headers defeat the fixed-format MX reader.
            from torchtitan_npu.extensions.mx_storage_reader.ao_hf_storage import MxAOHuggingFaceStorageReader

            ao_reader = MxAOHuggingFaceStorageReader(path)
            if ao_reader.detect():
                ao_reader.read_metadata()
                return ao_reader

            from torchtitan_npu.extensions.mx_storage_reader.hf_storage import MXHuggingFaceStorageReader

            reader = MXHuggingFaceStorageReader(path)
            reader.read_metadata()
            if reader._mx_tensors:
                return reader
        return super().get_hf_storage_reader(path, from_quantized)

    def _validate_mtp_key(self, key: str, prefix: str) -> None:
        if key.startswith(prefix) and int(self._first_number(key)) >= self._num_mtp_layers:
            raise ValueError(
                f"MTP key {key} is not present in the model config, which owns {self._num_mtp_layers} stage(s)."
            )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        delegated = {}
        sinks = {}
        for key, value in state_dict.items():
            if is_lora_parameter(key):
                continue
            self._validate_mtp_key(key, "mtp_layers.")
            if (
                key.startswith("layers.")
                and key.endswith(".moe.expert_bias_E")
                and int(self._first_number(key)) in self._hash_layers
            ):
                continue
            if key.endswith(".attention.attn_sink"):
                hf_key = key.replace("mtp_layers.", "mtp.", 1).replace(".attention.attn_sink", ".attn.attn_sink")
                sinks[hf_key] = value
            else:
                delegated[key] = value
        result = super().to_hf(delegated)
        result.update(sinks)
        return result

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        delegated = {}
        sinks = {}
        for key, value in hf_state_dict.items():
            self._validate_mtp_key(key, "mtp.")
            if (
                key.startswith("layers.")
                and key.endswith(".ffn.gate.bias")
                and int(self._first_number(key)) in self._hash_layers
            ):
                continue
            if key.endswith(".attn.attn_sink"):
                local_key = key.replace("mtp.", "mtp_layers.", 1).replace(".attn.attn_sink", ".attention.attn_sink")
                sinks[local_key] = value
            else:
                delegated[key] = value
        result = super().from_hf(delegated)
        result.update(sinks)
        return result
