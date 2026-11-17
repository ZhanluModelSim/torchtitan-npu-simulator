# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 configuration binding for the shared DeepSeek LoRA projections."""

from dataclasses import dataclass, replace

import torch
from torchtitan.config import derive

from torchtitan_npu.models.deepseek_v4.lora import (
    DEEPSEEK_V4_LORA_TARGETS,
    DeepSeekV4LoRAConverter,
    LoRAOptions,
    is_lora_parameter,
)
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

from .engram.host import HostEngramTable
from .model import DeepSeekV41Model, DeepSeekV41MultimodalModel
from .state_dict_adapter import DeepSeekV41StateDictAdapter


class _LoRAHostEngramTable(HostEngramTable):
    @dataclass(kw_only=True, slots=True)
    class Config(HostEngramTable.Config):
        pass

    def forward(self, input_ids_BL, positions_BL=None, **kwargs):
        # The host lookup has a live autograd anchor even when its weight is frozen.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.weight.requires_grad):
            return super().forward(input_ids_BL, positions_BL, **kwargs)


class DeepSeekV41LoRAConverter(DeepSeekV4LoRAConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4LoRAConverter.Config):
        include_mtp: bool = False

    def attach_model_config(self, model_config, lora_config):
        for layer in model_config.layers:
            if layer.engram is not None and isinstance(layer.engram.table, HostEngramTable.Config):
                layer.engram.table = derive(layer.engram.table, _LoRAHostEngramTable.Config)
        if isinstance(model_config, DeepSeekV41MultimodalModel.Config):
            return derive(model_config, _MultimodalLoRAConfig, lora=lora_config)
        if isinstance(model_config, DeepSeekV41Model.Config):
            return derive(model_config, _TextLoRAConfig, lora=lora_config)
        raise TypeError("DeepSeekV41LoRAConverter requires a V4.1 model config")


@dataclass(kw_only=True, slots=True)
class _TextLoRAConfig(DeepSeekV41Model.Config):
    lora: DeepSeekV4LoRAConverter.Config


@dataclass(kw_only=True, slots=True)
class _MultimodalLoRAConfig(DeepSeekV41MultimodalModel.Config):
    lora: DeepSeekV4LoRAConverter.Config


class DeepSeekV41LoRAStateDictAdapter(DeepSeekV4StateDictAdapter):
    """Reuse V4 PEFT export while delegating native/HF mapping to V4.1."""

    def __init__(self, model_config, hf_assets_path):
        # V4 construction also installs MTP/HF mappings. Keep those owned by V4.1
        # and initialize only the metadata consumed by the inherited PEFT methods.
        self._v41_adapter = DeepSeekV41StateDictAdapter(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.fqn_to_index_mapping = self._v41_adapter.fqn_to_index_mapping
        self._lora_config = replace(model_config.lora)
        targets = model_config.lora.target_modules
        self._lora_module_names = tuple(fqn for fqn, _, _, _ in model_config.traverse(LoRAOptions, recurse=True))
        self._peft_targets = self._lora_module_names or tuple(
            targets if targets is not None else DEEPSEEK_V4_LORA_TARGETS
        )

    def from_hf(self, hf_state_dict):
        return self._v41_adapter.from_hf(hf_state_dict)

    def to_hf(self, state_dict):
        base_state = {key: value for key, value in state_dict.items() if not is_lora_parameter(key)}
        return self._v41_adapter.to_hf(base_state)

    def get_hf_storage_reader(self, path, from_quantized=False):
        return self._v41_adapter.get_hf_storage_reader(path, from_quantized)
