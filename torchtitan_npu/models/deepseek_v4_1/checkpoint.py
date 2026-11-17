# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 adapter checkpoints and PEFT export with a fixed Engram base."""

from dataclasses import dataclass

import torch.distributed.checkpoint as dcp

from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager


class DeepSeekV41LoRACheckpointManager(DeepSeekV4PEFTCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4PEFTCheckpointManager.Config):
        last_save_model_only: bool = False
        save_training_state: bool = True

    def dcp_load(self, state_dict, checkpoint_id, from_hf=False, from_quantized=False):
        if checkpoint_id != self.initial_load_path and not from_hf:
            saved = dcp.FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata
            if not any(is_lora_parameter(name) for name in saved):
                raise ValueError("V4.1 LoRA resume requires adapter tensors; load the base through initial_load_path")
        return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)
