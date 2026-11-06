# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 adapter checkpoints and PEFT export with a fixed Engram base."""

from dataclasses import dataclass

from torchtitan_npu.models.common.deepseek_lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager


class DeepSeekV41LoRACheckpointManager(DeepSeekV4PEFTCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4PEFTCheckpointManager.Config):
        last_save_model_only: bool = False
        save_training_state: bool = True

    def _validate_native_resume(self, saved_keys: set[str]) -> None:
        super()._validate_native_resume(saved_keys)
        if not any(is_lora_parameter(name) for name in saved_keys):
            raise ValueError("V4.1 LoRA resume requires adapter tensors; load the base through initial_load_path")
