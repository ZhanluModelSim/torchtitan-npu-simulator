# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.6 SFT launch configuration, pending the Qwen 27B model migration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer


def run_qwen3_6_4k(*, seq_len: int | None = None) -> Trainer.Config:
    from torchtitan.config import derive
    from torchtitan.models.common.rope import RoPE

    from torchtitan_npu.models.qwen3_5 import NPUQwen35Model
    from torchtitan_npu_recipes.models.qwen3_5 import qwen35_27b_4k_sft

    # The base factory deliberately raises until the 27B model is migrated.
    config = qwen35_27b_4k_sft()
    if seq_len is None:
        seq_len = int(os.environ.get("SEQ_LEN", "4096"))
    if not 64 <= seq_len <= 4096 or seq_len % 64:
        raise ValueError("seq_len must be a multiple of 64 in [64, 4096]")
    hf_assets_path = os.environ.get("HF_ASSETS_PATH")
    if not hf_assets_path:
        raise ValueError("Set HF_ASSETS_PATH to the Qwen3.5/Qwen3.6 Hugging Face model directory")
    if not os.environ.get("DATA_FILES"):
        raise ValueError("Set DATA_FILES to the CANNBOT 4K SFT JSONL file")
    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.environ.get("DUMP_FOLDER", "./outputs/qwen3_6_4k_sft")
    config.override.imports = ["torchtitan_npu.override.qwen3_5.gated_delta.triton"]
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    assert isinstance(config.model, NPUQwen35Model.Config)
    config.model = derive(config.model, NPUQwen35Model.Config, max_context_length=seq_len)
    for _, rope, _, _ in config.model.traverse(RoPE.Config):
        rope.max_context_length = seq_len
    assert config.checkpointer is not None
    config.checkpointer.initial_load_path = os.environ.get("CHECKPOINT_INITIAL_LOAD_PATH", hf_assets_path)
    config.checkpointer.initial_load_in_hf = True
    config.checkpointer.initial_load_model_only = True
    return config
