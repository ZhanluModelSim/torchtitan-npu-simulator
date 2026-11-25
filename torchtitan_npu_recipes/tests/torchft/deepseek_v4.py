# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous TorchFT recipes for DeepSeek-V4 Flash on Ascend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu_recipes.models.deepseek_v4 import _flash_config
from torchtitan_npu_recipes.torchft.deepseek_v4 import _torchft_flash_config

if TYPE_CHECKING:
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


def deepseek_v4_flash_43layers_16experts_torchft() -> TrainerEx.Config:
    """The 16-expert Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(
        _flash_config(
            "deepseek_v4_flash_43layers_16experts",
            parallelism=ParallelismConfig(
                expert_parallel_degree=16, data_parallel_shard_degree=64, fsdp_reshard_after_forward="always"
            ),
            per_doc_alignment=int(os.environ.get("DSV4_PER_DOC_ALIGNMENT", 128)),
        ),
        expert_parallel_degree=16,
    )


def deepseek_v4_flash_8p_torchft() -> TrainerEx.Config:
    """Four Flash layers plus MTP on two EP4/FSDP4 replicas for 8-card validation."""
    base = _flash_config(
        "deepseek_v4_flash_43layers_16experts",
        num_layers=4,
        num_mtp_layers=1,
        seq_len=256,
        hf_assets_path="tests/assets/deepseek_v3",
        parallelism=ParallelismConfig(
            expert_parallel_degree=4, data_parallel_shard_degree=4, fsdp_reshard_after_forward="always"
        ),
        per_doc_alignment=int(os.environ.get("DSV4_PER_DOC_ALIGNMENT", 128)),
    )
    config = _torchft_flash_config(
        base,
        expert_parallel_degree=4,
        data_parallel_shard_degree=4,
        global_batch_size=16,
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    config.debug.seed = 42
    return config
