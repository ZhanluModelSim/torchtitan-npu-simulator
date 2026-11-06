# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torchtitan.config import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from torchtitan_npu.override.qwen3_5.gated_delta import ContextParallelGatedDeltaNet
from torchtitan_npu.override.qwen3_5.varlen_attention import AscVarlenInnerAttention


def test_long_text_cli_keeps_cp_and_selects_typed_cp_attention_and_gdn():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.qwen3_5.config_registry",
            "--config",
            "qwen35_27b_long_text_sft",
            "--hf-assets-path",
            "/tmp",
        ]
    )
    config.model.update_from_config(config=config)
    apply_overrides(config.override, config)
    assert config.parallelism.context_parallel_degree == 4
    configs = [cfg for _, cfg, _, _ in config.model.traverse(Module.Config, recurse=True)]
    cores = [cfg for cfg in configs if isinstance(cfg, CPInnerAttention.Config)]
    assert cores and all(cfg._owner is AscVarlenInnerAttention for cfg in cores)
    assert any(cfg._owner is ContextParallelGatedDeltaNet for cfg in configs)
    cu = torch.tensor([0, 2, 8], dtype=torch.int32)
    metadata = VarlenMetadata(cu, cu.clone(), max_q=6, max_k=6)
    assert AscVarlenInnerAttention.prepare_cp_metadata(metadata, permutation=None) is metadata
