# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Small optimizer integration recipes with explicit checkpoint ownership."""

from torchtitan.components.checkpointer import CheckpointManager

from torchtitan_npu.config import CompileConfig, ModelCompileConfig
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


def deepseek_v4_checkpoint():
    config = deepseek_v4_debugmodel()
    config.checkpointer = CheckpointManager.Config()
    return config


def deepseek_v4_aot_eager():
    config = deepseek_v4_debugmodel()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig(backend="aot_eager"))
    return config
