# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.models.deepseek_v4.graph_trainer import to_graph_trainer_config
from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model
from torchtitan_npu_recipes.models.deepseek_v4 import (
    deepseek_v4_debugmodel,
    deepseek_v4_flash,
    deepseek_v4_flash_43layers_16experts,
    deepseek_v4_pro,
    deepseek_v4_pro_61layers_32experts,
)

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def _graph_trainer_compile_config() -> GraphTrainerCompileConfig:
    """Compile settings shared by the GraphTrainer flash/pro config factories."""
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    return GraphTrainerCompileConfig(
        memory_policy="full",
        # Keeps recomputed forward mutations (fused partial RoPE) faithful;
        # see the functionalize_recompute_mutations patch module.
        pass_pipeline=_functionalize_mutations.PASS_PIPELINE_NAME,
        disable_passes=[
            "cudagraph_pass",
        ],
    )


def graph_trainer_deepseek_v4_debugmodel() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 debug model"""
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    config = to_graph_trainer_config(
        deepseek_v4_debugmodel(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = GraphTrainerCompileConfig(
        memory_policy="full",
        enable_passes=True,
        pass_pipeline=_functionalize_mutations.PASS_PIPELINE_NAME,
        disable_passes=[
            "cudagraph_pass",
        ],
    )
    return config


def graph_trainer_deepseek_v4_flash() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_flash_43layers_16experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash 43 Layers 16 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash_43layers_16experts(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro_61layers_32experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro 61 Layers 32 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro_61layers_32experts(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_flash_cpt() -> GraphTrainer.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = graph_trainer_deepseek_v4_flash()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config
