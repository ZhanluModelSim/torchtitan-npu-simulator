# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-model DeepSeek-V4 GraphTrainer recipe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan_npu.models.deepseek_v4.graph_trainer import to_graph_trainer_config
from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model
from torchtitan_npu_recipes.models import deepseek_v4 as model_recipes

if TYPE_CHECKING:
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


def _to_graph_config(base: TrainerEx.Config) -> TrainerEx.Config:
    """Convert the training recipe with the existing NPU graph policy and guard."""
    from torchtitan.config import derive
    from torchtitan.experiments.graph_trainer.configs import EpOverlapConfig, GraphTrainerCompileConfig

    try:
        from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx
    except TypeError as error:
        if "multiple bases have instance lay-out conflict" not in str(error):
            raise
        raise NotImplementedError(
            "NPU GraphTrainer remains deferred: its existing slotted configuration "
            "bases have incompatible instance layouts in TorchTitan 0.4.0."
        ) from error
    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    base.compile = None
    graph = to_graph_trainer_config(base, GraphTrainerDeepSeekV4Model.Config)
    graph.compile = GraphTrainerCompileConfig(
        memory_policy="full",
        pass_pipeline=f"{_functionalize_mutations.PASS_PIPELINE_NAME}+npu_auto_overlap",
        disable_passes=["cudagraph_pass"],
        ep_overlap=EpOverlapConfig(enabled=True, chunk_dim="seq", module_fqn="layers.*.feed_forward"),
    )
    return derive(graph, GraphTrainerEx.Config, extension=base.extension)


def graph_trainer_deepseek_v4_flash_cpt_4k_a5() -> TrainerEx.Config:
    """Build 128-rank Flash graph CPT on A5 (16 x 8 NPUs) with DeepEP and block-FP8; currently deferred."""
    base = model_recipes.deepseek_v4_flash_cpt_4k_a5()
    base.training.steps = 20
    # EP overlap currently requires one microbatch per train step.  Keep the
    # token budget sentinel until GraphTrainer lifts that schema restriction.
    base.training.num_tokens_per_train_step = -1
    return _to_graph_config(base)


def graph_trainer_deepseek_v4_debugmodel() -> TrainerEx.Config:
    """Debug-width GraphTrainer recipe used by the graph parallelize contract."""
    from torchtitan_npu_recipes.tests.models import deepseek_v4 as test_recipes

    return _to_graph_config(test_recipes.deepseek_v4_debugmodel())
