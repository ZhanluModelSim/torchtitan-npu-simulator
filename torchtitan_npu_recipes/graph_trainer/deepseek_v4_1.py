# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 text and single-image GraphTrainer recipes."""

from dataclasses import fields

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    MOE_BLOCK_FQN,
    EpOverlapConfig,
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.trainer import Trainer

from torchtitan_npu.models.deepseek_v4_1 import EngramArgs
from torchtitan_npu.models.deepseek_v4_1.model import (
    GraphTrainerDeepSeekV41Model,
    GraphTrainerDeepSeekV41MultimodalModel,
)
from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41GraphTrainer
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import functionalize_recompute_mutations
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer.chunked_loss import CoalescedChunkedLoss
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _make_trainer_config


def _make_graph_trainer_config(
    *,
    multimodal: bool = False,
    flavor: str | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
    engram_enabled: bool = True,
) -> DeepSeekV41GraphTrainer.Config:
    flavor = flavor or "deepseek_v4_1_flash_40layers_16experts_" + ("vision" if multimodal else "text")
    base = _make_trainer_config(
        flavor,
        engram=engram,
        engram_enabled=engram_enabled,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
    )
    base.training.num_tokens_per_train_step = -1
    if multimodal:
        base.override.imports.append("torchtitan_npu.override.deepseek_v4_1.vision.asc_unmasked_sdpa")
    upstream = Trainer.Config(**{f.name: getattr(base, f.name) for f in fields(Trainer.Config)})
    model_cls = GraphTrainerDeepSeekV41MultimodalModel if multimodal else GraphTrainerDeepSeekV41Model
    graph = to_graph_trainer_config(upstream, model_cls.Config)
    graph.loss = derive(graph.loss, CoalescedChunkedLoss.Config)
    graph.compile = GraphTrainerCompileConfig(
        ep_overlap=EpOverlapConfig(enabled=True, chunk_dim="seq", module_fqn=MOE_BLOCK_FQN),
        memory_policy="full",
        pass_pipeline=f"{functionalize_recompute_mutations.PASS_PIPELINE_NAME}+npu_auto_overlap",
        disable_passes=["cudagraph_pass"],
    )
    return DeepSeekV41GraphTrainer.Config(
        **{f.name: getattr(graph, f.name) for f in fields(graph)},
        extension=base.extension,
        engram_enabled=engram_enabled,
    )


def graph_trainer_deepseek_v4_1_flash_40layers_16experts_text(**kwargs) -> DeepSeekV41GraphTrainer.Config:
    return _make_graph_trainer_config(**kwargs)


def graph_trainer_deepseek_v4_1_flash_40layers_16experts_multimodal(**kwargs) -> DeepSeekV41GraphTrainer.Config:
    return _make_graph_trainer_config(multimodal=True, **kwargs)
