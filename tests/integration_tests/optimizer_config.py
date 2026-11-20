# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Small optimizer integration recipes with explicit checkpoint ownership."""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.config import CompileConfig, ModelCompileConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu_recipes.models.deepseek_v4 import (
    _dsv4_muon_profile,
    _enable_indexer_aux_loss,
)
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


def deepseek_v4_checkpoint(*, parallelism: ParallelismConfig | None = None, seq_len: int = 2048):
    config = deepseek_v4_debugmodel(parallelism=parallelism, seq_len=seq_len)
    config.checkpointer = CheckpointManager.Config()
    return config


def deepseek_v4_swap_checkpoint():
    config = deepseek_v4_checkpoint(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2, expert_parallel_degree=2, fsdp_reshard_after_forward="always"
        ),
        seq_len=128,
    )
    config.override.imports.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    config.hf_assets_path = "tests/assets/deepseek_v3"
    config.training.steps = 4
    config.training.max_context_length = 128
    config.training.num_tokens_per_microbatch_per_dp_rank = 128
    config.training.num_tokens_per_train_step = 256
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.metrics.save_tb_folder = f"tb_phase_{os.environ.get('TT_NPU_INTEGRATION_PHASE', '0')}"
    config.checkpointer.interval = 2
    config.checkpointer.last_save_model_only = False
    config.checkpointer.keep_latest_k = 0
    return config


def deepseek_v4_muon_swap_checkpoint():
    config = deepseek_v4_swap_checkpoint()
    _enable_indexer_aux_loss(config.model)
    for layer in config.model.layers:
        if layer.attention.indexer is not None:
            layer.attention.inner_attention.index_topk = 8
    profile = _dsv4_muon_profile(config.model)
    config.optim.optimizer.optimizers = [
        DistMuon.Config(
            pattern=profile.muon_pattern,
            compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
            bucket_configs=profile.bucket_configs,
            lr=1e-5,
            ns_steps=10,
            adjust_lr_fn="match_rms_adamw",
        ),
        AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
    ]
    return config


def deepseek_v4_aot_eager():
    config = deepseek_v4_debugmodel()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig(backend="aot_eager"))
    return config
