# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Eager DSV4 golden configurations owned by the integration suite."""

from torchtitan.components.optim import AdamW
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


def _golden_config(parallelism: ParallelismConfig):
    config = deepseek_v4_debugmodel(
        parallelism=parallelism,
        hf_assets_path="tests/assets/deepseek_v3",
    )
    config.training.steps = 30
    config.training.num_tokens_per_train_step = 4096
    config.optim.lr_scheduler.total_steps = 100
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.metrics.save_tb_folder = "tb"
    # Golden runs intentionally use the original AdamW reference optimizer.
    config.optim.optimizer = OptimizerConfig(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, betas=(0.9, 0.95), eps=1e-6)]
    )
    return config


def dsv4_golden_1rank():
    return _golden_config(ParallelismConfig(
        data_parallel_shard_degree=1, expert_parallel_degree=1, fsdp_reshard_after_forward="always"
    ))


def dsv4_golden_ep2_fsdp2():
    return _golden_config(ParallelismConfig(
        data_parallel_shard_degree=2, expert_parallel_degree=2, fsdp_reshard_after_forward="always"
    ))
