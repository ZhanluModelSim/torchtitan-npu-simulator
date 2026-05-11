# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtitan_npu.config.configs import (
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig,
    TrainingConfig,
)


def test_optimizer_defaults_expose_swap_config():
    config = OptimizerConfig()

    assert config.swap_optimizer is False
    assert config.swap_optimizer_times == 16
    assert config.name == "AdamW"


def test_parallelism_defaults_expose_custom_context_config():
    config = ParallelismConfig()

    assert config.enable_custom_context_parallel is False


def test_parallelism_accepts_custom_context_override():
    config = ParallelismConfig(enable_custom_context_parallel=True)

    assert config.enable_custom_context_parallel is True


def test_training_defaults_expose_npu_memory_ratio():
    config = TrainingConfig()

    assert config.torch_npu_memory_ratio == 1.0


def test_trainer_config_accepts_custom_sections():
    trainer_config = TrainerConfig(
        optimizer=OptimizerConfig(swap_optimizer=True, swap_optimizer_times=8),
        parallelism=ParallelismConfig(enable_custom_context_parallel=True),
        training=TrainingConfig(torch_npu_memory_ratio=0.8),
    )

    assert trainer_config.optimizer.swap_optimizer is True
    assert trainer_config.optimizer.swap_optimizer_times == 8
    assert trainer_config.parallelism.enable_custom_context_parallel is True
    assert trainer_config.training.torch_npu_memory_ratio == 0.8


def test_profiling_defaults_expose_custom_profile_fields():
    config = ProfilingConfig()

    assert config.profile_step_start == 0
    assert config.profile_step_end == 0
    assert config.profile_ranks == [-1]
    assert config.profile_record_shapes is True
    assert config.profile_with_memory is False
    assert config.profile_with_stack is False
    assert config.enable_online_parse is True


def test_trainer_config_uses_custom_config_types_by_default():
    trainer_config = TrainerConfig()

    assert isinstance(trainer_config.optimizer, OptimizerConfig)
    assert isinstance(trainer_config.parallelism, ParallelismConfig)
    assert isinstance(trainer_config.training, TrainingConfig)
    assert isinstance(trainer_config.profiling, ProfilingConfig)
