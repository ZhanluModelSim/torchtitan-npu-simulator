# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys


def test_ft_trainer_config_and_npu_lifecycle():
    program = """
from unittest.mock import Mock, patch
import torch
from torchtitan.config import ConfigManager

from torchtitan_npu.extensions.torchtitan.experiments.torchft import trainer as module
from torchtitan_npu.extensions.torchtitan import trainer as npu_trainer
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_flash, deepseek_v4_flash_8p_torchft

def make_config():
    return ConfigManager().parse_args([
        '--module', 'torchtitan_npu.models.deepseek_v4',
        '--config', 'deepseek_v4_flash_8p_torchft',
        '--fault-tolerance.group-size', '2',
        '--fault-tolerance.min-replica-size', '2',
        '--fault-tolerance.replica-id', '0',
        '--fault-tolerance.process-group-timeout-ms', '123456',
    ])

config = make_config()
assert isinstance(config, module.FaultTolerantTrainerEx.Config)
assert config.fault_tolerance.group_size == 2
assert config.fault_tolerance.min_replica_size == 2
assert config.fault_tolerance.replica_id == 0
assert config.fault_tolerance.process_group_timeout_ms == 123456
assert config.checkpoint.enable and config.checkpoint.enable_ft_dataloader_checkpoints
assert config.model_spec.model.n_layers == 4
assert config.parallelism.expert_parallel_degree == 4
assert config.parallelism.data_parallel_shard_degree == 4
assert isinstance(config.profiler, npu_trainer.CANNProfiler.Config)

full_config = ConfigManager().parse_args([
    '--module', 'torchtitan_npu.models.deepseek_v4',
    '--config', 'deepseek_v4_flash_torchft',
])
assert isinstance(full_config, module.FaultTolerantTrainerEx.Config)
assert full_config.model_spec.model.n_layers == 43
assert full_config.model_spec.model.layers[2].moe.num_experts == 256
assert full_config.parallelism.data_parallel_shard_degree == 64
assert full_config.parallelism.expert_parallel_degree == 64
assert full_config.checkpoint.enable_ft_dataloader_checkpoints
base_config = deepseek_v4_flash()
assert full_config.optimizer.param_groups == base_config.optimizer.param_groups
module.FaultTolerantTrainerEx._validate_supported_config(full_config)

small_expert_config = ConfigManager().parse_args([
    '--module', 'torchtitan_npu.models.deepseek_v4',
    '--config', 'deepseek_v4_flash_43layers_16experts_torchft',
])
assert isinstance(small_expert_config, module.FaultTolerantTrainerEx.Config)
assert small_expert_config.model_spec.model.layers[2].moe.num_experts == 16
assert small_expert_config.parallelism.expert_parallel_degree == 16
module.FaultTolerantTrainerEx._validate_supported_config(small_expert_config)

def set_path(root, path, value):
    target = root
    parts = path.split('.')
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)

unsupported = (
    ('checkpoint.enable', False, 'requires checkpoint'),
    ('checkpoint.enable_ft_dataloader_checkpoints', False, 'requires checkpoint'),
    ('model_spec.flavor', 'deepseek_v4_dense', 'DeepSeek-V4 Flash'),
    ('parallelism.data_parallel_replicate_degree', 2, 'elastic DP replicas'),
    ('parallelism.data_parallel_shard_degree', 1, 'requires both FSDP'),
    ('parallelism.expert_parallel_degree', 1, 'requires both FSDP'),
    ('parallelism.tensor_parallel_degree', 2, 'FSDP+EP only'),
)
for path, value, message in unsupported:
    candidate = deepseek_v4_flash_8p_torchft()
    set_path(candidate, path, value)
    with patch.object(module.TrainerEx, '__init__') as initialize:
        try:
            module.FaultTolerantTrainerEx(candidate)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f'accepted unsupported TorchFT configuration: {path}={value!r}')
        initialize.assert_not_called()

initialize_ft = Mock()
def initialize_once(trainer, cfg):
    initialize_ft(cfg)
    trainer.model_parts = []
    trainer.loss_fn = None
    trainer.gradient_accumulation_steps = 1
    trainer.ft_manager = Mock(
        spec=module.FTManagerEx,
        process_group=Mock(recovery=object()),
        manager=Mock(spec=module.ManagerEx),
    )
    trainer.lr_schedulers = Mock()

with patch.object(module.FaultTolerantTrainer, '__init__', initialize_once), \\
     patch.object(npu_trainer, 'set_allow_hf32') as set_hf32, \\
     patch.object(type(config.sdc), 'build') as build_sdc, \\
     patch.object(module, 'StepRecovery') as recovery:
    trainer = module.FaultTolerantTrainerEx(config)
    initialize_ft.assert_called_once_with(config)
    set_hf32.assert_called_once_with(config.training.extension.allow_hf32)
    build_sdc.assert_called_once_with(
        trainer_config=config, model_parts=[], gradient_accumulation_steps=1
    )
    assert trainer._sdc is build_sdc.return_value
    recovery.assert_called_once()

    for ft_manager, message in (
        (Mock(), 'requires FTManagerEx'),
        (Mock(spec=module.FTManagerEx, manager=Mock()), 'requires ManagerEx'),
    ):
        def initialize_wrong_manager(trainer, cfg):
            initialize_once(trainer, cfg)
            trainer.ft_manager = ft_manager

        recovery.reset_mock()
        with patch.object(module.FaultTolerantTrainer, '__init__', initialize_wrong_manager):
            try:
                module.FaultTolerantTrainerEx(config)
            except TypeError as error:
                assert message in str(error)
            else:
                raise AssertionError('accepted incompatible TorchFT manager')
        recovery.assert_not_called()

class AuxiliaryLoss(torch.nn.Module):
    # The production aux-loss state: a per-instance accumulator buffer and the
    # per-group step registers the optimizer pre-hook fills.
    group_acc = {'loss': 3.0}

    def __init__(self):
        super().__init__()
        self.register_buffer('instance_acc', torch.tensor(3.0), persistent=False)

auxiliary = AuxiliaryLoss()
module.FaultTolerantTrainerEx._reset_auxiliary_state(Mock(model_parts=[auxiliary]))
assert auxiliary.instance_acc.item() == 0
assert AuxiliaryLoss.group_acc == {}
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
