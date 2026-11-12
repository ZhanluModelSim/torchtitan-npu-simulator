# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
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
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash, deepseek_v4_flash_8p_torchft

def make_config():
    return ConfigManager().parse_args([
        '--module', 'torchtitan_npu_recipes.models.deepseek_v4',
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
assert config.checkpointer.enable_ft_dataloader_checkpoints
assert config.model.n_layers == 4
assert config.parallelism.expert_parallel_degree == 4
assert config.parallelism.data_parallel_shard_degree == 4
assert config.training.max_context_length == config.model.max_context_length == 256
assert config.training.num_tokens_per_microbatch_per_dp_rank == 256
assert config.training.num_tokens_per_train_step == 4096
assert config.training.num_tokens_per_train_step // (256 * 4 * 2) == 2
assert isinstance(config.profiler, npu_trainer.CANNProfiler.Config)

full_config = ConfigManager().parse_args([
    '--module', 'torchtitan_npu_recipes.models.deepseek_v4',
    '--config', 'deepseek_v4_flash_torchft',
])
assert isinstance(full_config, module.FaultTolerantTrainerEx.Config)
assert full_config.model.n_layers == 43
assert full_config.model.layers[2].moe.num_experts == 256
assert full_config.parallelism.data_parallel_shard_degree == 64
assert full_config.parallelism.expert_parallel_degree == 64
assert full_config.checkpointer.enable_ft_dataloader_checkpoints
base_config = deepseek_v4_flash()
assert full_config.optim.optimizer.optimizers == base_config.optim.optimizer.optimizers
module.FaultTolerantTrainerEx._validate_supported_config(full_config)

small_expert_config = ConfigManager().parse_args([
    '--module', 'torchtitan_npu_recipes.models.deepseek_v4',
    '--config', 'deepseek_v4_flash_43layers_16experts_torchft',
])
assert isinstance(small_expert_config, module.FaultTolerantTrainerEx.Config)
assert small_expert_config.model.layers[2].moe.num_experts == 16
assert small_expert_config.parallelism.expert_parallel_degree == 16
module.FaultTolerantTrainerEx._validate_supported_config(small_expert_config)

def set_path(root, path, value):
    target = root
    parts = path.split('.')
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)

unsupported = (
    ('sdc_replayer', object(), 'does not support the upstream SDC replay engine'),
    ('checkpointer.enable_ft_dataloader_checkpoints', False, 'requires checkpoint'),
    ('parallelism.data_parallel_replicate_degree', 2, 'elastic DP replicas'),
    ('parallelism.data_parallel_shard_degree', 1, 'requires both FSDP'),
    ('parallelism.expert_parallel_degree', 1, 'requires both FSDP'),
    ('parallelism.tensor_parallel_degree', 2, 'FSDP+EP only'),
)
for path, value, message in unsupported:
    candidate = deepseek_v4_flash_8p_torchft()
    set_path(candidate, path, value)
    with patch.object(npu_trainer.Trainer, '__init__') as initialize:
        try:
            module.FaultTolerantTrainerEx(candidate)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f'accepted unsupported TorchFT configuration: {path}={value!r}')
        initialize.assert_not_called()

initialize_ft = Mock()
def initialize_engine(trainer, cfg):
    initialize_ft(cfg)
    engine = Mock()
    engine.model_parts = []
    engine.optim = Mock(lr_schedulers=Mock())
    engine.ft_manager = Mock(
        spec=module.FTManagerEx,
        process_group=Mock(recovery=object()),
        manager=Mock(spec=module.ManagerEx),
    )
    trainer.engine = engine
    trainer.gradient_accumulation_steps = 1

with patch.object(npu_trainer.Trainer, '__init__', initialize_engine), \\
     patch.object(npu_trainer, 'set_allow_hf32') as set_hf32, \\
     patch.object(type(config.sdc), 'build') as build_sdc, \\
     patch.object(module, 'StepRecovery') as recovery:
    trainer = module.FaultTolerantTrainerEx(config)
    initialize_ft.assert_called_once_with(config)
    set_hf32.assert_called_once_with(config.training.extension.allow_hf32)
    build_sdc.assert_called_once_with(
        trainer_config=config, model_parts=[], gradient_accumulation_steps=1
    )
    assert trainer.engine._sdc is build_sdc.return_value
    recovery.assert_called_once()
    assert trainer.recovery is recovery.return_value
    assert trainer.anticipatory_schedule is None

    for ft_manager, message in (
        (Mock(), 'requires FTManagerEx'),
        (Mock(spec=module.FTManagerEx, manager=Mock()), 'requires ManagerEx'),
    ):
        def initialize_wrong_manager(trainer, cfg):
            initialize_engine(trainer, cfg)
            trainer.engine.ft_manager = ft_manager

        recovery.reset_mock()
        with patch.object(npu_trainer.Trainer, '__init__', initialize_wrong_manager):
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
module.FaultTolerantTrainerEx._reset_auxiliary_state(Mock(engine=Mock(model_parts=[auxiliary])))
assert auxiliary.instance_acc.item() == 0
assert AuxiliaryLoss.group_acc == {}
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_ft_train_step_consumes_collated_valid_tokens(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from torchtitan_npu.extensions.torchtitan.experiments.torchft import trainer as module

    parallel = SimpleNamespace(
        dp_enabled=False,
        dp_cp_enabled=False,
        pp_enabled=False,
        ep_enabled=False,
    )
    parallel.get_mesh = lambda _: None
    parallel.get_optional_mesh = lambda _: None
    microbatches = [
        SimpleNamespace(loss_token_counts=torch.tensor(2), routing_token_counts=torch.tensor(2)),
        SimpleNamespace(loss_token_counts=torch.tensor(1), routing_token_counts=torch.tensor(1)),
    ]
    observed = []
    result = SimpleNamespace(loss=torch.tensor(0.5))

    def forward_backward(*, microbatch_groups, global_valid_tokens):
        observed.append((microbatch_groups, global_valid_tokens.item()))
        return result

    schedulers = Mock()
    schedulers.schedulers = [Mock()]
    schedulers.schedulers[0].get_last_lr.return_value = [0.01]
    engine = Mock()
    engine.parallelism_context = parallel
    engine.num_completed_steps = 0
    engine.device = torch.device("cpu")
    engine.ntokens_seen = 3
    engine.optim = Mock(lr_schedulers=schedulers)
    engine.forward_backward = forward_backward
    engine.optim_step = Mock(return_value=torch.tensor(1.0))

    trainer = SimpleNamespace(
        engine=engine,
        metrics_processor=Mock(should_log=Mock(return_value=False)),
        gradient_accumulation_steps=2,
        num_pp_microbatches=1,
    )

    module.FaultTolerantTrainerEx._train_step(trainer, iter(microbatches))

    assert len(observed) == 1
    groups, valid_tokens = observed[0]
    assert groups == [[microbatches[0]], [microbatches[1]]]
    assert valid_tokens == 3
    engine.optim_step.assert_called_once_with()
    trainer.metrics_processor.log.assert_not_called()
