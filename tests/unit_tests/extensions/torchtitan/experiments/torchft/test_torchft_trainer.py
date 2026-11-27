# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys


def test_ft_trainer_config_and_npu_lifecycle(monkeypatch):
    monkeypatch.delenv("TORCHFT_REPLICA_ID", raising=False)
    monkeypatch.delenv("TORCHFT_MIN_REPLICA_SIZE", raising=False)
    program = """
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
import torchtitan.experiments.torchft.trainer as upstream
from torchtitan.config import ConfigLoader
from torchtitan.experiments.torchft.optimizer import TorchFTOptim

from torchtitan_npu.extensions.torchtitan.experiments import torchft as activation
from torchtitan_npu.extensions.torchtitan.experiments.torchft import trainer as module
from torchtitan_npu.extensions.torchtitan import trainer as npu_trainer
from torchtitan_npu_recipes.models.deepseek_v4 import _enable_indexer_aux_loss
from torchtitan_npu_recipes.tests.ci.deepseek_v4 import deepseek_v4_flash
from torchtitan_npu_recipes.tests.torchft.deepseek_v4 import deepseek_v4_flash_8p_torchft
from torchtitan_npu_recipes.torchft.deepseek_v4 import _torchft_flash_config
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimEx

engine_class = upstream.FaultTolerantTrainingEngine
allocation_hook = engine_class.__new__
activation.apply()
assert upstream.FaultTolerantTrainingEngine is engine_class
assert engine_class is module.FaultTolerantTrainingEngine
assert engine_class.__new__ is allocation_hook
assert issubclass(module.FaultTolerantTrainingEngineEx, engine_class)

def make_config():
    return ConfigLoader().load([
        '--module', 'torchtitan_npu_recipes.tests.torchft.deepseek_v4',
        '--config', 'deepseek_v4_flash_8p_torchft',
    ])

config = make_config()
assert isinstance(config, module.FaultTolerantTrainerEx.Config)
assert config.fault_tolerance.group_size == 2
assert config.fault_tolerance.min_replica_size == 2
assert config.fault_tolerance.replica_id == 0
with patch.dict('os.environ', TORCHFT_REPLICA_ID='1', TORCHFT_MIN_REPLICA_SIZE='1'):
    replacement_config = make_config()
    assert replacement_config.fault_tolerance.replica_id == replacement_config.fault_tolerance.min_replica_size == 1
    assert replacement_config.fault_tolerance.group_size == 2
assert config.checkpointer.enable_ft_dataloader_checkpoints
assert config.model.n_layers == 4
aux_loss = config.model.layers[2].attention.inner_attention.aux_loss
assert aux_loss is not None and aux_loss.coeff > 0
muon = config.optim.optimizer.optimizers[0]
for names in (muon.compute_sharding_by_fqn, [bucket.name for bucket in muon.bucket_configs]):
    assert {int(name.split('.')[1]) for name in names if name.startswith('layers.')} == set(range(4))
    assert {int(name.split('.')[1]) for name in names if name.startswith('mtp_layers.')} == {0}
assert config.parallelism.expert_parallel_degree == 4
assert config.parallelism.data_parallel_shard_degree == 4
assert config.training.max_context_length == config.model.max_context_length == 256
assert config.training.num_tokens_per_microbatch_per_dp_rank == 256
assert config.training.num_tokens_per_train_step == 4096
assert config.training.num_tokens_per_train_step // (256 * 4 * 2) == 2
assert isinstance(config.profiler, npu_trainer.CANNProfiler.Config)
assert not config.extension.quantization.enable_quantized_training
assert config.override.imports == ["torchtitan_npu.override.common.rope.workaround"]

full_config = ConfigLoader().load([
    '--module', 'torchtitan_npu_recipes.torchft.deepseek_v4',
    '--config', 'deepseek_v4_flash_torchft',
])
assert isinstance(full_config, module.FaultTolerantTrainerEx.Config)
assert full_config.model.n_layers == 43
assert full_config.model.layers[2].feed_forward.num_experts == 256
assert full_config.parallelism.data_parallel_shard_degree == 64
assert full_config.parallelism.expert_parallel_degree == 64
assert full_config.checkpointer.enable_ft_dataloader_checkpoints
base_config = deepseek_v4_flash()
_enable_indexer_aux_loss(base_config.model)
assert aux_loss == base_config.model.layers[2].attention.inner_attention.aux_loss
with patch.object(base_config, 'checkpointer', type(config.checkpointer)(folder='custom-ckpt')):
    converted = _torchft_flash_config(
        base_config, expert_parallel_degree=base_config.parallelism.expert_parallel_degree,
        data_parallel_shard_degree=base_config.parallelism.data_parallel_shard_degree,
    )
    assert converted.checkpointer.folder == 'custom-ckpt'
    assert converted.checkpointer.enable_ft_dataloader_checkpoints
assert full_config.optim.optimizer.optimizers == base_config.optim.optimizer.optimizers
assert full_config.optim.gradient_clipper == base_config.optim.gradient_clipper
assert full_config.optim.enable_cuda_graph == base_config.optim.enable_cuda_graph
assert isinstance(full_config.optim, TorchFTOptimEx.Config)
module.FaultTolerantTrainerEx._validate_supported_config(full_config)

small_expert_config = ConfigLoader().load([
    '--module', 'torchtitan_npu_recipes.tests.torchft.deepseek_v4',
    '--config', 'deepseek_v4_flash_43layers_16experts_torchft',
])
assert isinstance(small_expert_config, module.FaultTolerantTrainerEx.Config)
assert small_expert_config.model.layers[2].feed_forward.num_experts == 16
assert small_expert_config.parallelism.expert_parallel_degree == 16
assert not small_expert_config.extension.quantization.enable_quantized_training
assert small_expert_config.override.imports == ["torchtitan_npu.override.common.rope.workaround"]
module.FaultTolerantTrainerEx._validate_supported_config(small_expert_config)

# Preserve native callers after activating the NPU recipe in the same process.
native_config = upstream.FaultTolerantTrainer.Config(
    model=base_config.model, training=base_config.training, optim=TorchFTOptim.Config()
)
class CustomEngine(engine_class):
    pass

with patch.object(module.FaultTolerantTrainingEngine, '__init__', return_value=None) as initialize:
    for engine_config, constructor, expected_type in (
        (config, engine_class, module.FaultTolerantTrainingEngineEx),
        (native_config, engine_class, module.FaultTolerantTrainingEngine),
        (config, CustomEngine, CustomEngine),
    ):
        initialize.reset_mock()
        engine_args = dict(
            model_config=engine_config.model,
            max_num_documents=engine_config.dataloader.max_num_documents,
            output_dir=engine_config.dump_folder,
            fault_tolerance=engine_config.fault_tolerance,
        )
        engine = constructor(engine_config, **engine_args)
        assert type(engine) is expected_type
        initialize.assert_called_once_with(engine_config, **engine_args)

def set_path(root, path, value):
    target = root
    parts = path.split('.')
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)

unsupported = (
    ('optim', base_config.optim, 'requires TorchFTOptimEx'),
    ('optim.optimizer', base_config.optim.optimizer, 'requires TorchFTOptimEx'),
    ('sdc_replayer', object(), 'does not support the upstream SDC replay engine'),
    ('checkpointer.enable_ft_dataloader_checkpoints', False, 'requires checkpoint'),
    ('parallelism.data_parallel_replicate_degree', 2, 'unless disable_expert_replication is enabled'),
    ('parallelism.data_parallel_shard_degree', 1, 'requires both FSDP'),
    ('parallelism.expert_parallel_degree', 1, 'requires both FSDP'),
    ('parallelism.tensor_parallel_degree', 2, 'FSDP+EP only'),
)
for path, value, message in unsupported:
    candidate = deepseek_v4_flash_8p_torchft()
    set_path(candidate, path, value)
    with patch.object(module.FaultTolerantTrainer, '__init__') as initialize:
        try:
            module.FaultTolerantTrainerEx(candidate)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f'accepted unsupported TorchFT configuration: {path}={value!r}')
        initialize.assert_not_called()

from dataclasses import replace as dc_replace
from torchtitan.config import derive
from torchtitan_npu.config import ParallelismConfig as NPUParallelismConfig

expert_sharded_ft = make_config()
expert_sharded_ft = dc_replace(
    expert_sharded_ft,
    parallelism=derive(
        expert_sharded_ft.parallelism,
        NPUParallelismConfig,
        data_parallel_replicate_degree=2,
        disable_expert_replication=True,
    ),
)
assert expert_sharded_ft.parallelism.data_parallel_replicate_degree == 2
assert expert_sharded_ft.parallelism.disable_expert_replication is True
module.FaultTolerantTrainerEx._validate_supported_config(expert_sharded_ft)

dedup_muon = dc_replace(expert_sharded_ft.optim.optimizer.optimizers[0], enable_hsdp_replica_dedup=True)
dedup_optim = dc_replace(
    expert_sharded_ft.optim,
    optimizer=dc_replace(
        expert_sharded_ft.optim.optimizer,
        optimizers=[dedup_muon, *expert_sharded_ft.optim.optimizer.optimizers[1:]],
    ),
)
try:
    dc_replace(expert_sharded_ft, optim=dedup_optim)
except ValueError as error:
    assert 'incompatible with enable_hsdp_replica_dedup' in str(error)
else:
    raise AssertionError('accepted disable_expert_replication with enable_hsdp_replica_dedup')

# Overrides may replace the owner after the early recipe validation.
engine_config = deepseek_v4_flash_8p_torchft()
engine_config.optim.optimizer = base_config.optim.optimizer
invalid_engine = object.__new__(module.FaultTolerantTrainingEngineEx)
invalid_engine.config = engine_config
with patch.object(module.FaultTolerantTrainingEngine, '_initialize_optim') as initialize:
    try:
        invalid_engine._initialize_optim()
    except ValueError as error:
        assert 'requires TorchFTOptimEx' in str(error)
    else:
        raise AssertionError('accepted an optimizer owner replaced by overrides')
    initialize.assert_not_called()

config.fault_tolerance.replica_id = 1
config.override.imports = []
config.tokenizer = None
config.validator = None
ft = object.__new__(module.FTManagerEx)
ft.group_size, ft.replica_id, ft._manager = 2, 1, Mock(spec=module.ManagerEx)
ft.process_group = Mock()
mesh = Mock()
mesh.size.return_value, mesh.get_local_rank.return_value = 4, 1
engine = Mock(
    ft_manager=ft,
    model_parts=[],
    device=torch.device('cpu'),
    model_device_mem_stats=SimpleNamespace(max_reserved_gib=0.0, max_reserved_pct=0.0),
    parallelism_context=SimpleNamespace(dp_enabled=True, pp_enabled=False, get_mesh=lambda _: mesh),
)

# Exercise the actual FT initializer; only resource creation is mocked.
with patch.object(upstream, 'FaultTolerantTrainingEngine', return_value=engine), \\
     patch.object(type(config), 'maybe_log'), \\
     patch.object(type(config.dataloader), 'build') as build_data, \\
     patch.object(type(config.metrics), 'build') as build_metrics, \\
     patch.object(upstream, 'cuda_graphs_supported', return_value=False), \\
     patch.object(upstream.utils, 'get_peak_flops', return_value=1.0), \\
     patch.object(npu_trainer, 'set_allow_hf32') as set_hf32, \\
     patch.object(type(config.sdc), 'build') as build_sdc, \\
     patch.object(module, 'StepRecovery') as recovery:
    trainer = module.FaultTolerantTrainerEx(config)
    assert trainer.gradient_accumulation_steps == 2
    assert build_data.call_args.kwargs['dp_world_size'] == 8
    assert build_data.call_args.kwargs['dp_rank'] == 5
    assert build_metrics.call_args.kwargs['ft_replica_id'] == 1
    engine.initialize.assert_called_once()
    set_hf32.assert_called_once_with(config.training.extension.allow_hf32)
    build_sdc.assert_called_once_with(
        trainer_config=config, model_parts=[], gradient_accumulation_steps=2
    )
    assert trainer.engine._sdc is build_sdc.return_value
    recovery.assert_called_once()
    assert trainer.recovery is recovery.return_value
    assert trainer.anticipatory_schedule is None

    for ft_manager, message in (
        (Mock(), 'requires FTManagerEx'),
        (Mock(spec=module.FTManagerEx, manager=Mock()), 'requires ManagerEx'),
    ):
        ft_manager.get_dp_info = Mock(return_value=(8, 5))
        engine.ft_manager = ft_manager
        recovery.reset_mock()
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
        SimpleNamespace(loss_token_counts=torch.tensor([3, 2]), routing_token_counts=torch.tensor([4, 3])),
        SimpleNamespace(loss_token_counts=torch.tensor([2, 1]), routing_token_counts=torch.tensor([3, 2])),
    ]
    observed = []
    result = SimpleNamespace(loss=torch.tensor(0.5))

    def forward_backward(*, microbatch_groups, global_loss_token_counts, global_routing_token_counts):
        observed.append((microbatch_groups, global_loss_token_counts.tolist(), global_routing_token_counts.tolist()))
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
        recovery=Mock(),
        metrics_processor=Mock(should_log=Mock(return_value=False)),
        gradient_accumulation_steps=2,
        num_pp_microbatches=1,
    )

    module.FaultTolerantTrainerEx._train_step(trainer, iter(microbatches))

    assert len(observed) == 1
    groups, loss_tokens, routing_tokens = observed[0]
    assert groups == [[microbatches[0]], [microbatches[1]]]
    assert loss_tokens == [5, 3]
    assert routing_tokens == [7, 5]
    engine.optim_step.assert_called_once_with()
    trainer.recovery.mark_step_finished.assert_called_once_with()
    trainer.metrics_processor.log.assert_not_called()

    parallel.dp_enabled = parallel.dp_cp_enabled = True
    engine.ft_manager.loss_sync_pg.size.return_value = 2
    trainer.metrics_processor.should_log.return_value = True
    trainer.metrics_processor.step_last_log = 0
    monkeypatch.setattr(module.dist_utils, "dist_sum_tensor", lambda value, _: value * 2)
    monkeypatch.setattr(module.dist_utils, "dist_sum", lambda value, *_: value.item() * 4)
    monkeypatch.setattr(module.dist_utils, "dist_max", lambda value, *_: value.item())
    monkeypatch.setattr(module, "collect_aux_loss_metrics", lambda _: {"aux_loss": 0.25})
    engine.optim_step.side_effect = lambda: trainer.metrics_processor.log.assert_not_called() or torch.tensor(1.0)
    module.FaultTolerantTrainerEx._train_step(trainer, iter(microbatches))
    assert observed[1][1:] == ([10, 6], [14, 10])
    trainer.metrics_processor.reset.assert_called_once()
    log_args = trainer.metrics_processor.log.call_args
    assert log_args.args[1:3] == (1.0, 1.0)
    assert log_args.kwargs["extra_metrics"]["aux_loss"] == 0.25
