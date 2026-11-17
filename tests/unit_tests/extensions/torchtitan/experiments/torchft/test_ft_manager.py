# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys


def test_synchronous_npu_manager_contract():
    program = """
from unittest.mock import Mock, patch

from torchtitan_npu.extensions.torchtitan.experiments.torchft import ft_manager as module

with patch.object(module, 'ManagerEx') as manager, \\
     patch.object(module, 'ProcessGroupHCCLEx') as process_group, \\
     patch.object(module.torchft.process_group, 'ManagedProcessGroup') as managed:
    config = module.FTManagerEx.Config(
        replica_id=2, group_size=4, min_replica_size=2, process_group_timeout_ms=135000
    )
    assert not hasattr(config, 'enable_hccl_recovery')
    assert not hasattr(config, 'state_dict_on_training_thread')
    ft = module.FTManagerEx(config)
    assert not ft.use_async_quorum
    assert ft.group_size == config.group_size
    assert ft.replica_id == config.replica_id
    assert manager.call_args.kwargs['use_async_quorum'] is False
    assert manager.call_args.kwargs['init_sync'] is True
    assert manager.call_args.kwargs['min_replica_size'] == 2
    assert manager.call_args.kwargs['replica_id'] == 'torchtitan_ft_2'
    assert manager.call_args.kwargs['pg'] is process_group.return_value
    assert process_group.call_args.args[0].total_seconds() == 135
    managed.assert_called_once_with(manager.return_value)
    managed.return_value.register.assert_called_once_with('dp_replicate')
    assert ft.loss_sync_pg is managed.return_value

for overrides in (
    {'enable': False},
    {'process_group': 'gloo'},
    {'semi_sync_method': 'local_sgd'},
    {'group_size': 1},
    {'group_size': 2, 'min_replica_size': 3},
    {'group_size': 2, 'replica_id': 2},
):
    try:
        module.FTManagerEx(module.FTManagerEx.Config(**overrides))
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted unsupported FT configuration: {overrides}')
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120)


def test_hccl_process_group_creation_and_lifecycle():
    program = """
from datetime import timedelta
from unittest.mock import Mock, patch

import torch
from torchtitan_npu.extensions.torchtitan.experiments.torchft import process_group as module

with patch('torch_npu._C._distributed_c10d.ProcessGroupHCCL') as hccl, \\
     patch.object(module, 'ProcessGroup') as process_group, \\
     patch.object(torch.accelerator, 'current_device_index', return_value=0):
    pg = module.ProcessGroupHCCLEx(timedelta(seconds=90), recovery_timeout=timedelta(seconds=60))
    pg._quorum_id = 7
    pg._group_rank = 1
    pg._global_ranks = [0, 4]
    wrapper = pg._create_pg(Mock(), 1, 2)
    options = hccl.Options.return_value
    assert options._timeout == timedelta(seconds=180)
    assert options.group_id == 'torchft_quorum_7_rank_1'
    assert options.global_ranks_in_group == [0, 4]
    hccl.return_value._set_sequence_number_for_group.assert_called_once()
    wrapper._register_backend.assert_called_once_with(
        torch.device('npu'), process_group.BackendType.CUSTOM, hccl.return_value
    )

    pg._pg = wrapper
    wrapper._get_backend.return_value = hccl.return_value
    pg.shutdown()
    hccl.return_value.shutdown.assert_called_once()
    hccl.return_value.abort.assert_not_called()
    hccl.return_value.clear_workmeta_list.assert_not_called()
    assert pg._pg is None

    hccl.return_value.reset_mock()
    pg._pg = wrapper
    pg.abort()
    hccl.return_value.abort.assert_called_once()
    hccl.return_value.shutdown.assert_called_once()
    hccl.return_value.clear_workmeta_list.assert_called_once()
    assert pg._pg is None and isinstance(pg.errored(), RuntimeError)
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120)


def test_cross_replica_hook_waits_for_the_exact_all_reduce_work():
    program = """
from unittest.mock import Mock, patch

import torch
from torchtitan_npu.extensions.torchtitan.experiments.torchft import ft_manager as module

events = []
hooks = []
ft = module.FTManagerEx.__new__(module.FTManagerEx)
ft.replicate_pg = Mock()
ft.replicate_pg.size.return_value = 2
ft.process_group = Mock()
def install_hooks(parts, hook):
    hooks.append(hook)
    return 1

work = Mock()
def all_reduce(output, *, group, op, async_op):
    events.append(('all_reduce', group, op, async_op))
    return work

with patch.object(module, '_install_all_reduce_hooks', side_effect=install_hooks), \\
     patch.object(module.dist, 'all_reduce', side_effect=all_reduce):
    ft.maybe_set_all_reduce_hook([Mock()])
    hooks[0](torch.ones(1))

assert events == [('all_reduce', ft.replicate_pg, torch.distributed.ReduceOp.AVG, True)]
ft.process_group.wait_for_completion.assert_called_once_with(work)

ft.replicate_pg.size.return_value = 1
hooks[0](torch.ones(1))
assert len(events) == 1
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120)
