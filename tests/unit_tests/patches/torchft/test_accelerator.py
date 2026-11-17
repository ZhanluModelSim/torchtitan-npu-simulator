# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys


def test_patch_rebinds_preloaded_consumers_and_is_idempotent():
    program = """
from unittest.mock import Mock, patch
import sys
import torchft.utils as utils
import torchft.manager as manager
import torchft.process_group as process_group
import torchft.collectives as collectives
import torchft.futures as futures
import torchft.checkpointing.http_transport as transport

from torchtitan_npu.patches.torchft import accelerator

expected = {
    'get_stream_context': ('torchft.manager', 'torchft.process_group',
                           'torchft.collectives', 'torchft.futures',
                           'torchft.checkpointing.http_transport'),
    'record_event': ('torchft.process_group',),
    'synchronize': ('torchft.manager', 'torchft.process_group'),
}
for name, consumers in expected.items():
    assert getattr(utils, name) is getattr(accelerator, name)
    for consumer in consumers:
        assert getattr(sys.modules[consumer], name) is getattr(accelerator, name)

assert process_group.ProcessGroup._register is accelerator._register_process_group
accelerator.apply()
assert process_group.ProcessGroup._register is accelerator._register_process_group
with patch.object(accelerator.dist.Backend, 'register_backend') as register, \\
     patch.object(accelerator, '_current_accelerator_type', return_value='npu'):
    group = process_group.ProcessGroup(0, 1)
    group.getBackendName = Mock(return_value='torchft-hccl')
    assert group._register('dp_replicate') == 'torchft-hccl:dp_replicate'
    assert register.call_args.kwargs['devices'] == ['cpu', 'npu']
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120)


def test_plain_import_does_not_require_torchft_and_opt_in_explains_installation():
    program = """
import importlib.abc
import sys
from torchtitan.config import ConfigManager

class BlockTorchFT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torchft' or fullname.startswith('torchft.'):
            raise ModuleNotFoundError('TorchFT is unavailable', name=fullname)

sys.meta_path.insert(0, BlockTorchFT())
import torchtitan_npu
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

assert not any(name == 'torchft' or name.startswith('torchft.') for name in sys.modules)
common_args = [
    '--module', 'torchtitan_npu.models.deepseek_v4',
    '--config', 'deepseek_v4_flash_43layers_16experts',
]
config = ConfigManager().parse_args(common_args)
assert isinstance(config, TrainerEx.Config)
assert not any(name == 'torchft' or name.startswith('torchft.') for name in sys.modules)
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_flash_torchft
try:
    deepseek_v4_flash_torchft()
except ModuleNotFoundError as error:
    assert error.name == 'torchft'
    assert "pip install -e '.[torchft]'" in str(error)
else:
    raise AssertionError('Selecting a TorchFT recipe without its dependency must fail')
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
