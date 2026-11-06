# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU round trip through the local native-checkpoint integration."""

import copy
from contextlib import closing

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict

pytest.importorskip("torch_checkpointing")

from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer, ParamGroupConfig

from torchtitan_npu.extensions.torchtitan.components.torch_checkpointing import TorchCheckpointingManager

pytestmark = pytest.mark.cpu


class _Progress:
    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


@pytest.fixture(name="process_group")
def fixture_process_group(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_native_config_restores_full_state_after_async_and_final_save(tmp_path, process_group):
    def build_manager(*, load_only):
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Linear(2, 1, bias=False)
        optimizers = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.1})],
        ).build(model_parts=[model])
        scheduler = LRSchedulersContainer(optimizers, lr_lambda=lambda step: 1.0)
        progress, dataloader = _Progress(), _Progress()
        manager = TorchCheckpointingManager.Config(
            enable=True,
            folder=str(tmp_path / "checkpoint"),
            interval=1,
            load_only=load_only,
            last_save_model_only=False,
            initial_load_path=str(tmp_path / "unused-initial-hf") if load_only else None,
            initial_load_in_hf=load_only,
            initial_load_in_hf_quantized=load_only,
        ).build(
            storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
            dataloader=dataloader,
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=scheduler,
            states={"progress": progress},
            sd_adapter=None,
        )
        return manager, model, optimizers, scheduler, progress, dataloader

    manager, model, optimizers, scheduler, progress, dataloader = build_manager(load_only=False)
    expected = {}
    with closing(manager):
        for step in (1, 2):
            optimizers.zero_grad()
            model(torch.tensor([[0.25, -0.5]])).square().sum().backward()
            optimizers.step()
            scheduler.step()
            progress.step = dataloader.step = step
            expected[step] = copy.deepcopy(_stateful_to_state_dict(manager.states))
            assert manager.save(step, last_step=step == 2)
            manager.maybe_wait_for_staging()
            manager.maybe_wait_for_saving()

    restored, *_ = build_manager(load_only=True)
    with closing(restored):
        for requested, saved in ((1, 1), (-1, 2)):
            assert restored.load(requested)
            torch.testing.assert_close(_stateful_to_state_dict(restored.states), expected[saved], rtol=0, atol=0)
