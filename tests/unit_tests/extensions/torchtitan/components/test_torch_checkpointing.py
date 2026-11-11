# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU save, restore, and HF export through the native-checkpoint integration."""

import copy
import json
from contextlib import closing

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict

pytest.importorskip("torch_checkpointing")

from safetensors.torch import load_file
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.optim import AdamW, LRSchedulersContainer, OptimizersContainer

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
            optimizers=[AdamW.Config(pattern=".*", lr=0.1, fused=False)],
        ).build(model_parts=[model])
        scheduler = LRSchedulersContainer.Config().build(optimizers=optimizers, training_steps=3)
        progress, dataloader = _Progress(), _Progress()
        manager = TorchCheckpointingManager.Config(
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
            ema=None,
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


def test_native_final_save_exports_hf_weights_and_preserves_integer_buffers(tmp_path, process_group):
    class HFExportAdapter:
        fqn_to_index_mapping = {"hf.weight": 1, "hf.counter": 1}

        @staticmethod
        def to_hf(state_dict):
            return {"hf.weight": state_dict["weight"], "hf.counter": state_dict["counter"]}

    weights = torch.tensor([[0.123456, -0.654321]], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(weights)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=".*", lr=0.1, fused=False)],
    ).build(model_parts=[model])
    scheduler = LRSchedulersContainer.Config().build(optimizers=optimizers, training_steps=3)
    manager = TorchCheckpointingManager.Config(
        folder=str(tmp_path / "checkpoint"),
        last_save_in_hf=True,
        last_save_model_only=True,
        export_dtype="float16",
    ).build(
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
        dataloader=None,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=scheduler,
        ema=None,
        states={},
        sd_adapter=HFExportAdapter(),
    )

    with closing(manager):
        assert manager.save(2, last_step=True)

    checkpoint_dir = tmp_path / "checkpoint" / "step-2"
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"hf.weight", "hf.counter"}
    exported = {}
    for shard in set(index["weight_map"].values()):
        exported.update(load_file(checkpoint_dir / shard))
    assert set(exported) == {"hf.weight", "hf.counter"}
    assert exported["hf.weight"].dtype == torch.float16
    torch.testing.assert_close(exported["hf.weight"], weights.to(torch.float16), rtol=0, atol=0)
    assert exported["hf.counter"].dtype == torch.int64
    torch.testing.assert_close(exported["hf.counter"], torch.tensor(3, dtype=torch.int64), rtol=0, atol=0)
