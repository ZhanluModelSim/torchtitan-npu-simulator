# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist

from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager


class _TrainState:
    def __init__(self, step):
        self.step = torch.tensor(step)

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


def test_native_checkpoint_restores_model_optimizer_scheduler_and_extra_state(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        model.weight.grad = torch.tensor([[3.0]])
        optimizer.step()
        scheduler.step()
        expected_weight = model.weight.detach().clone()
        expected_moment = optimizer.state[model.weight]["exp_avg"].clone()
        expected_lr = scheduler.get_last_lr()
        state = _TrainState(5)
        manager = CheckpointManager.Config(interval=1, last_save_model_only=False).build(
            model_parts=[model],
            optimizers=optimizer,
            lr_schedulers=scheduler,
            ema=None,
            extra_states={"train_state": state},
            sd_adapter=None,
            base_folder=str(tmp_path),
        )

        assert manager._metadata_pg is None
        assert manager.save(1)
        model.weight.data.fill_(9.0)
        optimizer.state[model.weight]["exp_avg"].fill_(9.0)
        scheduler.last_epoch = 99
        state.step.fill_(99)
        assert manager.load(1)
        manager.close()

        torch.testing.assert_close(model.weight, expected_weight)
        torch.testing.assert_close(optimizer.state[model.weight]["exp_avg"], expected_moment)
        assert scheduler.last_epoch == 1
        assert scheduler.get_last_lr() == expected_lr
        assert state.step.item() == 5
    finally:
        dist.destroy_process_group()


class _ShardedTrainState:
    def __init__(self, value, step):
        self.value = value
        self.step = step

    def state_dict(self):
        return {"value": self.value, "step": self.step}

    def load_state_dict(self, state):
        self.value = state["value"]
        self.step = state["step"]


def _metadata_group_roundtrip_worker(rank, rendezvous, checkpoint_dir):
    from unittest.mock import patch

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from torchtitan_npu.extensions.torchtitan.components import checkpoint as checkpoint_module

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    manager = None
    try:
        mesh = init_device_mesh("cpu", (2,))
        state = _ShardedTrainState(distribute_tensor(torch.arange(8), mesh, [Shard(0)]), 7)
        expected_local = state.value.to_local().clone()
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        model.weight.grad = torch.tensor([[3.0]])
        optimizer.step()
        scheduler.step()
        expected_weight = model.weight.detach().clone()
        expected_moment = optimizer.state[model.weight]["exp_avg"].clone()
        get_backend = dist.get_backend

        # Simulate HCCL selection while retaining real two-rank Gloo collectives.
        def hccl_default(group=None):
            return "hccl" if group is None else get_backend(group)

        with (
            patch.object(dist, "get_backend", side_effect=hccl_default),
            patch.object(dist, "destroy_process_group", wraps=dist.destroy_process_group) as destroy_group,
            patch.object(checkpoint_module.dcp, "save", wraps=checkpoint_module.dcp.save) as save,
            patch.object(checkpoint_module.dcp, "load", wraps=checkpoint_module.dcp.load) as load,
        ):
            manager = CheckpointManager.Config(interval=1, last_save_model_only=False).build(
                model_parts=[model],
                optimizers=optimizer,
                lr_schedulers=scheduler,
                ema=None,
                extra_states={"train_state": state},
                sd_adapter=None,
                base_folder=checkpoint_dir,
            )
            assert manager._metadata_pg is not None
            assert get_backend(manager._metadata_pg) == "gloo"
            assert manager.save(1)
            assert save.call_args.kwargs["process_group"] is manager._metadata_pg
            model.weight.data.fill_(9.0)
            optimizer.state[model.weight]["exp_avg"].fill_(9.0)
            state.value.to_local().fill_(99)
            state.step = 99
            assert manager.load(1)
            assert load.call_args.kwargs["process_group"] is manager._metadata_pg
            torch.testing.assert_close(model.weight, expected_weight, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[model.weight]["exp_avg"], expected_moment, rtol=0, atol=0)
            torch.testing.assert_close(state.value.to_local(), expected_local, rtol=0, atol=0)
            assert state.step == 7
            assert scheduler.last_epoch == 1
            metadata_group = manager._metadata_pg
            manager.close()
            manager.close()
            destroy_group.assert_called_once_with(metadata_group)
            manager = None

        # Closing the owned metadata group must leave the training group intact.
        alive = torch.tensor(rank + 1)
        dist.all_reduce(alive)
        assert alive.item() == 3
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


def test_native_checkpoint_uses_owned_gloo_metadata_group_for_two_rank_restore(tmp_path):
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    setup = """
import sys
import types
from pathlib import Path
root = Path.cwd()
for name in (
    "torchtitan_npu", "torchtitan_npu.patches", "torchtitan_npu.patches.torch_npu",
    "torchtitan_npu.patches.torchtitan", "torchtitan_npu.patches.torchtitan.models",
    "torchtitan_npu.patches.torchtitan.models.common", "torchtitan_npu.extensions",
):
    module = types.ModuleType(name)
    module.__path__ = [str(root.joinpath(*name.split(".")))]
    sys.modules[name] = module
sys.modules["torch_npu"] = types.ModuleType("torch_npu")
import torchtitan_npu.patches.torchtitan.models.common.rope
import torchtitan_npu.patches.torchtitan.models.common.moe
import torchtitan_npu.patches.torchtitan.models.common.token_dispatcher
from tests.unit_tests.extensions.torchtitan.components.test_checkpoint_v04 import _metadata_group_roundtrip_worker
_metadata_group_roundtrip_worker(int(sys.argv[1]), sys.argv[2], sys.argv[3])
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), *sys.path])

    def run_rank(rank):
        return subprocess.run(
            [sys.executable, "-c", setup, str(rank), str(tmp_path / "rendezvous"), str(tmp_path)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_rank, range(2)))
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
