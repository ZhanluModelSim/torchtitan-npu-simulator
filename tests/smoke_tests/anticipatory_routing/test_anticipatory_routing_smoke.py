# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU recovery smoke covering loss spike detection, rollback and route replay."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke
logger = logging.getLogger(__name__)
_THIS_FILE = Path(__file__).resolve()
_NO_NPU_EXIT_CODE = 77
_WORLD_SIZE = 1


def test_loss_spike_rollback_and_route_replay(tmp_path):
    root = _THIS_FILE.parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    probe = subprocess.run(
        [sys.executable, str(_THIS_FILE), "--probe"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode == _NO_NPU_EXIT_CODE:
        pytest.skip(probe.stderr.strip() or "Ascend NPU is required")
    assert probe.returncode == 0, probe.stdout + probe.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={_WORLD_SIZE}",
            "--max-restarts=0",
            str(_THIS_FILE),
            "--worker",
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=360,
    )
    if result.returncode == _NO_NPU_EXIT_CODE:
        pytest.skip(result.stderr.strip() or "isolated anticipatory routing worker has no available Ascend NPU")
    assert result.returncode == 0, (
        f"isolated anticipatory routing worker failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _probe():
    try:
        import torch
        import torch_npu  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        logger.warning("Ascend NPU runtime is unavailable: %s", error)
        return _NO_NPU_EXIT_CODE
    if not torch.npu.is_available() or torch.npu.device_count() < _WORLD_SIZE:
        logger.warning("Anticipatory routing smoke requires an available Ascend NPU")
        return _NO_NPU_EXIT_CODE
    return 0


def _worker(folder):
    from contextlib import nullcontext
    from copy import deepcopy
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE

    from torchtitan_npu.extensions.experiment.anticipatory_routing.cache import RoutingMode
    from torchtitan_npu.extensions.experiment.anticipatory_routing.config import AnticipatoryRoutingConfig
    from torchtitan_npu.extensions.experiment.anticipatory_routing.detector import LossSpikeDetector
    from torchtitan_npu.extensions.experiment.anticipatory_routing.router import AnticipatoryHashRouter
    from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import AnticipatorySchedule, Phase
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

    class Loader:
        def __init__(self):
            self.position = 0
            self.injected = set()

        def __iter__(self):
            for index in range(self.position, 24):
                self.position = index + 1
                # Independent CPU storage, distinguishable at every microbatch.
                inputs = torch.tensor([[[index + 1.0, float(rank)]]])
                target = 0
                if rank == 0 and index in (10, 11) and index not in self.injected:
                    target = 1
                    self.injected.add(index)
                yield {"input": inputs}, torch.tensor([[target]])

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state):
            self.position = state["position"]
            # Fault injection is once per process, not restored with dataset position.

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routers = torch.nn.ModuleList()
            for _ in range(2):
                router = AnticipatoryHashRouter.__new__(AnticipatoryHashRouter)
                torch.nn.Module.__init__(router)
                router.gate = torch.nn.Identity()
                router.hash = False
                router.sorted_topk = False
                router.score_func = "sigmoid"
                router.num_experts, router.top_k = 2, 1
                router.num_expert_groups = None
                router.route_norm, router.route_scale = False, 1.0
                router._debug_force_load_balance = False
                self.routers.append(router)
            self.experts = torch.nn.Parameter(torch.tensor([[4.0, -4.0], [3.0, -3.0]]))
            self.register_buffer("counter", torch.zeros(()))
            self.recorded = {}
            self.replayed = []
            self.trained = []

        def forward(self, x):
            self.counter.add_(1)
            index = int(x[0, 0, 0].item()) - 1
            output = 0
            for layer, router in enumerate(self.routers):
                cache = router._anticipatory_cache
                choice = (index + layer + rank) % 2
                if cache.mode is RoutingMode.REPLAY:
                    choice = 1 - choice  # Force live top-k to disagree with captured IDs.
                logits = torch.tensor([[[5.0, 0.0] if choice == 0 else [0.0, 5.0]]], device=x.device)
                scores, ids, _ = router(logits)
                key = (index, layer)
                if cache.mode is RoutingMode.CAPTURE:
                    self.recorded[key] = (x.detach().cpu().clone(), ids.detach().cpu().clone())
                elif cache.mode is RoutingMode.REPLAY:
                    expected_x, expected_ids = self.recorded[key]
                    torch.testing.assert_close(x.cpu(), expected_x, rtol=0, atol=0)
                    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
                    self.replayed.append(key)
                # Actual returned IDs select trainable expert logits.
                output = output + (self.experts[ids] * scores.unsqueeze(-1)).sum(-2)
            if torch.is_grad_enabled():
                self.trained.append((index, cache.mode))
            return output / 2

    class Scheduler(torch.optim.lr_scheduler.StepLR):
        def get_metrics(self):
            return {"lr": self.get_last_lr()[0]}

    trainer = object.__new__(TrainerEx)
    trainer.device = torch.device(f"npu:{local_rank}")
    trainer.step = trainer.ntokens_seen = 0
    model = Model().to(trainer.device)
    trainer.model_parts = [model]
    trainer.dataloader = Loader()
    trainer.gradient_accumulation_steps = 2
    trainer.num_pipeline_parallel_microbatches = 1
    trainer.config = SimpleNamespace(
        anticipatory=AnticipatoryRoutingConfig(
            enable=True,
            delay_steps=2,
            active_steps=2,
            detector=LossSpikeDetector.Config(warmup_steps=4),
        ),
        training=SimpleNamespace(steps=10, disable_cuda_graphs=True, max_norm=100.0),
        checkpoint=SimpleNamespace(interval=2, keep_latest_k=0),
    )
    trainer.parallel_dims = SimpleNamespace(
        pp_enabled=False,
        dp_enabled=False,
        ep_enabled=False,
        dp_cp_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        data_loading_times=[],
        should_log=lambda step: False,
    )
    trainer.train_context = nullcontext
    trainer._sdc = SimpleNamespace(finalize_sdc_step=lambda: None)
    trainer.optimizers = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    trainer.lr_schedulers = Scheduler(trainer.optimizers, step_size=1, gamma=0.99)

    def process(inputs, labels):
        trainer.ntokens_seen += labels.numel()
        return inputs["input"], labels, {}

    trainer.post_dataloading_process = process
    trainer.loss_fn = lambda pred, labels, count, **kwargs: (
        F.cross_entropy(pred.flatten(0, 1), labels.flatten(), reduction="sum") / count,
        None,
    )
    trainer.fwd_bwd_fn = trainer._forward_backward_body

    class DiskCheckpoint:
        """DCP adapter with upstream flat model keys and resumable training state."""

        def __init__(self):
            self.folder = folder
            self.states = {MODEL: model}
            self.saved = []
            self.loaded = []

        def maybe_wait_for_staging(self):
            pass

        def maybe_wait_for_saving(self):
            pass

        def payload(self):
            state = dict(model.state_dict())
            state.update(
                {
                    OPTIMIZER: trainer.optimizers.state_dict(),
                    LR_SCHEDULER: trainer.lr_schedulers.state_dict(),
                    DATALOADER: trainer.dataloader.state_dict(),
                    TRAIN_STATE: trainer.state_dict(),
                }
            )
            return state

        def save(self, curr_step, last_step=False):
            dcp.save(self.payload(), checkpoint_id=str(Path(folder) / f"step-{curr_step}"))
            self.saved.append(curr_step)
            return True

        def load(self, *, step):
            state = deepcopy(self.payload())
            dcp.load(state, checkpoint_id=str(Path(folder) / f"step-{step}"))
            model.load_state_dict({key: state[key] for key in model.state_dict()})
            trainer.optimizers.load_state_dict(state[OPTIMIZER])
            trainer.lr_schedulers.load_state_dict(state[LR_SCHEDULER])
            trainer.dataloader.load_state_dict(state[DATALOADER])
            trainer.load_state_dict(state[TRAIN_STATE])
            torch.testing.assert_close(self.payload(), state, rtol=0, atol=0)
            self.loaded.append(step)
            return True

    trainer.checkpointer = DiskCheckpoint()
    trainer.anticipatory_schedule = AnticipatorySchedule(trainer)
    engine, schedule = trainer.anticipatory_schedule.engine, trainer.anticipatory_schedule
    stream = trainer.batch_generator(trainer.dataloader)
    transitions, blocked = [], []
    for _ in range(14):  # Bounded even if rollback regresses into a loop.
        if trainer.step >= 10:
            break
        trainer.step += 1
        trainer.train_step(stream)
        transitions.append((trainer.step, schedule.phase, len(schedule._queue)))
        if trainer.step == 4 and trainer.checkpointer.loaded:
            assert model.counter.item() == 8  # WARMUP buffer changes were restored.
            assert trainer.ntokens_seen == 8
            assert trainer.lr_schedulers.last_epoch == 4
            assert trainer.dataloader.position == 12
            for entry in schedule._queue:
                assert all(labels.device.type == "cpu" for _, labels in entry.microbatches)
                assert all(ids.device.type == "npu" for slot in entry.slots for ids in slot.values())
        if engine.suppress_checkpoint_saves:
            assert not trainer.checkpointer.save(trainer.step)
            blocked.append(trainer.step)
        elif trainer.step % 2 == 0:
            assert trainer.checkpointer.save(trainer.step, last_step=trainer.step == 10)

    torch.npu.synchronize()
    assert trainer.step == 10
    assert trainer.checkpointer.loaded == [4]
    assert trainer.checkpointer.saved == [2, 4, 8, 10]
    assert blocked == [4, 5, 6, 7]
    assert transitions[5:10] == [
        (4, Phase.ACTIVE, 2),
        (5, Phase.ACTIVE, 2),
        (6, Phase.DRAIN, 2),
        (7, Phase.DRAIN, 1),
        (8, Phase.NORMAL, 0),
    ]
    assert [i for i, _ in model.trained] == list(range(12)) + list(range(8, 20))
    assert [mode for _, mode in model.trained[12:]] == [RoutingMode.REPLAY] * 8 + [RoutingMode.OFF] * 4
    assert model.replayed == [(i, layer) for i in range(8, 16) for layer in range(2)]
    assert len(model.recorded) == len(model.replayed)
    assert model.counter.item() == trainer.ntokens_seen == 20
    assert trainer.lr_schedulers.last_epoch == 10
    assert schedule._num_rollbacks == 1 and not schedule._queue
    assert schedule.cache.mode is RoutingMode.OFF
    assert not engine.failed and not engine.suppress_checkpoint_saves
    logger.info("Anticipatory routing smoke passed: loss spike -> rollback -> WARMUP -> ACTIVE -> DRAIN -> NORMAL")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        raise SystemExit(_probe())
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Usage: test_anticipatory_routing_smoke.py --worker CHECKPOINT_DIR")
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size() == _WORLD_SIZE
        raise SystemExit(_worker(sys.argv[2]))
    finally:
        dist.destroy_process_group()
