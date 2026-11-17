# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CLI/DeepSeek/checkpointer recovery test; instrumentation is test-only."""

from functools import wraps


def main():
    import torchtitan_npu  # noqa: F401
    from torchtitan.train import main as train_main
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
    from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
    from torchtitan_npu.extensions.experiment.anticipatory_routing.router import AnticipatoryHashRouter
    from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import Phase

    original_init, original_step, original_train = TrainerEx.__init__, TrainerEx.train_step, TrainerEx.train

    @wraps(original_init)
    def initialize(self, config):
        original_init(self, config)
        assert isinstance(config, TrainerEx.Config)
        assert isinstance(self.checkpointer, CheckpointManager)
        assert self.anticipatory_schedule is not None
        routers = [m for part in self.model_parts for m in part.modules() if isinstance(m, AnticipatoryHashRouter)]
        assert routers, "CLI router override did not construct dynamic anticipatory routers"
        assert all(m._anticipatory_cache is self.anticipatory_schedule.cache for m in routers)
        self._test_injected = False
        self._test_phases = []
        self._test_rollbacks = []
        original_loss = self.loss_fn

        def perturbed_loss(*args, **kwargs):
            loss, metrics = original_loss(*args, **kwargs)
            # Exercise the real detector with a changed training objective;
            # do not inject its verdict or replace checkpoint/load methods.
            if self.step == 6 and not self._test_injected:
                loss = loss * 100
                self._test_injected = True
            return loss, metrics

        self.loss_fn = perturbed_loss

    @wraps(original_step)
    def step(self, data_iterator):
        entry_step = self.step
        before = self.anticipatory_schedule.phase
        result = original_step(self, data_iterator)
        after = self.anticipatory_schedule.phase
        self._test_phases.append((before, after))
        if self.step < entry_step:
            self._test_rollbacks.append((entry_step, self.step))
            assert self.anticipatory_schedule._queue, "Rollback did not execute inline WARMUP"
        return result

    @wraps(original_train)
    def train(self):
        original_train(self)
        schedule = self.anticipatory_schedule
        assert self._test_injected
        assert len(self._test_rollbacks) == 1 and self._test_rollbacks[0][0] == 6
        assert 0 < self._test_rollbacks[0][1] < 6
        assert (Phase.NORMAL, Phase.ACTIVE) in self._test_phases
        assert (Phase.ACTIVE, Phase.DRAIN) in self._test_phases
        assert (Phase.DRAIN, Phase.NORMAL) in self._test_phases
        assert self.step == self.config.training.steps
        assert schedule._num_rollbacks == 1 and not schedule._queue
        assert not schedule.engine.failed and not schedule.engine.suppress_checkpoint_saves

    TrainerEx.__init__, TrainerEx.train_step, TrainerEx.train = initialize, step, train
    try:
        train_main()
    finally:
        TrainerEx.__init__, TrainerEx.train_step, TrainerEx.train = original_init, original_step, original_train


if __name__ == "__main__":
    main()
