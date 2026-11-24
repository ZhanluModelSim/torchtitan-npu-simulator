# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import TorchFTCommitTimeoutError
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery import step_recovery
from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import (
    FaultTolerantTrainerEx,
    FaultTolerantTrainingEngineEx,
)


def test_device_recovery_discards_attempt_without_loading_training_state(monkeypatch):
    events = []
    hccl = Mock(enabled=True, optimizer_update_completed=False)
    hccl.recover.side_effect = lambda _: events.append("device restarted") or True
    manager = Mock()
    manager.discard_interrupted_step.side_effect = lambda _: events.append("attempt discarded")
    state = object.__new__(FaultTolerantTrainingEngineEx)
    state.num_completed_steps = 14
    state.ntokens_seen = 0
    state.optim = Mock()
    state.load_state_dict = Mock()
    monkeypatch.setattr(step_recovery, "reset_failed_iteration", lambda _: events.append("iteration reset"))
    monkeypatch.setattr(step_recovery.torch_npu.npu, "empty_cache", lambda: events.append("cache released"))
    recovery = step_recovery.StepRecovery(
        hccl,
        manager=manager,
        model_parts=[],
        train_state=state,
        finish_discarded_step=state.finish_discarded_step,
        max_consecutive_recoveries=3,
        reset_auxiliary_state=lambda: events.append("auxiliary reset"),
    )
    error = RuntimeError("interrupted collective")
    step = Mock(side_effect=error)
    iterator = iter([])

    recovery.run_step(step, iterator)

    assert events == ["device restarted", "iteration reset", "auxiliary reset", "cache released", "attempt discarded"]
    state.load_state_dict.assert_not_called()
    manager.restore_committed_state.assert_not_called()
    manager.discard_interrupted_step.assert_called_once_with(error)
    assert state.num_completed_steps == 15
    state.optim.lr_schedulers.step.assert_called_once_with()
    state.optim.step.assert_not_called()
    state.optim.optimizers.step.assert_not_called()
    state.optim.ema.step.assert_not_called()
    step.assert_called_once_with(iterator)


def test_commit_timeout_exits_instead_of_discarding_and_continuing():
    hccl = Mock(enabled=True, optimizer_update_completed=False)
    hccl.recover.return_value = False
    manager, state, schedulers = Mock(), Mock(), Mock()
    recovery = step_recovery.StepRecovery(
        hccl,
        manager=manager,
        model_parts=[],
        train_state=state,
        finish_discarded_step=schedulers.step,
        max_consecutive_recoveries=3,
        reset_auxiliary_state=Mock(),
    )
    error = TorchFTCommitTimeoutError("restart the entire replica")
    with pytest.raises(TorchFTCommitTimeoutError) as raised:
        recovery.run_step(Mock(side_effect=error), iter([]))
    assert raised.value is error
    manager.discard_interrupted_step.assert_not_called()
    schedulers.step.assert_not_called()


@pytest.mark.parametrize("commit_authorized", [False, True])
def test_completed_rejected_step_clears_transient_auxiliary_state(monkeypatch, commit_authorized):
    from torchtitan.models.common.aux_loss import AuxLoss

    # This engine has no model parts. Ignore losses registered by earlier model builds.
    monkeypatch.setattr(AuxLoss, "_group_counts", {})
    engine = object.__new__(FaultTolerantTrainingEngineEx)
    engine.num_completed_steps = 14
    engine.optim = Mock()
    engine.optim.lr_schedulers.schedulers = [Mock(get_last_lr=Mock(return_value=[0.01]))]
    engine.parallelism_context = SimpleNamespace(dp_enabled=False, dp_cp_enabled=False)
    engine.device = torch.device("cpu")
    engine.ntokens_seen = 1
    engine.loss = torch.tensor(1.0)
    engine.forward_backward = Mock(return_value=SimpleNamespace(loss=engine.loss))
    hccl = Mock(optimizer_update_completed=commit_authorized)
    hccl.recover.return_value = True
    manager = Mock(commit_authorized=commit_authorized)
    auxiliary_reset, moe_reset = Mock(), Mock()
    monkeypatch.setattr(step_recovery, "reset_rejected_moe_state", moe_reset)
    finish_discarded_step = Mock(wraps=engine.finish_discarded_step)
    recovery = step_recovery.StepRecovery(
        hccl,
        manager=manager,
        model_parts=[],
        train_state=engine,
        finish_discarded_step=finish_discarded_step,
        max_consecutive_recoveries=3,
        reset_auxiliary_state=auxiliary_reset,
    )
    trainer = SimpleNamespace(
        engine=engine,
        recovery=recovery,
        metrics_processor=Mock(should_log=Mock(return_value=True)),
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
    )

    def optimizer_attempt(*args, **kwargs):
        engine.optim.lr_schedulers.step()
        return torch.tensor(1.0)

    engine.optim.step.side_effect = optimizer_attempt
    recovery.run_step(
        lambda iterator: FaultTolerantTrainerEx._train_step(trainer, iterator),
        iter([SimpleNamespace(loss_token_counts=torch.tensor(1), routing_token_counts=torch.tensor(1))]),
    )
    assert engine.num_completed_steps == 15
    engine.optim.lr_schedulers.step.assert_called_once_with()
    assert moe_reset.call_count == (0 if commit_authorized else 1)
    assert auxiliary_reset.call_count == (0 if commit_authorized else 1)
    finish_discarded_step.assert_not_called()

    if not commit_authorized:
        # A late error after a rejected, finalized attempt must not consume it twice.
        auxiliary_reset.reset_mock()
        moe_reset.reset_mock()
        engine.optim.lr_schedulers.step.reset_mock()
        trainer.metrics_processor.log.reset_mock()
        error = RuntimeError("FORCE STOP after rejected optimizer step")
        trainer.metrics_processor.log.side_effect = error
        with pytest.raises(RuntimeError, match="finalized step requires restarting") as raised:
            recovery.run_step(
                lambda iterator: FaultTolerantTrainerEx._train_step(trainer, iterator),
                iter([SimpleNamespace(loss_token_counts=torch.tensor(1), routing_token_counts=torch.tensor(1))]),
            )
        assert raised.value.__cause__ is error
        assert engine.num_completed_steps == 16
        engine.optim.lr_schedulers.step.assert_called_once_with()
        trainer.metrics_processor.log.assert_called_once()
        manager.discard_interrupted_step.assert_not_called()
        finish_discarded_step.assert_not_called()
        auxiliary_reset.assert_not_called()
        moe_reset.assert_not_called()
