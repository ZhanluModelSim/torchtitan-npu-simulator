# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from unittest.mock import Mock

import pytest

from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import TorchFTCommitTimeoutError
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery import step_recovery


def test_device_recovery_discards_attempt_without_loading_training_state(monkeypatch):
    events = []
    hccl = Mock(enabled=True, optimizer_update_completed=False)
    hccl.recover.side_effect = lambda _: events.append("device restarted") or True
    manager = Mock()
    manager.discard_interrupted_step.side_effect = lambda _: events.append("attempt discarded")
    state = Mock()
    state.state_dict.return_value = {"step": 1}
    schedulers = Mock()
    monkeypatch.setattr(step_recovery, "reset_failed_iteration", lambda _: events.append("iteration reset"))
    monkeypatch.setattr(step_recovery.torch_npu.npu, "empty_cache", lambda: events.append("cache released"))
    recovery = step_recovery.StepRecovery(
        hccl,
        manager=manager,
        model_parts=[],
        train_state=state,
        lr_schedulers=schedulers,
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
    schedulers.step.assert_called_once_with()
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
        lr_schedulers=schedulers,
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
    hccl = Mock(enabled=True, optimizer_update_completed=commit_authorized)
    manager = Mock(commit_authorized=commit_authorized)
    auxiliary_reset = Mock()
    moe_reset = Mock()
    monkeypatch.setattr(step_recovery, "reset_rejected_moe_state", moe_reset)
    recovery = step_recovery.StepRecovery(
        hccl,
        manager=manager,
        model_parts=[],
        train_state=Mock(),
        lr_schedulers=Mock(),
        max_consecutive_recoveries=3,
        reset_auxiliary_state=auxiliary_reset,
    )

    result = recovery.run_step(lambda _: "done", iter([]))

    assert result == "done"
    assert moe_reset.call_count == (0 if commit_authorized else 1)
    assert auxiliary_reset.call_count == (0 if commit_authorized else 1)
