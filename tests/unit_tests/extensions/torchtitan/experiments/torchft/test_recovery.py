# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""CPU-observable work ownership and the NPU recovery handshake."""

import threading
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from torch.distributed import distributed_c10d
from torchft.work import _DummyWork

from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import _restore_tensor_leaves, _snapshot_to_cpu
from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery import step_recovery
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.hccl import HcclRecovery, HcclRecoveryError


def test_cpu_snapshot_survives_optimizer_counter_and_nested_metadata_updates():
    state = {"step": torch.tensor(2.0), "groups": [{"lr": 0.1}]}
    snapshot = _snapshot_to_cpu(state)
    state["step"].add_(1)
    state["groups"][0]["lr"] = 0.01
    restored = _restore_tensor_leaves(snapshot, state)
    restored["step"].add_(1)

    assert snapshot["step"].item() == 2
    assert snapshot["groups"] == [{"lr": 0.1}]
    assert restored["step"].item() == 3
    assert restored["groups"] == [{"lr": 0.1}]


def test_recovery_rejects_mismatched_tensor_shape_before_loading():
    with pytest.raises(ValueError, match="shape or dtype"):
        _restore_tensor_leaves({"weight": torch.ones(2)}, {"weight": torch.zeros(3)})


def test_reset_failed_iteration_clears_cached_fsdp_gradients(monkeypatch):
    class FakeFSDPModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sharded = torch.nn.Parameter(torch.ones(1))
            self.sharded.grad = torch.ones(1)
            unsharded = torch.nn.Parameter(torch.ones(1))
            unsharded.grad = torch.ones(1)
            self.fsdp_param = SimpleNamespace(
                _unsharded_param=unsharded,
                unsharded_accumulated_grad=torch.ones(1),
            )
            self.state = SimpleNamespace(_fsdp_param_groups=[SimpleNamespace(fsdp_params=[self.fsdp_param])])
            self.reset_count = 0

        def _get_fsdp_state(self):
            return self.state

        def reset_iter_state(self):
            self.reset_count += 1

    monkeypatch.setattr(step_recovery, "FSDPModule", FakeFSDPModule)
    model = FakeFSDPModule()
    model.nested = FakeFSDPModule()

    step_recovery.reset_failed_iteration([model])

    for module in (model, model.nested):
        assert module.sharded.grad is None
        assert module.fsdp_param._unsharded_param.grad is None
        assert module.fsdp_param.unsharded_accumulated_grad is None
    assert model.reset_count == 1


def test_rejected_step_clears_moe_token_counts(monkeypatch):
    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("tokens_per_expert_E", torch.tensor([3.0, 5.0]))

    monkeypatch.setattr(step_recovery, "MoE", FakeMoE)
    model = torch.nn.Module()
    model.moe = FakeMoE()

    step_recovery.reset_rejected_moe_state([model])

    assert torch.count_nonzero(model.moe.tokens_per_expert_E).item() == 0


@pytest.fixture
def deferred_timers(monkeypatch):
    timers = []

    class DeferredTimer:
        """Control the clock while retaining the production timeout callback."""

        def __init__(self, interval, callback, args):
            self.interval = interval
            self.callback = lambda: callback(*args)
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(threading, "Timer", DeferredTimer)
    return timers


def test_new_quorum_gets_warmup_timeout_until_first_update(monkeypatch, deferred_timers):
    monkeypatch.setattr(torch_npu.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    pg = ProcessGroupHCCLEx(
        timeout=timedelta(milliseconds=35),
        recovery_timeout=timedelta(seconds=1),
        warmup_timeout=timedelta(milliseconds=90),
    )
    assert pg.recovery.watchdog_timeout == timedelta(milliseconds=90, seconds=31)

    pg.recovery.group_created()
    pg._wrap_work(Mock(), object())
    assert deferred_timers[-1].interval == 0.09

    pg.recovery.apply_optimizer(lambda: None)
    pg.recovery.begin_attempt()
    pg._wrap_work(Mock(), object())
    assert deferred_timers[-1].interval == 0.035

    pg.recovery.group_created()
    pg._wrap_work(Mock(), object())
    assert deferred_timers[-1].interval == 0.09
    pg.recovery.work_tracker.clear()


@pytest.mark.parametrize("completed", [False, True], ids=["unfinished-work", "completed-work"])
def test_only_unfinished_device_work_requests_recovery(monkeypatch, deferred_timers, completed):
    requests = []
    monkeypatch.setattr(
        HcclRecovery, "_schedule_recovery", lambda self, token, generation: requests.append((token, generation))
    )
    monkeypatch.setattr(torch.npu, "Event", lambda: SimpleNamespace(record=lambda: None, query=lambda: completed))
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    work = pg._wrap_work(Mock(wait=Mock(return_value=True)), object())

    work.wait()
    deferred_timers[0].callback()

    assert requests == ([] if completed else [(0, 0)])
    assert pg.recovery.work_tracker.has_active_work() is not completed
    pg.recovery.work_tracker.clear()


def test_wait_for_completion_retires_only_the_selected_hccl_work(monkeypatch, deferred_timers):
    monkeypatch.setattr("torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.hccl.time.sleep", lambda _: None)
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    inner_work = Mock(is_completed=Mock(side_effect=[False, True]))
    tracked_work = pg._wrap_work(inner_work, object())
    other_work = Mock(is_completed=Mock(return_value=False))
    pg._wrap_work(other_work, object())
    outer_work = SimpleNamespace(_work=tracked_work, wait=Mock(return_value=True))

    pg.wait_for_completion(outer_work)

    assert inner_work.is_completed.call_count == 2
    other_work.is_completed.assert_not_called()
    outer_work.wait.assert_called_once_with()
    assert pg.recovery.work_tracker.has_active_work()
    pg.recovery.work_tracker.clear()


def test_wait_for_completion_accepts_dummy_work_after_quorum_error(deferred_timers):
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))

    pg.wait_for_completion(_DummyWork(torch.ones(1)))

    assert not pg.recovery.work_tracker.has_active_work()


def test_timeout_from_a_committed_step_cannot_stop_the_next_step(monkeypatch, deferred_timers):
    stop = Mock()
    monkeypatch.setattr(torch_npu.npu, "stop_device", stop)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    pg._pg = SimpleNamespace()
    old_generation = pg.recovery.work_tracker.generation
    pg.recovery.work_tracker.track(timedelta(seconds=1))
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1, foreach=False)
    parameter.grad = torch.ones(1)

    pg.recovery.apply_optimizer(optimizer.step)
    assert pg.recovery.optimizer_update_completed
    pg.recovery.begin_attempt()
    assert not pg.recovery.optimizer_update_completed
    pg.recovery.work_tracker.track(timedelta(seconds=1))
    # A timeout worker may already have been scheduled before the commit.
    pg.recovery._recover_device(None, old_generation)

    torch.testing.assert_close(parameter, torch.tensor([0.9]))
    stop.assert_not_called()
    assert deferred_timers[0].cancelled
    assert pg.recovery.work_tracker.has_active_work()
    pg.recovery.work_tracker.clear()


@pytest.mark.parametrize("restart_fails", [False, True], ids=["device-ready", "restart-fails"])
def test_recovery_waits_for_the_training_thread_before_restarting_device(monkeypatch, deferred_timers, restart_fails):
    calls = []
    stopped = threading.Event()
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=2))
    pg.recovery._device_index = 0
    backend = SimpleNamespace(
        abort_hccl_comm=lambda reason: calls.append(("abort", reason)),
        clear_workmeta_list=lambda: calls.append("clear"),
        abort=Mock(),
        shutdown=Mock(),
    )
    pg._pg = SimpleNamespace(_get_backend=lambda device: backend)
    pg.recovery.work_tracker.track(timedelta(seconds=1))
    healthy_pg = Mock()
    healthy_pg._device_types = [torch.device("npu")]
    cpu_pg = Mock()
    cpu_pg._device_types = [torch.device("cpu")]

    def resume(device):
        assert not pg.recovery._device_ready.is_set()
        calls.append(("resume-healthy-hccl", device))

    healthy_pg._get_backend.return_value.resume_hccl_comm.side_effect = resume
    monkeypatch.setattr(distributed_c10d, "_pg_map", {healthy_pg: None, cpu_pg: None})

    def stop(device):
        calls.append(("stop", device))
        stopped.set()

    def restart(device, **kwargs):
        assert pg.recovery._stop_observed.is_set()
        calls.append(("restart", device))
        assert kwargs == {"rebuild_all_resources": True, "disable_tensor_unsafe_check": True}
        if restart_fails:
            raise RuntimeError("restart failed")

    monkeypatch.setattr(torch_npu.npu, "stop_device", stop)
    monkeypatch.setattr(torch_npu.npu, "restart_device", restart)
    monkeypatch.setattr(
        distributed_c10d, "_cleanup_process_group_global_state", lambda group: calls.append("unregister")
    )

    pg.recovery._schedule_recovery(None, pg.recovery.work_tracker.generation)
    assert stopped.wait(2)
    with pytest.raises(HcclRecoveryError):
        pg.abort(errored=False)
    backend.abort.assert_not_called()
    backend.shutdown.assert_not_called()
    assert not pg.recovery.recover(RuntimeError("unrelated model error"))
    assert not pg.recovery._stop_observed.is_set()
    if restart_fails:
        with pytest.raises(RuntimeError, match="device recovery failed") as raised:
            pg.recovery.recover(RuntimeError("NPU FORCE STOP"))
        assert str(raised.value.__cause__) == "restart failed"
    else:
        assert pg.recovery.recover(RuntimeError("NPU FORCE STOP"))
        assert pg.recovery.num_device_recoveries == 1
        assert pg._pg is None
        assert not pg.recovery.work_tracker.has_active_work()
    expected = [("stop", 0), ("abort", "reinit"), "clear", "unregister", ("restart", 0)]
    if not restart_fails:
        expected.append(("resume-healthy-hccl", 0))
    assert calls == expected
    cpu_pg._get_backend.assert_not_called()


def test_interrupted_attempt_cannot_enter_optimizer_update(deferred_timers):
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    pg.recovery._recovery_requested.set()
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.ones(1)

    with pytest.raises(HcclRecoveryError):
        pg.recovery.apply_optimizer(optimizer.step)

    torch.testing.assert_close(parameter, torch.ones(1))


def test_late_timeout_waits_for_authorized_optimizer_to_complete(monkeypatch, deferred_timers):
    entered = threading.Event()
    finish = threading.Event()
    recovery_finished = threading.Event()
    stop = Mock()
    monkeypatch.setattr(torch_npu.npu, "stop_device", stop)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    pg._pg = SimpleNamespace()
    generation = pg.recovery.work_tracker.generation
    pg.recovery.work_tracker.track(timedelta(seconds=1))

    def update():
        entered.set()
        assert finish.wait(2)

    def late_timeout():
        pg.recovery._recover_device(None, generation)
        recovery_finished.set()

    update_thread = threading.Thread(target=pg.recovery.apply_optimizer, args=(update,))
    timer_thread = threading.Thread(target=late_timeout)
    update_thread.start()
    try:
        assert entered.wait(2)
        timer_thread.start()
        assert not recovery_finished.wait(0.05)
        stop.assert_not_called()
    finally:
        finish.set()
        update_thread.join(2)
        if timer_thread.ident is not None:
            timer_thread.join(2)
    assert not update_thread.is_alive() and not timer_thread.is_alive()
    assert recovery_finished.is_set()
    assert pg.recovery.optimizer_update_completed
    stop.assert_not_called()


def test_completed_future_does_not_request_device_recovery(monkeypatch, deferred_timers):
    requests = []
    monkeypatch.setattr(
        HcclRecovery, "_schedule_recovery", lambda self, token, generation: requests.append((token, generation))
    )
    monkeypatch.setattr(torch.npu, "Event", lambda: SimpleNamespace(record=lambda: None, query=lambda: True))
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    future = torch.futures.Future()
    work = pg._wrap_work(Mock(get_future=lambda: future), object())

    work.get_future()
    future.set_result([torch.ones(1)])
    deferred_timers[0].callback()

    assert requests == []
    assert not pg.recovery.work_tracker.has_active_work()


def test_completed_timed_out_work_cannot_stop_new_work_in_the_same_step(monkeypatch, deferred_timers):
    requests = []
    stop = Mock()
    monkeypatch.setattr(torch_npu.npu, "stop_device", stop)
    monkeypatch.setattr(
        HcclRecovery, "_schedule_recovery", lambda self, token, generation: requests.append((token, generation))
    )
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    pg._pg = SimpleNamespace()
    old_token = pg.recovery.work_tracker.track(timedelta(seconds=1))

    deferred_timers[0].callback()
    pg.recovery.work_tracker.complete(old_token)
    pg.recovery.work_tracker.track(timedelta(seconds=1))
    pg.recovery._recover_device(*requests[0])

    stop.assert_not_called()
    assert pg.recovery.work_tracker.has_active_work()
    pg.recovery.work_tracker.clear()
