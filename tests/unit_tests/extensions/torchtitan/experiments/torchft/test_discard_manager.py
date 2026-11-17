# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from concurrent.futures import Future
from datetime import timedelta
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torchft.manager import Manager, WorldSizeMode
from torchft.process_group import ProcessGroupWrapper

from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx, TorchFTCommitTimeoutError
from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx


def test_begin_step_leaves_quorum_failure_for_commit_vote():
    events = []
    manager = SimpleNamespace(
        _initial_quorum_barrier_pending=False,
        _quorum_future=None,
        _quorum_id=-1,
        _group_world_size=1,
        _snapshot_requested=Event(),
        _snapshot_ready=Event(),
        start_quorum=lambda: events.append("healing completed"),
        errored=Mock(),
    )

    ManagerEx._begin_step(manager)

    assert events == ["healing completed"]
    manager.errored.assert_not_called()
    assert manager.commit_authorized is False


def test_initial_barrier_runs_only_once_across_local_quorum_reset(monkeypatch):
    events = []
    manager = SimpleNamespace(
        _initial_quorum_barrier_pending=True,
        _quorum_future=None,
        _quorum_id=-1,
        _group_world_size=2,
        _snapshot_requested=Event(),
        _snapshot_ready=Event(),
        start_quorum=lambda: events.append("quorum"),
        errored=lambda: None,
    )
    monkeypatch.setattr("torch.distributed.barrier", lambda: events.append("replica barrier"))
    ManagerEx._begin_step(manager)
    assert events == ["replica barrier", "quorum"]
    assert manager._initial_quorum_barrier_pending is False
    assert manager._healing_snapshot is None
    assert not manager._snapshot_ready.is_set()
    assert not hasattr(manager, "recovery_state")
    manager._quorum_id = 7
    manager._quorum_id = -1  # Only this rank invalidated its HCCL communicator.
    ManagerEx._begin_step(manager)
    assert events == ["replica barrier", "quorum", "quorum"]


def test_failed_configure_retries_quorum_when_no_process_group_exists():
    manager = SimpleNamespace(
        _initial_quorum_barrier_pending=False,
        _quorum_future=None,
        _quorum_id=7,
        _pg=SimpleNamespace(_get_process_group=lambda: None),
        _snapshot_requested=Event(),
        _snapshot_ready=Event(),
        start_quorum=Mock(),
    )

    ManagerEx._begin_step(manager)

    assert manager._quorum_id == -1
    manager.start_quorum.assert_called_once_with()


def test_begin_step_waits_for_pending_healing_before_resetting_snapshot_events():
    future = Future()
    events = []
    requested, ready = Event(), Event()
    requested.set()
    ready.set()

    def wait_quorum():
        events.append(("wait", requested.is_set(), ready.is_set()))
        future.set_result(None)

    manager = SimpleNamespace(
        _initial_quorum_barrier_pending=False,
        _quorum_future=future,
        _quorum_id=-1,
        _snapshot_requested=requested,
        _snapshot_ready=ready,
        wait_quorum=wait_quorum,
        start_quorum=lambda: events.append(("start", requested.is_set(), ready.is_set())),
    )

    ManagerEx._begin_step(manager)

    assert events == [("wait", True, True), ("start", False, False)]


def test_rejected_recovery_votes_failure_for_a_fresh_quorum(monkeypatch):
    manager = object.__new__(ManagerEx)
    future = Future()
    future.set_result(None)
    manager._quorum_future = future
    manager._quorum_timeout = timedelta(seconds=1)
    manager._snapshot_requested = Event()
    manager._snapshot_ready = Event()
    manager._pg = Mock()
    manager._pg.errored.return_value = None
    manager._client = Mock()
    manager._client.should_commit.return_value = False
    manager._checkpoint_transport = Mock()
    manager._logger = Mock()
    manager.commits_logger = Mock()
    manager._recovery_event = object()
    manager._healing = False
    manager._errored = None
    manager._min_replica_size = 2
    manager._participating_replica_world_size = 2
    manager._group_rank = 0
    manager._replica_id = "torchtitan_ft_0"
    manager._quorum_id = 7
    manager._step = 3
    manager._timeout = timedelta(seconds=1)
    manager._commit_failures = 0
    manager._max_retries = None
    manager.commit_authorized = False
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    manager.discard_interrupted_step(RuntimeError("interrupted HCCL work"))

    assert manager._client.should_commit.call_args.args == (0, 3, False)
    assert manager._commit_failures == 1
    assert manager._quorum_id == -1
    assert manager._recovery_event is None


def test_same_membership_reconfiguration_uses_failure_vote_for_a_fresh_prefix(monkeypatch):
    pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    pg._used_store_addresses.add("localhost:1234/torchft/7/0")
    manager = object.__new__(ManagerEx)
    manager._pg = pg
    manager._client = Mock()
    manager._checkpoint_transport = Mock()
    manager._group_rank = 0
    manager._group_world_size = 1
    manager._replica_id = "torchtitan_ft_0"
    manager._step = 1
    manager._init_sync = False
    manager._commit_failures = 0
    manager._quorum_id = -1
    manager._min_replica_size = 2
    manager._replica_world_size_mode = WorldSizeMode.DYNAMIC
    manager._use_async_quorum = False
    manager._healing = False
    manager._recovery_event = None
    manager._timeout = timedelta(seconds=1)
    manager._max_retries = None
    manager._original_fr_dump_temp_file = None
    manager._logger = Mock()
    manager.quorum_logger = Mock()
    manager.commits_logger = Mock()
    future = Future()
    future.set_result(None)
    manager._quorum_future = future
    manager._snapshot_requested = Event()
    manager._snapshot_ready = Event()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    def quorum(**request):
        quorum_id = 8 if request["commit_failures"] else 7
        return SimpleNamespace(
            quorum_id=quorum_id,
            replica_rank=0,
            replica_world_size=2,
            recover_src_manager_address=None,
            store_address="localhost:1234",
            max_step=1,
            max_replica_rank=0,
            max_world_size=2,
            heal=False,
            replica_ids=["torchtitan_ft_0", "torchtitan_ft_1"],
        )

    manager._client._quorum.side_effect = quorum
    manager._client.should_commit.return_value = False
    with patch.object(ProcessGroupWrapper, "configure") as configure:
        manager._async_quorum(False, False, timedelta(seconds=1), -1)
        assert manager.errored() is not None
        assert "fresh store prefix" in str(manager.errored())
        assert not manager.should_commit()
        assert manager._commit_failures == 1

        manager._quorum_id = -1  # A failed configure did not create a process group.
        manager._errored = None
        manager._async_quorum(False, False, timedelta(seconds=1), -1)

    assert manager._client._quorum.call_args.kwargs["commit_failures"] == 1
    assert configure.call_count == 1
    assert configure.call_args.args[0] == "localhost:1234/torchft/8/0"
    assert manager.errored() is None


def test_healing_snapshot_is_exported_on_the_training_thread_only_when_requested(monkeypatch):
    manager = object.__new__(ManagerEx)
    manager._quorum_future = Future()
    manager._quorum_timeout = timedelta(seconds=1)
    manager._snapshot_requested = Event()
    manager._snapshot_ready = Event()
    manager._snapshot_error = None
    manager._healing_snapshot = None
    snapshot = {"user": {}, "torchft": {"step": 4}}
    export_threads = []
    manager._export_snapshot = lambda: export_threads.append(get_ident()) or snapshot
    monkeypatch.setattr(Manager, "wait_quorum", lambda self: self._quorum_future.result(timeout=1))
    results = []

    manager._quorum_future.set_result(None)
    manager.wait_quorum()
    assert export_threads == []
    manager._quorum_future = Future()

    def request_snapshot():
        try:
            results.append(manager._manager_state_dict())
        finally:
            manager._quorum_future.set_result(None)

    worker = Thread(target=request_snapshot)
    worker.start()
    manager.wait_quorum()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert results == [snapshot]
    assert export_threads == [get_ident()]


def test_healing_snapshot_export_error_reaches_the_requester(monkeypatch):
    manager = object.__new__(ManagerEx)
    manager._quorum_future = Future()
    manager._quorum_timeout = timedelta(seconds=1)
    manager._snapshot_requested = Event()
    manager._snapshot_ready = Event()
    manager._snapshot_error = None
    manager._healing_snapshot = None
    manager._export_snapshot = Mock(side_effect=ValueError("export failed"))
    monkeypatch.setattr(Manager, "wait_quorum", lambda self: self._quorum_future.result(timeout=1))
    errors = []

    def request_snapshot():
        try:
            manager._manager_state_dict()
        except RuntimeError as error:
            errors.append(error)
        finally:
            manager._quorum_future.set_result(None)

    worker = Thread(target=request_snapshot)
    worker.start()
    manager.wait_quorum()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0].__cause__, ValueError)
    manager._export_snapshot.assert_called_once()


def test_commit_rpc_timeout_is_fatal_and_closes_healing_publication(monkeypatch):
    manager = object.__new__(ManagerEx)
    manager.report_error = Mock()
    manager._checkpoint_transport = Mock()
    manager.commit_authorized = False
    error = TimeoutError("response deadline expired")
    monkeypatch.setattr(Manager, "should_commit", Mock(side_effect=error))

    with pytest.raises(TorchFTCommitTimeoutError, match="entire replica") as raised:
        manager.should_commit()

    assert raised.value.__cause__ is error
    assert not manager.commit_authorized
    manager.report_error.assert_called_once_with(error)
    manager._checkpoint_transport.disallow_checkpoint.assert_called_once_with()


def test_authorized_update_cannot_be_discarded():
    manager = SimpleNamespace(commit_authorized=True, should_commit=Mock())

    with pytest.raises(RuntimeError, match="authorized optimizer"):
        ManagerEx.discard_interrupted_step(manager, RuntimeError("stop"))

    manager.should_commit.assert_not_called()
