# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torchtitan_simulator.capture import storage_lifetimes
from torchtitan_simulator.capture.storage_lifetimes import StorageLifetimeTracker


def _register(tracker, tensor, key="allocation"):
    storage = tensor.untyped_storage()
    native = StorageWeakRef(storage)
    tracker.register(key, native, storage=storage)
    return native


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_view_keeps_storage_live_until_last_alias_is_gone(device):
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device=device)
    view = tensor.view(2, 2)
    native = _register(tracker, tensor)
    _register(tracker, view)
    del tensor
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    del view
    tracker.poll()
    assert native.expired()
    assert len(tracker.records()) == 1
    assert tracker.records()[0].release_seq >= 0


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_cpp_autograd_saved_value_keeps_storage_live(device):
    tracker = StorageLifetimeTracker()
    value = torch.ones(4, device=device, requires_grad=True).sin()
    loss = value.square().sum()
    native = _register(tracker, value)
    del value
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    loss.backward()
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0


def test_cpp_dlpack_owner_keeps_storage_live_without_python_tensor():
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4)
    native = _register(tracker, tensor)
    capsule = torch.utils.dlpack.to_dlpack(tensor)
    del tensor
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    del capsule
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0


def test_early_notification_remains_pending_until_native_release():
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device="meta")
    native = _register(tracker, tensor)
    notification = tracker._wrapper_refs["allocation"]
    notification.__callback__(notification)
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    del tensor
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0


def test_unweakrefable_storage_falls_back_to_native_checks(monkeypatch, caplog):
    original_ref = weakref.ref

    def storage_without_weakrefs(value, callback=None):
        if isinstance(value, torch.UntypedStorage):
            raise TypeError("Storage runtime without Python weak references")
        return original_ref(value, callback)

    monkeypatch.setattr(storage_lifetimes, "weakref", SimpleNamespace(ref=storage_without_weakrefs))
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device="meta")
    native = _register(tracker, tensor)
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    del tensor
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0
    assert "checking its native lifetime at every boundary" in caplog.text


def test_gc_cycle_release_is_observed_at_next_boundary():
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device="meta")
    storage = tensor.untyped_storage()
    storage.cycle = storage
    native = _register(tracker, tensor)
    del tensor, storage
    tracker.poll()
    assert not native.expired()
    assert tracker.records()[0].release_seq == -1
    gc.collect()
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0


def test_callbacks_do_not_retain_tracker_or_storage_after_close():
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device="meta")
    native = _register(tracker, tensor)
    tracker_ref = weakref.ref(tracker)
    tracker.close()
    assert tracker.records()[0].release_seq == -1
    del tracker
    assert tracker_ref() is None
    assert not native.expired()
    del tensor
    assert native.expired()


def test_cross_thread_release_is_observed_at_next_boundary():
    tracker = StorageLifetimeTracker()
    tensor = torch.empty(4, device="meta")
    native = _register(tracker, tensor)
    owners = [tensor]
    del tensor
    thread = threading.Thread(target=owners.clear)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    tracker.poll()
    assert native.expired()
    assert tracker.records()[0].release_seq >= 0


def test_storage_generations_keep_distinct_release_records():
    tracker = StorageLifetimeTracker()
    for generation in range(100):
        tensor = torch.empty(4, device="meta")
        native = _register(tracker, tensor, key=f"generation_{generation}")
        del tensor
        tracker.poll()
        assert native.expired()
        del native
    records = tracker.records()
    assert len(records) == 100
    assert all(record.release_seq >= 0 for record in records)
    assert [record.release_seq for record in records] == sorted({record.release_seq for record in records})
