# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe storage ownership without keeping tensors or storages alive."""

import logging
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from torchtitan_simulator.memory.records import StorageLifetimeRecord


class StorageLifetimeTracker:
    def __init__(self):
        self._active: dict[str, StorageWeakRef] = {}
        self._records: dict[str, StorageLifetimeRecord] = {}
        self._pending: set[str] = set()
        self._wrapper_refs: dict[str, weakref.ReferenceType[torch.UntypedStorage]] = {}
        self._warned_missing_weakref = False

    def register(self, key: str, storage_weak_ref: StorageWeakRef, *, storage: torch.UntypedStorage) -> None:
        if key in self._records:
            return
        self._records[key] = StorageLifetimeRecord(key)
        self._active[key] = storage_weak_ref
        owner = weakref.ref(self)

        def notify(_ref):
            tracker = owner()
            if tracker is not None and key in tracker._active:
                tracker._pending.add(key)

        try:
            self._wrapper_refs[key] = weakref.ref(storage, notify)
        except TypeError:
            # Unsupported wrappers stay pending, preserving the old full-scan
            # semantics for these allocations rather than losing their release.
            self._pending.add(key)
            if not self._warned_missing_weakref:
                logging.getLogger(__name__).warning(
                    "Storage wrapper does not support weak references; checking its native lifetime at every boundary"
                )
                self._warned_missing_weakref = True

    def poll(self):
        from torchtitan_simulator.capture.dispatch_capture import _seq_counter

        if not self._pending:
            return
        # A wrapper notification is only a candidate. C++/autograd owners may
        # outlive it, so retain early notifications until the native ref expires.
        expired = [key for key in tuple(self._pending) if self._active[key].expired()]
        if expired:
            seq = next(_seq_counter)
            for key in expired:
                self._records[key].release_seq = seq
                self._active.pop(key)
                self._pending.discard(key)
                self._wrapper_refs.pop(key, None)

    def close(self):
        self.poll()
        self._active.clear()
        self._wrapper_refs.clear()
        self._pending.clear()

    def records(self) -> list[StorageLifetimeRecord]:
        return list(self._records.values())
