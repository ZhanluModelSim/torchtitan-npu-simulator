# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe storage ownership without keeping tensors or storages alive."""

from torch.multiprocessing.reductions import StorageWeakRef

from torchtitan_simulator.memory.records import StorageLifetimeRecord


class StorageLifetimeTracker:
    def __init__(self):
        self._active: dict[str, StorageWeakRef] = {}
        self._records: dict[str, StorageLifetimeRecord] = {}

    def register(self, key: str, storage_weak_ref: StorageWeakRef) -> None:
        if key not in self._records:
            self._records[key] = StorageLifetimeRecord(key)
            self._active[key] = storage_weak_ref

    def poll(self):
        from torchtitan_simulator.capture.dispatch_capture import _seq_counter

        expired = [key for key, ref in self._active.items() if ref.expired()]
        if expired:
            seq = next(_seq_counter)
            for key in expired:
                self._records[key].release_seq = seq
                self._active.pop(key)

    def close(self):
        self.poll()
        self._active.clear()

    def records(self) -> list[StorageLifetimeRecord]:
        return list(self._records.values())
