# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Track unfinished HCCL work without letting old timers affect a new step."""

import threading
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch_npu
from torchft.process_group import _WorkAcceleratorTimeout


class HcclWorkTracker:
    def __init__(self, on_timeout):
        self.boundary_lock = threading.RLock()
        self._lock = threading.Lock()
        self._on_timeout = on_timeout
        self._generation = 0
        self._next_token = 0
        self._active: dict[int, tuple[threading.Timer, Any]] = {}

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def track(self, timeout: timedelta):
        self.retire_completed()
        with self._lock:
            token = self._next_token
            self._next_token += 1
            generation = self._generation
            timer = threading.Timer(timeout.total_seconds(), self._expire, args=(token, generation))
            timer.daemon = True
            self._active[token] = (timer, None)
            timer.start()
            return token

    def record_completion(self, token):
        event = torch_npu.npu.Event()
        event.record()
        with self._lock:
            if token in self._active:
                timer, _ = self._active[token]
                self._active[token] = (timer, event)

    def complete(self, token):
        with self._lock:
            entry = self._active.pop(token, None)
        if entry is not None:
            entry[0].cancel()

    def retire_completed(self):
        with self._lock:
            entries = tuple(self._active.items())
        for token, (_, event) in entries:
            if event is not None:
                try:
                    if event.query():
                        self.complete(token)
                except RuntimeError:
                    # A stopped device cannot confirm that the work completed.
                    pass

    def has_active_work(self, token=None):
        self.retire_completed()
        with self._lock:
            return bool(self._active) if token is None else token in self._active

    def clear(self):
        with self.boundary_lock, self._lock:
            timers = [timer for timer, _ in self._active.values()]
            self._active.clear()
            self._generation += 1
        for timer in timers:
            timer.cancel()

    def _expire(self, token, generation):
        self.retire_completed()
        with self._lock:
            if generation != self._generation or token not in self._active:
                return
        self._on_timeout(token, generation)


class TrackedHcclWork(_WorkAcceleratorTimeout):
    _work: Any

    def __init__(self, pg, work, timeout, *, tracker):
        super().__init__(pg, work, timeout)
        self._tracker = tracker
        self._token = self._tracker.track(timeout)

    @classmethod
    @contextmanager
    def _stream_timeout(cls, pg, timeout):
        # The owner timer is generation-scoped; the upstream timer is not.
        yield

    def wait(self, timeout=None):
        completed = super().wait(timeout)
        if completed:
            self._tracker.record_completion(self._token)
        return completed

    def is_completed(self):
        completed = self._work.is_completed()
        if completed:
            # WorkHCCL.is_completed() queries this collective's HCCL end event,
            # so it can retire the timer without synchronizing the compute stream.
            self._tracker.complete(self._token)
        return completed

    def get_future(self):
        future = self._work.get_future()
        tracker, token = self._tracker, self._token

        def record_completion(done):
            done.wait()
            tracker.record_completion(token)

        future.add_done_callback(record_completion)
        return future
