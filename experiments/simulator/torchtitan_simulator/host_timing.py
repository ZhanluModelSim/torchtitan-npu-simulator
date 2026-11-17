# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-invocation host wall-clock diagnostics, independent of device estimates."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator


class HostStageTiming(TypedDict):
    wall_seconds: float
    calls: int


class HostTimingSummary(TypedDict):
    clock: str
    scope: str
    status: str
    stages: dict[str, HostStageTiming]


@dataclass
class HostTimings:
    """Accumulate inclusive scopes locally; nested stages must not be summed."""

    stages: dict[str, HostStageTiming] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name: str, wall_seconds: float, *, calls: int = 1) -> None:
        previous = self.stages.get(name, {"wall_seconds": 0.0, "calls": 0})
        self.stages[name] = {
            "wall_seconds": previous["wall_seconds"] + wall_seconds,
            "calls": previous["calls"] + calls,
        }

    def summary(self, status: Literal["completed", "failed"]) -> HostTimingSummary:
        stages: dict[str, HostStageTiming] = {
            name: {"wall_seconds": value["wall_seconds"], "calls": value["calls"]}
            for name, value in self.stages.items()
        }
        return {
            "clock": "time.perf_counter",
            "scope": "host wall time; inclusive nested stages; not device execution time",
            "status": status,
            "stages": stages,
        }
