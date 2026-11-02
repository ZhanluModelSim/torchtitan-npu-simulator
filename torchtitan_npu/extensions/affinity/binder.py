# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Unified binder: rebind all four reserved thread types to planned cores.

For main: first call the official torch_npu API (sets kStartMainThreadBind),
then syscall to the planned core. For acl/release/watchdog: find tid by comm
name, then syscall to the planned core.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .planner import Plan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BindResult:
    """Outcome of a bind_all call."""

    success: bool
    bound: dict[str, int]  # thread_type -> core
    failures: list[str]  # thread types that failed
    tids: dict[str, int] = field(default_factory=dict)  # thread_type -> tid


def _current_mask() -> frozenset[int]:
    return frozenset(os.sched_getaffinity(0))


def _find_tid_by_comm(comm: str) -> int | None:
    """Find a thread ID by its comm name in /proc/self/task."""
    task_root = Path("/proc/self/task")
    try:
        for entry in task_root.iterdir():
            comm_file = entry / "comm"
            try:
                if comm_file.read_text().strip() == comm:
                    return int(entry.name)
            except (OSError, ValueError):
                continue
    except OSError:
        return None
    return None


def _bind_tid(tid: int, cores: frozenset[int], label: str) -> bool:
    """Bind a thread to the given cores, verify, and log."""
    try:
        os.sched_setaffinity(tid, cores)
    except OSError as exc:
        logger.warning("[affinity] bind failed for %s(tid=%d): %s", label, tid, exc)
        return False
    try:
        actual = frozenset(os.sched_getaffinity(tid))
        if not actual <= cores:
            logger.warning(
                "[affinity] bind for %s(tid=%d) verified mask=%s != expected=%s",
                label,
                tid,
                sorted(actual),
                sorted(cores),
            )
            return False
    except OSError:
        pass
    logger.info("[affinity] bound %s(tid=%d) → core %s", label, tid, sorted(cores))
    return True


def _official_reset() -> bool:
    """Call torch_npu's _npu_reset_thread_affinity (official API for main)."""
    try:
        import torch_npu

        api = getattr(torch_npu._C, "_npu_reset_thread_affinity", None)
        if api is None:
            logger.info("[affinity] official API unavailable; syscall only")
            return False
        api()
        logger.info("[affinity] official API: kStartMainThreadBind=true")
        return True
    except Exception as exc:  # never crash training
        logger.warning("[affinity] official API call failed: %s", exc)
        return False


_COMM_MAP = {
    "acl": "acl_thread",
    "release": "release_thread",
    "watchdog": "hccl_watchdog_t",
}


def bind_main(info_cores: frozenset[int]) -> BindResult:
    """Bind only the main thread (mode 1 assist)."""
    _official_reset()
    core = next(iter(info_cores)) if len(info_cores) == 1 else -1
    mask = info_cores if core < 0 else frozenset({core})
    ok = _bind_tid(0, mask, "main")
    return BindResult(
        success=ok,
        bound={"main": core} if ok else {},
        failures=[] if ok else ["main"],
        tids={"main": os.getpid()},
    )


def bind_all(p: Plan) -> BindResult:
    """Rebind all four reserved thread types to planned cores (mode 2)."""
    bound: dict[str, int] = {}
    tids: dict[str, int] = {}
    failures: list[str] = []

    # main: official API first (arms torch_npu flag), then reposition
    _official_reset()
    main_core = p.assignments.get("main", -1)
    if main_core >= 0:
        if _bind_tid(0, frozenset({main_core}), "main"):
            bound["main"] = main_core
            tids["main"] = os.getpid()
        else:
            failures.append("main")

    # acl / release / watchdog: find tid by comm, syscall to planned core
    for thread_type in ("acl", "release", "watchdog"):
        core = p.assignments.get(thread_type, -1)
        if core < 0:
            continue
        comm = _COMM_MAP[thread_type]
        tid = _find_tid_by_comm(comm)
        if tid is None:
            logger.info("[affinity] %s (comm=%s) not found; skipping", thread_type, comm)
            failures.append(thread_type)
            continue
        if _bind_tid(tid, frozenset({core}), thread_type):
            bound[thread_type] = core
            tids[thread_type] = tid
        else:
            failures.append(thread_type)

    ok = len(failures) == 0
    logger.info(
        "[affinity] verified: %s",
        " ".join(f"{k}=[{v}]" for k, v in sorted(bound.items())),
    )
    return BindResult(success=ok, bound=bound, failures=failures, tids=tids)
