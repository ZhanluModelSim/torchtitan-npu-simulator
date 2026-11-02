# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Gap-filler daemon: rebind every unmanaged thread to this rank's core pool.

Scans /proc/self/task each round (1s warmup, then 5s). A thread is unmanaged
when its mask is not where this extension placed or exempted it: floating on
every core, stuck on the pinned main's core, matching a shape produced by
the official per-type binding geometry, or narrowed since the previous
round. Reserved threads (main/acl/release/watchdog/daemon), threads already
on the pool, and deliberately narrow masks are left untouched. Threads that
need rebinding for _BLACKLIST_THRESHOLD consecutive rounds are given up on
(circuit breaker).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .planner import Plan
    from .topology import TopologyInfo

logger = logging.getLogger(__name__)

_ROUND_INTERVAL = 5.0
_WARMUP_INTERVAL = 1.0
_WARMUP_ROUNDS = 30
_BLACKLIST_THRESHOLD = 3


@dataclass(frozen=True)
class RebindRules:
    """Static classification inputs shared by every per-thread decision."""

    main_tid: int
    reserved_tids: frozenset[int]
    usable: frozenset[int]
    pool: frozenset[int]
    main_mask: frozenset[int] | None
    patterns: frozenset[frozenset[int]]


def is_unmanaged(rules: RebindRules, tid: int, mask: frozenset[int], prev: frozenset[int] | None) -> bool:
    """True when the thread is not where this extension placed or exempted it.

    unmanaged = floating on every core / stuck on main's core / matching a
    known official binding shape / narrowed since the last round. The
    disposition is uniform: rebind to the rank pool.
    """
    if tid == rules.main_tid or tid in rules.reserved_tids or mask == rules.pool:
        return False
    return (
        mask == rules.usable
        or mask == rules.main_mask
        or mask in rules.patterns
        or (prev is not None and len(mask) < len(prev))
    )


class AffinityDaemon:
    """Background thread that periodically fixes thread affinity."""

    def __init__(
        self,
        topology: TopologyInfo,
        plan: Plan,
        *,
        reserved_tids: frozenset[int] = frozenset(),
        victim_masks: frozenset[frozenset[int]] = frozenset(),
    ) -> None:
        main_core = plan.assignments.get("main", -1)
        self._rules = RebindRules(
            main_tid=os.getpid(),
            reserved_tids=frozenset(reserved_tids),
            usable=topology.usable,
            pool=frozenset(plan.other_pool),
            main_mask=frozenset({main_core}) if main_core >= 0 else None,
            patterns=frozenset(victim_masks),
        )
        self._stop_event = threading.Event()
        self._blacklist: dict[int, int] = {}  # tid → consecutive-rescue strike count
        self._prev_masks: dict[int, frozenset[int]] = {}
        self._rescued_prev_round: set[int] = set()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="affinity-daemon", daemon=True)
        self._thread.start()
        logger.info(
            "[affinity] daemon started: pool=%d cores, victim_sigs=%d",
            len(self._rules.pool),
            len(self._rules.patterns),
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("[affinity] daemon stopped")

    def _run(self) -> None:
        # get_native_id() is the kernel TID that /proc/self/task exposes;
        # get_ident() is a Python-internal handle that never matches it.
        self._rules = replace(self._rules, reserved_tids=self._rules.reserved_tids | {threading.get_native_id()})
        rounds = 0
        while not self._stop_event.wait(_WARMUP_INTERVAL if rounds < _WARMUP_ROUNDS else _ROUND_INTERVAL):
            try:
                self._one_round()
            except Exception as exc:  # never crash training
                logger.warning("[affinity] daemon round failed: %s", exc)
            rounds += 1

    def _one_round(self) -> None:
        task_root = Path("/proc/self/task")

        try:
            tids = [int(entry.name) for entry in task_root.iterdir() if entry.name.isdigit()]
        except OSError:
            return

        current: dict[int, frozenset[int]] = {}
        unmanaged = 0
        bound_this_round: set[int] = set()

        for tid in tids:
            try:
                mask = frozenset(os.sched_getaffinity(tid))
            except OSError:
                continue
            current[tid] = mask

            if not is_unmanaged(self._rules, tid, mask, self._prev_masks.get(tid)):
                continue

            # circuit breaker: give up only after _BLACKLIST_THRESHOLD
            # consecutive rounds of rescuing the same thread
            strikes = self._blacklist.get(tid, 0)
            if strikes >= _BLACKLIST_THRESHOLD:
                continue

            try:
                os.sched_setaffinity(tid, self._rules.pool)
            except OSError as exc:
                logger.warning("[affinity] widen tid=%d failed: %s", tid, exc)
                continue

            unmanaged += 1
            bound_this_round.add(tid)
            logger.debug("[affinity] rebind tid=%d mask=%s", tid, sorted(mask))

            if tid in self._rescued_prev_round:
                strikes += 1
                self._blacklist[tid] = strikes
                if strikes >= _BLACKLIST_THRESHOLD:
                    logger.warning(
                        "[affinity] tid=%d needed rescue %d consecutive rounds; blacklisted",
                        tid,
                        strikes,
                    )
            else:
                self._blacklist.pop(tid, None)

        # prune per-round state: only keep threads that still exist
        self._prev_masks = current
        self._blacklist = {t: s for t, s in self._blacklist.items() if t in current}
        self._rescued_prev_round = bound_this_round

        if unmanaged:
            hard = sum(1 for s in self._blacklist.values() if s >= _BLACKLIST_THRESHOLD)
            logger.info("[affinity] round: unmanaged=%d blacklisted=%d", unmanaged, hard)
