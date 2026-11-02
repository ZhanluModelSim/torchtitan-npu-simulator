# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Deterministic core-allocation planner for the affinity extension.

Pure function: takes a TopologyInfo and produces a Plan. No side effects.

Sort keys: the cross-rank split orders cores by (freq desc, core_id asc) —
identical on every rank; NPU locality ((local desc, freq desc, core_id asc))
is applied only afterwards, when reordering this rank's own slice.

Rank isolation: with world_size > 1, the rank-agnostic ranked list is split
across node-local ranks by stride (rank r takes ranked[r::world_size]).
Because the split order must not depend on per-rank inputs (npu_id), rank
domains are disjoint by construction; folding locality into the pre-split
order breaks the partition when per-rank local sets differ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import TopologyInfo

logger = logging.getLogger(__name__)

RESERVED_THREAD_TYPES = ("main", "acl", "release", "watchdog")


@dataclass(frozen=True)
class Plan:
    """Immutable allocation plan."""

    assignments: dict[str, int]
    reserved: frozenset[int]
    other_pool: frozenset[int]
    numa_aware: bool


def plan(topology: TopologyInfo, *, npu_id: int | None = None, rank: int = 0, world_size: int = 1) -> Plan:
    cores = sorted(topology.usable)
    if not cores:
        raise RuntimeError("[affinity] no usable cores")

    if world_size < 1:
        world_size = 1
    if not 0 <= rank < world_size:
        logger.warning("[affinity] rank %d out of range [0, %d); using 0", rank, world_size)
        rank = 0

    local = None
    numa_aware = False
    if npu_id is not None:
        local = topology.local_cores(npu_id)
        if local:
            numa_aware = True

    # phase 1 (rank-agnostic): every rank must produce the same ranked list,
    # or the strided split below does not partition the core set; NPU
    # locality must NOT be part of this key.
    ranked = sorted(cores, key=lambda core: (-topology.freq.get(core, 0), core))
    domain = ranked[rank::world_size] if world_size > 1 else ranked

    # phase 2 (rank-local): reorder only this rank's own slice so reserved
    # threads land on NPU-local cores first; disjointness is unaffected.
    if numa_aware and local is not None:
        domain = sorted(domain, key=lambda core: (0 if core in local else 1, -topology.freq.get(core, 0), core))

    if len(domain) < len(RESERVED_THREAD_TYPES):
        logger.warning(
            "[affinity] rank %d/%d: only %d cores for %d types; all share",
            rank,
            world_size,
            len(domain),
            len(RESERVED_THREAD_TYPES),
        )
        return Plan(
            assignments=dict.fromkeys(RESERVED_THREAD_TYPES, -1),
            reserved=frozenset(),
            other_pool=frozenset(domain),
            numa_aware=numa_aware,
        )

    assignments: dict[str, int] = {}
    used: set[int] = set()
    for thread_type, core in zip(RESERVED_THREAD_TYPES, domain, strict=False):
        assignments[thread_type] = core
        used.add(core)

    other_pool = frozenset(c for c in domain if c not in used)
    mode = "NPU-local + freq" if numa_aware else "freq-ranked"
    logger.info(
        "[affinity] plan rank=%d/%d (%s): %s | domain=%d cores | other_pool=%d",
        rank,
        world_size,
        mode,
        assignments,
        len(domain),
        len(other_pool),
    )

    return Plan(assignments=assignments, reserved=frozenset(used), other_pool=other_pool, numa_aware=numa_aware)
