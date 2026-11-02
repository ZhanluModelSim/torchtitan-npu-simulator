# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AffinityExtension: automatic CPU affinity for torchtitan-npu training.

Entry point mounted at TrainerEx.__init__() tail. Parses CPU_AFFINITY_CONF,
probes topology, plans optimal placement, rebinds threads, and starts a
gap-filler daemon for threads the official mechanism cannot reach.

Configuration (sub-options use colon, matching torch_npu convention):

    The turbo sub-option is the extension's opt-in switch. Without it the
    extension stays completely inactive and official semantics run untouched:

    CPU_AFFINITY_CONF=0 / unset / other   → extension disabled (official only)
    CPU_AFFINITY_CONF=1 / 2 (no turbo)    → extension disabled (turbo off by default)
    CPU_AFFINITY_CONF=1,turbo:<spec>      → assist: bind main to this rank's device slice
    CPU_AFFINITY_CONF=2,turbo:<spec>      → full: topology + planner + binder + daemon
    turbo:auto / turbo:0-7 / turbo:0,3,5  → opt in (auto-detect or manual cores)
    turbo:off                             → stay inactive (explicit default)

Never raises: affinity problems log and disable themselves.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

from .topology import TopologyInfo, TurboSpec, probe

if sys.platform == "linux":
    from .binder import bind_all, bind_main
    from .daemon import AffinityDaemon
    from .planner import plan as make_plan

logger = logging.getLogger(__name__)


def _conf_tokens(conf: str) -> dict[str, list[str]]:
    """Split CPU_AFFINITY_CONF into sub-options.

    Mirrors torch_npu's scanner: a segment containing a colon starts a new
    sub-option ("lazy_bind:0"); a bare comma segment continues the previous
    sub-option's value list ("turbo:0,3,5"). Bare segments before any
    sub-option (the leading mode, e.g. "2") are dropped.
    """
    tokens: dict[str, list[str]] = {}
    current: str | None = None
    for segment in conf.split(","):
        segment = segment.strip()
        if not segment:
            continue
        head, sep, tail = segment.partition(":")
        if sep and head:
            current = head
            tokens.setdefault(current, []).append(tail)
        elif current is not None:
            tokens[current].append(segment)
    return tokens


@dataclass(frozen=True)
class _Intent:
    """Parsed CPU_AFFINITY_CONF result."""

    mode: int  # 0=off, 1=assist, 2=full
    turbo: TurboSpec


def _turbo_mode_of(values: list[str]) -> str:
    """Classify turbo sub-option values: "auto" | "off" | "manual"."""
    stripped = [v.strip() for v in values if v.strip()]
    if not stripped:
        return "off"
    if all(v in ("off", "none") for v in stripped):
        return "off"
    if stripped == ["auto"]:
        return "auto"
    return "manual"


def _parse_turbo(conf: str, usable: frozenset[int]) -> TurboSpec:
    """Parse turbo:<spec> from CPU_AFFINITY_CONF."""
    values = _conf_tokens(conf).get("turbo")
    if not values:
        return TurboSpec(mode="auto")
    mode = _turbo_mode_of(values)
    if mode != "manual":
        return TurboSpec(mode=mode)
    # try to parse as core list
    cores: set[int] = set()
    for part in values:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit():
                cores.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cores.add(int(part))
    if not cores:
        logger.warning("[affinity] turbo:%r is not a valid core spec; using auto", values)
        return TurboSpec(mode="auto")
    valid = cores & usable
    if not valid:
        logger.warning("[affinity] turbo spec has no usable cores; using auto")
        return TurboSpec(mode="auto")
    if valid != cores:
        logger.warning("[affinity] turbo spec includes unavailable cores; using intersection")
    return TurboSpec(mode="manual", cores=frozenset(valid))


def _parse_conf() -> _Intent:
    raw = os.environ.get("CPU_AFFINITY_CONF", "").strip()
    if not raw or raw == "0":
        return _Intent(mode=0, turbo=TurboSpec(mode="off"))
    # extract leading mode digits
    head = raw.split(",")[0].strip()
    if head == "1":
        mode = 1
    elif head.isdigit() and int(head) >= 2:
        mode = 2
        if int(head) > 2:
            logger.info("[affinity] CONF mode %s treated as 2", head)
    elif head.startswith("mode:"):
        val = head.split(":", 1)[1]
        mode = int(val) if val.isdigit() else 0
    else:
        mode = 0
    # parse turbo (need usable cores for validation, defer to start());
    # absent turbo means the extension stays completely inactive
    turbo = TurboSpec(mode="off")
    values = _conf_tokens(raw).get("turbo")
    if values is not None:
        # manual core specs are validated later in _start via _parse_turbo
        turbo = TurboSpec(mode=_turbo_mode_of(values))
    return _Intent(mode=mode, turbo=turbo)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else None


def _local_rank_world() -> tuple[int, int]:
    """Node-local rank/world from torchrun (LOCAL_RANK/LOCAL_WORLD_SIZE).

    Multi-node safe: each node partitions only among its own processes.
    """
    world = _env_int("LOCAL_WORLD_SIZE") or 1
    rank = _env_int("LOCAL_RANK") or 0
    if world < 1:
        world = 1
    if not 0 <= rank < world:
        logger.warning("[affinity] LOCAL_RANK=%d invalid (world=%d); using 0", rank, world)
        rank = 0
    return rank, world


def _device_count() -> int:
    """Visible NPU count: Python API first, ASCEND_RT_VISIBLE_DEVICES fallback."""
    n: int | None = None
    try:
        import torch_npu

        getters = (
            getattr(torch_npu.npu, "device_count", None),
            getattr(torch_npu, "device_count", None),
        )
        for getter in getters:
            if getter is None:
                continue
            try:
                v = getter()
            except Exception:
                continue
            if isinstance(v, int) and v > 0:
                n = v
                break
    except Exception:
        pass
    if n is None:
        raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        if raw:
            n = len([p for p in raw.split(",") if p.strip().isdigit()])
    return n if isinstance(n, int) and n > 0 else 1


def _victim_signatures(info: TopologyInfo) -> frozenset[frozenset[int]]:
    """Masks that torch_npu's native per-type binding may pin workers to.

    torch_npu slices online cores per device (block = base // devices) and,
    in mode >= 2, pins device-touching threads onto the slice head cores
    (one per reserved type) or the slice remainder (OTHER_THREAD). Both the
    host-online view and the usable (cpuset) view are considered because
    sysconf may report either depending on the container.
    """
    sigs: set[frozenset[int]] = set()
    devices = _device_count()
    bases = {n for n in (os.cpu_count(), len(info.usable)) if n and n > 0}
    for base in bases:
        block = base // devices
        if block < 1:
            continue
        for d in range(devices):
            lo = d * block
            hi = min((d + 1) * block, base)
            if hi - lo < 5:
                # torch_npu binds every type to the whole slice when tiny
                whole = frozenset(c for c in range(lo, hi) if c in info.usable)
                if whole and whole != info.usable:
                    sigs.add(whole)
                continue
            for k in range(4):
                core = lo + k
                if core in info.usable:
                    sigs.add(frozenset({core}))
            rest = frozenset(c for c in range(lo + 4, hi) if c in info.usable)
            if rest:
                sigs.add(rest)
    return frozenset(sigs)


class AffinityExtension:
    """Main entry point. start() at TrainerEx init tail, stop() at close()."""

    def __init__(self) -> None:
        self._daemon: AffinityDaemon | None = None

    @classmethod
    def start(cls) -> AffinityExtension | None:
        """Parse config, probe topology, plan, bind, start daemon. Never raises."""
        if sys.platform != "linux":
            logger.info("[affinity] non-linux host; disabled")
            return None

        try:
            return cls()._start()
        except Exception as exc:  # never fail training
            logger.warning("[affinity] disabled after unexpected error: %s", exc)
            return None

    def _start(self) -> AffinityExtension | None:
        intent = _parse_conf()
        if intent.mode == 0:
            logger.info("[affinity] extension disabled (mode=0)")
            return None
        if intent.turbo.mode == "off":
            logger.info("[affinity] extension disabled (turbo off; official semantics only)")
            return None

        logger.info("[affinity] mode=%d, turbo=%s", intent.mode, intent.turbo.mode)

        # probe topology (turbo:manual needs usable cores for validation)
        info = probe(turbo=intent.turbo if intent.turbo.mode != "manual" else None)

        # re-validate turbo manual with actual usable cores
        turbo = intent.turbo
        if turbo.mode == "manual" and not turbo.cores:
            raw = os.environ.get("CPU_AFFINITY_CONF", "")
            turbo = _parse_turbo(raw, info.usable)
            if turbo.mode == "manual":
                info = probe(turbo=turbo)

        if intent.mode == 1:
            # assist: bind main to this rank's contiguous share of usable
            # cores, matching the official device-slice layout; without turbo
            # main stays floating (lazy chain dead) or eager-pinned
            rank, world = _local_rank_world()
            cores = sorted(info.usable)
            block = len(cores) // world
            main_region = frozenset(cores[rank * block : (rank + 1) * block])
            result = bind_main(main_region)
            logger.info("[affinity] assist mode: main bind %s", "ok" if result.success else "failed")
            return self

        # mode 2: full pipeline
        npu_id = None
        try:
            import torch_npu

            npu_id = torch_npu.npu.current_device()
        except Exception:
            pass

        rank, world = _local_rank_world()
        p = make_plan(info, npu_id=npu_id, rank=rank, world_size=world)
        bind_result = bind_all(p)

        self._daemon = AffinityDaemon(
            info,
            p,
            reserved_tids=frozenset(bind_result.tids.values()),
            victim_masks=_victim_signatures(info),
        )
        self._daemon.start()

        if bind_result.failures:
            logger.warning("[affinity] bind failures: %s", bind_result.failures)
        else:
            logger.info("[affinity] all %d threads bound", len(bind_result.bound))

        return self

    def stop(self) -> None:
        if self._daemon is not None:
            self._daemon.stop()
            self._daemon = None
