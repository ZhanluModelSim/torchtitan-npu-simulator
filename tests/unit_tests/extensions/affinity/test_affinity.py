# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for the affinity extension's pure-logic components."""

import time
from dataclasses import replace

from torchtitan_npu.extensions.affinity import AffinityExtension, _parse_conf, _parse_turbo, _victim_signatures
from torchtitan_npu.extensions.affinity.binder import BindResult
from torchtitan_npu.extensions.affinity.daemon import AffinityDaemon, RebindRules, is_unmanaged
from torchtitan_npu.extensions.affinity.planner import plan
from torchtitan_npu.extensions.affinity.topology import TopologyInfo, TurboSpec, _parse_cpu_list, _probe_usable


class TestTopologyInfo:
    def test_uniform_freq_no_tiers(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={i: 3000000 for i in range(8)},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert not info.has_freq_tiers()

    def test_heterogeneous_freq_tiers(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={**{i: 2600000 for i in range(4)}, **{i: 3500000 for i in range(4, 8)}},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert info.has_freq_tiers()

    def test_no_numa_affinity_when_none(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert not info.has_numa_affinity()

    def test_local_cores_lookup(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={},
            numa_nodes={0: frozenset(range(4)), 1: frozenset(range(4, 8))},
            npu_local_cores={0: frozenset(range(4))},
        )
        assert info.has_numa_affinity()
        assert info.local_cores(0) == frozenset(range(4))
        assert info.local_cores(1) is None

    def test_top_freq_cores_tiers(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={0: 2600000, 1: 2600000, 2: 3500000, 3: 3500000},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert info.top_freq_cores() == frozenset({2, 3})

    def test_top_freq_cores_uniform(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={i: 3000000 for i in range(8)},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert info.top_freq_cores() == frozenset()

    def test_top_freq_cores_empty(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        assert info.top_freq_cores() == frozenset()


class TestParseConf:
    def _parse(self, monkeypatch, value):
        monkeypatch.setenv("CPU_AFFINITY_CONF", value)
        return _parse_conf()

    def test_turbo_defaults_to_off(self, monkeypatch):
        intent = self._parse(monkeypatch, "2")
        assert intent.mode == 2
        assert intent.turbo.mode == "off"

    def test_turbo_explicit_off(self, monkeypatch):
        intent = self._parse(monkeypatch, "2,turbo:off")
        assert intent.mode == 2
        assert intent.turbo.mode == "off"

    def test_turbo_auto(self, monkeypatch):
        intent = self._parse(monkeypatch, "2,turbo:auto")
        assert intent.mode == 2
        assert intent.turbo.mode == "auto"

    def test_turbo_manual_deferred(self, monkeypatch):
        intent = self._parse(monkeypatch, "2,turbo:0-7")
        assert intent.mode == 2
        assert intent.turbo.mode == "manual"

    def test_turbo_off_terminated_by_next_suboption(self, monkeypatch):
        """A following sub-option must not extend the turbo:off value."""
        intent = self._parse(monkeypatch, "2,turbo:off,lazy_bind:0")
        assert intent.mode == 2
        assert intent.turbo.mode == "off"

    def test_turbo_none_terminated_by_device_range(self, monkeypatch):
        intent = self._parse(monkeypatch, "1,turbo:none,npu0:0-3")
        assert intent.mode == 1
        assert intent.turbo.mode == "off"

    def test_turbo_auto_terminated_by_next_suboption(self, monkeypatch):
        intent = self._parse(monkeypatch, "2,turbo:auto,npu_affine:1")
        assert intent.mode == 2
        assert intent.turbo.mode == "auto"

    def test_turbo_manual_terminated_by_next_suboption(self, monkeypatch):
        intent = self._parse(monkeypatch, "2,turbo:0-3,lazy_bind:0")
        assert intent.mode == 2
        assert intent.turbo.mode == "manual"

    def test_unset_disables(self, monkeypatch):
        monkeypatch.delenv("CPU_AFFINITY_CONF", raising=False)
        intent = _parse_conf()
        assert intent.mode == 0
        assert intent.turbo.mode == "off"


class TestExtensionGate:
    def test_turbo_off_disables_start(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "2")
        assert AffinityExtension.start() is None

    def test_mode_zero_disables_start(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "0")
        assert AffinityExtension.start() is None

    def test_turbo_off_with_suboptions_disables_start(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "2,turbo:off,lazy_bind:0")
        assert AffinityExtension.start() is None


class TestTurboSpec:
    def test_auto_default(self):
        t = TurboSpec(mode="auto")
        assert t.mode == "auto"
        assert t.cores == frozenset()


class TestParseTurbo:
    def _usable(self):
        return frozenset(range(40))

    def test_range_spec(self):
        t = _parse_turbo("2,turbo:0-7", self._usable())
        assert t.mode == "manual"
        assert t.cores == frozenset(range(8))

    def test_scattered_list(self):
        t = _parse_turbo("2,turbo:0,3,5", self._usable())
        assert t.mode == "manual"
        assert t.cores == frozenset({0, 3, 5})

    def test_list_terminated_by_next_suboption(self):
        t = _parse_turbo("2,turbo:0-3,npu_affine:1", self._usable())
        assert t.mode == "manual"
        assert t.cores == frozenset(range(4))

    def test_off_terminated_by_next_suboption(self):
        assert _parse_turbo("2,turbo:off,lazy_bind:0", self._usable()).mode == "off"

    def test_scattered_list_terminated_by_next_suboption(self):
        t = _parse_turbo("2,turbo:0,3,5,lazy_bind:0", self._usable())
        assert t.mode == "manual"
        assert t.cores == frozenset({0, 3, 5})

    def test_range_terminated_by_next_suboption(self):
        t = _parse_turbo("2,turbo:0-3,force:1", self._usable())
        assert t.mode == "manual"
        assert t.cores == frozenset(range(4))

    def test_auto_and_off(self):
        assert _parse_turbo("2,turbo:auto", self._usable()).mode == "auto"
        assert _parse_turbo("2,turbo:off", self._usable()).mode == "off"

    def test_absent(self):
        assert _parse_turbo("2", self._usable()).mode == "auto"


class TestParseCpuList:
    def test_single(self):
        assert _parse_cpu_list("5") == frozenset({5})

    def test_range(self):
        assert _parse_cpu_list("0-7") == frozenset(range(8))

    def test_scattered(self):
        assert _parse_cpu_list("0,3,5,7") == frozenset({0, 3, 5, 7})

    def test_mixed(self):
        assert _parse_cpu_list("0-3,8,10-11") == frozenset({0, 1, 2, 3, 8, 10, 11})

    def test_empty(self):
        assert _parse_cpu_list("") == frozenset()

    def test_invalid(self):
        assert _parse_cpu_list("abc") == frozenset()


class TestPlanner:
    def _make_topo(self, cores, freq=None, npu_local=None):
        return TopologyInfo(
            usable=frozenset(cores),
            freq=freq or {c: 0 for c in cores},
            numa_nodes={0: frozenset(cores)},
            npu_local_cores=npu_local,
        )

    def test_homogeneous_ascending(self):
        """On homogeneous cores, allocation should be ascending id order."""
        info = self._make_topo(range(8))
        p = plan(info)
        assert p.assignments == {"main": 0, "acl": 1, "release": 2, "watchdog": 3}
        assert p.other_pool == frozenset(range(4, 8))
        assert not p.numa_aware

    def test_heterogeneous_freq_first(self):
        """High-freq cores should be assigned to high-priority types."""
        info = self._make_topo(range(16), freq={**{i: 2600000 for i in range(8)}, **{i: 3500000 for i in range(8, 16)}})
        p = plan(info)
        assert p.assignments == {"main": 8, "acl": 9, "release": 10, "watchdog": 11}
        assert p.other_pool == frozenset(list(range(8)) + list(range(12, 16)))

    def test_numa_local_preferred(self):
        """NPU-local cores should rank first."""
        info = self._make_topo(range(16), npu_local={0: frozenset(range(8, 16))})
        p = plan(info, npu_id=0)
        assert p.numa_aware
        # top 4 cores should all be from the local set (8-15)
        assert all(p.assignments[t] >= 8 for t in ("main", "acl", "release", "watchdog"))

    def test_partial_turbo(self):
        """Partial turbo cores (0-3 high) should get top assignments."""
        info = self._make_topo(range(16), freq={**{i: 3500000 for i in range(4)}, **{i: 3000000 for i in range(4, 16)}})
        p = plan(info)
        assert p.assignments == {"main": 0, "acl": 1, "release": 2, "watchdog": 3}

    def test_insufficient_cores(self):
        """Fewer than 4 cores → all types share."""
        info = self._make_topo(range(3))
        p = plan(info)
        assert all(v == -1 for v in p.assignments.values())
        assert p.other_pool == frozenset(range(3))

    def test_reserved_set(self):
        info = self._make_topo(range(8))
        p = plan(info)
        assert p.reserved == frozenset({0, 1, 2, 3})


class TestPlannerRankIsolation:
    def _make_topo(self, cores, freq=None):
        return TopologyInfo(
            usable=frozenset(cores),
            freq=freq or {c: 0 for c in cores},
            numa_nodes={0: frozenset(cores)},
            npu_local_cores=None,
        )

    def test_two_ranks_disjoint(self):
        """Rank domains must partition usable cores with zero overlap."""
        info = self._make_topo(range(8))
        p0 = plan(info, rank=0, world_size=2)
        p1 = plan(info, rank=1, world_size=2)
        d0 = set(p0.assignments.values()) | set(p0.other_pool)
        d1 = set(p1.assignments.values()) | set(p1.other_pool)
        assert d0 & d1 == set()
        assert d0 | d1 == set(range(8))

    def test_many_ranks_pairwise_disjoint(self):
        info = self._make_topo(range(12))
        domains = []
        for r in range(3):
            p = plan(info, rank=r, world_size=3)
            domains.append(set(p.assignments.values()) | set(p.other_pool))
        for i in range(3):
            for j in range(i + 1, 3):
                assert domains[i] & domains[j] == set()

    def test_stride_spreads_freq_tiers(self):
        """Both ranks draw reserved cores from the high-freq tier."""
        info = self._make_topo(
            range(16),
            freq={**{i: 2600000 for i in range(8)}, **{i: 3500000 for i in range(8, 16)}},
        )
        for r in (0, 1):
            p = plan(info, rank=r, world_size=2)
            assert all(p.assignments[t] >= 8 for t in ("main", "acl", "release", "watchdog"))

    def test_world_size_one_unchanged(self):
        info = self._make_topo(range(8))
        p = plan(info, rank=0, world_size=1)
        assert p.assignments == {"main": 0, "acl": 1, "release": 2, "watchdog": 3}
        assert p.other_pool == frozenset(range(4, 8))

    def test_rank_out_of_range_falls_back(self):
        info = self._make_topo(range(8))
        p = plan(info, rank=5, world_size=2)
        assert p.assignments == plan(info, rank=0, world_size=2).assignments

    def test_domain_too_small_shares(self):
        info = self._make_topo(range(8))
        p = plan(info, rank=3, world_size=8)
        assert all(v == -1 for v in p.assignments.values())
        assert p.other_pool == frozenset({3})

    def _make_numa_topo(self, cores, local):
        return TopologyInfo(
            usable=frozenset(cores),
            freq={c: 0 for c in cores},
            numa_nodes={0: frozenset(cores)},
            npu_local_cores=local,
        )

    def test_numa_aware_two_ranks_disjoint(self):
        """Per-rank NPU locality must not break the cross-rank partition.

        Regression: unequal per-rank local sets (7 vs 9 cores) made the old
        locality-aware pre-split ranking collide the two domains completely.
        """
        info = self._make_numa_topo(range(16), {0: frozenset(range(7)), 1: frozenset(range(7, 16))})
        p0 = plan(info, npu_id=0, rank=0, world_size=2)
        p1 = plan(info, npu_id=1, rank=1, world_size=2)
        assert p0.numa_aware and p1.numa_aware
        d0 = set(p0.assignments.values()) | set(p0.other_pool)
        d1 = set(p1.assignments.values()) | set(p1.other_pool)
        assert d0 & d1 == set()
        assert d0 | d1 == set(range(16))

    def test_numa_aware_many_ranks_pairwise_disjoint(self):
        info = self._make_numa_topo(range(12), {0: frozenset(range(5)), 1: frozenset(range(5, 12))})
        domains = []
        for r in range(3):
            p = plan(info, npu_id=r, rank=r, world_size=3)
            domains.append(set(p.assignments.values()) | set(p.other_pool))
        for i in range(3):
            for j in range(i + 1, 3):
                assert domains[i] & domains[j] == set()

    def test_reserved_prefer_local_within_domain(self):
        """After the split, reserved cores come from this rank's local set first."""
        info = self._make_numa_topo(range(16), {0: frozenset(range(7)), 1: frozenset(range(7, 16))})
        p1 = plan(info, npu_id=1, rank=1, world_size=2)
        assert set(p1.assignments.values()) <= set(range(7, 16))


class TestIsUnmanaged:
    _RULES = RebindRules(
        main_tid=100,
        reserved_tids=frozenset({101, 102}),
        usable=frozenset(range(8)),
        pool=frozenset(range(4, 8)),
        main_mask=frozenset({0}),
        patterns=frozenset({frozenset({1})}),
    )

    def _is_unmanaged(self, mask, prev=None, tid=999, **kw):
        rules = replace(self._RULES, **kw) if kw else self._RULES
        return is_unmanaged(rules, tid, frozenset(mask), frozenset(prev) if prev is not None else None)

    def test_reserved_tid(self):
        assert not self._is_unmanaged(frozenset({1}), tid=101)

    def test_main_tid(self):
        assert not self._is_unmanaged(frozenset({0}), tid=100)

    def test_floating(self):
        assert self._is_unmanaged(range(8))

    def test_at_pool(self):
        assert not self._is_unmanaged(range(4, 8))

    def test_on_main_core(self):
        assert self._is_unmanaged(frozenset({0}))

    def test_matching_pattern(self):
        assert self._is_unmanaged(frozenset({1}))

    def test_narrowed_since_last_round(self):
        assert self._is_unmanaged(frozenset({2}), prev=range(4, 8))

    def test_widened(self):
        assert not self._is_unmanaged(range(4, 8), prev=frozenset({2}))

    def test_deliberately_narrow(self):
        assert not self._is_unmanaged(frozenset({3}))


class TestBindResult:
    def test_tids_default_empty(self):
        r = BindResult(success=True, bound={}, failures=[])
        assert r.tids == {}


class TestVictimSignatures:
    def _make_topo(self, cores):
        return TopologyInfo(
            usable=frozenset(cores),
            freq={},
            numa_nodes={0: frozenset(cores)},
            npu_local_cores=None,
        )

    def test_single_device(self, monkeypatch):
        import torchtitan_npu.extensions.affinity as aff

        monkeypatch.setattr(aff, "_device_count", lambda: 1)
        monkeypatch.setattr(aff.os, "cpu_count", lambda: 8)
        sigs = _victim_signatures(self._make_topo(range(8)))
        assert frozenset({0}) in sigs
        assert frozenset({3}) in sigs
        assert frozenset(range(4, 8)) in sigs

    def test_two_devices_small_block(self, monkeypatch):
        import torchtitan_npu.extensions.affinity as aff

        monkeypatch.setattr(aff, "_device_count", lambda: 2)
        monkeypatch.setattr(aff.os, "cpu_count", lambda: 8)
        sigs = _victim_signatures(self._make_topo(range(8)))
        # block = 8 // 2 = 4 < 5 → whole-slice signatures per device
        assert frozenset(range(4)) in sigs
        assert frozenset(range(4, 8)) in sigs

    def test_two_devices_block20(self, monkeypatch):
        """The production geometry: 40 cores, 2 devices, block=20 per slice."""
        import torchtitan_npu.extensions.affinity as aff

        monkeypatch.setattr(aff, "_device_count", lambda: 2)
        monkeypatch.setattr(aff.os, "cpu_count", lambda: 40)
        sigs = _victim_signatures(self._make_topo(range(40)))
        # device-0 slice [0..19]: head cores and remainder
        assert frozenset({0}) in sigs
        assert frozenset({3}) in sigs
        assert frozenset(range(4, 20)) in sigs
        # device-1 slice [20..39]: head cores and remainder
        assert frozenset({20}) in sigs
        assert frozenset({23}) in sigs
        assert frozenset(range(24, 40)) in sigs


class TestProbeUsable:
    """The usable-core universe must survive main being pre-pinned (eager)."""

    def _setup_sys(self, root):
        (root / "devices/system/cpu").mkdir(parents=True)
        (root / "devices/system/cpu/online").write_text("0-7")
        (root / "fs/cgroup").mkdir(parents=True)

    def test_cgroup_wins_over_pinned_main(self, tmp_path, monkeypatch):
        self._setup_sys(tmp_path)
        (tmp_path / "fs/cgroup/cpuset.cpus.effective").write_text("0-7")
        # main already pinned to a single core by eager binding
        monkeypatch.setattr(
            "torchtitan_npu.extensions.affinity.topology.os.sched_getaffinity",
            lambda pid: {0},
        )
        assert _probe_usable(tmp_path) == frozenset(range(8))

    def test_v1_cgroup_path(self, tmp_path, monkeypatch):
        self._setup_sys(tmp_path)
        (tmp_path / "fs/cgroup/cpuset").mkdir()
        (tmp_path / "fs/cgroup/cpuset/cpuset.cpus").write_text("2-5")
        monkeypatch.setattr(
            "torchtitan_npu.extensions.affinity.topology.os.sched_getaffinity",
            lambda pid: {0},
        )
        assert _probe_usable(tmp_path) == frozenset(range(2, 6))

    def test_affinity_fallback_without_cgroup(self, tmp_path, monkeypatch):
        self._setup_sys(tmp_path)
        monkeypatch.setattr(
            "torchtitan_npu.extensions.affinity.topology.os.sched_getaffinity",
            lambda pid: frozenset(range(8)),
        )
        assert _probe_usable(tmp_path) == frozenset(range(8))


class TestCircuitBreaker:
    def _make_daemon(self):
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        p = plan(info, rank=0, world_size=1)
        return AffinityDaemon(info, p, reserved_tids=frozenset({100}))

    def _fake_proc(self, monkeypatch, masks):
        import torchtitan_npu.extensions.affinity.daemon as daemon_mod

        calls = []
        monkeypatch.setattr(
            daemon_mod.os,
            "sched_getaffinity",
            lambda tid: masks.get(tid, frozenset(range(8))),
            raising=False,
        )
        monkeypatch.setattr(
            daemon_mod.os,
            "sched_setaffinity",
            lambda tid, cores: calls.append(tid),
            raising=False,
        )

        class FakeEntry:
            def __init__(self, name):
                self.name = name

        class FakePath:
            def __init__(self, path):
                self._path = str(path)

            def iterdir(self):
                if "task" in self._path:
                    return [FakeEntry(t) for t in ("100", "101", "102")]
                return []

        monkeypatch.setattr(daemon_mod, "Path", FakePath)
        return calls

    def test_consecutive_rescues_blacklist_after_threshold(self, monkeypatch):
        d = self._make_daemon()
        # 101 simulates instant revert: mask is always full regardless of binds
        masks = {100: frozenset({0}), 101: frozenset(range(8)), 102: frozenset(range(4, 8))}
        calls = self._fake_proc(monkeypatch, masks)
        for _ in range(6):
            d._one_round()
        # rescued rounds 1-4 (strikes 0,1,2,3), skipped from round 5 on
        assert calls.count(101) == 4
        assert 102 not in calls
        assert 100 not in calls
        assert d._blacklist[101] >= 3

    def test_occasional_renarrow_never_blacklists(self, monkeypatch):
        d = self._make_daemon()
        masks = {100: frozenset({0}), 101: frozenset(range(8)), 102: frozenset(range(4, 8))}
        calls = self._fake_proc(monkeypatch, masks)
        d._one_round()  # rescue 101
        masks[101] = frozenset(range(4, 8))
        d._one_round()  # 101 settles at pool
        masks[101] = frozenset(range(8))
        d._one_round()  # narrows again later — not consecutive
        masks[101] = frozenset(range(4, 8))
        d._one_round()
        masks[101] = frozenset(range(8))
        d._one_round()
        assert calls.count(101) == 3
        assert d._blacklist.get(101, 0) == 0


class TestDaemonReservesItself:
    def test_daemon_native_tid_in_reserved(self, monkeypatch):
        """The daemon thread must reserve its kernel TID, not its Python ident.

        /proc/self/task exposes kernel TIDs; threading.get_ident() is a
        Python-internal handle that never matches, which used to leave the
        daemon unreserved so its first round re-classified itself.
        """
        import torchtitan_npu.extensions.affinity.daemon as daemon_mod

        class FakePath:
            def __init__(self, path):
                self._path = str(path)

            def iterdir(self):
                return []  # hermetic: no real /proc scanning in _one_round

        monkeypatch.setattr(daemon_mod, "Path", FakePath)
        info = TopologyInfo(
            usable=frozenset(range(8)),
            freq={},
            numa_nodes={0: frozenset(range(8))},
            npu_local_cores=None,
        )
        p = plan(info, rank=0, world_size=1)
        d = AffinityDaemon(info, p)
        try:
            d.start()
            assert d._thread is not None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if d._thread.native_id in d._rules.reserved_tids:
                    break
                time.sleep(0.05)
            assert d._thread.native_id in d._rules.reserved_tids
        finally:
            d.stop()
