# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU topology probing for the affinity extension.

Reads host sysfs and (when available) the DCMI device management library
to build the facts the planner needs.

Probing layers (each degrades independently):

| layer              | method                              | fallback on failure     |
|--------------------|-------------------------------------|-------------------------|
| core inventory     | cgroup cpuset ∩ /sys/cpu/online     | affinity ∩ online, then os.cpu_count() |
| frequency layering | cpufreq or turbo:manual override    | no layering             |
| NUMA nodes         | /sys/.../node/nodeN/cpulist         | single node             |
| NPU-local cores    | DCMI → sysfs PCI → npu-smi topo     | None (no local bias)    |

The core inventory prefers the cgroup cpuset over the calling thread's
affinity mask: torch_npu's eager binding (lazy_bind:0) pins the main thread
to a single core during device init, and probing afterwards would otherwise
collapse the whole universe to that one core.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TOPO_INFO_MAX_LENGTH = 256
_TURBO_SENTINEL_FREQ = 9999999


@dataclass(frozen=True)
class TopologyInfo:
    """Immutable snapshot of host CPU topology."""

    usable: frozenset[int]
    freq: dict[int, int]
    numa_nodes: dict[int, frozenset[int]]
    npu_local_cores: dict[int, frozenset[int]] | None

    def has_freq_tiers(self) -> bool:
        return len(set(self.freq.values())) > 1

    def top_freq_cores(self) -> frozenset[int]:
        """Cores in the highest frequency tier (empty when uniform or unreadable)."""
        if not self.freq or not self.has_freq_tiers():
            return frozenset()
        top = max(self.freq.values())
        return frozenset(c for c, f in self.freq.items() if f == top)

    def has_numa_affinity(self) -> bool:
        return self.npu_local_cores is not None and len(self.numa_nodes) > 1

    def local_cores(self, npu_id: int) -> frozenset[int] | None:
        if self.npu_local_cores is None:
            return None
        return self.npu_local_cores.get(npu_id)


@dataclass(frozen=True)
class TurboSpec:
    """Parsed turbo sub-option from CPU_AFFINITY_CONF."""

    mode: str  # "auto" | "off" | "manual"
    cores: frozenset[int] = frozenset()


def _read(path: Path) -> str | None:
    try:
        text = path.read_text().strip()
        return text if text else None
    except (OSError, PermissionError):
        return None


def _parse_cpu_list(text: str) -> frozenset[int]:
    cores: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                cores.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cores.add(int(part))
    return frozenset(cores)


# ---------------------------------------------------------------------------
# layer 1: core inventory
# ---------------------------------------------------------------------------


def _cgroup_cpuset(sys_root: Path) -> frozenset[int] | None:
    """Pod-level CPU allocation from the cgroup cpuset (v2, then v1).

    Unlike the calling thread's affinity mask, this is immune to any binding
    applied before we probe.
    """
    for rel in ("fs/cgroup/cpuset.cpus.effective", "fs/cgroup/cpuset/cpuset.cpus"):
        text = _read(sys_root / rel)
        if text:
            cores = _parse_cpu_list(text)
            if cores:
                return cores
    return None


def _probe_usable(sys_root: Path) -> frozenset[int]:
    cgroup = _cgroup_cpuset(sys_root)
    if cgroup is not None:
        allowed = cgroup
    elif hasattr(os, "sched_getaffinity"):
        allowed = frozenset(os.sched_getaffinity(0))
    else:
        allowed = frozenset()
    online_text = _read(sys_root / "devices/system/cpu/online")
    online = _parse_cpu_list(online_text) if online_text else frozenset()

    if allowed and online:
        usable = allowed & online
        if usable:
            return usable
        logger.warning("[affinity] allowed∩online empty; using allowed")
        return allowed
    if online:
        return online
    if allowed:
        return allowed
    count = os.cpu_count() or 0
    if count:
        logger.info("[affinity] cpu/online unreadable; using os.cpu_count()=%d", count)
        return frozenset(range(count))
    raise RuntimeError("[affinity] cannot determine any usable CPU set")


# ---------------------------------------------------------------------------
# layer 2: frequency
# ---------------------------------------------------------------------------


def _probe_freq(sys_root: Path, usable: frozenset[int], turbo: TurboSpec | None) -> dict[int, int]:
    if turbo is not None and turbo.mode == "off":
        logger.info("[affinity] turbo:off; no frequency layering")
        return {}
    if turbo is not None and turbo.mode == "manual":
        logger.info("[affinity] turbo: manual high-freq cores %s", sorted(turbo.cores))
        return {c: _TURBO_SENTINEL_FREQ if c in turbo.cores else 0 for c in usable}

    freq: dict[int, int] = {}
    for core in sorted(usable):
        candidates = []
        for name in ("cpuinfo_max_freq", "scaling_max_freq"):
            text = _read(sys_root / f"devices/system/cpu/cpu{core}/cpufreq/{name}")
            if text and text.isdigit() and int(text) > 0:
                candidates.append(int(text))
        if candidates:
            freq[core] = min(candidates)
    if not freq:
        logger.info("[affinity] cpufreq unreadable; no frequency layering")
    return freq


# ---------------------------------------------------------------------------
# layer 3: NUMA nodes
# ---------------------------------------------------------------------------


def _probe_numa_nodes(sys_root: Path, usable: frozenset[int]) -> dict[int, frozenset[int]]:
    nodes: dict[int, frozenset[int]] = {}
    node_root = sys_root / "devices/system/node"
    if not node_root.is_dir():
        logger.info("[affinity] NUMA layout unreadable; treating all cores as one domain")
        return nodes
    for entry in sorted(node_root.iterdir()):
        m = re.match(r"node(\d+)$", entry.name)
        if not m or not entry.is_dir():
            continue
        text = _read(entry / "cpulist")
        if not text:
            continue
        cores = _parse_cpu_list(text) & usable
        if cores:
            nodes[int(m.group(1))] = cores
    if len(nodes) <= 1:
        logger.info("[affinity] single NUMA node (%d nodes)", len(nodes))
    return nodes


# ---------------------------------------------------------------------------
# layer 4: NPU-local cores (multi-method)
# ---------------------------------------------------------------------------

_NPU_PCI_VENDOR = "0x19e5"
_NPU_PCI_CLASS = "0x120000"

_DCMI_PROTOS = {
    "DcmiInit": [],
    "DcmiGetCardNumList": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "DcmiGetDeviceIdInCard": [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ],
    "DcmiGetAffinityCpuInfoByDeviceId": [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)],
}


def _load_dcmi() -> ctypes.CDLL | None:
    for name in ("libdcmi.so", "libdcmi.so.1", "dcmi"):
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            for fn, argtypes in _DCMI_PROTOS.items():
                func = getattr(lib, fn, None)
                if func is None:
                    return None
                func.argtypes = argtypes
                func.restype = ctypes.c_int
            return lib
        except OSError:
            continue
    return None


def _probe_npu_local_dcmi(usable: frozenset[int]) -> dict[int, frozenset[int]] | None:
    lib = _load_dcmi()
    if lib is None:
        return None
    try:
        if lib.DcmiInit() != 0:
            return None
        card_count = ctypes.c_int(0)
        card_list = (ctypes.c_int * 16)()
        if lib.DcmiGetCardNumList(ctypes.byref(card_count), card_list, 16) != 0:
            return None
        result: dict[int, frozenset[int]] = {}
        for i in range(card_count.value):
            card_id = card_list[i]
            device_id_max = ctypes.c_int(0)
            mcu_id = ctypes.c_int(0)
            cpu_id = ctypes.c_int(0)
            if (
                lib.DcmiGetDeviceIdInCard(
                    card_id, ctypes.byref(device_id_max), ctypes.byref(mcu_id), ctypes.byref(cpu_id)
                )
                != 0
            ):
                continue
            device_id = max(0, device_id_max.value - 1)
            buf = ctypes.create_string_buffer(_TOPO_INFO_MAX_LENGTH)
            length = ctypes.c_int(_TOPO_INFO_MAX_LENGTH)
            if lib.DcmiGetAffinityCpuInfoByDeviceId(card_id, device_id, buf, ctypes.byref(length)) == 0:
                cpu_str = buf.value.decode("utf-8", errors="replace").strip()
                if cpu_str:
                    cores = _parse_cpu_list(cpu_str) & usable
                    if cores:
                        result[i] = cores
        if result:
            logger.info("[affinity] NPU-local (DCMI): %s", {k: len(v) for k, v in result.items()})
            return result
        return None
    except Exception:
        return None


def _find_npu_bdfs(sys_root: Path) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    pci_root = sys_root / "bus/pci/devices"
    if not pci_root.is_dir():
        return results
    bdfs = []
    for entry in sorted(pci_root.iterdir()):
        if _read(entry / "vendor") == _NPU_PCI_VENDOR and _read(entry / "class") == _NPU_PCI_CLASS:
            bdfs.append(entry.name)
    return [(bdf, idx) for idx, bdf in enumerate(bdfs)]


def _probe_npu_local_sysfs_pci(
    sys_root: Path, numa_nodes: dict[int, frozenset[int]], usable: frozenset[int]
) -> dict[int, frozenset[int]] | None:
    bdfs = _find_npu_bdfs(sys_root)
    if not bdfs:
        return None
    result: dict[int, frozenset[int]] = {}
    for bdf, dev_idx in bdfs:
        numa_text = _read(sys_root / f"bus/pci/devices/{bdf}/numa_node")
        if numa_text is None or not numa_text.lstrip("-").isdigit():
            return None
        node = int(numa_text)
        if node < 0:
            return None
        cores = numa_nodes.get(node)
        if cores is None:
            lc_text = _read(sys_root / f"bus/pci/devices/{bdf}/local_cpulist")
            if lc_text:
                cores = _parse_cpu_list(lc_text) & usable
        if cores:
            result[dev_idx] = cores
    if result:
        logger.info("[affinity] NPU-local (sysfs PCI): %s", {k: len(v) for k, v in result.items()})
        return result
    return None


def _probe_npu_local_npu_smi(usable: frozenset[int]) -> dict[int, frozenset[int]] | None:
    try:
        proc = subprocess.run(["npu-smi", "info", "-t", "topo"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    result: dict[int, frozenset[int]] = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(?:NUMA\d+\s+)?([\d,\-]+)\s*$", line)
        if m:
            cores = _parse_cpu_list(m.group(2)) & usable
            if cores:
                result[int(m.group(1))] = cores
    if result:
        logger.info("[affinity] NPU-local (npu-smi): %s", {k: len(v) for k, v in result.items()})
        return result
    return None


def _probe_npu_local(
    sys_root: Path, numa_nodes: dict[int, frozenset[int]], usable: frozenset[int]
) -> dict[int, frozenset[int]] | None:
    result = _probe_npu_local_dcmi(usable)
    if result is not None:
        return result
    result = _probe_npu_local_sysfs_pci(sys_root, numa_nodes, usable)
    if result is not None:
        return result
    result = _probe_npu_local_npu_smi(usable)
    if result is not None:
        return result
    logger.info("[affinity] NPU-local undetectable; freq-only ranking")
    return None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def probe(sys_root: Path | None = None, *, turbo: TurboSpec | None = None) -> TopologyInfo:
    """Probe host topology. Never raises — affinity problems log and disable."""
    root = sys_root or Path("/sys")
    usable = _probe_usable(root)
    freq = _probe_freq(root, usable, turbo)
    numa_nodes = _probe_numa_nodes(root, usable)
    npu_local = _probe_npu_local(root, numa_nodes, usable)

    parts = [f"{len(usable)} cores"]
    if len(set(freq.values())) > 1:
        parts.append(f"freq {min(freq.values())}-{max(freq.values())}")
    else:
        parts.append("uniform freq")
    if npu_local:
        parts.append(f"NPU-local: { {k: len(v) for k, v in npu_local.items()} }")
    else:
        parts.append("no NPU-local")
    logger.info("[affinity] topology: %s", ", ".join(parts))

    return TopologyInfo(usable=usable, freq=freq, numa_nodes=numa_nodes, npu_local_cores=npu_local)
