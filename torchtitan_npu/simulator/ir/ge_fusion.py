# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GE graph-mode fusion profile: consume an offline-captured GE fused graph
(produced by Phase 1 ``scripts/ge_fusion_capture.py`` on a real NPU) to
populate ``StepGraph.fused_regions``.

Why offline + consume (not run fusion in the simulator)
-------------------------------------------------------
The simulator captures L0 ops under meta-device ``__torch_dispatch__``, which
yields an *over-decomposed* graph (~75% of nodes are aten primitives like
``aten.mul``/``aten.unsqueeze`` that a real NPU's Graph Engine fuses away).
GE's op fusion happens inside the ``aclgrph`` compile pipeline, whose
``build_initialize`` needs the NPU runtime/driver -- so fusion cannot run in a
no-NPU simulator. Instead we interface with GE's *output*: Phase 1 captures
the GE-compiled (fused) graph as a profile, and this module (Phase 2) loads it
and maps it onto the captured L0 ops so the simulator models real fused
kernels.

Profile sources
---------------
* JSON profile (``load_fusion_profile_json``): the portable contract. Phase 1
  emits one entry per fused GE node with its ``fused_op_type`` and the
  ``original_op_seq_idxs`` (the captured L0 ``OpNode.seq_idx`` values that
  merged into it). This carries the original->fused mapping explicitly.
* ``.air`` graph (``load_fusion_profile_air``): GE's native serialized graph.
  ``extract_graph_topology`` walks it to recover fused-node types + data edges
  (producer/consumer). ``.air`` alone does not carry the original->fused op
  mapping, so this yields topology only -- useful for fused-op cost typing
  even without per-op attribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FusedNode:
    """One GE fused kernel in the compiled graph."""

    node_id: int
    fused_op_type: str
    # captured L0 OpNode.seq_idx values that merged into this fused kernel.
    # Empty for a raw-.air topology node (no per-op attribution available).
    original_op_seq_idxs: list[int] = field(default_factory=list)
    # producer FusedNode.node_id values (data-edge inputs), in port order.
    input_fused_ids: list[int] = field(default_factory=list)
    output_bytes: int = 0


@dataclass
class GEFusionProfile:
    """A loaded GE fusion profile: the list of fused nodes + a seq_idx->fused
    lookup for fast mapping onto a captured StepGraph."""

    graph_name: str
    fused_nodes: list[FusedNode]
    seq_to_fused: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seq_to_fused:
            for fn in self.fused_nodes:
                for s in fn.original_op_seq_idxs:
                    self.seq_to_fused[s] = fn.node_id


@dataclass
class FusedRegion:
    """A fusion region written into ``StepGraph.fused_regions``: a group of
    captured L0 OpNode ids that the real NPU runs as one GE kernel."""

    region_id: int
    fused_op_type: str
    op_ids: list[int]
    # bytes of intermediate tensors eliminated by fusion (not materialized).
    eliminated_intermediates_bytes: int = 0
    # True when the region is a single unfused op (GE kept it as-is).
    is_unfused: bool = False


def load_fusion_profile_json(path: str | Path) -> GEFusionProfile:
    """Load the portable JSON fusion profile emitted by Phase 1.

    Expected schema::

        {"graph_name": str,
         "fused_nodes": [
           {"node_id": int, "fused_op_type": str,
            "original_op_seq_idxs": [int, ...],
            "input_fused_ids": [int, ...],
            "output_bytes": int}, ...]}
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    fused_nodes = [
        FusedNode(
            node_id=fn["node_id"],
            fused_op_type=fn["fused_op_type"],
            original_op_seq_idxs=list(fn.get("original_op_seq_idxs", [])),
            input_fused_ids=list(fn.get("input_fused_ids", [])),
            output_bytes=int(fn.get("output_bytes", 0)),
        )
        for fn in raw.get("fused_nodes", [])
    ]
    return GEFusionProfile(graph_name=raw.get("graph_name", ""), fused_nodes=fused_nodes)


def extract_graph_topology(graph: Any) -> list[FusedNode]:
    """Walk a ``ge.graph.Graph`` and recover the fused-node topology: node
    type, data-edge producers (``input_fused_ids`` by node index), output
    bytes (best-effort from the first output tensor attr).

    ``.air`` does not carry original->fused op attribution, so
    ``original_op_seq_idxs`` is left empty here -- populate via the JSON
    profile if you need StepGraph mapping.
    """
    nodes = graph.get_all_nodes()
    name_to_idx = {n.name: i for i, n in enumerate(nodes)}
    out: list[FusedNode] = []
    for i, n in enumerate(nodes):
        ntype = n.type
        if ntype in ("Data", "Const", "NetOutput", "RefData", "AippData"):
            continue  # I/O/constant leaf, not a fusion region
        ins: list[int] = []
        for port in range(n.get_inputs_size()):
            try:
                prod, _ = n.get_in_data_nodes_and_port_indexes(port)
            except RuntimeError:
                continue
            j = name_to_idx.get(prod.name)
            if j is not None and j != i:
                ins.append(j)
        out_bytes = 0
        try:
            oattr = n.get_output_attr(0)
            if oattr is not None:
                shape = getattr(oattr, "shape", None) or []
                dt = getattr(oattr, "data_type", None)
                out_bytes = _tensor_bytes(shape, dt)
        except Exception:
            pass
        out.append(FusedNode(node_id=i, fused_op_type=ntype, input_fused_ids=ins,
                             output_bytes=out_bytes))
    return out


def load_fusion_profile_air(path: str | Path) -> GEFusionProfile:
    """Load a GE ``.air`` graph and extract its fused-node topology. ``ge`` is
    imported lazily so this module stays importable without CANN installed."""
    import ge  # noqa: F401  (lazy; triggers ge.graph availability)
    from ge.graph import Graph

    g = Graph(str(Path(path).stem))
    g.load_from_air(str(path))
    fused_nodes = extract_graph_topology(g)
    return GEFusionProfile(graph_name=g.name, fused_nodes=fused_nodes)


def apply_ge_fusion_profile(step_graph: Any, profile: GEFusionProfile) -> list[FusedRegion]:
    """Map a GE fusion profile onto a captured ``StepGraph`` and populate its
    ``fused_regions`` field. Each L0 ``OpNode`` is assigned to exactly one
    region: ops listed in a ``FusedNode.original_op_seq_idxs`` join that fused
    region; any unattributed op becomes a single-op unfused region.

    Returns the list of regions (also assigned to ``step_graph.fused_regions``).
    """
    nodes = step_graph.nodes
    seq_to_opid = {n.seq_idx: n.op_id for n in nodes.values()}
    # regions keyed by fused node id (or op_id for unfused singletons)
    regions: list[FusedRegion] = []
    by_fused: dict[int, FusedRegion] = {}
    for fn in profile.fused_nodes:
        op_ids = [seq_to_opid[s] for s in fn.original_op_seq_idxs if s in seq_to_opid]
        if not op_ids:
            continue
        r = FusedRegion(region_id=len(regions), fused_op_type=fn.fused_op_type,
                        op_ids=op_ids, is_unfused=len(op_ids) == 1)
        by_fused[fn.node_id] = r
        regions.append(r)
    # intermediate elimination = sum of output bytes of region-internal edges
    # (edges whose producer & consumer are both in the same region).
    succ = {n.op_id: list(n.successors) for n in nodes.values()}
    opid_to_region = {oid: r.region_id for r in regions for oid in r.op_ids}
    for r in regions:
        internal = 0
        for oid in r.op_ids:
            for s in succ.get(oid, []):
                if opid_to_region.get(s) == r.region_id:
                    n = nodes[oid]
                    internal += getattr(n, "peak_mem", 0)
        r.eliminated_intermediates_bytes = internal
    # unattributed ops -> singleton unfused regions (preserved as-is by GE)
    attributed = {oid for r in regions for oid in r.op_ids}
    for n in nodes.values():
        if n.op_id in attributed:
            continue
        r = FusedRegion(region_id=len(regions), fused_op_type=n.op_type,
                        op_ids=[n.op_id], is_unfused=True)
        regions.append(r)
        attributed.add(n.op_id)
    step_graph.fused_regions = regions
    return regions


def fusion_summary(step_graph: Any) -> dict[str, Any]:
    """Before/after fusion statistics for a StepGraph that has had
    ``apply_ge_fusion_profile`` applied."""
    nodes = step_graph.nodes
    regions = getattr(step_graph, "fused_regions", []) or []
    fused = [r for r in regions if not r.is_unfused]
    eliminated = sum(r.eliminated_intermediates_bytes for r in fused)
    return {
        "step_id": step_graph.step_id,
        "step_type": step_graph.step_type,
        "l0_nodes": len(nodes),
        "fused_regions": len(regions),
        "fused_multi_op_regions": len(fused),
        "unfused_singletons": len(regions) - len(fused),
        "compression_ratio": (len(nodes) / len(regions)) if regions else 0.0,
        "eliminated_intermediates_bytes": eliminated,
        "fused_op_types": {r.fused_op_type for r in fused},
    }


def _tensor_bytes(shape: Any, dtype: Any) -> int:
    numel = 1
    for d in (shape or []):
        numel *= int(d)
    # map common GE DataType enum value->bytes; fall back to 4 (float32)
    sizes = {0: 4, 1: 8, 2: 2, 3: 1, 4: 1, 6: 4, 7: 4, 9: 8, 10: 4, 11: 1, 12: 2}
    return numel * sizes.get(getattr(dtype, "value", None) if dtype is not None else None, 4)
