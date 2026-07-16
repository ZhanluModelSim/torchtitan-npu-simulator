# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GE graph-mode fusion: achieve the fusion effect in a pure no-NPU
environment by reproducing GE's fusion behavior on the captured L0 DAG.

Why a host-only pass (not running real GE)
------------------------------------------
The simulator captures L0 ops under meta-device ``__torch_dispatch__``, which
yields an *over-decomposed* graph (~75% of nodes are aten primitives like
``aten.mul``/``aten.unsqueeze`` that a real NPU's Graph Engine fuses away).
GE's op fusion happens inside the ``aclgrph`` compile pipeline. We verified
that this pipeline is *partially* accessible host-only:
  * ``build_initialize({"ge.socVersion": "Ascend910B"})`` succeeds without NPU.
  * ``DUMP_GE_GRAPH=1`` dumps every compile stage (PreRunBegin ->
    AfterBuiltinFusionPass -> OptimizeOriginalGraph -> ... -> BeforeBuild) as
    parseable proto-text.
BUT real GE fusion still cannot be harvested host-only:
  * ``build_model`` fails at device-kernel codegen for real ops (matmul/norm).
  * The BuiltinFusionPass / OptimizeOriginalGraph passes do NOT trigger fusion
    on ``ge.es``-built graphs -- verified across all 10 stage dumps (residual
    Add+LayerNorm stayed separate at every stage).
So we reproduce GE's fusion *behavior* via a host-only graph-rewrite on the
captured DAG, targeting GE's authentic fused op types from the ``ge.es.nn``
catalog (AddRmsNorm, AddLayerNorm, FusedMatMul, AdamApplyOneWithDecay). This
delivers the fusion EFFECT (compression + eliminated intermediates) with zero
NPU dependence.

Two consumption modes
---------------------
* Host-only pass (``build_ge_fusion_profile``): the primary pure-no-NPU path.
  Walks the captured StepGraph, pattern-matches GE fusion rules, emits a
  GEFusionProfile with authentic fused op types.
* Optional loaders (``load_fusion_profile_json``/``_air``): cross-check a
  profile captured elsewhere (e.g. a real-NPU GE dump) against the host pass.
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


# ---------------------------------------------------------------------------
# Host-only fusion pass (GE-catalog-grounded)
# ---------------------------------------------------------------------------
# Real GE fusion cannot run host-only (build_model fails at device-kernel
# codegen for real ops, and the BuiltinFusionPass / OptimizeOriginalGraph
# passes do not trigger on ge.es-built graphs -- verified across all compile-
# stage dumps). So the pure-no-NPU path reproduces GE's fusion *behavior* via a
# graph-rewrite on the captured L0 DAG, targeting GE's authentic fused op
# types from the ge.es.nn catalog (AddRmsNorm, AddLayerNorm, FusedMatMul,
# AdamApplyOneWithDecay). This gives the fusion EFFECT (compression +
# eliminated intermediates) without any NPU hardware or offline capture.

# captured L0 op_type -> classification
_NORM_OPS = {"rms_norm", "layer_norm"}
_MATMUL_OPS = {"matmul", "bmm"}
# elementwise / shape / reduction primitives that GE folds into fusion regions
_FUSIBLE = {
    "unknown", "view", "reshape", "transpose", "cat", "split", "gelu", "silu",
    "swiglu", "moe_re_routing", "einsum",
}
# optimizer foreach sub-ops (aten primitives of _fused_adamw_ decomposition)
_FOREACH_OPT = {"zeros", "zeros_like", "zero_", "mean", "sub", "sign", "mul",
                "add", "add_", "select", "floor_divide", "_to_copy", "clone",
                "detach", "unsqueeze"}
# GE fused-op-type targets (authentic names from the ge.es.nn catalog)
_GE_ADD_RMS = "AddRmsNorm"
_GE_ADD_LN = "AddLayerNorm"
_GE_FUSED_MM = "FusedMatMul"
_GE_ELT = "ElementwiseFusion"
_GE_ADAMW = "AdamApplyOneWithDecay"


def _raw(n: Any) -> str:
    return str(n.annotations.get("raw_op_type", ""))


def build_ge_fusion_profile(step_graph: Any) -> GEFusionProfile:
    """Host-only fusion pass: walk the captured StepGraph DAG and emit a
    GEFusionProfile whose fused op types are GE's authentic catalog names.

    Patterns (GE fusion behavior):
      * residual Add + rms_norm/layer_norm -> AddRmsNorm / AddLayerNorm
      * matmul/bmm + trailing bias-Add -> FusedMatMul
      * consecutive single-consumer elementwise/shape chains -> ElementwiseFusion
      * optimizer foreach-adamw sub-ops -> AdamApplyOneWithDecay
    Anchors (attention/rope/softmax/comm/adamw_step) are kept as single-op
    regions (GE preserves them).
    """
    nodes = {n.op_id: n for n in step_graph.nodes.values()}
    order = sorted(step_graph.nodes.values(), key=lambda n: n.seq_idx)
    succ = {n.op_id: [int(s) for s in n.successors] for n in order}
    pred = {n.op_id: [int(p) for p in n.predecessors] for n in order}
    visited: set[int] = set()
    fused: list[FusedNode] = []
    run: list[int] = []  # pending elementwise chain

    def flush_run():
        nonlocal run
        if run:
            fused.append(FusedNode(node_id=len(fused), fused_op_type=_GE_ELT,
                                   original_op_seq_idxs=[nodes[o].seq_idx for o in run]))
            run = []

    def single_succ(oid: int) -> int | None:
        s = succ.get(oid, [])
        return s[0] if len(s) == 1 else None

    for n in order:
        if n.op_id in visited:
            continue
        t = n.op_type
        if t in _NORM_OPS:
            # residual Add + norm fusion: merge immediate elementwise "add"
            # predecessor that has this norm as its single successor.
            add_pred = None
            for p in pred.get(n.op_id, []):
                pn = nodes.get(p)
                if pn is None or pn.op_id in visited:
                    continue
                if pn.op_type in _FUSIBLE and "add" in _raw(pn) and single_succ(p) == n.op_id:
                    add_pred = p
                    break
            flush_run()
            seqs = [nodes[add_pred].seq_idx, n.seq_idx] if add_pred else [n.seq_idx]
            fused.append(FusedNode(node_id=len(fused),
                                   fused_op_type=_GE_ADD_RMS if t == "rms_norm" else _GE_ADD_LN,
                                   original_op_seq_idxs=seqs))
            visited.add(n.op_id)
            if add_pred:
                visited.add(add_pred)
            continue
        if t in _MATMUL_OPS:
            # matmul + trailing bias-Add -> FusedMatMul
            flush_run()
            bias_s = None
            for s in succ.get(n.op_id, []):
                sn = nodes.get(s)
                if sn is None or sn.op_id in visited:
                    continue
                if sn.op_type in _FUSIBLE and "add" in _raw(sn) and len(pred.get(s, [])) <= 2:
                    bias_s = s
                    break
            seqs = [n.seq_idx] + ([nodes[bias_s].seq_idx] if bias_s else [])
            fused.append(FusedNode(node_id=len(fused), fused_op_type=_GE_FUSED_MM,
                                   original_op_seq_idxs=seqs))
            visited.add(n.op_id)
            if bias_s:
                visited.add(bias_s)
            continue
        # optimizer foreach sub-op block
        if step_graph.step_type.upper() == "OPTIMIZER" and (
                t in _FOREACH_OPT or (t == "unknown" and _raw(n) in _FOREACH_OPT)):
            run.append(n.op_id)
            visited.add(n.op_id)
            continue
        if t in _FUSIBLE:
            run.append(n.op_id)
            visited.add(n.op_id)
            continue
        # anchor: single-op region
        flush_run()
        fused.append(FusedNode(node_id=len(fused), fused_op_type=t,
                               original_op_seq_idxs=[n.seq_idx]))
        visited.add(n.op_id)
    # optimizer foreach run -> single AdamApplyOneWithDecay
    if run and step_graph.step_type.upper() == "OPTIMIZER":
        fused.append(FusedNode(node_id=len(fused), fused_op_type=_GE_ADAMW,
                               original_op_seq_idxs=[nodes[o].seq_idx for o in run]))
        run = []
    flush_run()
    return GEFusionProfile(graph_name=step_graph.step_id, fused_nodes=fused)
