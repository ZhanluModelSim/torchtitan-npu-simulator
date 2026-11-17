# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One NPU profiling calibration round for CV-chunk FX scheduling."""

from __future__ import annotations

import json
import operator
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    is_c10d_functional_node,
)
from torchtitan.tools.logging import logger

from .cann_kernel_metrics import CannKernelMetrics, classify_mixed_core_times
from .cann_launches import associate_cann_launches
from .cann_profiler import cann_profile, isolated_cann_profiler_work_path, parse_cann_profile
from .profile_overlap import SCOPE, summarize_profile

if TYPE_CHECKING:
    from pathlib import Path

_TAG_NAMESPACE = "cv_parallel"
_TAG_VERSION = 1

# These are device-side synchronization/control tasks rather than useful work
# that a compute node can contribute to hiding communication.  Collective cost
# intentionally follows a different rule below: its full top-level HCOM
# launch-to-ready interval must retain NOTIFY_WAIT/NOTIFY_RECORD and protocol
# synchronization because that entire dependency latency is overlapable.
_DEVICE_CONTROL_TASK_PREFIXES = (
    "EVENT_",
    "NOTIFY_",
    "PLACE_HOLDER",
    "PROFILING_",
    "STREAM_",
    "WRITE_VALUE",
)
_DEVICE_CONTROL_TASK_TYPES = {"COMMUNICATION"}

_DEEPEP_COMMUNICATION_SCHEMAS = frozenset(
    {"deepep::dispatch", "deepep::combine", "deepep::dispatch_backward", "deepep::combine_backward"}
)


@dataclass(frozen=True)
class RankLocalProfileCosts:
    """Rank-local costs keyed by stable FX node identity."""

    local_ms: dict[str, float]
    local_resources: dict[str, str] = field(default_factory=dict)


_AIC_RESOURCE = "aic"
_AIV_RESOURCE = "aiv"
_MIX_RESOURCE = "mix"
_OTHER_RESOURCE = "other"
_NON_CV_RESOURCE = "non_cv"


_ALWAYS_METADATA_ONLY_TARGETS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.sym_size.int,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unbind.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
    torch.ops.aten.view.dtype,
    torch.ops.aten._unsafe_view.default,
}


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return dict(custom) if isinstance(custom, dict) else {}


def _metadata_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    return tuple(leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor))


def _reshape_shares_input_storage(node: fx.Node) -> bool:
    if not node.args or not isinstance(node.args[0], fx.Node):
        return False
    inputs = _metadata_tensors(node.args[0].meta.get("val"))
    outputs = _metadata_tensors(node.meta.get("val"))
    if len(inputs) != 1 or len(outputs) != 1:
        return False
    try:
        if inputs[0].untyped_storage()._cdata == outputs[0].untyped_storage()._cdata:
            return True
        # FX metadata propagation can recreate independent fake storages.
        # A valid reshape of a contiguous strided tensor still cannot copy.
        return inputs[0].layout == torch.strided and inputs[0].is_contiguous()
    except (AttributeError, NotImplementedError, RuntimeError):
        return False


def _is_metadata_only_compute_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target is operator.getitem:
        if len(node.args) != 2 or node.kwargs or type(node.args[1]) is not int:
            return False
        source = node.args[0]
        # Only tuple/list unpacking is host-only; Tensor indexing may launch
        # device work and must retain its input completion wait.
        return isinstance(source, fx.Node) and isinstance(source.meta.get("val"), (tuple, list))
    if node.target in _ALWAYS_METADATA_ONLY_TARGETS:
        return True
    return node.target == torch.ops.aten.reshape.default and _reshape_shares_input_storage(node)


def assign_stable_node_tags(gm: fx.GraphModule) -> dict[str, fx.Node]:
    """Preserve source-node identity and order through event emission and reordering."""
    tagged: dict[str, fx.Node] = {}
    for ordinal, node in enumerate(gm.graph.nodes):
        node_id = f"cv_v{_TAG_VERSION}_g0000_n{ordinal:06d}"
        custom = _custom_meta(node)
        metadata = custom.get(_TAG_NAMESPACE)
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(
            {
                "node_id": node_id,
                "canonical_ordinal": ordinal,
            }
        )
        custom[_TAG_NAMESPACE] = metadata
        node.meta["custom"] = custom
        tagged[node_id] = node
    return tagged


def node_id(node: fx.Node) -> str | None:
    metadata = _custom_meta(node).get(_TAG_NAMESPACE)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("node_id")
    return value if isinstance(value, str) else None


def canonical_order(gm: fx.GraphModule) -> dict[fx.Node, int]:
    result: dict[fx.Node, int] = {}
    for node in gm.graph.nodes:
        metadata = _custom_meta(node).get(_TAG_NAMESPACE)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"FX node {node.name} has no auto-overlap tag")
        ordinal = metadata.get("canonical_ordinal")
        if not isinstance(ordinal, int):
            raise RuntimeError(f"FX node {node.name} has no canonical ordinal")
        result[node] = ordinal
    return result


def _schema_arguments(node: fx.Node) -> dict[str, Any]:
    """Bind an OpOverload's positional/keyword arguments without executing it."""
    schema = getattr(node.target, "_schema", None)
    arguments = getattr(schema, "arguments", ())
    bound = {argument.name: value for argument, value in zip(arguments, node.args, strict=False)}
    bound.update(node.kwargs)
    return bound


def _collective_type(node: fx.Node) -> str:
    schema = getattr(node.target, "_schema", None)
    name = getattr(schema, "name", None)
    return str(name) if name is not None else str(node.target)


def is_deepep_communication_node(node: fx.Node) -> bool:
    schema = getattr(getattr(node, "target", None), "_schema", None)
    return getattr(schema, "name", None) in _DEEPEP_COMMUNICATION_SCHEMAS


def _collective_group(node: fx.Node) -> str:
    # Opaque DeepEP handles are not comparable across ranks; retain one
    # conservative logical group for all communication stages.
    if is_deepep_communication_node(node):
        return "deepep"
    bound = _schema_arguments(node)
    for name in ("group_name", "group", "tag"):
        if name in bound:
            return str(bound[name])
    # Functional collectives normally expose group_name in their schema. Keep
    # an explicit fallback so an unsupported/new schema forms its own group
    # instead of being silently merged with every other collective.
    return f"unknown:{_collective_type(node)}"


def _is_collective_launch(node: fx.Node) -> bool:
    return is_c10d_functional_node(node) and node.target != torch.ops._c10d_functional.wait_tensor.default


def collective_order_keys(gm: fx.GraphModule) -> dict[fx.Node, tuple[int, str, int]]:
    """Identify collectives in source order before any schedule permutation."""
    ordinals: dict[str, int] = {}
    counts: dict[tuple[int, str], int] = defaultdict(int)
    keys = {}
    for node in sorted(gm.graph.nodes, key=canonical_order(gm).__getitem__):
        if _is_collective_launch(node) or is_deepep_communication_node(node):
            # First appearance identifies local groups independently of hashes.
            group = ordinals.setdefault(_collective_group(node), len(ordinals))
            collective = _collective_type(node)
            keys[node] = (group, collective, counts[group, collective])
            counts[group, collective] += 1
    return keys


def _collective_count_manifest(gm: fx.GraphModule) -> tuple[tuple[int, str, int], ...]:
    """Count FX collective launches by logical local group and collective type."""
    counts = Counter((group, collective) for group, collective, _ in collective_order_keys(gm).values())
    return tuple(sorted((group, collective, count) for (group, collective), count in counts.items()))


def _validate_collective_counts_across_ranks(gm: fx.GraphModule) -> None:
    """Require only equal per-group/per-type FX communication stage counts."""
    import torch.distributed as dist

    manifest = _collective_count_manifest(gm)
    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[tuple[tuple[int, str, int], ...] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, manifest)
    if any(item != manifest for item in gathered):
        raise RuntimeError(f"CV parallel FX collective counts differ across ranks: manifests={gathered}")


# CompositeImplicitAutograd operators may not emit a CPU scope bearing their
# registered schema name. Keep the expansion aliases explicit and small: an
# alias is accepted only as part of the full-graph ordered match below.
_HOST_OP_ALIASES = {
    "torchtitan::deterministic_scatter_add": ("aten::scatter_add",),
    # Some native captures expose only the wrapped NPU dispatcher scope.
    "torchtitan_npu::npu_moe_token_permute": ("npu::npu_moe_token_permute",),
}


def _host_op_names(node: fx.Node) -> tuple[str, ...]:
    """Return native profiler CPU scope names expected for one FX call."""
    target = node.target
    schema = getattr(target, "_schema", None)
    schema_name = getattr(schema, "name", None)
    if isinstance(schema_name, str):
        return (schema_name, *_HOST_OP_ALIASES.get(schema_name, ()))
    if node.op == "call_method" and isinstance(target, str):
        return (f"aten::{target}",)
    return ()


def _match_fx_nodes_to_host_scopes(
    trace: dict[str, Any] | list[Any],
    gm: fx.GraphModule,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Match FX calls to native CPU scopes without instrumenting the graph.

    The generated FX forward executes calls in graph order on one host lane.
    CANN already records those dispatcher scopes and ``torch_to_npu`` flows.
    Match the complete FX call sequence (not only the scheduler cost domain),
    since skipping unrelated calls could consume an identically named later
    operation. Device execution order is deliberately not used here.

    Unsupported calls remain unmatched without shifting subsequent identities.
    Their costs are estimated from local measurements by the planner.
    """
    raw_events = trace.get("traceEvents", ()) if isinstance(trace, dict) else trace
    events = [event for event in raw_events if isinstance(event, dict)]
    specs: list[tuple[str, tuple[str, ...]]] = []
    for node in gm.graph.nodes:
        stable_id = node_id(node)
        names = _host_op_names(node)
        if stable_id is not None and names:
            specs.append((stable_id, names))
    specs_by_names: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for stable_id, names in specs:
        specs_by_names[tuple(names)].append(stable_id)
    by_lane: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if (
            event.get("ph") == "X"
            and event.get("cat") == "cpu_op"
            and event.get("name") != SCOPE
            and not str(event.get("name", "")).startswith("ProfilerStep#")
            and "ts" in event
            and "dur" in event
        ):
            by_lane[event.get("pid"), event.get("tid")].append(event)

    def match_lane(
        host_events: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        intervals = sorted(
            ((Decimal(str(event["ts"])), Decimal(str(event["dur"])), event) for event in host_events),
            key=lambda item: (item[0], -item[1]),
        )
        # Native operators include their implementation's dispatcher scopes
        # (reshape -> view, index_put -> clone, etc.). Only outer calls are
        # candidates for the flat FX forward; retain the original ranges so
        # device association still includes launches from all descendants.
        outer_events = []
        outer_end = Decimal("-Infinity")
        for begin, duration, event in intervals:
            end = begin + duration
            if begin < outer_end and end <= outer_end:
                continue
            outer_events.append(event)
            outer_end = end
        host_events = outer_events
        positions: dict[str, list[int]] = defaultdict(list)
        for index, event in enumerate(host_events):
            positions[str(event.get("name"))].append(index)
        cursor = 0
        matched: dict[str, dict[str, Any]] = {}
        missing: set[str] = set()
        for stable_id, names in specs:
            choices = []
            for name in names:
                indices = positions.get(name, ())
                offset = bisect_left(indices, cursor)
                if offset < len(indices):
                    choices.append(indices[offset])
            if not choices:
                missing.add(stable_id)
                continue
            index = min(choices)
            matched[stable_id] = host_events[index]
            cursor = index + 1
        # A greedy ordered match cannot identify which repeated FX node owns a
        # host scope when that scope name's count differs (for example because
        # a composite op omitted its outer scope or emitted an identical inner
        # scope). Drop the complete ambiguous name group instead of silently
        # shifting every later node identity onto the wrong CANN task.
        for names, stable_ids in specs_by_names.items():
            host_count = len({index for name in names for index in positions.get(name, ())})
            if host_count == len(stable_ids):
                continue
            for stable_id in stable_ids:
                matched.pop(stable_id, None)
                missing.add(stable_id)
        return matched, [stable_id for stable_id, _ in specs if stable_id in missing]

    candidates = []
    for lane, host_events in by_lane.items():
        matched, missing = match_lane(host_events)
        candidates.append((len(matched), lane, matched, missing))
    if not candidates:
        return {}, {"expected": len(specs), "matched": 0, "missing": len(specs), "lane": None}
    candidates.sort(key=lambda item: item[0], reverse=True)
    count, lane, matched, missing = candidates[0]
    return matched, {
        "expected": len(specs),
        "matched": count,
        "missing": len(missing),
        "lane": lane,
        "missing_examples": missing[:8],
    }


def _device_interval_union_us(events: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (
            Decimal(str(event["ts"])),
            Decimal(str(event["ts"])) + Decimal(str(event["dur"])),
        )
        for event in events
    )
    if not intervals:
        return 0.0
    total = Decimal(0)
    begin, end = intervals[0]
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return float(total + end - begin)


def _is_non_collective_device_work(event: dict[str, Any]) -> bool:
    """Keep kernels and copies, but reject synchronization/control events.

    CANN async flows can associate a native host scope with both its actual AIC/AIV,
    AICPU or copy task and an EVENT_WAIT inserted to satisfy an earlier stream
    dependency.  The wait is real in the profiled schedule, but assigning it to
    the following compute node would falsely turn (for example) a 70-us Mul
    into a 20-ms movable compute node.  Task types vary slightly between CANN
    versions, so inspect both the structured type and the event name.

    This deliberately uses a control-task denylist instead of a kernel
    allowlist: future full-graph scheduling must continue to measure new device
    kernel and memcpy task types rather than silently treating them as zero.
    """
    # The caller must first establish device provenance. Keep this additional
    # guard so host queue ranges never become movable compute if reused.
    if str(event.get("cat", "")).lower() in {"cpu_op", "python_function", "enqueue", "dequeue"}:
        return False
    name = str(event.get("name", "")).upper()
    args = event.get("args", {})
    task_type = ""
    if isinstance(args, dict):
        task_type = str(args.get("Task Type", args.get("task_type", ""))).upper()

    # A nested/accidentally associated communication activity is not compute
    # work. Actual collective launch nodes are handled by the HCOM path.
    if name.startswith("HCOM_"):
        return False
    if task_type in _DEVICE_CONTROL_TASK_TYPES:
        return False
    return not any(
        value.startswith(prefix) for value in (task_type, name) for prefix in _DEVICE_CONTROL_TASK_PREFIXES if value
    )


def _device_work_resource(event: dict[str, Any]) -> str:
    """Classify one associated CANN device task conservatively."""
    args = event.get("args", {})
    task_type = str(args.get("Task Type", args.get("task_type", ""))) if isinstance(args, dict) else ""
    # Kernel-name substrings such as "DMA" in GroupedMatmul do not establish
    # data movement; only explicit hardware task types release compute units.
    if task_type.upper() in {"MEMCPY", "SDMA", "DMA", "ASYNC_MEMCPY", "MEMCPY_ASYNC", "AI_CPU", "AICPU"}:
        return _NON_CV_RESOURCE
    descriptor = f"{task_type} {event.get('name', '')}".upper()

    def has_token(*tokens: str) -> bool:
        return any(re.search(rf"(?:^|[^A-Z0-9]){re.escape(token)}(?:[^A-Z0-9]|$)", descriptor) for token in tokens)

    # A raw MIX task type cannot prove that the secondary core is idle.
    # Only per-launch counters may refine the MIX classification.
    if has_token("MIX_AIC", "MIX_AIV", "MIX"):
        return _MIX_RESOURCE
    if has_token("AI_VECTOR_CORE", "AIVEC", "AIV", "VECTOR_CORE"):
        return _AIV_RESOURCE
    if has_token("AI_CORE", "AICORE", "AIC", "CUBE"):
        return _AIC_RESOURCE
    return _OTHER_RESOURCE


def _combined_device_resource(events: list[dict[str, Any]], *, kernel_metrics: CannKernelMetrics | None = None) -> str:
    resources = {_device_work_resource(event) for event in events}
    if not resources:
        return _OTHER_RESOURCE
    resources.discard(_NON_CV_RESOURCE)
    if not resources:
        return _NON_CV_RESOURCE
    # Explicit DMA/AICPU tasks do not reserve compute units. Unrecognized
    # tasks remain conservative, including when paired with a known kernel.
    if _OTHER_RESOURCE in resources:
        return _OTHER_RESOURCE if resources == {_OTHER_RESOURCE} else _MIX_RESOURCE
    if resources == {_AIC_RESOURCE}:
        return _AIC_RESOURCE
    if resources == {_AIV_RESOURCE}:
        return _AIV_RESOURCE
    # A separate pure-Cube launch cannot leave the Cube chain merely because
    # the surrounding FX node spends more time on Vector work.
    if _AIC_RESOURCE in resources:
        return _MIX_RESOURCE
    if kernel_metrics is not None:
        return (
            classify_mixed_core_times(
                [
                    kernel_metrics.core_metrics(event)
                    for event in events
                    if _device_work_resource(event) != _NON_CV_RESOURCE
                ]
            )
            or _MIX_RESOURCE
        )
    return _MIX_RESOURCE


class _DeviceTaskIndex:
    """Exact launch lookup with a lazy, nested-interval index per device lane."""

    def __init__(self, events: list[dict[str, Any]]):
        self.by_start: dict[tuple[Any, Any, Decimal], list[dict[str, Any]]] = defaultdict(list)
        self.by_lane: dict[tuple[Any, Any], list[tuple[Decimal, dict[str, Any]]]] = defaultdict(list)
        self.ranges: dict[tuple[Any, Any], tuple[list[Decimal], list[Decimal], list[Decimal]]] = {}
        for event in events:
            lane = event.get("pid"), event.get("tid")
            start = Decimal(str(event["ts"]))
            self.by_start[(*lane, start)].append(event)
            self.by_lane[lane].append((start, event))

    def lookup(self, finish: dict[str, Any]) -> list[dict[str, Any]]:
        lane = finish.get("pid"), finish.get("tid")
        timestamp = Decimal(str(finish["ts"]))
        exact = self.by_start.get((*lane, timestamp))
        if exact is not None:
            return exact
        entries = self.by_lane.get(lane, [])
        if lane not in self.ranges:
            entries.sort(key=lambda item: item[0])
            starts = [start for start, _ in entries]
            ends = [start + Decimal(str(event["dur"])) for start, event in entries]
            prefix_ends = []
            latest = Decimal("-Infinity")
            for end in ends:
                latest = max(latest, end)
                prefix_ends.append(latest)
            self.ranges[lane] = starts, ends, prefix_ends
        starts, ends, prefix_ends = self.ranges[lane]
        index = bisect_right(starts, timestamp) - 1
        result = []
        # Prefix maxima retain enclosing HCOM/kernel ranges when a shorter
        # nested task has already ended. A nearest-start lookup would miss them.
        while index >= 0 and prefix_ends[index] > timestamp:
            if ends[index] > timestamp:
                result.append(entries[index][1])
            index -= 1
        return result


def _extract_cann_node_measurements(
    trace: dict[str, Any] | list[Any],
    profile_nodes: dict[str, fx.Node],
    node_scopes: dict[str, dict[str, Any]],
    *,
    kernel_metrics: CannKernelMetrics | None = None,
    kernel_observations: dict[str, list[tuple[float, float, str, str]]] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Map native CPU dispatcher scopes to device work, never host dequeue.

    One host scope can own both enqueue_to_dequeue and torch_to_npu flows.
    The former ends at a CPU Dequeue@aclnn* range (including host-side waits),
    not at a device task. Follow only the direct torch_to_npu flow and verify
    its destination against process metadata before measuring any interval.
    """
    raw_events = trace.get("traceEvents", ()) if isinstance(trace, dict) else trace
    events = [event for event in raw_events if isinstance(event, dict)]
    complete = [event for event in events if event.get("ph") == "X" and "ts" in event and "dur" in event]
    device_pids = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") in {"Ascend Hardware", "Communication"}
    }
    if not device_pids:
        # Host activity cannot substitute for unidentified device work.
        logger.warning("CV parallel CANN device process metadata missing; no device costs extracted")
        return {}, {}
    origin = min(
        (Decimal(str(event["ts"])) for event in complete if event.get("pid") in device_pids), default=Decimal(0)
    )

    def flow_key(event):
        # IDs are not assumed unique across flow categories/names.
        return event.get("cat"), event.get("name"), event.get("id")

    flows = [
        event
        for event in events
        if event.get("cat") == "async_npu" and event.get("name") == "torch_to_npu" and event.get("id") is not None
    ]
    # torch_npu emits adjacent (start, finish) pairs and uses device timestamp
    # as flow ID. Different streams can have the same ID: a dict keyed only
    # by ID would overwrite one finish and assign another stream's task.
    flow_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in flows:
        flow_groups[flow_key(event)].append(event)
    flow_finishes: dict[int, dict[str, Any]] = {}
    for index, start in enumerate(flows):
        if start.get("ph") != "s":
            continue
        following = flows[index + 1] if index + 1 < len(flows) else {}
        if following.get("ph") == "f" and flow_key(following) == flow_key(start):
            flow_finishes[id(start)] = following
        else:
            group = flow_groups[flow_key(start)]
            if len(group) == 2:
                finishes = [event for event in group if event.get("ph") == "f"]
                if len(finishes) == 1:
                    flow_finishes[id(start)] = finishes[0]
    starts_by_lane: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in flows:
        if event.get("ph") == "s" and "ts" in event:
            starts_by_lane.setdefault((event.get("pid"), event.get("tid")), []).append(event)
    start_times_by_lane: dict[tuple[Any, Any], list[Decimal]] = {}
    for lane, starts in starts_by_lane.items():
        starts.sort(key=lambda event: Decimal(str(event["ts"])))
        start_times_by_lane[lane] = [Decimal(str(event["ts"])) for event in starts]
    # A flow finish timestamp is the associated device task's start. Index by
    # device lane and an exact timestamp, then retain a containment fallback
    # for profiler versions that stringify timestamps at different precision.
    # Float epoch timestamps lose sub-us precision. In particular a short
    # EVENT_WAIT can collapse onto the following kernel's start timestamp.
    device_tasks = _DeviceTaskIndex(
        [
            event
            for event in complete
            if event.get("pid") in device_pids
            and str(event.get("cat", "")).lower() not in {"cpu_op", "python_function", "enqueue", "dequeue"}
        ]
    )

    costs: dict[str, float] = {}
    resources: dict[str, str] = {}
    missing_device_work: list[str] = []
    cann_launches = associate_cann_launches(events, node_scopes)
    for stable_id, node in profile_nodes.items():
        assert stable_id == node_id(node)
        host_scope = node_scopes.get(stable_id)
        if _is_metadata_only_compute_node(node):
            costs[stable_id] = 0.0
            resources[stable_id] = _OTHER_RESOURCE
            continue
        if host_scope is None:
            # Python-only graph plumbing such as operator.getitem has neither
            # a dispatcher CPU scope nor device work. It is deliberately not
            # part of host-order coverage, so do not emit thousands of false
            # missing-scope warnings. A node with an expected native scope is
            # a real mapping failure and remains visible.
            if _host_op_names(node):
                logger.warning(
                    "CV parallel CANN host scope missing: node_id=%s node=%s target=%s",
                    stable_id,
                    node.name,
                    node.target,
                )
            continue
        begin = Decimal(str(host_scope["ts"]))
        end = begin + Decimal(str(host_scope["dur"]))
        associated: list[dict[str, Any]] = []
        lane = host_scope.get("pid"), host_scope.get("tid")
        start_times = start_times_by_lane.get(lane, ())
        starts = starts_by_lane.get(lane, ())
        # Index host launches by lane/time rather than scanning every flow for
        # every FX node. This matters for full-model traces with 10k+ scopes.
        for start in starts[bisect_left(start_times, begin) : bisect_left(start_times, end)]:
            finish = flow_finishes.get(id(start))
            if finish is None or "ts" not in finish:
                continue
            associated.extend(device_tasks.lookup(finish))
        associated.extend(cann_launches.get(stable_id, ()))
        associated = list({id(event): event for event in associated}.values())

        if _is_collective_launch(node):
            hcom = [event for event in associated if str(event.get("name", "")).lower().startswith("hcom_")]
            if not hcom:
                logger.warning(
                    "CV parallel CANN HCOM association missing: node_id=%s node=%s target=%s associated=%s",
                    stable_id,
                    node.name,
                    node.target,
                    [event.get("name") for event in associated],
                )
                continue
            # HCCL may expose parallel links/nested tasks. The longest HCOM
            # interval represents launch-to-ready communication latency. Keep
            # its NOTIFY_WAIT/NOTIFY_RECORD and protocol time: all of it is a
            # valid window for computation to overlap with the collective.
            cost_us = max(float(event["dur"]) for event in hcom)
            resource = _OTHER_RESOURCE
        else:
            device_work = [event for event in associated if _is_non_collective_device_work(event)]
            if not device_work:
                # No mapping is not evidence of zero runtime. The planner
                # estimates missing costs from this rank's measured work.
                # Proven metadata-only nodes are handled above.
                missing_device_work.append(f"{stable_id}:{node.name}")
                continue
            cost_us = _device_interval_union_us(device_work)
            resource = _combined_device_resource(device_work, kernel_metrics=kernel_metrics)
            if kernel_observations is not None:
                kernel_observations[stable_id] = [
                    (
                        float(Decimal(str(event["ts"])) - origin) / 1000,
                        float(Decimal(str(event["ts"])) - origin + Decimal(str(event["dur"]))) / 1000,
                        str((event.get("pid"), event.get("tid"))),
                        _combined_device_resource([event], kernel_metrics=kernel_metrics),
                    )
                    for event in device_work
                ]
        costs[stable_id] = max(0.0, cost_us / 1000.0)
        resources[stable_id] = resource
    if missing_device_work:
        logger.info(
            "CV parallel CANN nodes without measured device work: "
            "count=%d fallback=rank_local_cost_estimate examples=%s",
            len(missing_device_work),
            missing_device_work[:8],
        )
    return costs, resources


def profile_whole_graph_costs(
    gm: fx.GraphModule,
    run_candidate,
    *,
    profile_root: Path,
    kernel_observations=None,
    profile_name="calibration",
    source_gm=None,
    kernel_summary=None,
) -> RankLocalProfileCosts:
    """Replay an isolated graph and map native dispatcher scopes to device costs.

    Stable FX tags identify nodes across materialization and reordering.
    """
    # Validate the source graph, excluding emitted stream/event operations.
    _validate_collective_counts_across_ranks(gm if source_gm is None else source_gm)
    cpu_rng_before = torch.random.get_rng_state().clone()
    npu_rng_before = torch.npu.get_rng_state().clone()  # pyrefly: ignore [missing-attribute]
    profile_nodes = {
        stable_id: node
        for node in gm.graph.nodes
        if (stable_id := node_id(node)) is not None and node.op in {"call_function", "call_method", "call_module"}
    }
    logger.info(
        "CV parallel whole-graph CANN calibration starting: "
        "graph_nodes=%d profile_cost_nodes=%d "
        "instrumentation=none mapping=fx_host_order_torch_to_npu",
        len(tuple(gm.graph.nodes)),
        len(profile_nodes),
    )
    with (
        torch.random.fork_rng(
            devices=[torch.npu.current_device()],  # pyrefly: ignore [missing-attribute]
            device_type="npu",
        ),
        isolated_cann_profiler_work_path(profile_root, prefix=profile_name) as work_path,
    ):
        # Clone calibration inputs and snapshot mutable state before opening
        # the profiler. Otherwise those safety operations become part of the
        # graph makespan and their dispatcher scopes can shift FX-to-CANN
        # matching. Synchronize once so asynchronous snapshot copies cannot
        # spill into the measured session.
        run_candidate.prepare()
        torch.npu.synchronize()  # pyrefly: ignore [missing-attribute]
        try:
            with cann_profile(), torch.profiler.record_function(SCOPE):
                result = run_candidate(gm)
                torch.npu.synchronize()  # pyrefly: ignore [missing-attribute]
            # Calibration gradients are unused; release them before parsing.
            del result
        finally:
            # State equality checks and buffer restoration may launch their
            # own NPU work, so keep them outside the measured profiler scope.
            run_candidate.finalize()
        trace_path = parse_cann_profile(work_path) / "trace_view.json"
        trace = None
        kernel_metrics = None
        if kernel_summary is not None or profile_nodes:
            with trace_path.open(encoding="utf-8") as trace_file:
                # Preserve sub-us epoch precision before matching launches/overlap.
                trace = json.load(trace_file, parse_float=Decimal)
            kernel_metrics = CannKernelMetrics(work_path)
        if kernel_summary is not None:
            kernel_summary.update(
                summarize_profile(
                    trace_path.parent,
                    trace=trace,
                    kernel_metrics=kernel_metrics,
                )
            )
        if profile_nodes:
            assert trace is not None
            host_scopes, match_stats = _match_fx_nodes_to_host_scopes(trace, gm)
            costs, resources = _extract_cann_node_measurements(
                trace,
                profile_nodes,
                host_scopes,
                kernel_metrics=kernel_metrics,
                kernel_observations=kernel_observations,
            )
        else:
            host_scopes = {}
            costs, resources = {}, {}
            match_stats = {
                "expected": 0,
                "matched": 0,
                "missing": 0,
                "lane": None,
                "missing_examples": (),
            }
        requested_host_ids = {stable_id for stable_id, node in profile_nodes.items() if _host_op_names(node)}
        requested_matched_ids = requested_host_ids & host_scopes.keys()
        logger.info(
            "CV parallel host-order mapping: "
            "expected_fx_host_ops=%d matched_fx_host_ops=%d missing_fx_host_ops=%d "
            "host_lane=%s requested=%d requested_host_ops=%d "
            "requested_host_ops_matched=%d python_only_requested=%d "
            "extracted_costs=%d missing_examples=%s",
            match_stats["expected"],
            match_stats["matched"],
            match_stats["missing"],
            match_stats["lane"],
            len(profile_nodes),
            len(requested_host_ids),
            len(requested_matched_ids),
            len(profile_nodes) - len(requested_host_ids),
            len(costs),
            match_stats.get("missing_examples", ()),
        )
        if requested_matched_ids != requested_host_ids:
            logger.warning(
                "CV parallel host-order mapping missing requested native scopes: "
                "missing=%d examples=%s; unmeasured work uses local fallback costs",
                len(requested_host_ids - requested_matched_ids),
                sorted(requested_host_ids - requested_matched_ids)[:8],
            )
    if not torch.equal(cpu_rng_before, torch.random.get_rng_state()) or not torch.equal(
        npu_rng_before,
        torch.npu.get_rng_state(),  # pyrefly: ignore [missing-attribute]
    ):
        raise RuntimeError("whole-graph calibration changed CPU or NPU RNG state")
    logger.info(
        "CV parallel whole-graph calibration preserved RNG state: cpu=True npu=True measured_nodes=%d",
        len(costs),
    )
    return RankLocalProfileCosts(local_ms=costs, local_resources=resources)
