# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed chunk streams, dependency barriers and profile-guided event emission."""

from __future__ import annotations

import functools
import heapq
import operator
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree
from torchtitan.experiments.graph_trainer.ep_pass_utils import is_c10d_functional_node
from torchtitan.experiments.graph_trainer.passes import construct_default_graph_passes
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline

from . import CV_PARALLEL_PIPELINE
from .batch_chunk import (
    _RUNTIME_CONTEXT,
    _make_run_candidate,
    _requires_full_stream_barrier,
    configure_batch_chunk_passes,
)
from .whole_graph_runtime_profile import (
    _is_metadata_only_compute_node,
    assign_stable_node_tags,
    is_deepep_communication_node,
)


@dataclass
class Schedule:
    order: list[fx.Node]
    lanes: dict[fx.Node, int]
    dependencies: dict[fx.Node, set[fx.Node]]
    start_ms: dict[fx.Node, float]
    end_ms: dict[fx.Node, float]
    algorithm: str = "canonical"


def _is_stream_barrier(node):
    return (
        node.op == "output"
        or is_c10d_functional_node(node)
        or is_deepep_communication_node(node)
        or (_requires_full_stream_barrier(node) and not _is_private_temporary_mutation(node))
    )


def dependency_only_schedule(gm):
    """Alternate ready lanes without resource costs, priorities or resource fences.

    Tensor dependencies, per-lane FIFO and collective/state barriers remain
    mandatory, including the free graph's conservative input completion joins.
    """
    nodes = list(gm.graph.nodes)
    ordinal = {node: index for index, node in enumerate(nodes)}
    lanes = {node: int(node.meta.get("chunk_id") == 1) for node in nodes}
    dependencies = {node: set(node.all_input_nodes) for node in nodes}
    tails = [None, None]
    fence = None
    for node in nodes:
        lane = lanes[node]
        if tails[lane] is not None:
            dependencies[node].add(tails[lane])
        if fence is not None:
            dependencies[node].add(fence)
        if _is_stream_barrier(node):
            dependencies[node].update(tail for tail in tails if tail is not None)
            fence = node
        tails[lane] = node
    successors = {node: [] for node in nodes}
    remaining = {node: len(dependencies[node]) for node in nodes}
    ready = [[], []]
    for node in nodes:
        for parent in dependencies[node]:
            successors[parent].append(node)
        if not remaining[node]:
            heapq.heappush(ready[lanes[node]], ordinal[node])
    order = []
    lane = 0
    while ready[0] or ready[1]:
        if not ready[lane]:
            lane = 1 - lane
        node = nodes[heapq.heappop(ready[lane])]
        order.append(node)
        for child in successors[node]:
            remaining[child] -= 1
            if not remaining[child]:
                heapq.heappush(ready[lanes[child]], ordinal[child])
        lane = 1 - lane
    if len(order) != len(nodes):
        raise ValueError("Free dual-stream graph has a dependency cycle")
    return Schedule(
        order=order,
        lanes=lanes,
        dependencies=dependencies,
        start_ms=dict.fromkeys(nodes, 0.0),
        end_ms=dict.fromkeys(nodes, 0.0),
        algorithm="dependency_only",
    )


def _is_private_temporary_mutation(node):
    """A sole consumer may update its fresh lane-local allocation in place."""
    if node.op != "call_function" or node.target not in {
        torch.ops.aten.scatter_.value,
        torch.ops.aten.index_put_.default,
        torch.ops.aten.logical_and_.default,
        torch.ops.aten.add_.Tensor,
    }:
        return False
    source = node.args[0]
    if not isinstance(source, fx.Node) or source.op != "call_function":
        return False
    schema = getattr(source.target, "_schema", None)
    return (
        schema is not None
        and schema.name.startswith("aten::")
        and not schema.is_mutable
        # _unsafe_view has no schema alias annotation but still shares storage.
        and not _is_metadata_only_compute_node(source)
        and len(schema.returns) == 1
        and schema.returns[0].alias_info is None
        and set(source.users) == {node}
        and source.meta.get("chunk_id") in (0, 1)
        and source.meta.get("chunk_id") == node.meta.get("chunk_id")
    )


# These kernels allocate their outputs and consume explicit tensor inputs on
# the current stream. Keep this list exact: e.g. DeepEP also declares no input
# mutation, but its communication/handle effects are not represented by FX edges.
# Repo wrappers live in ops/ascendc/moe_token_{permute,unpermute}.py; native
# schemas live in torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml.
_REORDERABLE_NPU_SCHEMAS = frozenset(
    {
        "npu::npu_dynamic_block_mx_quant",
        "npu::npu_dynamic_mx_quant",
        "npu::npu_dynamic_mx_quant_with_dual_axis",
        "npu::npu_dynamic_quant",
        "npu::npu_grouped_dynamic_mx_quant",
        "npu::npu_grouped_matmul",
        "npu::npu_quant_matmul",
        "npu::npu_rms_norm",
        "npu::npu_rms_norm_backward",
        "npu::npu_moe_token_permute",
        "npu::npu_moe_token_permute_grad_v2",
        "npu::npu_moe_token_unpermute",
        "npu::npu_moe_token_unpermute_grad",
        "torchtitan_npu::npu_moe_token_permute",
        "torchtitan_npu::npu_moe_token_permute_backward",
        "torchtitan_npu::npu_moe_token_unpermute",
        "torchao_npu::mx_last_dim_fake_quantize",
    }
)


def _is_reorderable_npu_call(node):
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return False
    schema = node.target._schema
    return (
        schema.name in _REORDERABLE_NPU_SCHEMAS
        and not schema.overload_name
        and not schema.is_mutable
        and all(arg.alias_info is None for arg in (*schema.arguments, *schema.returns))
        and not _requires_full_stream_barrier(node)
    )


def _is_npu_tuple_getitem(node):
    # Moving a tuple extraction is necessary to expose the following compute.
    # Do not generalize this to Python __getitem__ or Tensor indexing.
    if node.op != "call_function" or node.target is not operator.getitem or len(node.args) != 2 or node.kwargs:
        return False
    source, index = node.args
    if not isinstance(source, fx.Node) or not isinstance(source.target, torch._ops.OpOverload):
        return False
    num_returns = len(source.target._schema.returns)
    return (
        _is_reorderable_npu_call(source)
        and type(index) is int
        and num_returns > 1
        and -num_returns <= index < num_returns
    )


def _can_reorder(node):
    if _is_reorderable_npu_call(node) or _is_npu_tuple_getitem(node):
        return True
    # Unreviewed custom/Python calls and alias writes still delimit windows.
    schema = getattr(node.target, "_schema", None)
    return (
        node.op == "call_function"
        and schema is not None
        and schema.name.startswith("aten::")
        and not schema.is_mutable
        and not _requires_full_stream_barrier(node)
    )


def event_frontiers(schedule):
    """Join device-work frontiers; metadata only forwards its dependencies."""
    clocks = [[0, 0], [0, 0]]
    producers = [[], []]
    frontiers = {}
    waits = {}
    for n in schedule.order:
        required = [0, 0]
        for p in schedule.dependencies[n]:
            required[0] = max(required[0], frontiers[p][0])
            required[1] = max(required[1], frontiers[p][1])
        metadata = n.op in {"placeholder", "get_attr"} or _is_metadata_only_compute_node(n)
        if metadata and not _is_stream_barrier(n):
            # Do not mark the lane as synchronized by host-only work. Its
            # consumers inherit this logical dependency, including any gates.
            frontiers[n] = required
            continue
        lane = schedule.lanes[n]
        other = 1 - lane
        if required[other] > clocks[lane][other]:
            p = producers[other][required[other] - 1]
            waits[n] = p
            clocks[lane] = [max(a, b) for a, b in zip(clocks[lane], frontiers[p], strict=True)]
        if n.op != "output":
            producers[lane].append(n)
            clocks[lane][lane] = len(producers[lane])
        frontiers[n] = clocks[lane].copy()
    return waits


class DualStreamRuntime(torch.nn.Module):
    """One auxiliary stream, reusable per-edge events, allocator ownership."""

    def __init__(self):
        super().__init__()
        self.streams = None
        self.events = {}
        self._event_topology = None
        self.active_lane = 0

    def bind_event_topology(self, topology):
        """Bind reusable device events to one immutable producer/wait graph."""
        if self._event_topology is None:
            self._event_topology = topology
        elif self._event_topology != topology:
            raise RuntimeError("DualStreamRuntime cannot be reused across different event topologies")

    def synchronize_for_topology_rebind(self):
        """Drain the old graph before reusing its event pool for a new topology."""
        if self.streams is not None:
            main, _ = self.streams
            torch.npu.synchronize(main.device)  # pyrefly: ignore [missing-attribute]
            torch.npu.set_stream(main)  # pyrefly: ignore [missing-attribute]
            self.active_lane = 0
        self._event_topology = None

    def begin(self, inputs):
        anchor = next(
            (v for v in pytree.tree_leaves(inputs) if isinstance(v, torch.Tensor) and v.device.type == "npu"), None
        )
        if anchor is None:
            return None
        main = torch.npu.current_stream(anchor.device)  # pyrefly: ignore [missing-attribute]
        if self.streams is None:
            self.streams = (main, torch.npu.Stream(device=anchor.device))  # pyrefly: ignore [missing-attribute]
        elif self.streams[0] != main:
            raise RuntimeError("CV parallel graph must be invoked on its original stream")
        self.streams[1].wait_stream(main)
        self.active_lane = 0
        return anchor.device

    def enter(self, lane, event_id, inputs, device):
        if device is None:
            return device
        assert self.streams is not None
        stream = self.streams[lane]
        if lane != self.active_lane:
            torch.npu.set_stream(stream)  # pyrefly: ignore [missing-attribute]
            self.active_lane = lane
        if event_id >= 0:
            stream.wait_event(self.events[event_id])
        for value in pytree.tree_leaves(inputs):
            if isinstance(value, torch.Tensor) and value.device == device:
                value.record_stream(stream)
        return device

    def record(self, lane, event_id, values, device):
        if device is not None:
            if event_id not in self.events:
                self.events[event_id] = torch.npu.Event()  # pyrefly: ignore [missing-attribute]
            event = self.events[event_id]
            assert self.streams is not None
            event.record(self.streams[lane])
        return device

    def finish(self, outputs, device):
        if device is not None:
            assert self.streams is not None
            main, auxiliary = self.streams
            main.wait_stream(auxiliary)
            torch.npu.set_stream(main)  # pyrefly: ignore [missing-attribute]
            for value in pytree.tree_leaves(outputs):
                if isinstance(value, torch.Tensor) and value.device == device:
                    value.record_stream(main)
        return outputs


def materialize(gm, schedule, *, runtime=None):
    # finish() joins the auxiliary stream; output-only events have no consumer.
    waits = {n: p for n, p in event_frontiers(schedule).items() if n.op != "output"}
    producers = set(waits.values())
    event_ids = {n: i for i, n in enumerate(n for n in schedule.order if n in producers)}
    runtime = runtime if runtime is not None else DualStreamRuntime()
    if isinstance(runtime, DualStreamRuntime):
        runtime.bind_event_topology(
            tuple(
                (
                    event_ids[source],
                    source.name,
                    schedule.lanes[source],
                    target.name,
                    schedule.lanes[target],
                )
                for target, source in waits.items()
            )
        )
    # GraphModule only needs the source graph long enough to copy referenced
    # attributes from ``gm``. The emitted graph replaces it below, so cloning
    # a full training graph here only adds startup time and peak host memory.
    root = fx.GraphModule(gm, gm.graph)
    root.add_module("_cv_parallel_runtime", runtime)
    graph = fx.Graph()
    mapped = {}
    previous_lane = None
    recorded_inputs = [set(), set()]
    storage_lanes: dict[fx.Node, set[int]] = {}
    # Placeholder ordering is the public traced graph signature.
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            mapped[n] = graph.node_copy(n, lambda p: mapped[p])
            storage_lanes[n] = {0}
    runtime = graph.get_attr("_cv_parallel_runtime")
    token = graph.call_method("begin", (runtime, tuple(mapped.values())))
    for n in schedule.order:
        if n.op == "placeholder":
            continue
        if n.op == "output":
            output = fx.map_arg(n.args[0], lambda p: mapped[p])
            graph.output(graph.call_method("finish", (runtime, output, token)))
            continue
        lane = schedule.lanes[n]
        p = waits.get(n)
        event_id = event_ids[p] if p is not None else -1
        metadata_only = _is_metadata_only_compute_node(n)
        input_storage_lanes: set[int] = set().union(*(storage_lanes[v] for v in n.all_input_nodes))
        schema = getattr(n.target, "_schema", None)
        if n.op == "get_attr":
            storage_lanes[n] = {0}
        elif metadata_only:
            storage_lanes[n] = input_storage_lanes
        elif schema is None or any(result.alias_info is not None for result in schema.returns):
            storage_lanes[n] = input_storage_lanes | {lane}
        else:
            storage_lanes[n] = {lane}
        # record_stream registers a storage's stream use until deallocation;
        # repeating it at every consumer adds host dispatch overhead. Track FX
        # values only, without retaining runtime tensors past their last use.
        inputs_to_record = [
            v
            for v in n.all_input_nodes
            if not metadata_only and storage_lanes[v] - {lane} and v not in recorded_inputs[lane]
        ]
        recorded_inputs[lane].update(inputs_to_record)
        cross_inputs = tuple(mapped[v] for v in inputs_to_record)
        # Views do not use the device. Record their storage only at the actual
        # compute consumer; recording at the view can delay allocator reuse on
        # a stream that never reads the storage. Event producers still enter
        # their assigned lane to preserve the schedule's completion frontier.
        needs_stream = not metadata_only or p is not None or n in event_ids
        if needs_stream and (lane != previous_lane or p is not None or cross_inputs):
            token = graph.call_method("enter", (runtime, lane, event_id, cross_inputs, token))
            previous_lane = lane
        mapped[n] = graph.node_copy(n, lambda v: mapped[v])
        if n in event_ids:
            token = graph.call_method("record", (runtime, lane, event_ids[n], (mapped[n],), token))
    graph.lint()
    root.graph = graph
    root.recompile()
    return root, waits


def schedule_pass(gm, example_inputs, *, profile_root: Path, runtime_context=None):
    """Profile and schedule a complete F/B graph already lowered into two chunks.

    The caller supplies chunk_id=0/1, valid data/state dependencies and live
    GraphTrainer inputs. Chunk lowering and attention metadata are external
    to this pass; profile_root is bound by the owning pipeline.
    """
    if runtime_context is None:
        raise RuntimeError("CV parallel profiling requires live training inputs")
    assign_stable_node_tags(gm)
    chunks = {n.meta.get("chunk_id") for n in gm.graph.nodes if n.op == "call_function"}
    if not {0, 1}.issubset(chunks):
        raise RuntimeError("Fixed chunk streams require both chunks in the final FX graph")
    run = _make_run_candidate(runtime_context)
    from .schedule_calibration import calibrate_and_select_schedule

    return calibrate_and_select_schedule(gm, run, profile_root=profile_root)


@register_pass_pipeline(CV_PARALLEL_PIPELINE)
def cv_parallel_pipeline(traced_result, config, *, parallel_dims=None):
    if config.compile.backend != "aot_eager":
        raise ValueError("The fixed chunk stream probe currently requires aot_eager")
    if not config.compile.ep_overlap.enabled:
        raise ValueError("cv_parallel requires EP chunking to be enabled")
    passes = construct_default_graph_passes(traced_result, config, parallel_dims=parallel_dims)
    context_factory = _RUNTIME_CONTEXT.get()
    if context_factory is None:
        raise RuntimeError("cv_parallel requires its trainer-instance runtime context")
    runtime_context = context_factory()
    passes = configure_batch_chunk_passes(passes, config, runtime_context=runtime_context)
    profile_root = Path(os.getenv("CV_PARALLEL_PROFILING_DIR", str(Path(config.dump_folder) / "profiling")))
    passes.append(
        functools.partial(
            schedule_pass,
            profile_root=profile_root,
            runtime_context=runtime_context,
        )
    )
    return passes
