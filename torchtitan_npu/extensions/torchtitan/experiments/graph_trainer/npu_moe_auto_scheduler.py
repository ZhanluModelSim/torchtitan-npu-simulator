# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cost-driven scheduler for two-chunk GraphTrainer MoE regions."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.fx as fx
import torch_npu
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch.utils._ordered_set import OrderedSet
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_EXCHANGE,
    _EP_TOKEN_COUNT_SYNC,
    _EP_TOKEN_EXCHANGE,
    _get_module_fqn,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    ChunkedRegion,
    collect_chunked_regions,
    is_c10d_functional_node,
    is_module_fqn_inside_root,
    ordered_nodes,
)
from torchtitan.tools.logging import logger

from .compute_benchmark import (
    _GROUPED_MM_TARGET,
    PERMUTATION_TARGETS,
    ComputeBenchmarker,
    _is_npu_benchmark_compute_node,
)
from .utils import (
    is_metadata_only_compute_node,
    node_id,
    resolve_collective_process_group,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_TOKEN_EXCHANGE_PHASES = {"dispatch", "combine"}
_ACTIVE_REORDER_MIN_COST_MS = 0.02
_SLACK_FILLER_PREPARATION_FACTOR = 2.0
_FILLER_MIN_COST_MS = 0.05
_COLLECTIVE_EVENT_RUNS = 5


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _exchange_label(node: fx.Node) -> str | None:
    if node.target != torch.ops._c10d_functional.all_to_all_single.default:
        return None
    if _custom_meta(node).get(_EP_TOKEN_COUNT_EXCHANGE) == "dispatch":
        return "count"
    phase = _custom_meta(node).get(_EP_TOKEN_EXCHANGE)
    if phase in _TOKEN_EXCHANGE_PHASES:
        return str(phase)
    return str(node.target)


def _is_token_count_sync_copy(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten._to_copy.default
        and _custom_meta(node).get(_EP_TOKEN_COUNT_SYNC) == "dispatch"
    )


def _is_zero_cost_schedulable_node(node: fx.Node) -> bool:
    """Return pure MoE body work that may move without standalone timing."""
    return _is_token_count_sync_copy(node) or (
        node.op == "call_function" and not node.is_impure() and not is_c10d_functional_node(node)
    )


@dataclass(frozen=True)
class _Exchange:
    label: str
    launch: fx.Node
    waits: tuple[fx.Node, ...]


@dataclass(frozen=True)
class _GreedyScheduleContext:
    region: ChunkedRegion
    nodes: tuple[fx.Node, ...]
    ordered_launches: list[fx.Node]
    scheduled_set: set[fx.Node]
    launch_set: set[fx.Node]
    wait_to_launch: dict[fx.Node, fx.Node]
    count_launches: set[fx.Node]
    launch_ancestors: dict[fx.Node, set[fx.Node]]
    projected_ancestors: dict[fx.Node, set[fx.Node]]
    compute_costs: dict[fx.Node, float]
    collective_costs: dict[fx.Node, float]
    inflight: list[fx.Node]
    remaining: dict[fx.Node, float]


@dataclass(frozen=True)
class MoeRegionSchedule:
    """The assigned costs and requested topological order for one MoE region."""

    region: ChunkedRegion
    nodes: tuple[fx.Node, ...]
    compute_cost_ms: dict[fx.Node, float]
    collective_cost_ms: dict[fx.Node, float]
    compute_cost_count: int
    cost_fallback_count: int
    estimated_hidden_ms: float
    estimated_exposed_ms: float


class NpuMoeAutoOverlapScheduler:
    """Greedily reorder selected two-chunk MoE regions using measured costs.

    Each instance performs one dependency-safe scheduling round. Costs can come
    from standalone operator benchmarks or a whole-graph profile.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        *,
        module_pattern: str,
        compute_cost_fn: Callable[[fx.Node], float] | None = None,
        collective_cost_fn: Callable[[fx.Node], float] | None = None,
        align_across_ranks: bool = True,
        canonical_order_override: dict[fx.Node, int] | None = None,
        use_profile_compute_costs: bool = False,
        debug: bool = False,
    ) -> None:
        self.gm = gm
        self.graph = gm.graph
        self.module_pattern = module_pattern
        self.original_order = canonical_order_override or ordered_nodes(gm)
        self.regions = collect_chunked_regions(gm, module_pattern=module_pattern)
        self._region_nodes = self._collect_region_compute_nodes()
        self._communication_ids = self._original_communication_ids()
        self._uses_default_compute_benchmark = compute_cost_fn is None
        self._uses_standalone_benchmark = self._uses_default_compute_benchmark or collective_cost_fn is None
        self._uses_default_collective_benchmark = collective_cost_fn is None
        if use_profile_compute_costs and compute_cost_fn is None:
            raise ValueError("profile compute costs require a cost callback")
        self._uses_profile_compute_costs = use_profile_compute_costs
        self.compute_cost_fn = compute_cost_fn or self._benchmark_compute
        self.collective_cost_fn = collective_cost_fn or self._benchmark_collective
        self.align_across_ranks = align_across_ranks
        self.debug = debug
        self._compute_benchmarker = ComputeBenchmarker()
        self._uniform_a2a_assumptions: dict[fx.Node, tuple[int, bool]] = {}
        self._uniform_gmm_token_assumptions: dict[fx.Node, int] = {}
        self._uniform_permutation_assumptions: dict[fx.Node, tuple[int, int]] = {}
        self._cann_collective_costs: dict[fx.Node, float] = {}
        self._cost_fallback_nodes: set[fx.Node] = set()
        self._has_run = False
        self.costs_by_node_id: dict[int, float] = {}

    def _original_communication_ids(self) -> dict[fx.Node, tuple[int, int]]:
        """Identify launches by original group/communication order, not FX IDs."""
        from .whole_graph_benchmark import _collective_group

        groups: dict[str, int] = {}
        counts: dict[int, int] = defaultdict(int)
        result = {}
        for node in sorted(self.graph.nodes, key=self.original_order.__getitem__):
            if _exchange_label(node) is None:
                continue
            group = _collective_group(node)
            group_id = groups.setdefault(group, len(groups))
            result[node] = (group_id, counts[group_id])
            counts[group_id] += 1
        return result

    @staticmethod
    def _gather_order_values(value: Any) -> list[Any]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return [value]
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _validate_communication_manifest(self) -> None:
        """Validate that communication launch sequences match across ranks."""
        manifest = (
            [(identity, str(node.target), _exchange_label(node)) for node, identity in self._communication_ids.items()],
            [
                (
                    region.root_fqn,
                    region.is_backward,
                    [
                        self._communication_ids[node]
                        for node in self._region_nodes[(region.root_fqn, region.is_backward)]
                        if node in self._communication_ids
                    ],
                )
                for region in self.regions
            ],
        )
        manifests = self._gather_order_values(manifest)
        if any(item != manifest for item in manifests):
            raise RuntimeError("NPU auto-overlap original communication manifests differ across ranks")

    def _validate_world_communication_groups(self) -> None:
        """Reject subgroup communication until scheduling is group-aware."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for node in self._communication_ids:
            process_group = resolve_collective_process_group(node)
            group_size = dist.get_world_size(process_group)
            if group_size != world_size:
                raise RuntimeError(
                    "NPU auto-overlap currently requires every graph "
                    "communication group used for scheduling to span WORLD: "
                    f"node={node.name} group_size={group_size} "
                    f"world_size={world_size}"
                )

    def _agree_communication_swap(
        self,
        pair: tuple[fx.Node, fx.Node],
        eligible: bool,
        delta: float,
        has_gain: bool,
    ) -> bool:
        identity = tuple(self._communication_ids[node] for node in pair)
        votes = self._gather_order_values((identity, eligible, delta, has_gain))
        if any(vote[0] != identity for vote in votes):
            raise RuntimeError("NPU auto-overlap communication decision boundaries differ across ranks")
        all_eligible = all(vote[1] for vote in votes)
        mean_delta = sum(vote[2] for vote in votes) / len(votes)
        any_gain = any(vote[3] for vote in votes)
        return all_eligible and any_gain and mean_delta > 1e-9

    def _collect_region_compute_nodes(self) -> dict[tuple[str, bool], tuple[fx.Node, ...]]:
        """Collect chunked and shared compute for each MoE region.

        Include unchunked work inside the same module so cost measurement and
        ordering cover all compute that can affect overlap.
        """
        domains = {
            (region.root_fqn, region.is_backward): {
                node for body in region.bodies_by_chunk.values() for node in body.nodes
            }
            for region in self.regions
        }
        owned = set().union(*domains.values()) if domains else set()
        # Assign shared nodes to the most specific matching region.
        keys = sorted(domains, key=lambda key: (-len(key[0]), key))
        for node in self.graph.nodes:
            if node in owned or node.op != "call_function" or node.is_impure() or is_c10d_functional_node(node):
                continue
            fqn = _get_module_fqn(node)
            if not fqn:
                continue
            backward = bool(node.meta.get("chunked_region_is_backward", _is_backward_node(node)))
            for root, direction in keys:
                if backward == direction and is_module_fqn_inside_root(fqn, root):
                    domains[(root, direction)].add(node)
                    owned.add(node)
                    break
        return {key: tuple(sorted(nodes, key=self.original_order.__getitem__)) for key, nodes in domains.items()}

    def _can_benchmark_compute_node(self, node: fx.Node) -> bool:
        # Metadata-only nodes have no device cost
        if is_metadata_only_compute_node(node):
            return False
        # Permutation and GMM may have dynamic routed-token shapes, using
        # balanced-routing assumptions to materialize inputs.
        if str(node.target) in PERMUTATION_TARGETS:
            return node in self._uniform_permutation_assumptions
        if str(node.target) == _GROUPED_MM_TARGET:
            return (
                node.op == "call_function"
                and not node.is_impure()
                and (_is_npu_benchmark_compute_node(node) or node in self._uniform_gmm_token_assumptions)
            )
        return _is_npu_benchmark_compute_node(node)

    def _benchmark_compute(self, node: fx.Node) -> float:
        target = str(node.target)
        result = self._compute_benchmarker.benchmark(
            node,
            assumed_tokens=self._uniform_gmm_token_assumptions.get(node),
            permutation_assumption=self._uniform_permutation_assumptions.get(node),
        )
        if result is None:
            raise RuntimeError(f"cannot materialize benchmark inputs for {node.name} ({target})")
        return float(result[0])

    def _compute_cost_or_zero(self, node: fx.Node) -> float:
        try:
            return float(self.compute_cost_fn(node))
        except Exception as error:
            self._cost_fallback_nodes.add(node)
            logger.warning(
                "NPU auto-overlap could not obtain a cost for compute node %s (%s); using zero cost: %s",
                node.name,
                node.target,
                error,
            )
            return 0.0

    def _benchmark_collective(self, node: fx.Node) -> float:
        """Measure one functional collective through completion with events."""
        cann_cost = self._cann_collective_costs.get(node)
        if cann_cost is not None:
            return cann_cost

        from .collective_benchmark import (
            benchmark_collective_with_npu_events,
        )

        assumption = self._uniform_a2a_assumptions.get(node)
        runtime_ms, _ = benchmark_collective_with_npu_events(
            node,
            nruns=_COLLECTIVE_EVENT_RUNS,
            uniform_dispatch_rows=assumption[0] if assumption else None,
            reverse_uniform_splits=assumption[1] if assumption else False,
        )
        if runtime_ms is None:
            raise RuntimeError(f"NPU auto-overlap could not benchmark collective {node.name} ({node.target})")
        return float(runtime_ms)

    def _profile_collective_costs_with_cann(self) -> None:
        """Profile missing collective signatures and cache their per-node costs."""
        from .collective_benchmark import (
            benchmark_collectives_with_cann_profiler,
        )

        requests: list[tuple[fx.Node, int | None, bool]] = []
        for region in self.regions:
            nodes = set(self._region_nodes[(region.root_fqn, region.is_backward)])
            exchanges = self._collect_exchanges(nodes)
            if not exchanges:
                continue
            for exchange in exchanges:
                assumption = self._uniform_a2a_assumptions.get(exchange.launch)
                requests.append(
                    (
                        exchange.launch,
                        assumption[0] if assumption else None,
                        assumption[1] if assumption else False,
                    )
                )
        if not requests:
            return
        try:
            self._cann_collective_costs.update(
                benchmark_collectives_with_cann_profiler(
                    requests,
                    nruns=_COLLECTIVE_EVENT_RUNS,
                    retain_profile=self.debug,
                )
            )
        except Exception as error:
            self._cann_collective_costs.clear()
            logger.warning(
                "NPU auto-overlap CANN collective benchmark failed; falling back to NPU events: %s",
                error,
            )

    def _collect_exchanges(self, region_nodes: set[fx.Node]) -> tuple[_Exchange, ...]:
        """Pair every recognized functional collective with all its region waits."""
        from torch._inductor.fx_passes.bucketing import (
            _get_collective_node_from_wait,
            is_wait_tensor,
        )

        waits_by_launch: dict[fx.Node, list[fx.Node]] = defaultdict(list)
        for node in sorted(region_nodes, key=self.original_order.__getitem__):
            if not is_wait_tensor(node):
                continue
            # Recomputed graphs may nest wait_tensor nodes. Map every wait in
            # the chain to the same launch; dependencies preserve their order,
            # and the launch's remaining communication cost is charged once.
            inner_wait = node
            while inner_wait.args and isinstance(inner_wait.args[0], fx.Node) and is_wait_tensor(inner_wait.args[0]):
                inner_wait = inner_wait.args[0]
            launch = _get_collective_node_from_wait(inner_wait)
            if launch is not None and launch in region_nodes and _exchange_label(launch) is not None:
                waits_by_launch[launch].append(node)

        exchanges: list[_Exchange] = []
        for launch, waits in sorted(waits_by_launch.items(), key=lambda item: self.original_order[item[0]]):
            label = _exchange_label(launch)
            assert label is not None
            exchanges.append(
                _Exchange(
                    label=label,
                    launch=launch,
                    waits=tuple(sorted(waits, key=self.original_order.__getitem__)),
                )
            )
        return tuple(exchanges)

    @staticmethod
    def _a2a_input_descriptor(
        node: fx.Node,
    ) -> tuple[int | None, tuple[int, ...] | None, torch.dtype, Any] | None:
        """Return static input information without materializing an FX node."""
        from torch._inductor.fx_passes.node_runtime_estimation import get_hint

        if node.target is not torch.ops._c10d_functional.all_to_all_single.default:
            return None
        value = node.args[0]
        if isinstance(value, fx.Node):
            value = value.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            return None
        hints = [get_hint(dim) for dim in value.shape]
        rows = int(hints[0]) if hints[0] is not None else None
        tail = cast("tuple[int, ...]", tuple(hints[1:])) if all(dim is not None for dim in hints[1:]) else None
        group = node.args[3] if len(node.args) > 3 else node.kwargs.get("group_name")
        return rows, tail, value.dtype, group

    def _set_uniform_a2a_assumptions(
        self,
        region: ChunkedRegion,
        exchanges: tuple[_Exchange, ...],
    ) -> None:
        """Derive one balanced standalone-benchmark workload per MoE chunk.

        The dispatch A2A's routed-token count is reused for dispatch/combine
        A2A, GMM, and permutation inputs; combine reverses the dispatch splits.
        """
        # Group data exchanges by chunk; count exchanges only provide metadata.
        by_chunk: dict[int, list[_Exchange]] = defaultdict(list)
        for exchange in exchanges:
            chunk_id = exchange.launch.meta.get("chunk_id")
            if isinstance(chunk_id, int) and exchange.label in _TOKEN_EXCHANGE_PHASES:
                by_chunk[chunk_id].append(exchange)

        for chunk_id, chunk_exchanges in by_chunk.items():
            # Use dispatch as the source of truth for routed rows and EP group.
            dispatch_descriptors = [
                descriptor
                for exchange in chunk_exchanges
                if exchange.label == "dispatch"
                and (descriptor := self._a2a_input_descriptor(exchange.launch)) is not None
                and descriptor[0] is not None
            ]
            dispatch_rows = {descriptor[0] for descriptor in dispatch_descriptors}
            dispatch_groups = {descriptor[3] for descriptor in dispatch_descriptors}
            if len(dispatch_rows) != 1 or len(dispatch_groups) != 1:
                logger.warning(
                    "NPU auto-overlap cannot infer one uniform routed-token "
                    "count for %s chunk%d: dispatch_rows=%s groups=%s; using node-local "
                    "shape materialization",
                    region.root_fqn,
                    chunk_id,
                    dispatch_rows,
                    dispatch_groups,
                )
                continue
            rows = next(iter(dispatch_rows))
            assert rows is not None
            # Count exchange shape determines the expert count for permutation.
            count_descriptors = [
                self._a2a_input_descriptor(exchange.launch)
                for exchange in exchanges
                if exchange.label == "count" and exchange.launch.meta.get("chunk_id") == chunk_id
            ]
            expert_counts = {
                descriptor[0] * descriptor[1][0]
                for descriptor in count_descriptors
                if descriptor is not None
                and descriptor[0] is not None
                and descriptor[1] is not None
                and len(descriptor[1]) == 1
            }
            if len(expert_counts) == 1:
                experts = next(iter(expert_counts))
                for node in region.bodies_by_chunk[chunk_id].nodes:
                    if str(node.target) in PERMUTATION_TARGETS:
                        # The benchmark derives source-token rows from this
                        # routed-token total and the node's top-k width.
                        self._uniform_permutation_assumptions[node] = (rows, experts)
            # Reuse dispatch splits for data A2As and reverse them for combine.
            for exchange in chunk_exchanges:
                descriptor = self._a2a_input_descriptor(exchange.launch)
                if descriptor is None:
                    continue
                node_rows, _node_tail, _node_dtype, node_group = descriptor
                if node_group not in dispatch_groups:
                    logger.warning(
                        "NPU auto-overlap did not pair %s with the uniform "
                        "dispatch assumption for %s chunk%d: incompatible group",
                        exchange.launch.name,
                        region.root_fqn,
                        chunk_id,
                    )
                    continue
                if node_rows is not None and node_rows != rows:
                    logger.warning(
                        "NPU auto-overlap did not override static A2A input "
                        "rows for %s in %s chunk%d: node_rows=%d, "
                        "dispatch_rows=%d",
                        exchange.launch.name,
                        region.root_fqn,
                        chunk_id,
                        node_rows,
                        rows,
                    )
                    continue
                self._uniform_a2a_assumptions[exchange.launch] = (
                    rows,
                    exchange.label == "combine",
                )
            # Give every GMM in the chunk the same routed-token total.
            gmm_nodes = [
                node for node in region.bodies_by_chunk[chunk_id].nodes if str(node.target) == _GROUPED_MM_TARGET
            ]
            for node in gmm_nodes:
                self._uniform_gmm_token_assumptions[node] = rows

    @staticmethod
    def _align_values(values: list[float]) -> list[float]:
        """Take cross-rank medians for one shared positional node order."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return values
        gathered: list[list[float] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, values)
        if any(item is None or len(item) != len(values) for item in gathered):
            raise RuntimeError("NPU MoE benchmark node lists differ across ranks")
        concrete_values = cast("list[list[float]]", gathered)
        return [float(statistics.median(item[index] for item in concrete_values)) for index in range(len(values))]

    def _debug_benchmark_costs(
        self,
        region: ChunkedRegion,
        kind: str,
        nodes: tuple[fx.Node, ...],
        local_values: list[float],
        aligned_values: list[float],
    ) -> None:
        if not self.debug:
            return
        for node, local_ms, aligned_ms in zip(
            nodes,
            local_values,
            aligned_values,
            strict=True,
        ):
            logger.info(
                "NPU auto-overlap standalone benchmark: region=%s phase=%s "
                "kind=%s node_id=%s target=%s local_ms=%.6f aligned_ms=%.6f",
                region.root_fqn,
                "backward" if region.is_backward else "forward",
                kind,
                node_id(node),
                node.target,
                local_ms,
                aligned_ms,
            )

    def _measure_region(
        self,
        region: ChunkedRegion,
    ) -> tuple[
        tuple[fx.Node, ...],
        tuple[_Exchange, ...],
        dict[fx.Node, float],
        dict[fx.Node, float],
    ]:
        """Collect region nodes and obtain their compute and collective costs."""
        if set(region.bodies_by_chunk) != {0, 1}:
            raise ValueError(f"NPU auto-overlap requires chunks 0/1 for {region.root_fqn}")
        nodes = self._region_nodes[(region.root_fqn, region.is_backward)]
        node_set = set(nodes)
        exchanges = self._collect_exchanges(node_set)
        if not exchanges:
            raise ValueError(f"NPU auto-overlap found no EP A2A in {region.root_fqn}")
        compute_cost_nodes = tuple(node for node in nodes if self._can_benchmark_compute_node(node))
        zero_cost_compute_nodes = tuple(
            node for node in nodes if node not in compute_cost_nodes and _is_zero_cost_schedulable_node(node)
        )
        # Profile callbacks supply measured or fallback costs directly. Let
        # them price zero-cost schedulable nodes without input materialization.
        if self._uses_profile_compute_costs:
            compute_cost_nodes = tuple(
                node for node in nodes if node in compute_cost_nodes or node in zero_cost_compute_nodes
            )
            zero_cost_compute_nodes = ()

        local_compute_values = [self._compute_cost_or_zero(node) for node in compute_cost_nodes]
        compute_values = local_compute_values
        # The CANN prepass has already cached costs by collective launch.
        # Missing launches fall back to per-node event timing here.
        local_collective_values = [self.collective_cost_fn(exchange.launch) for exchange in exchanges]
        collective_values = local_collective_values
        if self.align_across_ranks:
            # The first standalone round benchmarks the original, unscheduled
            # graph. Matched MoE regions have the same canonical node order
            # across ranks, so their costs can be aligned positionally.
            compute_values = self._align_values(compute_values)
            collective_values = self._align_values(collective_values)
        if self._uses_default_compute_benchmark:
            self._debug_benchmark_costs(
                region,
                "compute",
                compute_cost_nodes,
                local_compute_values,
                compute_values,
            )
        if self._uses_default_collective_benchmark:
            self._debug_benchmark_costs(
                region,
                "collective",
                tuple(exchange.launch for exchange in exchanges),
                local_collective_values,
                collective_values,
            )
        return (
            nodes,
            exchanges,
            {
                **dict.fromkeys(zero_cost_compute_nodes, 0.0),
                **dict(zip(compute_cost_nodes, compute_values, strict=True)),
            },
            dict(
                zip(
                    (exchange.launch for exchange in exchanges),
                    collective_values,
                    strict=True,
                )
            ),
        )

    def _apply_token_count_d2h_protocol(
        self,
        nodes: tuple[fx.Node, ...],
        exchanges: tuple[_Exchange, ...],
        projected_ancestors: dict[fx.Node, set[fx.Node]],
    ) -> None:
        """Rewrite two-chunk token-count D2H copies to share one final host sync."""
        # Original (one host sync per chunk):
        #   chunk0: count0 -> D2H0 async -> D2H1 sync -> ...
        #   chunk1: count1 -> D2H2 async -> D2H3 sync -> ...
        # Rewritten (one shared host sync):
        #   count0 -> count1 -> D2H0/1/2 async -> D2H3 sync -> ...
        first_data_launch_by_chunk: dict[int, fx.Node] = {}
        for exchange in exchanges:
            chunk_id = exchange.launch.meta.get("chunk_id")
            if exchange.label in _TOKEN_EXCHANGE_PHASES and chunk_id in (0, 1):
                first_data_launch_by_chunk.setdefault(chunk_id, exchange.launch)

        sync_copies = sorted(
            (node for node in nodes if _is_token_count_sync_copy(node)),
            key=self.original_order.__getitem__,
        )
        copy_chunks = {copy.meta.get("chunk_id") for copy in sync_copies}
        if len(sync_copies) != 4 or copy_chunks != {0, 1}:
            return

        # Enqueue all copies in canonical order and synchronize only the last.
        for copy_index, copy in enumerate(sync_copies):
            if copy_index:
                projected_ancestors[copy].add(sync_copies[copy_index - 1])
            kwargs = dict(copy.kwargs)
            kwargs["non_blocking"] = copy_index + 1 != len(sync_copies)
            copy.kwargs = kwargs

        # Each data launch keeps its own split-list materialization closure,
        # but host consumers wait for the shared final D2H synchronization.
        copy_set = set(sync_copies)
        for data_launch in first_data_launch_by_chunk.values():
            post_sync_nodes = {
                node
                for node in projected_ancestors[data_launch]
                if node not in copy_set and any(copy in projected_ancestors[node] for copy in copy_set)
            }
            for node in post_sync_nodes:
                projected_ancestors[node].add(sync_copies[-1])
            projected_ancestors[data_launch].add(sync_copies[-1])

        # Restore the transitive closure after adding the D2H protocol edges so
        # launch preparation includes every node along the new dependency path.
        closed_ancestors: dict[fx.Node, set[fx.Node]] = {}
        closing: set[fx.Node] = set()

        def close_ancestors(node: fx.Node) -> set[fx.Node]:
            if node in closed_ancestors:
                return closed_ancestors[node]
            if node in closing:
                raise RuntimeError("NPU auto-overlap synthetic dependencies form a cycle")
            closing.add(node)
            expanded = set(projected_ancestors[node])
            for dependency in tuple(expanded):
                expanded.update(close_ancestors(dependency))
            closing.remove(node)
            closed_ancestors[node] = expanded
            return expanded

        try:
            for node in nodes:
                projected_ancestors[node] = close_ancestors(node)
        finally:
            # The recursive closure captures itself and the full projected
            # ancestor map. TorchTitan disables automatic cyclic GC, so leaving
            # this cell populated retains one full map per region and round.
            # Empty it even on failure; the map can then die by refcount when
            # regional scheduling finishes.
            del close_ancestors

    def _select_next_launch(
        self,
        context: _GreedyScheduleContext,
        active_remaining: float,
    ) -> fx.Node | None:
        """Choose the better order for the next two independent collective launches.

        For candidate order ``comm0 -> comm1``:
        ``score = gain - idle - uncovered``, where ``gain`` is compute
        unlocked by ``comm0``, ``idle`` is communication queue idle time
        while inputs are prepared, and ``uncovered`` is communication time
        not hidden by available compute. Zero-gain pairs retain order.
        """
        nodes = context.nodes
        ordered_launches = context.ordered_launches
        scheduled_set = context.scheduled_set
        launch_set = context.launch_set
        wait_to_launch = context.wait_to_launch
        count_launches = context.count_launches
        launch_ancestors = context.launch_ancestors
        projected_ancestors = context.projected_ancestors
        compute_costs = context.compute_costs
        collective_costs = context.collective_costs

        pending = [node for node in ordered_launches if node not in scheduled_set]
        if not pending:
            return None
        first = pending[0]
        if len(pending) < 2:
            return first
        second = pending[1]
        pair = (first, second)
        prep = {node: launch_ancestors[node] - scheduled_set for node in pair}

        def is_ordinary_compute(node: fx.Node) -> bool:
            return (
                node not in launch_set
                and node not in wait_to_launch
                and not node.is_impure()
                and not _is_token_count_sync_copy(node)
            )

        # Keep the original order when the pair or its input preparation
        # crosses communication group, dependency, host synchronization,
        # or another side effect.
        eligible = (
            count_launches.isdisjoint(pair)
            and self._communication_ids[first][0] == self._communication_ids[second][0]
            and first not in launch_ancestors[second]
            and second not in launch_ancestors[first]
            and all(is_ordinary_compute(node) for dependencies in prep.values() for node in dependencies)
        )

        def compute_sum(subset: set[fx.Node]) -> float:
            return sum(max(0.0, compute_costs.get(node, 0.0)) for node in nodes if node in subset)

        # Additional executable compute can cover the first communication,
        # and common preparation must not be counted twice.
        preparation = prep[first] | prep[second]
        cover_available = scheduled_set | preparation
        cover_nodes: set[fx.Node] = set()
        if eligible:
            for node in nodes:
                if node in cover_available:
                    continue
                if is_ordinary_compute(node) and projected_ancestors[node] <= cover_available:
                    cover_available.add(node)
                    cover_nodes.add(node)
        cover_ms = compute_sum(cover_nodes)

        def evaluate(left: fx.Node, right: fx.Node) -> tuple[float, float, float]:
            p0 = compute_sum(prep[left])
            p1 = compute_sum(prep[right] - prep[left])
            window = max(active_remaining - p0, 0.0) + collective_costs[left]
            idle = max(p0 - active_remaining, 0.0) + max(p1 - window, 0.0)
            uncovered = max(window - p1 - cover_ms, 0.0)
            completed_launches = (launch_set & scheduled_set) | {left}
            available = scheduled_set | prep[left] | {left}
            gain = 0.0
            preparation = prep[left] | prep[right]
            for node in nodes:
                if node in available:
                    continue
                if node in wait_to_launch:
                    if wait_to_launch[node] in completed_launches and projected_ancestors[node] <= available:
                        available.add(node)
                elif node not in launch_set and not node.is_impure() and projected_ancestors[node] <= available:
                    available.add(node)
                    if left in projected_ancestors[node] and node not in preparation:
                        gain += max(0.0, compute_costs.get(node, 0.0))
            return gain, idle, uncovered

        gain01, idle01, uncovered01 = evaluate(first, second) if eligible else (0.0,) * 3
        gain10, idle10, uncovered10 = evaluate(second, first) if eligible else (0.0,) * 3
        # All ranks vote, including ranks whose local graph forbids the swap.
        # Never branch around synchronization on local dependencies.
        swap = self._agree_communication_swap(
            pair,
            eligible,
            (gain10 - idle10 - uncovered10) - (gain01 - idle01 - uncovered01),
            has_gain=gain01 > 0.0 or gain10 > 0.0,
        )
        if swap:
            a, b = ordered_launches.index(first), ordered_launches.index(second)
            ordered_launches[a], ordered_launches[b] = second, first
            return second
        return first

    @staticmethod
    def _estimated_complete_launches(context: _GreedyScheduleContext) -> set[fx.Node]:
        """Include only issued launches with a zero remaining queue prefix."""
        completed = (context.launch_set & context.scheduled_set) - set(context.inflight)
        for launch in context.inflight:
            if context.remaining[launch] > 0:
                break
            completed.add(launch)
        return completed

    def _can_prepare_filler(self, context: _GreedyScheduleContext, node: fx.Node) -> bool:
        """Allow only zero-cost setup and waits whose launches have completed."""
        completed_launches = self._estimated_complete_launches(context)
        for dependency in context.projected_ancestors[node] - context.scheduled_set:
            if dependency in context.wait_to_launch:
                if context.wait_to_launch[dependency] not in completed_launches:
                    return False
            elif not is_metadata_only_compute_node(dependency) or context.compute_costs.get(dependency, 0.0) != 0:
                return False
        return True

    def _select_slack_filler(
        self,
        context: _GreedyScheduleContext,
        next_launch: fx.Node,
        active_remaining: float,
    ) -> tuple[fx.Node, ...]:
        """Select one filler chain without delaying launch preparation.

        Reserve a safety-scaled sum of unfinished preparation costs from the
        active communication window because one compute stream serializes them.
        """
        # Do not use the latency-critical count/D2H startup as filler slack
        if any(node in context.count_launches and context.remaining[node] > 0 for node in context.inflight):
            return ()
        frontier = context.launch_ancestors[next_launch] - context.scheduled_set
        preparation = sum(max(0.0, context.compute_costs.get(node, 0.0)) for node in frontier)
        if preparation == 0:
            # Advance zero-cost plumbing or a blocking wait immediately
            return ()
        preparation_budget = preparation * _SLACK_FILLER_PREPARATION_FACTOR
        slack = active_remaining - preparation_budget
        if slack <= 0:
            return ()

        candidates = []
        for node in context.nodes:
            cost = context.compute_costs.get(node, 0.0)
            if node in context.scheduled_set or node in frontier or cost <= _FILLER_MIN_COST_MS or cost >= slack:
                continue
            if node.is_impure() or node in context.launch_set or node in context.wait_to_launch:
                continue
            if not self._can_prepare_filler(context, node):
                continue
            candidates.append(node)
        if not candidates:
            return ()

        # Choose the earliest feasible filler and include its zero-cost setup
        # and waits whose launches are estimated complete.
        chosen = min(candidates, key=self.original_order.__getitem__)
        setup = context.projected_ancestors[chosen] - context.scheduled_set
        return (*sorted(setup, key=self.original_order.__getitem__), chosen)

    def _debug_schedule_decision(
        self,
        context: _GreedyScheduleContext,
        action: str,
        node: fx.Node | None,
        next_launch: fx.Node | None,
        active_remaining: float,
    ) -> None:
        """Log one greedy scheduling decision and its current queue state."""
        if not self.debug:
            return
        if node is None:
            cost = 0.0
            stable_id = None
            name = target = "<none>"
        else:
            cost = context.compute_costs.get(node, context.collective_costs.get(node, 0.0))
            if node in context.wait_to_launch:
                cost = max(0.0, context.remaining[context.wait_to_launch[node]])
            stable_id = node_id(node)
            name = node.name
            target = str(node.target)
        next_ref = "<none>" if next_launch is None else f"{node_id(next_launch)}:{next_launch.name}"
        logger.info(
            "NPU auto-overlap schedule: region=%s phase=%s scheduled_nodes=%d "
            "action=%s node_id=%s node=%s target=%s cost_ms=%.6f "
            "next_launch=%s inflight_collectives=%d communication_remaining_ms=%.6f",
            context.region.root_fqn,
            "backward" if context.region.is_backward else "forward",
            len(context.scheduled_set),
            action,
            stable_id,
            name,
            target,
            cost,
            next_ref,
            len(context.inflight),
            active_remaining,
        )

    def _greedy_schedule(
        self,
        region: ChunkedRegion,
        nodes: tuple[fx.Node, ...],
        exchanges: tuple[_Exchange, ...],
        compute_costs: dict[fx.Node, float],
        collective_costs: dict[fx.Node, float],
    ) -> MoeRegionSchedule:
        """Build one dependency-safe order that hides in-flight EP traffic."""
        node_set = set(nodes)
        index = self.original_order
        launch_set = {exchange.launch for exchange in exchanges}
        wait_to_launch = {wait: exchange.launch for exchange in exchanges for wait in exchange.waits}
        count_launches = {exchange.launch for exchange in exchanges if exchange.label == "count"}
        successors: dict[fx.Node, list[fx.Node]] = defaultdict(list)
        in_degree: dict[fx.Node, int] = {}

        # Preserve dependencies that leave and re-enter the region. If
        # A(region) -> X(outside) -> B(region), treat A as B's ancestor;
        # otherwise local reordering could add B -> A and cycle the full graph.
        projected_ancestors: dict[fx.Node, set[fx.Node]] = {}
        for graph_node in self.graph.nodes:
            ancestors: set[fx.Node] = set()
            for dep in graph_node.all_input_nodes:
                ancestors.update(projected_ancestors[dep])
                if dep in node_set:
                    ancestors.add(dep)
            projected_ancestors[graph_node] = ancestors

        self._apply_token_count_d2h_protocol(
            nodes,
            exchanges,
            projected_ancestors,
        )

        for node in nodes:
            deps = projected_ancestors[node]
            in_degree[node] = len(deps)
            for dep in deps:
                successors[dep].append(node)

        ordered_launches = sorted(launch_set, key=index.__getitem__)
        launch_ancestors = {launch: projected_ancestors[launch] for launch in launch_set}

        ready: set[fx.Node] = {node for node in nodes if in_degree[node] == 0}
        scheduled: list[fx.Node] = []
        scheduled_set: set[fx.Node] = set()
        inflight: list[fx.Node] = []
        remaining = dict(collective_costs)
        hidden_ms = 0.0
        exposed_ms = 0.0
        preparing_next_launch = False
        next_launch: fx.Node | None = None

        def earliest(candidates: list[fx.Node]) -> fx.Node:
            """Choose the earliest candidate in canonical graph order."""
            return min(candidates, key=index.__getitem__)

        def schedule(node: fx.Node) -> None:
            scheduled.append(node)
            scheduled_set.add(node)
            ready.remove(node)
            for user in successors[node]:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    ready.add(user)

        def account_compute_overlap(node: fx.Node) -> None:
            nonlocal hidden_ms
            available = compute_costs.get(node, 0.0)
            for collective in inflight:
                overlap = min(available, max(0.0, remaining[collective]))
                remaining[collective] -= overlap
                available -= overlap
                hidden_ms += overlap
                if available <= 0:
                    break

        def complete_collective_wait(node: fx.Node) -> None:
            nonlocal exposed_ms
            launch = wait_to_launch[node]
            exposed_ms += max(0.0, remaining[launch])
            remaining[launch] = 0.0
            if launch in inflight:
                # Launches share one ordered EP queue, but waits can be
                # scheduled out of order. Waiting for a launch also retires
                # every earlier launch from the in-flight queue.
                launch_index = inflight.index(launch)
                for prior in inflight[:launch_index]:
                    exposed_ms += max(0.0, remaining[prior])
                    remaining[prior] = 0.0
                # Mutate in place so the scheduling context sees the current
                # queue.
                del inflight[: launch_index + 1]
            schedule(node)

        def schedule_filler(node: fx.Node) -> None:
            assert node in ready
            if node in wait_to_launch:
                complete_collective_wait(node)
            else:
                account_compute_overlap(node)
                schedule(node)

        schedule_context = _GreedyScheduleContext(
            region=region,
            nodes=nodes,
            ordered_launches=ordered_launches,
            scheduled_set=scheduled_set,
            launch_set=launch_set,
            wait_to_launch=wait_to_launch,
            count_launches=count_launches,
            launch_ancestors=launch_ancestors,
            projected_ancestors=projected_ancestors,
            compute_costs=compute_costs,
            collective_costs=collective_costs,
            inflight=inflight,
            remaining=remaining,
        )

        while ready:
            active_remaining = sum(max(0.0, remaining[node]) for node in inflight)
            # Keep preparing the same launch; choose again only after it is
            # issued.
            if next_launch is None or next_launch in scheduled_set:
                # Compare the next two eligible collectives by overlap score.
                # In DSV4, for example, moving score0 before dispatch1 can
                # unlock reroute0 and GMM0 early enough to overlap dispatch1.
                next_launch = self._select_next_launch(schedule_context, active_remaining)
                preparing_next_launch = False
                self._debug_schedule_decision(
                    schedule_context, "select_launch", next_launch, next_launch, active_remaining
                )
            ready_compute = [node for node in ready if compute_costs.get(node, 0.0) >= _ACTIVE_REORDER_MIN_COST_MS]
            if next_launch is not None and next_launch in ready:
                # Launch a ready collective immediately to avoid a gap in the
                # communication queue.
                self._debug_schedule_decision(
                    schedule_context, "launch_collective", next_launch, next_launch, active_remaining
                )
                inflight.append(next_launch)
                schedule(next_launch)
                continue

            # After count D2H, dispatch preparation contains many small kernels
            # and can become host-bound. Use independent larger kernels to fill
            # the resulting compute-stream gaps.
            if next_launch is not None and active_remaining > 0 and not preparing_next_launch:
                filler = self._select_slack_filler(schedule_context, next_launch, active_remaining)
                if filler:
                    self._debug_schedule_decision(
                        schedule_context, "slack_filler", filler[-1], next_launch, active_remaining
                    )
                    for node in filler:
                        schedule_filler(node)
                    continue

            # Once this launch's dependency preparation begins, do not reopen
            # budgeted slack filling.
            if next_launch is not None and not preparing_next_launch:
                self._debug_schedule_decision(
                    schedule_context, "begin_launch_preparation", next_launch, next_launch, active_remaining
                )
                preparing_next_launch = True

            critical_non_wait = [
                node
                for node in ready
                if next_launch is not None and node in launch_ancestors[next_launch] and node not in wait_to_launch
            ]
            if critical_non_wait:
                node = earliest(critical_non_wait)
                self._debug_schedule_decision(schedule_context, "prepare_launch", node, next_launch, active_remaining)
                account_compute_overlap(node)
                schedule(node)
                continue

            critical_count_waits = [
                node
                for node in ready
                if next_launch is not None
                and node in launch_ancestors[next_launch]
                and node in wait_to_launch
                and wait_to_launch[node] in count_launches
            ]
            if critical_count_waits:
                # The count wait starts D2H and split-list preparation for the
                # next data A2A. Complete it immediately; unrelated compute is
                # more valuable after that larger communication is launched.
                wait = earliest(critical_count_waits)
                self._debug_schedule_decision(
                    schedule_context, "complete_count_wait", wait, next_launch, active_remaining
                )
                complete_collective_wait(wait)
                continue

            if ready_compute and active_remaining > 0 and next_launch is None:
                node = earliest(ready_compute)
                self._debug_schedule_decision(schedule_context, "ready_compute", node, next_launch, active_remaining)
                account_compute_overlap(node)
                schedule(node)
                continue

            critical_waits = [
                node
                for node in ready
                if next_launch is not None and node in launch_ancestors[next_launch] and node in wait_to_launch
            ]
            if critical_waits:
                wait = earliest(critical_waits)
                launch = wait_to_launch[wait]
                # If this launch or an earlier queued launch still has unhidden
                # communication time, run independent compute before its wait.
                communication_pending = False
                if launch in inflight:
                    for collective in inflight:
                        communication_pending |= remaining[collective] > 0
                        if communication_pending or collective is launch:
                            break
                candidates = []
                critical_ancestors = launch_ancestors[next_launch] if next_launch is not None else set()
                if communication_pending:
                    for node in nodes:
                        cost = compute_costs.get(node, 0.0)
                        if (
                            node in scheduled_set
                            or node in critical_ancestors
                            or node in launch_set
                            or node in wait_to_launch
                            or node.is_impure()
                            or cost <= _FILLER_MIN_COST_MS
                        ):
                            continue
                        if self._can_prepare_filler(schedule_context, node):
                            candidates.append(node)
                if candidates:
                    node = earliest(candidates)
                    self._debug_schedule_decision(schedule_context, "wait_filler", node, next_launch, active_remaining)
                    setup = projected_ancestors[node] - scheduled_set
                    for filler in (*sorted(setup, key=index.__getitem__), node):
                        schedule_filler(filler)
                    continue
                self._debug_schedule_decision(
                    schedule_context, "complete_critical_wait", wait, next_launch, active_remaining
                )
                complete_collective_wait(wait)
                continue

            ready_non_wait = [node for node in ready if node not in wait_to_launch]
            if ready_non_wait:
                node = earliest(ready_non_wait)
                self._debug_schedule_decision(schedule_context, "ready_non_wait", node, next_launch, active_remaining)
                account_compute_overlap(node)
                schedule(node)
                continue

            ready_waits = [node for node in ready if node in wait_to_launch]
            if ready_waits:
                wait = earliest(ready_waits)
                self._debug_schedule_decision(
                    schedule_context, "complete_ready_wait", wait, next_launch, active_remaining
                )
                complete_collective_wait(wait)
                continue

            # Defensive fallback
            node = earliest(list(ready))
            self._debug_schedule_decision(schedule_context, "fallback", node, next_launch, active_remaining)
            schedule(node)

        if len(scheduled) != len(nodes):
            raise RuntimeError(f"NPU auto-overlap could not schedule {region.root_fqn}")
        return MoeRegionSchedule(
            region=region,
            nodes=tuple(scheduled),
            compute_cost_ms=compute_costs,
            collective_cost_ms=collective_costs,
            compute_cost_count=len(compute_costs),
            cost_fallback_count=sum(node in self._cost_fallback_nodes for node in nodes),
            estimated_hidden_ms=hidden_ms,
            estimated_exposed_ms=exposed_ms,
        )

    def _ordering_deps(
        self,
        schedules: list[MoeRegionSchedule],
    ) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Convert regional orders into globally acyclic dependencies."""
        proposed: list[tuple[fx.Node, fx.Node]] = []
        for schedule in schedules:
            previous: fx.Node | None = None
            for node in schedule.nodes:
                if previous is not None:
                    proposed.append((previous, node))
                previous = node

        deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
        for previous, node in proposed:
            deps.setdefault(node, OrderedSet()).add(previous)

        # Region schedules are derived independently. Cross-layer edges can
        # make otherwise legal regional choices conflict, but dropping an
        # ordering edge could also remove a synthetic D2H correctness
        # dependency. Reject the combined schedule instead.
        from torch._dynamo.graph_deduplication import _has_cycle

        if _has_cycle(self.graph, deps):
            raise RuntimeError("NPU auto-overlap regional schedules conflict with the full-graph DAG")
        return deps

    def run(self) -> fx.GraphModule:
        """Obtain costs for all matched regions, reorder, and recompile the graph."""
        if self._has_run:
            raise RuntimeError(
                "NPU auto-overlap scheduler instances are one-shot; "
                "create a fresh instance for another scheduling round"
            )
        self._has_run = True
        if not self.regions:
            raise ValueError(f"NPU auto-overlap found no chunked regions matching {self.module_pattern!r}")
        self._validate_world_communication_groups()
        self._validate_communication_manifest()

        def measure_and_schedule() -> list[MoeRegionSchedule]:
            schedules: list[MoeRegionSchedule] = []
            # Standalone benchmarks use balanced synthetic inputs for
            # dynamic-shape dispatch, combine, GMM, permutation, and related
            # nodes.
            if not self._uses_profile_compute_costs or self._uses_default_collective_benchmark:
                for region in self.regions:
                    region_nodes = set(self._region_nodes[(region.root_fqn, region.is_backward)])
                    self._set_uniform_a2a_assumptions(
                        region,
                        self._collect_exchanges(region_nodes),
                    )
            if self._uses_default_collective_benchmark:
                # Deduplicate collective signatures across all regions and
                # populate per-launch CANN costs before regional scheduling.
                # Event timing remains the fallback for missing launches.
                self._profile_collective_costs_with_cann()
            for region in self.regions:
                nodes, exchanges, compute_costs, collective_costs = self._measure_region(region)
                schedules.append(
                    self._greedy_schedule(
                        region,
                        nodes,
                        exchanges,
                        compute_costs,
                        collective_costs,
                    )
                )
            return schedules

        if self._uses_standalone_benchmark:
            # Standalone operator benchmarks materialize synthetic inputs. Keep
            # them from advancing the real training step's CPU and NPU RNG.
            with torch.random.fork_rng(
                devices=[torch_npu.npu.current_device()],
                device_type="npu",
            ):
                schedules = measure_and_schedule()
        else:
            schedules = measure_and_schedule()

        ordering_deps = self._ordering_deps(schedules)
        # Publish this round's costs by stable node ID. The outer auto-overlap
        # pass uses them as fallback costs for the next round.
        for schedule in schedules:
            self.costs_by_node_id.update(
                {
                    stable_id: cost
                    for node, cost in (schedule.compute_cost_ms | schedule.collective_cost_ms).items()
                    if (stable_id := node_id(node)) is not None
                }
            )
        _stable_topological_sort(self.graph, ordering_deps)
        self.graph.lint()
        self.gm.recompile()

        # Preserve each region's selected launch order through the global sort,
        # then compare per-group launch order across ranks. FX node IDs,
        # compute nodes, and wait placement may differ under uneven splitting
        # (e.g. DSV4 MHC).
        final_order = ordered_nodes(self.gm)
        for schedule in schedules:
            launches = set(schedule.collective_cost_ms)
            planned = [node for node in schedule.nodes if node in launches]
            actual = sorted(launches, key=final_order.__getitem__)
            if planned != actual:
                raise RuntimeError("NPU auto-overlap global sort changed the selected communication order")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            by_group: dict[int, list[int]] = defaultdict(list)
            for node in self.graph.nodes:
                if node in self._communication_ids:
                    group, ordinal = self._communication_ids[node]
                    by_group[group].append(ordinal)
            signature = sorted(by_group.items())
            digest = hashlib.sha256(repr(signature).encode()).hexdigest()
            digests: list[str | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(digests, digest)
            if any(value != digest for value in digests):
                raise RuntimeError("NPU auto-overlap selected different communication orders across ranks")
        for node, deps in ordering_deps.items():
            if any(final_order[dep] >= final_order[node] for dep in deps):
                raise RuntimeError(
                    f"NPU auto-overlap failed to materialize an accepted ordering dependency for {node.name}"
                )
        logger.info(
            "NPU auto-overlap scheduled %d region(s): "
            "compute_costs=%d cost_fallbacks=%d collectives=%d "
            "hidden_ms=%.4f exposed_ms=%.4f",
            len(schedules),
            sum(schedule.compute_cost_count for schedule in schedules),
            sum(schedule.cost_fallback_count for schedule in schedules),
            sum(len(schedule.collective_cost_ms) for schedule in schedules),
            sum(schedule.estimated_hidden_ms for schedule in schedules),
            sum(schedule.estimated_exposed_ms for schedule in schedules),
        )
        return self.gm
