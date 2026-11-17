# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Candidate generation and regional CV scheduling search."""

from __future__ import annotations

import copy
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import pairwise
from typing import Any

from torchtitan.tools.logging import logger

from . import schedule_simulation
from .cv_parallel import event_frontiers
from .schedule_scoring import _local_coverage_score, _overlap_metrics, predicted_coverage_improved
from .schedule_simulation import ScheduleSimulation, _compute_union_ms, _ReorderProposal, _ReorderWindowOrder
from .whole_graph_runtime_profile import node_id

_REORDER_TEMPLATE_CACHE_NODE_LIMIT = 262144
_EXHAUSTIVE_REGION_NODE_LIMIT = 512
_BOUNDED_REGION_MAX_PASSES = 8
_BOUNDED_REORDER_NODE_LIMIT_PER_KIND = 16
_BOUNDED_REORDER_DESTINATION_LIMIT = 24
_BOUNDED_CV_CANDIDATE_LIMIT = 2048
_BOUNDED_CV_SOURCE_LIMIT = 32

_COVERAGE_FIELDS = ("cv_ms", "mixed_cv_ms", "cc_ms", "vv_ms")


def _bounded_destinations(size, position, limit):
    """Cover nearby, exponentially distant and global move positions."""
    if size - 1 <= limit:
        return tuple(i for i in range(size) if i != position)
    selected = {0, size - 1}
    selected.discard(position)
    nearby_limit = max(2, limit // 2)
    distance = 1
    while len(selected) < nearby_limit and distance < size:
        for destination in (position - distance, position + distance):
            if 0 <= destination < size and destination != position:
                selected.add(destination)
                if len(selected) >= nearby_limit:
                    break
        distance *= 2
    denominator = max(1, limit - 1)
    for slot in range(limit):
        destination = round(slot * (size - 1) / denominator)
        if destination != position:
            selected.add(destination)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for destination in range(size):
            if destination != position:
                selected.add(destination)
            if len(selected) >= limit:
                break
    return tuple(sorted(selected))


class _WindowPlanner:
    """Search policies composed with an explicit simulation state."""

    def __init__(self, gm, profile, kernel_observations, *, reference=None):
        self.simulation = ScheduleSimulation(gm, profile, kernel_observations, reference=reference)
        self.bounded_search = len(self.simulation.nodes) > _EXHAUSTIVE_REGION_NODE_LIMIT

    def search(self):
        """Search barrier regions exhaustively unless a large region needs bounded work."""
        # Mandatory stream barriers cannot be crossed by a permutation or an
        # optional event. Searching their regions independently preserves the
        # complete legal candidate set while keeping every trial local.
        stops = [self.simulation.rank[i] + 1 for i in self.simulation.order if i in self.simulation.barriers]
        if not stops or stops[-1] != len(self.simulation.nodes):
            stops.append(len(self.simulation.nodes))
        bounds = list(pairwise([0, *stops]))
        region_of = {
            i: ordinal for ordinal, (begin, stop) in enumerate(bounds) for i in self.simulation.order[begin:stop]
        }
        lanes_by_region = [[[], []] for _ in bounds]
        for lane, nodes in enumerate(self.simulation.lanes):
            for i in nodes:
                lanes_by_region[region_of[i]][lane].append(i)
        barriers_by_region = [[] for _ in bounds]
        for i in self.simulation.barriers:
            barriers_by_region[region_of[i]].append(i)
        windows = defaultdict(list)
        for lane, first, size in self.simulation.windows:
            window = self.simulation.lanes[lane][first : first + size]
            ordinal = region_of[window[0]]
            assert all(region_of[i] == ordinal for i in window), "A reorder window crossed a mandatory barrier"
            windows[ordinal].append((lane, window[0], size))

        self.simulation.stats["num_search_regions"] = len(bounds)
        self.simulation.stats["largest_region_nodes"] = max(stop - begin for begin, stop in bounds)
        started = progress = time.perf_counter()
        times = dict.fromkeys(("reorder", "cv", "prune"), 0.0)
        logger.info(
            "CV parallel regional search: nodes=%d reorder_windows=%d regions=%d largest_region_nodes=%d "
            "exhaustive_region_node_limit=%d",
            len(self.simulation.nodes),
            len(self.simulation.windows),
            len(bounds),
            self.simulation.stats["largest_region_nodes"],
            _EXHAUSTIVE_REGION_NODE_LIMIT,
        )

        def log_progress(ordinal):
            nonlocal progress
            now = time.perf_counter()
            if now - progress < 10 and ordinal != len(bounds):
                return
            logger.info(
                "CV parallel regional search regions=%d/%d elapsed_seconds=%.3f passes=%d "
                "cv_candidates=%d cv_trials=%d reorder_trials=%d reorder_edits=%d wait_edits=%d",
                ordinal,
                len(bounds),
                now - started,
                self.simulation.stats["num_passes"],
                self.simulation.stats["num_cv_candidates"],
                self.simulation.stats["num_trials"],
                self.simulation.stats["num_reorder_trials"],
                self.simulation.stats["num_reorder_edits"],
                self.simulation.stats["num_cv_edits"],
            )
            progress = now

        lanes = [[], []]
        global_coverage = self.simulation.coverage.copy()
        global_span = self.simulation.span
        for ordinal, (begin, stop) in enumerate(bounds, 1):
            indices = sorted(self.simulation.order[begin:stop])
            region_windows = windows[ordinal - 1]
            region_lanes = lanes_by_region[ordinal - 1]
            if not region_windows:
                phase_kinds = [set(), set()]
                for i in indices:
                    phase_kinds[self.simulation.lane_of[i]].update(kind for _, _, kind in self.simulation.phases[i])
                has_alignment_pair = (
                    ("aiv" in phase_kinds[0] or "mix_aiv" in phase_kinds[0]) and "aic" in phase_kinds[1]
                ) or (("aiv" in phase_kinds[1] or "mix_aiv" in phase_kinds[1]) and "aic" in phase_kinds[0])
                if not has_alignment_pair:
                    for lane in (0, 1):
                        lanes[lane].extend(region_lanes[lane])
                    self.simulation.stats["num_passes"] += 1
                    self.simulation.stats["num_search_regions_fast_skipped"] += 1
                    log_progress(ordinal)
                    continue
            index = {node: local for local, node in enumerate(indices)}
            local = copy.copy(self)
            local.simulation = copy.copy(self.simulation)
            local.simulation.nodes = [self.simulation.nodes[i] for i in indices]
            local.bounded_search = len(local.simulation.nodes) > _EXHAUSTIVE_REGION_NODE_LIMIT
            local.simulation.barriers = {index[i] for i in barriers_by_region[ordinal - 1]}
            local.simulation.lane_of = [self.simulation.lane_of[i] for i in indices]
            local.simulation.lanes = [[index[i] for i in region_lanes[lane]] for lane in (0, 1)]
            local.simulation.base_parents = [
                {index[p] for p in self.simulation.base_parents[i] if p in index} for i in indices
            ]
            positions = {node: position for lane in local.simulation.lanes for position, node in enumerate(lane)}
            local.simulation.windows = [(lane, positions[index[first]], size) for lane, first, size in region_windows]
            local.simulation.durations = [self.simulation.durations[i] for i in indices]
            local.simulation.phases = [self.simulation.phases[i] for i in indices]
            local.simulation.gates = {}
            local.simulation.stats = Counter()
            local.simulation.reorder_edits = []
            local.simulation.pruned_gates = []
            local.simulation.benefit_by_method = {method: Counter() for method in ("reorder", "wait_event")}
            # Node indices and phase tables are region-local. Do not inherit
            # whole-graph cache entries through the shallow planner copy.
            local.simulation.shifted_phase_cache = {}
            local.simulation.reorder_template_cache = {}
            local.simulation.reorder_template_cache_nodes = 0
            local.simulation.state_version = 0
            local.simulation.trial_cache = {}
            local.simulation.reorder_trial_cache = {}
            local.simulation._current_score_cache = None
            local.simulation._non_fifo_ancestor_bits = None
            local.simulation._node_bits = None
            local.simulation._non_fifo_gate_signature = None
            local.simulation._reset_trial_workspaces()
            local.simulation.refresh()
            local.simulation.span_limit = local.simulation.span
            initial_local_coverage = local.simulation.coverage.copy()
            initial_local_span = local.simulation.span
            local.simulation.score_context = {
                "coverage": [
                    total - regional for total, regional in zip(global_coverage, initial_local_coverage, strict=True)
                ],
                "cube_work": self.simulation.cube_work,
                "vector_work": self.simulation.vector_work,
                "span": global_span - initial_local_span,
            }
            # The context is installed after refresh because it depends on the
            # freshly measured regional baseline.
            local.simulation._current_score_cache = None
            local.simulation.stats["num_bounded_search_regions"] += int(local.bounded_search)

            local_passes = 0
            while True:
                local_passes += 1
                local.simulation.stats["num_passes"] += 1
                tick = time.perf_counter()
                reordered = local.reorder()
                times["reorder"] += time.perf_counter() - tick
                tick = time.perf_counter()
                aligned = local.refine()
                times["cv"] += time.perf_counter() - tick
                if not (reordered or aligned):
                    break
                if local.bounded_search and local_passes >= _BOUNDED_REGION_MAX_PASSES:
                    local.simulation.stats["num_pass_limited_regions"] += 1
                    break

            tick = time.perf_counter()
            local.prune_gates()
            times["prune"] += time.perf_counter() - tick
            for lane in (0, 1):
                lanes[lane].extend(indices[i] for i in local.simulation.lanes[lane])
            self.simulation.gates.update(
                {indices[target]: indices[source] for target, source in local.simulation.gates.items()}
            )
            self.simulation.reorder_edits.extend(local.simulation.reorder_edits)
            self.simulation.pruned_gates.extend(local.simulation.pruned_gates)
            for method, values in local.simulation.benefit_by_method.items():
                self.simulation.benefit_by_method[method].update(values)
            global_coverage = [
                total + final - initial
                for total, final, initial in zip(
                    global_coverage,
                    local.simulation.coverage,
                    initial_local_coverage,
                    strict=True,
                )
            ]
            global_span += local.simulation.span - initial_local_span
            self.simulation.stats["num_region_refreshes"] += local.simulation.stats.pop("num_full_refreshes")
            self.simulation.stats.update(local.simulation.stats)

            log_progress(ordinal)

        self.simulation.lanes = lanes
        self.simulation.refresh()
        self.materialized_order = self.simulation.order.copy()
        logger.info(
            "CV parallel regional search caches: shifted_phase_hits=%d reorder_template_hits=%d "
            "cv_trial_hits=%d reorder_trial_hits=%d trial_entries_invalidated=%d",
            self.simulation.stats["num_shifted_phase_cache_hits"],
            self.simulation.stats["num_reorder_template_cache_hits"],
            self.simulation.stats["num_trial_cache_hits"],
            self.simulation.stats["num_reorder_trial_cache_hits"],
            self.simulation.stats["num_trial_cache_entries_invalidated"],
        )
        return times

    def candidates(self):
        candidates = {}
        for lane in (0, 1):
            partners = [phase for phase in self.simulation.absolute[1 - lane] if phase[2] == "aic"]
            positions = [self.simulation.positions[j] for _, _, _, j, _ in partners]
            dependencies = [self.simulation.clocks[j][lane] for _, _, _, j, _ in partners]
            for phase in self.simulation.absolute[lane]:
                start, end, kind, i, p = phase
                if kind not in {"aiv", "mix_aiv"}:
                    continue
                covered = self.simulation.timelines[1 - lane]["aic"].overlap(start, end)
                if round(end - start - covered, 9) <= 0:
                    continue
                left = bisect_right(positions, self.simulation.clocks[i][1 - lane])
                right = bisect_left(dependencies, self.simulation.positions[i])
                for a, b, _, j, q in partners[left:right]:
                    capacity = min(end - start, b - a)
                    gain = capacity - max(0.0, min(end, b) - max(start, a))
                    if round(gain, 9) <= 0:
                        continue
                    pair = (i, p, j, q)
                    candidates[pair] = (kind != "aiv", -gain)
        self.simulation.stats["num_cv_candidates"] += len(candidates)
        return sorted(candidates, key=candidates.__getitem__)

    def refine(self):
        accepted = 0
        attempted: dict[int, set[int]] = defaultdict(set)
        source_ranges = {}
        candidates = self.candidates()
        if self.bounded_search and len(candidates) > _BOUNDED_CV_CANDIDATE_LIMIT:
            self.simulation.stats["num_cv_candidates_budget_skipped"] += len(candidates) - _BOUNDED_CV_CANDIDATE_LIMIT
            candidates = candidates[:_BOUNDED_CV_CANDIDATE_LIMIT]
        for i, p, j, q in candidates:
            a, b, _ = self.simulation.phases[i][p]
            c, d, _ = self.simulation.phases[j][q]
            a, b, c, d = (
                a + self.simulation.starts[i],
                b + self.simulation.starts[i],
                c + self.simulation.starts[j],
                d + self.simulation.starts[j],
            )
            overlap = max(0.0, min(b, d) - max(a, c))
            if round(min(b - a, d - c) - overlap, 9) <= 0:
                continue
            target, later = (i, j) if a < c else (j, i)
            target_phase, later_phase = (p, q) if target == i else (q, p)
            ta, tb, _ = self.simulation.phases[target][target_phase]
            la, lb, _ = self.simulation.phases[later][later_phase]
            la, lb = la + self.simulation.starts[later], lb + self.simulation.starts[later]
            lane = self.simulation.lane_of[later]
            minimum = max(self.simulation.starts[target], la + overlap - tb)
            left = bisect_right(self.simulation.lane_ends[lane], minimum)
            right = self.simulation.positions[later] - 1
            range_key = (lane, left, right)
            sources = source_ranges.get(range_key)
            if sources is None:
                unique_sources = {}
                for source in self.simulation.lanes[lane][left:right]:
                    if self.simulation.nodes[source].op != "placeholder":
                        unique_sources.setdefault(self.simulation.ends[source], source)
                sources = tuple(unique_sources.items())
                source_ranges[range_key] = sources
            else:
                self.simulation.stats["num_cv_source_range_cache_hits"] += 1
            best: tuple[float, ...] | None = None
            chosen: int | None = None

            def source_score(item, target_start=ta, target_end=tb, later_start=la, later_end=lb):
                end = item[0]
                return (
                    -max(0.0, min(end + target_end, later_end) - max(end + target_start, later_start)),
                    end,
                )

            available_sources = []
            target_lane = self.simulation.lane_of[target]
            target_position = self.simulation.positions[target]
            target_remaining = self.simulation.remaining_ms[target]
            attempted_sources = attempted[target]
            for item in sources:
                end, source = item
                if source in attempted_sources:
                    continue
                if self.simulation.clocks[source][target_lane] >= target_position:
                    attempted_sources.add(source)
                    self.simulation.stats["num_cycle_rejected"] += 1
                    self.simulation.stats["num_cv_cycle_bound_rejected"] += 1
                    continue
                if round(end - self.simulation.starts[target], 9) <= 0:
                    attempted_sources.add(source)
                    self.simulation.stats["num_cv_noop_rejected"] += 1
                    continue
                if round(end + target_remaining - self.simulation.span_limit, 9) > 0:
                    attempted_sources.add(source)
                    self.simulation.stats["num_span_rejected"] += 1
                    self.simulation.stats["num_cv_span_bound_rejected"] += 1
                    continue
                available_sources.append(item)
            ordered_sources = sorted(available_sources, key=source_score)
            if self.bounded_search and len(ordered_sources) > _BOUNDED_CV_SOURCE_LIMIT:
                self.simulation.stats["num_cv_sources_budget_skipped"] += (
                    len(ordered_sources) - _BOUNDED_CV_SOURCE_LIMIT
                )
                ordered_sources = ordered_sources[:_BOUNDED_CV_SOURCE_LIMIT]
            for _, source in ordered_sources:
                attempted_sources.add(source)
                # ``attempted`` already makes this pair unique in the current
                # state; accepted edits refresh the state before it can repeat.
                score = self.simulation.trial(source, target, cache_result=False)
                if score is not None and (best is None or score > best):
                    best, chosen = score, source
            if chosen is not None:
                before = self.simulation._benefit_snapshot()
                self.simulation.gates[target] = chosen
                self.simulation.refresh()
                self.simulation._record_benefit("wait_event", before)
                accepted += 1
                self.simulation.stats["num_cv_edits"] += 1
                attempted.clear()
                source_ranges.clear()
        return accepted

    def prune_gates(self):
        """Drop optional alignment gates without sacrificing CV or completion time."""
        # Transitive gates cannot affect starts or coverage. Remove all gates
        # that remain implied by the current DAG, then rebuild derived state
        # once instead of doing one full refresh per gate.
        implied = []
        for target in sorted(self.simulation.gates, key=self.simulation.rank.__getitem__, reverse=True):
            source = self.simulation.gates[target]
            self.simulation.stats["num_gate_prune_trials"] += 1
            if not self.simulation._gate_has_alternate_path(source, target):
                continue
            implied.append((source, target))
            del self.simulation.gates[target]
            lane = self.simulation.lane_of[target]
            position = self.simulation.positions[target] - 1
            fifo_parent = self.simulation.lanes[lane][position - 1] if position else None
            if source not in self.simulation.base_parents[target] and source != fifo_parent:
                self.simulation.parents[target].discard(source)
            self.simulation.pruned_gates.append(
                {
                    "source": node_id(self.simulation.nodes[source]),
                    "target": node_id(self.simulation.nodes[target]),
                    "reason": "implied_dependency",
                    "cv_delta_ms": 0.0,
                    "cube_mix_aiv_delta_ms": 0.0,
                    "cc_delta_ms": 0.0,
                    "vv_delta_ms": 0.0,
                    "span_reduction_ms": 0.0,
                    "compute_union_reduction_ms": 0.0,
                }
            )
            self.simulation.stats["num_gates_pruned"] += 1
            self.simulation.stats["num_gate_implied_prunes"] += 1
        if implied:
            self.simulation.refresh()

        union = _compute_union_ms(self.simulation.absolute)
        for target in sorted(self.simulation.gates, key=self.simulation.rank.__getitem__, reverse=True):
            source = self.simulation.gates[target]
            candidate = copy.copy(self)
            candidate.simulation = copy.copy(self.simulation)
            candidate.simulation.gates = self.simulation.gates.copy()
            del candidate.simulation.gates[target]
            candidate.simulation.refresh()
            current_metrics = self.simulation._scoring_metrics(self.simulation.coverage, self.simulation.span)
            candidate_metrics = candidate.simulation._scoring_metrics(
                candidate.simulation.coverage, candidate.simulation.span
            )
            if (
                candidate_metrics["cube_covered_by_vector_percent"] + 1e-9
                < current_metrics["cube_covered_by_vector_percent"]
                or candidate_metrics["vector_covered_by_cube_percent"] + 1e-9
                < current_metrics["vector_covered_by_cube_percent"]
                or candidate_metrics["cc_ms"] > current_metrics["cc_ms"] + 1e-9
                or candidate_metrics["vv_ms"] > current_metrics["vv_ms"] + 1e-9
            ):
                continue
            if round(candidate.simulation.span - self.simulation.span, 9) > 0:
                continue
            candidate_union = _compute_union_ms(candidate.simulation.absolute)
            # Preserving CV alone can still lower its share by losing other
            # useful overlap. Keep the compute union from growing as well.
            if round(candidate_union - union, 9) > 0:
                continue
            self.simulation.pruned_gates.append(
                {
                    "source": node_id(self.simulation.nodes[source]),
                    "target": node_id(self.simulation.nodes[target]),
                    "reason": "implied_dependency"
                    if candidate.simulation.clocks[target][self.simulation.lane_of[source]]
                    >= candidate.simulation.positions[source]
                    else "no_predicted_cv_benefit",
                    "cv_delta_ms": candidate.simulation.coverage[0] - self.simulation.coverage[0],
                    "cube_mix_aiv_delta_ms": candidate.simulation.coverage[1] - self.simulation.coverage[1],
                    "cc_delta_ms": candidate.simulation.coverage[2] - self.simulation.coverage[2],
                    "vv_delta_ms": candidate.simulation.coverage[3] - self.simulation.coverage[3],
                    "span_reduction_ms": self.simulation.span - candidate.simulation.span,
                    "compute_union_reduction_ms": union - candidate_union,
                }
            )
            self.simulation.stats["num_gates_pruned"] += 1
            before = self.simulation._benefit_snapshot()
            self.simulation.__dict__.update(candidate.simulation.__dict__)
            self.simulation._record_benefit("wait_event", before)
            union = candidate_union

    def _reorder_proposal_records(
        self,
        sequence: list[int],
        moved_node: int,
        *,
        window_order: _ReorderWindowOrder | None = None,
        moved_position: int | None = None,
        destinations: tuple[int, ...] | None = None,
    ) -> tuple[_ReorderProposal, ...]:
        """Return move templates with reusable hashes and exact changed bounds."""
        window_order = _ReorderWindowOrder(tuple(sequence)) if window_order is None else window_order
        key = (window_order, moved_node, destinations)
        cached = self.simulation.reorder_template_cache.get(key)
        if cached is not None:
            cached_weight, cached_records = cached
            # Keep frequently reused templates resident without changing their
            # deterministic candidate order.
            del self.simulation.reorder_template_cache[key]
            self.simulation.reorder_template_cache[key] = (cached_weight, cached_records)
            self.simulation.stats["num_reorder_template_cache_hits"] += 1
            return cached_records

        self.simulation.stats["num_reorder_template_cache_misses"] += 1
        size = len(sequence)
        position = sequence.index(moved_node) if moved_position is None else moved_position
        ordered_destinations = tuple(i for i in range(size) if i != position) if destinations is None else destinations
        destination_set = None if destinations is None else set(destinations)
        repeated_scans = sum(abs(position - destination) for destination in ordered_destinations)
        self.simulation.stats["num_reorder_dependency_nodes_skipped"] += max(0, repeated_scans - (size - 1))

        moving_by_destination = {}
        moving = {moved_node}
        needed = self.simulation.base_parents[moved_node].copy()
        for destination in range(position - 1, -1, -1):
            candidate = sequence[destination]
            if candidate in needed:
                moving.add(candidate)
                needed.update(self.simulation.base_parents[candidate])
            if destination_set is None or destination in destination_set:
                moving_by_destination[destination] = frozenset(moving)
        moving = {moved_node}
        for destination in range(position + 1, size):
            candidate = sequence[destination]
            if self.simulation.base_parents[candidate] & moving:
                moving.add(candidate)
            if destination_set is None or destination in destination_set:
                moving_by_destination[destination] = frozenset(moving)
        self.simulation.stats["num_reorder_dependency_nodes_scanned"] += size - 1

        record_list: list[_ReorderProposal] = []
        for destination in ordered_destinations:
            left, right = sorted((position, destination))
            segment = sequence[left : right + 1]
            moving = moving_by_destination[destination]
            moved, rest = [], []
            for i in segment:
                (moved if i in moving else rest).append(i)
            replacement = moved + rest if destination < position else rest + moved
            if replacement == segment:
                proposal = window_order.sequence
                record_list.append(_ReorderProposal(proposal, size, -1))
                continue
            proposal = tuple(sequence[:left] + replacement + sequence[right + 1 :])
            changed_left: int | None = None
            changed_right: int | None = None
            for offset, pair in enumerate(zip(segment, replacement, strict=True)):
                if pair[0] != pair[1]:
                    changed_left = offset if changed_left is None else changed_left
                    changed_right = offset
            assert changed_left is not None and changed_right is not None
            record_list.append(_ReorderProposal(proposal, left + changed_left, left + changed_right))
        records = tuple(record_list)

        weight = sum(len(record.sequence) for record in records)
        if 0 < weight <= _REORDER_TEMPLATE_CACHE_NODE_LIMIT:
            while (
                self.simulation.reorder_template_cache
                and self.simulation.reorder_template_cache_nodes + weight > _REORDER_TEMPLATE_CACHE_NODE_LIMIT
            ):
                oldest = next(iter(self.simulation.reorder_template_cache))
                evicted_weight, _ = self.simulation.reorder_template_cache.pop(oldest)
                self.simulation.reorder_template_cache_nodes -= evicted_weight
                self.simulation.stats["num_reorder_template_cache_evictions"] += 1
            self.simulation.reorder_template_cache[key] = (weight, records)
            self.simulation.reorder_template_cache_nodes += weight
        return records

    def reorder(self):
        accepted = 0
        non_fifo_ancestors = self.simulation._non_fifo_ancestors()
        node_bits = self.simulation._node_bits
        lane_has_vector_work = [
            any(phase[2] in {"aiv", "mix_aiv"} for phase in self.simulation.absolute[lane]) for lane in (0, 1)
        ]
        for lane, first, size in self.simulation.windows:
            sequence = self.simulation.lanes[lane][first : first + size]
            members = set(sequence)
            coupled = any(
                any(
                    self.simulation.lane_of[child] != lane and not members <= self.simulation.base_parents[child]
                    for child in self.simulation.children[i]
                )
                for i in sequence
            )
            vector_potential = {}
            vectors = []
            for i in sequence:
                uncovered = [
                    b
                    - a
                    - self.simulation.timelines[1 - lane]["aic"].overlap(
                        self.simulation.starts[i] + a, self.simulation.starts[i] + b
                    )
                    for a, b, kind in self.simulation.phases[i]
                    if kind in {"aiv", "mix_aiv"}
                ]
                if any(round(value, 9) > 0 for value in uncovered):
                    vectors.append(i)
                    vector_potential[i] = sum(max(0.0, value) for value in uncovered)
            vector_nodes = set(vectors)
            cube_potential = {}
            cubes = []
            if lane_has_vector_work[1 - lane]:
                for i in sequence:
                    if i in vector_nodes:
                        continue
                    uncovered = [
                        b
                        - a
                        - self.simulation.timelines[1 - lane]["aiv"].overlap(
                            self.simulation.starts[i] + a, self.simulation.starts[i] + b
                        )
                        for a, b, kind in self.simulation.phases[i]
                        if kind == "aic"
                    ]
                    if any(round(value, 9) > 0 for value in uncovered):
                        cubes.append(i)
                        cube_potential[i] = sum(max(0.0, value) for value in uncovered)
            if self.bounded_search:
                for candidates, potential in ((vectors, vector_potential), (cubes, cube_potential)):
                    if len(candidates) <= _BOUNDED_REORDER_NODE_LIMIT_PER_KIND:
                        continue
                    selected = set(
                        sorted(candidates, key=lambda i: potential[i], reverse=True)[
                            :_BOUNDED_REORDER_NODE_LIMIT_PER_KIND
                        ]
                    )
                    self.simulation.stats["num_reorder_nodes_budget_skipped"] += len(candidates) - len(selected)
                    candidates[:] = [i for i in candidates if i in selected]
            if not vectors and not cubes:
                continue
            evaluated: set[_ReorderProposal] = set()
            _, best = self.simulation._current_scoring_state()
            chosen: tuple[int, ...] | None = None
            chosen_node = None
            chosen_before: list[int] | None = None
            local_score = self.simulation._window_score(sequence, lane=lane, first=first)
            local_score = None if local_score is None else _local_coverage_score(local_score)
            window_members = frozenset(sequence)
            window_upper_rank = max(self.simulation.rank[i] for i in sequence)
            window_order = _ReorderWindowOrder(tuple(sequence))
            window_positions = {node: position for position, node in enumerate(sequence)}
            for moved_node in (*vectors, *cubes):
                moved_position = window_positions[moved_node]
                destinations = None
                if self.bounded_search:
                    destinations = _bounded_destinations(size, moved_position, _BOUNDED_REORDER_DESTINATION_LIMIT)
                    self.simulation.stats["num_reorder_destinations_budget_skipped"] += size - 1 - len(destinations)
                for candidate in self._reorder_proposal_records(
                    sequence,
                    moved_node,
                    window_order=window_order,
                    moved_position=moved_position,
                    destinations=destinations,
                ):
                    proposal = candidate.sequence
                    if candidate.left > candidate.right:
                        continue
                    if candidate in evaluated:
                        self.simulation.stats["num_reorder_cache_hits"] += 1
                        continue
                    evaluated.add(candidate)
                    # Every changed FIFO edge is present in the materialized
                    # schedule. If its child is already a non-FIFO ancestor of its
                    # parent, the proposal must cycle regardless of timing or
                    # optional gates, so reject it before replaying phases.
                    if non_fifo_ancestors is not None:
                        assert node_bits is not None
                        lane_nodes = self.simulation.lanes[lane]
                        last_child_offset = min(candidate.right + 1, len(proposal))
                        has_base_cycle = False
                        for child_offset in range(candidate.left, last_child_offset + 1):
                            lane_position = first + child_offset
                            if lane_position >= len(lane_nodes):
                                break
                            child = (
                                proposal[child_offset] if child_offset < len(proposal) else lane_nodes[lane_position]
                            )
                            if child_offset:
                                parent = proposal[child_offset - 1]
                            elif first:
                                parent = lane_nodes[first - 1]
                            else:
                                continue
                            if non_fifo_ancestors[parent] & node_bits[child]:
                                has_base_cycle = True
                                break
                        if has_base_cycle:
                            self.simulation.stats["num_reorder_cycle_rejected"] += 1
                            self.simulation.stats["num_reorder_non_fifo_cycle_rejected"] += 1
                            self.simulation.stats["num_reorder_non_fifo_cycle_prefilter_rejected"] += 1
                            continue
                    # Local replay is exact when this lane catches up and every
                    # directly affected cross-lane consumer keeps its start.
                    # Cyclic proposals are also safe to reject on a local
                    # no-gain result because the full DAG would reject them.
                    localized_end = []
                    replayed_ends = {}
                    score = self.simulation._window_score(
                        proposal,
                        lane=lane,
                        first=first,
                        changed_bounds=(candidate.left, candidate.right),
                        localized_end=localized_end,
                        replayed_ends=replayed_ends,
                    )
                    if score is None:
                        continue
                    localized = not coupled
                    if coupled and localized_end:
                        cross_children = {
                            child
                            for i, end in replayed_ends.items()
                            if end != self.simulation.ends[i]
                            for child in self.simulation.children[i]
                            if self.simulation.lane_of[child] != lane
                        }
                        localized = all(
                            max(
                                (replayed_ends.get(p, self.simulation.ends[p]) for p in self.simulation.parents[child]),
                                default=0.0,
                            )
                            == self.simulation.starts[child]
                            for child in cross_children
                        )
                    if localized and (local_score is None or _local_coverage_score(score) <= local_score):
                        self.simulation.stats["num_reorder_localized_rejected"] += 1
                        continue
                    score = self.simulation._reorder_trial(
                        lane,
                        first,
                        proposal,
                        changed_bounds=(candidate.left, candidate.right),
                        window_members=window_members,
                        window_upper_rank=window_upper_rank,
                        # ``evaluated`` already deduplicates this exact order;
                        # every accepted edit invalidates the scheduling state.
                        cache_result=False,
                    )
                    if score is not None and score > best:
                        best, chosen, chosen_node, chosen_before = score, proposal, moved_node, sequence
            if chosen is not None:
                assert chosen_node is not None and chosen_before is not None
                before = self.simulation._benefit_snapshot()
                self.simulation.lanes[lane][first : first + size] = chosen
                self.simulation.refresh()
                self.simulation._record_benefit("reorder", before)
                self.simulation.reorder_edits.append(
                    {
                        "vector" if chosen_node in vectors else "cube": node_id(self.simulation.nodes[chosen_node]),
                        "before": [node_id(self.simulation.nodes[i]) for i in chosen_before],
                        "after": [node_id(self.simulation.nodes[i]) for i in chosen],
                    }
                )
                accepted += 1
                self.simulation.stats["num_reorder_edits"] += 1
        return accepted


def build_profile_schedule(
    gm,
    profile,
    *,
    kernel_observations=None,
    reference=None,
):
    """Build one edited free-based plan; the caller owns free fallback."""
    started = time.perf_counter()
    planner = _WindowPlanner(gm, profile, kernel_observations or {}, reference=reference)
    search_started = time.perf_counter()
    search_times = planner.search()
    search_seconds = time.perf_counter() - search_started
    nodes = planner.simulation.nodes
    schedule = replace(
        planner.simulation.reference,
        order=[nodes[i] for i in planner.materialized_order],
        dependencies={nodes[i]: {nodes[p] for p in parents} for i, parents in enumerate(planner.simulation.parents)},
        start_ms={n: planner.simulation.starts[i] for i, n in enumerate(nodes)},
        end_ms={n: planner.simulation.ends[i] for i, n in enumerate(nodes)},
        algorithm="profile_guided_cv",
    )
    before = {n: p for n, p in event_frontiers(planner.simulation.reference).items() if n.op != "output"}
    after = {n: p for n, p in event_frontiers(schedule).items() if n.op != "output"}
    added, removed = len(after.keys() - before.keys()), len(before.keys() - after.keys())
    retargeted = sum(before[n] != after[n] for n in before.keys() & after.keys())
    reordered_nodes = sum(
        original != current
        for original_lane, current_lane in zip(planner.simulation.original_lanes, planner.simulation.lanes, strict=True)
        for original, current in zip(original_lane, current_lane, strict=True)
    )
    attribution = planner.simulation.benefit_attribution()
    optimization_summary = {
        "attribution_method": attribution["method"],
        "reorder": {
            "actions": len(planner.simulation.reorder_edits),
            "reordered_nodes": reordered_nodes,
            "estimated_benefit": attribution["reorder"],
        },
        "wait_event": {
            "accepted_alignment_edits": planner.simulation.stats["num_cv_edits"],
            "alignment_gates_kept": len(planner.simulation.gates),
            "waits_added": added,
            "waits_removed": removed,
            "waits_retargeted": retargeted,
            "event_records_total": len(set(after.values())),
            "estimated_benefit": attribution["wait_event"],
        },
        "total_estimated_benefit": attribution["total"],
    }
    free_prediction = _overlap_metrics(
        planner.simulation.initial_coverage,
        cube_work=planner.simulation.cube_work,
        vector_work=planner.simulation.vector_work,
        compute_union=planner.simulation.initial_compute_union,
        span=planner.simulation.initial_span,
    )
    scheduled_prediction = planner.simulation.overlap_metrics(
        compute_union=_compute_union_ms(planner.simulation.absolute),
        span=planner.simulation.span,
    )
    coverage_improved = predicted_coverage_improved(free_prediction, scheduled_prediction)
    vector_rows = planner.simulation.vector_coverage()
    bounded_regions = planner.simulation.stats["num_bounded_search_regions"]
    stats: dict[str, Any] = {
        "algorithm": schedule.algorithm,
        "search_mode": "adaptive_bounded_region_greedy" if bounded_regions else "exhaustive_region_greedy",
        "search_budget": {
            "exhaustive_region_node_limit": _EXHAUSTIVE_REGION_NODE_LIMIT,
            "bounded_region_max_passes": _BOUNDED_REGION_MAX_PASSES,
            "reorder_moved_nodes_per_kind": _BOUNDED_REORDER_NODE_LIMIT_PER_KIND,
            "reorder_destinations_per_node": _BOUNDED_REORDER_DESTINATION_LIMIT,
            "cv_candidates_per_pass": _BOUNDED_CV_CANDIDATE_LIMIT,
            "cv_sources_per_candidate": _BOUNDED_CV_SOURCE_LIMIT,
        }
        if bounded_regions
        else None,
        "search_cache_limits": {
            "shifted_phase_entries": schedule_simulation._SHIFTED_PHASE_CACHE_LIMIT,
            "trial_entries_per_state": schedule_simulation._TRIAL_CACHE_LIMIT,
            "reorder_template_node_references": _REORDER_TEMPLATE_CACHE_NODE_LIMIT,
        },
        "profile_source": "dependency_only",
        "cost_source": "rank_local_dependency_only",
        "objective": "balanced_directional_cv_then_pure_cv_then_min_same_type_overlap",
        "span_constraint": "each_mandatory_barrier_region_no_longer_than_dependency_only_prediction",
        "cc_policy": "non_increasing_per_accepted_edit",
        "vv_policy": "non_increasing; no_vv_search",
        "prediction_scope": "kernel_phases_within_fx; fixed_free_durations_not_a_contention_model",
        "initialization_wall_seconds": search_started - started,
        "search_wall_seconds": search_seconds,
        "search_stage_seconds": search_times,
        "changed_from_free": bool(added or removed or retargeted or reordered_nodes),
        "predicted_coverage_improved": coverage_improved,
        "prediction": {"free": free_prediction, "scheduled": scheduled_prediction},
        "cost_sources": dict(Counter(planner.simulation.cost_sources)),
        "num_ambiguous_kernel_intervals": planner.simulation.num_ambiguous_intervals,
        "num_waits_free": len(before),
        "num_waits": len(after),
        "num_waits_added": added,
        "num_waits_removed": removed,
        "num_waits_retargeted": retargeted,
        "num_events": len(set(after.values())),
        "num_reordered_nodes": reordered_nodes,
        "reorder_changes": planner.simulation.reorder_edits,
        "optimization_summary": optimization_summary,
        "predicted_cv_ms": planner.simulation.coverage[0],
        "predicted_cube_mix_aiv_ms": planner.simulation.coverage[1],
        "predicted_cc_ms": planner.simulation.coverage[2],
        "predicted_vv_ms": planner.simulation.coverage[3],
        "predicted_free_cv_ms": planner.simulation.initial_coverage[0],
        "predicted_free_cc_ms": planner.simulation.initial_coverage[2],
        "predicted_free_vv_ms": planner.simulation.initial_coverage[3],
        "predicted_cube_covered_by_vector_percent": scheduled_prediction["cube_covered_by_vector_percent"],
        "predicted_vector_covered_by_cube_percent": scheduled_prediction["vector_covered_by_cube_percent"],
        "predicted_free_cube_covered_by_vector_percent": free_prediction["cube_covered_by_vector_percent"],
        "predicted_free_vector_covered_by_cube_percent": free_prediction["vector_covered_by_cube_percent"],
        "predicted_span_ms": planner.simulation.span,
        "predicted_free_span_ms": planner.simulation.initial_span,
        "window_refinement": {
            **planner.simulation.stats,
            "num_gates": len(planner.simulation.gates),
            "gates": [
                {"source": node_id(nodes[s]), "target": node_id(nodes[t])} for t, s in planner.simulation.gates.items()
            ],
            "pruned_gates": planner.simulation.pruned_gates,
        },
        "vector_coverage": vector_rows,
        "vector_coverage_totals": {
            key: sum(row[key] for row in vector_rows)
            for key in ("vector_work_ms", "cube_covered_ms", "not_cube_covered_ms", "vector_covered_ms", "uncovered_ms")
        },
        "wait_changes": [
            {
                "target": node_id(n),
                "free_source": node_id(before[n]) if n in before else None,
                "scheduled_source": node_id(after[n]) if n in after else None,
            }
            for n in schedule.order
            if before.get(n) != after.get(n)
        ],
        "planning_wall_seconds": time.perf_counter() - started,
    }
    logger.info(
        "CV parallel planned optimization: reordered_nodes=%d reorder_actions=%d "
        "wait_events_added=%d wait_events_removed=%d wait_events_retargeted=%d alignment_gates=%d "
        "estimated_reorder_benefit=%s estimated_wait_event_benefit=%s",
        reordered_nodes,
        len(planner.simulation.reorder_edits),
        added,
        removed,
        retargeted,
        len(planner.simulation.gates),
        attribution["reorder"],
        attribution["wait_event"],
    )
    return schedule, stats
