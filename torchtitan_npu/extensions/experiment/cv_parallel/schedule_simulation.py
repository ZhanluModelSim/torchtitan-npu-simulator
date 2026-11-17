# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local schedule state, incremental replay and prediction caches."""

from __future__ import annotations

import heapq
import math
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import accumulate, islice, pairwise
from typing import Any, cast

from .cv_parallel import _can_reorder, _is_stream_barrier, dependency_only_schedule
from .schedule_scoring import _metric_score, _overlap_metrics, candidate_score
from .whole_graph_runtime_profile import _is_metadata_only_compute_node, node_id

_COVERAGE_CACHE_LIMIT = 16384
_SHIFTED_PHASE_CACHE_LIMIT = 65536
_TRIAL_CACHE_LIMIT = 65536
_ANCESTOR_NODE_LIMIT = 8192
_CACHE_MISS = object()

_COVERAGE_FIELDS = ("cv_ms", "mixed_cv_ms", "cc_ms", "vv_ms")
_ATTRIBUTION_FIELDS = (*_COVERAGE_FIELDS, "span_ms")


def _evict_oldest_cache_quarter(cache, limit):
    """Make bounded-cache room without discarding the still-useful majority."""
    if len(cache) < limit:
        return 0
    keys = list(islice(cache, max(1, limit // 4)))
    for key in keys:
        del cache[key]
    return len(keys)


class _ReorderProposal:
    """A full proposal with exact changed bounds and a reusable tuple hash."""

    __slots__ = ("_hash", "left", "right", "sequence")

    def __init__(self, sequence: tuple[int, ...], left: int, right: int):
        self.sequence = sequence
        self.left = left
        self.right = right
        self._hash = 0 if left > right else hash(sequence)

    def __hash__(self):
        return self._hash

    def __eq__(self, other: object):
        if not isinstance(other, _ReorderProposal):
            return NotImplemented
        return self.sequence == other.sequence


class _ReorderWindowOrder:
    """An exact window order whose tuple hash is computed once."""

    __slots__ = ("_hash", "sequence")

    def __init__(self, sequence: tuple[int, ...]):
        self.sequence = sequence
        self._hash = hash(sequence)

    def __hash__(self):
        return self._hash

    def __eq__(self, other: object):
        if not isinstance(other, _ReorderWindowOrder):
            return NotImplemented
        return self.sequence == other.sequence


class _Timeline:
    def __init__(self, *, cache=False):
        self.starts, self.ends, self.prefix = [], [], [0.0]
        self.cache = {} if cache else None

    def append(self, start, end):
        self.starts.append(start)
        self.ends.append(end)
        self.prefix.append(self.prefix[-1] + end - start)
        if self.cache:
            self.cache.clear()

    def covered_until(self, timestamp):
        if self.cache is not None:
            cached = self.cache.get(timestamp)
            if cached is not None:
                return cached
        index = bisect_right(self.ends, timestamp)
        covered = self.prefix[index] + (max(0.0, timestamp - self.starts[index]) if index < len(self.starts) else 0.0)
        if self.cache is not None:
            # Bound memory independently of the number of candidate trials.
            _evict_oldest_cache_quarter(self.cache, _COVERAGE_CACHE_LIMIT)
            self.cache[timestamp] = covered
        return covered

    def overlap(self, start, end):
        if not self.ends:
            return 0.0
        return self.covered_until(end) - self.covered_until(start)


def _timelines(phases, *, cache=False):
    result = defaultdict(lambda: _Timeline(cache=cache))
    for start, end, kind, *_ in phases:
        result[kind].append(start, end)
    return result


def _coverage(phases, other, *, end_ms=math.inf):
    """Return disjoint pure CV, mixed CV, CC and VV overlap durations."""
    result = [0.0, 0.0, 0.0, 0.0]
    aic_overlap = other["aic"].overlap
    aiv_overlap = other["aiv"].overlap
    mix_aic_overlap = other["mix_aic"].overlap
    mix_aiv_overlap = other["mix_aiv"].overlap
    for start, end, kind, *_ in phases:
        end = min(end, end_ms)
        if end <= start:
            continue
        if kind == "aic":
            result[0] += aiv_overlap(start, end)
            result[1] += mix_aiv_overlap(start, end)
            result[2] += aic_overlap(start, end)
        elif kind == "aiv":
            result[0] += aic_overlap(start, end)
            result[1] += mix_aic_overlap(start, end)
            result[3] += aiv_overlap(start, end)
        elif kind == "mix_aiv":
            result[1] += aic_overlap(start, end)
        elif kind == "mix_aic":
            result[1] += aiv_overlap(start, end)
    return result


def _compute_union_ms(lanes):
    total = end = 0.0
    for start, stop, *_ in heapq.merge(*lanes):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _kernel_phases(intervals):
    """Keep internal C/V phases and gaps, without treating a MIX kernel as pure work."""
    if not intervals:
        return [], 0.0
    origin = min(start for start, _, _, _ in intervals)
    events = defaultdict(list)
    for start, end, _, kind in set(intervals):
        if not all(math.isfinite(value) for value in (start, end)) or end < start:
            raise ValueError("Invalid kernel phase interval")
        if end > start:
            events[start - origin].append((kind, 1))
            events[end - origin].append((kind, -1))
    active, phases = Counter(), []
    previous = 0.0
    for timestamp in sorted(events):
        kinds = {kind for kind, count in active.items() if count > 0 and kind != "non_cv"}
        if timestamp > previous and kinds:
            kind = next(iter(kinds)) if len(kinds) == 1 else "mix"
            if phases and phases[-1][1] == previous and phases[-1][2] == kind:
                phases[-1] = (phases[-1][0], timestamp, kind)
            else:
                phases.append((previous, timestamp, kind))
        for kind, delta in events[timestamp]:
            active[kind] += delta
        previous = timestamp
    return phases, previous


class ScheduleSimulation:
    """Own graph state and caches for dependency-safe candidate replay."""

    def __init__(self, gm, profile, kernel_observations, *, reference=None):
        self.reference = dependency_only_schedule(gm) if reference is None else reference
        self.nodes = self.reference.order
        index = {node: i for i, node in enumerate(self.nodes)}
        self.barriers = {index[node] for node in self.nodes if _is_stream_barrier(node)}
        self.lane_of = [self.reference.lanes[node] for node in self.nodes]
        self.lanes = [[i for i, lane in enumerate(self.lane_of) if lane == k] for k in (0, 1)]
        self.positions = {i: p + 1 for lane in self.lanes for p, i in enumerate(lane)}
        self.base_parents: list[set[int]] = [{index[p] for p in self.reference.dependencies[n]} for n in self.nodes]
        self.original_lanes = [lane.copy() for lane in self.lanes]
        windows: list[list[int]] = []
        pending = [[], []]
        for node in gm.graph.nodes:
            if _can_reorder(node):
                pending[self.reference.lanes[node]].append(index[node])
            else:
                windows.extend(window for window in pending if len(window) > 1)
                pending = [[], []]
        original_children = defaultdict(set)
        for child, parents in enumerate(self.base_parents):
            for parent in parents:
                original_children[parent].add(child)
        for window in windows:
            anchors = self.base_parents[window[0]].copy()
            for previous, current in pairwise(window):
                if self.nodes[previous] not in self.nodes[current].all_input_nodes:
                    self.base_parents[current].discard(previous)
            for i in window:
                self.base_parents[i].update(anchors)
            # Keep FIFO/effect boundaries behind the whole window. A reviewed
            # pure consumer on the other lane needs only its actual inputs.
            for child in original_children[window[-1]]:
                if self.lane_of[child] == self.lane_of[window[-1]] or not _can_reorder(self.nodes[child]):
                    self.base_parents[child].update(window)
        self.windows = [(self.lane_of[window[0]], self.positions[window[0]] - 1, len(window)) for window in windows]
        by_target = defaultdict(list)
        for node in self.nodes:
            value = profile.local_ms.get(node_id(node), 0.0)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid free profile cost: {node.name}={value}")
            if value > 0:
                by_target[node.target].append(value)
        positive = [value for values in by_target.values() for value in values]
        if not positive:
            raise ValueError("Free profile contains no measured device work")
        fallback = statistics.median(positive)
        owners = Counter(interval for rows in kernel_observations.values() for interval in set(rows))
        self.durations, self.phases, self.cost_sources = [], [], []
        self.num_ambiguous_intervals = sum(count > 1 for count in owners.values())
        for node in self.nodes:
            sid = node_id(node)
            metadata = node.op in {"placeholder", "get_attr", "output"} or _is_metadata_only_compute_node(node)
            resource = "metadata" if metadata else profile.local_resources.get(sid, "other")
            measured = profile.local_ms.get(sid)
            duration = (
                0.0
                if metadata
                else measured
                if measured is not None
                else statistics.median(by_target[node.target] or [fallback])
            )
            source = "metadata" if metadata else "free_measured" if measured is not None else "free_estimated"
            intervals = kernel_observations.get(sid, ())
            # Ambiguous ownership is not additional work. Keep its time occupied,
            # but do not optimize coverage against an invented duplicate phase.
            intervals = [
                (a, b, stream, kind if owners[a, b, stream, kind] == 1 else "other") for a, b, stream, kind in intervals
            ]
            phases, envelope = _kernel_phases(intervals)
            if metadata:
                phases = []
            elif intervals:
                duration = max(duration, envelope)
                source = "free_kernel_envelope"
            elif resource in {"aic", "aiv", "mix_aic", "mix_aiv"} and duration > 0:
                phases = [(0.0, duration, resource)]
            self.durations.append(duration)
            self.phases.append(phases)
            self.cost_sources.append(source)
        self.gates = {}
        self.stats = Counter()
        self.reorder_edits = []
        self.pruned_gates = []
        self.benefit_by_method = {method: Counter() for method in ("reorder", "wait_event")}
        self.score_context: dict[str, Any] | None = None
        self.shifted_phase_cache = {}
        self.reorder_template_cache: dict[
            tuple[_ReorderWindowOrder, int, tuple[int, ...] | None],
            tuple[int, tuple[_ReorderProposal, ...]],
        ] = {}
        self.reorder_template_cache_nodes = 0
        self.state_version = 0
        self.trial_cache = {}
        self.reorder_trial_cache = {}
        self._current_score_cache = None
        self._non_fifo_ancestor_bits: list[int] | None = None
        self._node_bits: list[int] | None = None
        self._non_fifo_gate_signature: tuple[tuple[int, int], ...] | None = None
        self._reset_trial_workspaces()
        self.refresh()
        self.span_limit = self.span
        self.initial_coverage = self.coverage.copy()
        self.initial_span = self.span
        self.initial_compute_union = _compute_union_ms(self.absolute)

    def overlap_metrics(self, coverage=None, *, compute_union=None, span=None):
        return _overlap_metrics(
            self.coverage if coverage is None else coverage,
            cube_work=self.cube_work,
            vector_work=self.vector_work,
            compute_union=compute_union,
            span=span,
        )

    def _current_scoring_state(self):
        """Return non-regression guards and score for the current search state."""
        cached = self._current_score_cache
        if cached is not None:
            return cached
        metrics = self._scoring_metrics(self.coverage, self.span)
        guards = (
            metrics["cube_covered_by_vector_percent"],
            metrics["vector_covered_by_cube_percent"],
            metrics["cc_ms"],
            metrics["vv_ms"],
        )
        cached = (guards, _metric_score(metrics))
        self._current_score_cache = cached
        return cached

    def _scoring_metrics(self, coverage, span):
        """Evaluate a regional state against the current whole-graph totals."""
        context = self.score_context
        if context is None:
            return self.overlap_metrics(coverage, span=span)
        return _overlap_metrics(
            [outside + local for outside, local in zip(context["coverage"], coverage, strict=True)],
            cube_work=context["cube_work"],
            vector_work=context["vector_work"],
            span=context["span"] + span,
        )

    def _benefit_snapshot(self):
        metrics = self.overlap_metrics(span=self.span)
        return {field: metrics[field] for field in _ATTRIBUTION_FIELDS}

    def _record_benefit(self, method, before):
        after = self._benefit_snapshot()
        self.benefit_by_method[method].update({field: after[field] - before[field] for field in _ATTRIBUTION_FIELDS})

    def benefit_attribution(self):
        """Attribute the final predicted delta to reorder and wait edits.

        CV/CC/VV changes telescope along the accepted search path. Directional
        ratios are nonlinear, so use a two-factor Shapley estimate over the
        accumulated raw deltas. The synthetic one-method states need not be
        independently materializable; this is diagnostic attribution only.
        """

        def state(methods):
            coverage = self.initial_coverage.copy()
            span = self.initial_span
            for method in methods:
                values = self.benefit_by_method[method]
                for index, field in enumerate(_COVERAGE_FIELDS):
                    coverage[index] += values[field]
                span += values["span_ms"]
            return _overlap_metrics(
                coverage,
                cube_work=self.cube_work,
                vector_work=self.vector_work,
                span=span,
            )

        baseline = state(())
        reorder = state(("reorder",))
        wait = state(("wait_event",))
        final = self.overlap_metrics(span=self.span)

        def shapley(field, method):
            own, other = (reorder, wait) if method == "reorder" else (wait, reorder)
            return 0.5 * ((own[field] - baseline[field]) + (final[field] - other[field]))

        def benefits(method):
            return {
                "cv_gain_ms": shapley("cv_ms", method),
                "mixed_cv_gain_ms": shapley("mixed_cv_ms", method),
                "cc_reduction_ms": -shapley("cc_ms", method),
                "vv_reduction_ms": -shapley("vv_ms", method),
                "span_reduction_ms": -shapley("span_ms", method),
                "cube_coverage_gain_percentage_points": shapley("cube_covered_by_vector_percent", method),
                "vector_coverage_gain_percentage_points": shapley("vector_covered_by_cube_percent", method),
            }

        total = {
            "cv_gain_ms": final["cv_ms"] - baseline["cv_ms"],
            "mixed_cv_gain_ms": final["mixed_cv_ms"] - baseline["mixed_cv_ms"],
            "cc_reduction_ms": baseline["cc_ms"] - final["cc_ms"],
            "vv_reduction_ms": baseline["vv_ms"] - final["vv_ms"],
            "span_reduction_ms": baseline["span_ms"] - final["span_ms"],
            "cube_coverage_gain_percentage_points": final["cube_covered_by_vector_percent"]
            - baseline["cube_covered_by_vector_percent"],
            "vector_coverage_gain_percentage_points": final["vector_covered_by_cube_percent"]
            - baseline["vector_covered_by_cube_percent"],
        }
        return {
            "method": "two_factor_shapley_from_sequential_predicted_deltas",
            "reorder": benefits("reorder"),
            "wait_event": benefits("wait_event"),
            "total": total,
        }

    def _candidate_score(self, coverage, span):
        current, current_score = self._current_scoring_state()
        score = candidate_score(current, current_score, self._scoring_metrics(coverage, span))
        if score is None:
            self.stats["num_coverage_or_same_type_rejected"] += 1
        return score

    def refresh(self):
        self.stats["num_full_refreshes"] += 1
        self.stats["num_schedule_states"] += 1
        self.state_version += 1
        gate_signature = tuple(sorted(self.gates.items()))
        if gate_signature != self._non_fifo_gate_signature:
            self._non_fifo_ancestor_bits = None
            self._non_fifo_gate_signature = gate_signature
        # Trial scores depend on clocks, starts, coverage and gates. A refresh
        # creates a new scheduling state, so no result may cross this boundary.
        self.stats["num_trial_cache_entries_invalidated"] += len(self.trial_cache) + len(self.reorder_trial_cache)
        self.trial_cache = {}
        self.reorder_trial_cache = {}
        self._current_score_cache = None
        self.window_cache = {}
        self.window_coverage_cache = {}
        size = len(self.nodes)
        self.positions = {i: p + 1 for lane in self.lanes for p, i in enumerate(lane)}
        self.parents = [
            parents | ({self.gates[i]} if i in self.gates else set()) for i, parents in enumerate(self.base_parents)
        ]
        for lane in self.lanes:
            for previous, current in pairwise(lane):
                self.parents[current].add(previous)
        self.children = [[] for _ in self.nodes]
        remaining = [len(parents) for parents in self.parents]
        for i, parents in enumerate(self.parents):
            for parent in parents:
                self.children[parent].append(i)
        self.starts, self.ends, self.order = [0.0] * size, [0.0] * size, []
        ready = [(0.0, i) for i, count in enumerate(remaining) if not count]
        heapq.heapify(ready)
        while ready:
            start, i = heapq.heappop(ready)
            self.starts[i], self.ends[i] = start, start + self.durations[i]
            self.order.append(i)
            for child in self.children[i]:
                remaining[child] -= 1
                if not remaining[child]:
                    heapq.heappush(ready, (max(self.ends[p] for p in self.parents[child]), child))
        if len(self.order) != size:
            raise RuntimeError("CV window dependency cycle")
        self.rank = {i: rank for rank, i in enumerate(self.order)}
        self.span = max(self.ends, default=0.0)
        self.remaining_ms = self.durations.copy()
        for i in reversed(self.order):
            self.remaining_ms[i] += max((self.remaining_ms[c] for c in self.children[i]), default=0.0)
        self.clocks = [[0, 0] for _ in self.nodes]
        for i in self.order:
            for parent in self.parents[i]:
                self.clocks[i][0] = max(self.clocks[i][0], self.clocks[parent][0])
                self.clocks[i][1] = max(self.clocks[i][1], self.clocks[parent][1])
            self.clocks[i][self.lane_of[i]] = self.positions[i]
        self.absolute = self.shifted(self.starts)
        self.timelines = [_timelines(phases, cache=True) for phases in self.absolute]
        self.coverage = _coverage(self.absolute[0], self.timelines[1])
        if getattr(self, "_work_phases", None) is not self.phases:
            self.cube_work = sum(b - a for phases in self.phases for a, b, kind in phases if kind == "aic")
            self.vector_work = sum(b - a for phases in self.phases for a, b, kind in phases if kind == "aiv")
            self._work_phases = self.phases
        else:
            self.stats["num_work_summaries_reused"] += 1
        self.lane_ends = [[self.ends[i] for i in lane] for lane in self.lanes]
        # After a join, both FIFO tails depend on the completed prefix. Moving
        # that prefix only translates the suffix; its internal coverage is fixed.
        self.joins = set()
        for i in self.order:
            lane = self.lane_of[i]
            other = self.lanes[1 - lane]
            following = self.clocks[i][1 - lane]
            if following == len(other) or self.clocks[other[following]][lane] >= self.positions[i]:
                self.joins.add(i)
        self.join_ranks = sorted(self.rank[i] for i in self.joins)

    def _shifted_node_phases(self, i, start):
        if not self.phases[i]:
            return ()
        key = (i, start)
        cached = self.shifted_phase_cache.get(key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached
        self.stats["num_shifted_phase_cache_misses"] += 1
        shifted = tuple((start + a, start + b, kind, i, p) for p, (a, b, kind) in enumerate(self.phases[i]))
        if _SHIFTED_PHASE_CACHE_LIMIT > 0:
            evicted = _evict_oldest_cache_quarter(self.shifted_phase_cache, _SHIFTED_PHASE_CACHE_LIMIT)
            if evicted:
                self.stats["num_shifted_phase_cache_evictions"] += evicted
            self.shifted_phase_cache[key] = shifted
        return shifted

    def shifted(self, starts):
        result = [[], []]
        requests = 0
        misses_before = self.stats["num_shifted_phase_cache_misses"]
        if isinstance(starts, dict):
            for i in starts:
                requests += bool(self.phases[i])
                result[self.lane_of[i]].extend(self._shifted_node_phases(i, starts[i]))
            for phases in result:
                phases.sort()
        else:
            # FIFO edges make node starts monotonic within each lane, and a
            # node's phases are already ordered. Preserve that order directly
            # instead of sorting every phase again after each accepted edit.
            for lane, nodes in enumerate(self.lanes):
                for i in nodes:
                    requests += bool(self.phases[i])
                    result[lane].extend(self._shifted_node_phases(i, starts[i]))
            self.stats["num_full_phase_sorts_skipped"] += sum(map(len, result))
        misses = self.stats["num_shifted_phase_cache_misses"] - misses_before
        self.stats["num_shifted_phase_cache_hits"] += requests - misses
        return result

    def _store_trial_result(self, *, reorder, key, value):
        if key is None:
            return value
        cache = self.reorder_trial_cache if reorder else self.trial_cache
        if _TRIAL_CACHE_LIMIT > 0:
            if len(cache) >= _TRIAL_CACHE_LIMIT:
                cache.clear()
                self.stats["num_trial_cache_clears"] += 1
            cache[key] = value
        return value

    def _reset_trial_workspaces(self):
        """Allocate reusable candidate-propagation storage for the current nodes."""
        size = len(self.nodes)
        self._trial_generation = 0
        self._trial_queued_marks = [0] * size
        self._trial_start_marks = [0] * size
        self._trial_start_values = [0.0] * size
        self._trial_changed_nodes: list[int] = []
        self._trial_ready: list[int] = []

        self._reorder_generation = 0
        self._reorder_dfs_generation = 0
        self._reorder_end_marks = [0] * size
        self._reorder_end_values = [0.0] * size
        self._reorder_queued_marks = [0] * size
        self._reorder_active_marks = [0] * size
        self._reorder_start_values = [0.0] * size
        self._reorder_changed_nodes: list[int] = []
        self._reorder_pending: list[int] = []
        self._reorder_ready: list[int] = []
        self._reorder_stack: list[int] = []

    def _non_fifo_ancestors(self):
        """Return ancestors through dependencies that a reorder cannot remove."""
        if len(self.nodes) > _ANCESTOR_NODE_LIMIT:
            return None
        cached = self._non_fifo_ancestor_bits
        if cached is not None:
            return cached
        node_bits = self._node_bits
        if node_bits is None:
            node_bits = [1 << i for i in range(len(self.nodes))]
            self._node_bits = node_bits
        ancestors: list[int] = [0] * len(self.nodes)
        for i in self.order:
            bits: int = 0
            parents = self.base_parents[i]
            gate = self.gates.get(i)
            for parent in parents if gate is None else (*parents, gate):
                parent_index = cast("int", parent)
                bits |= ancestors[parent_index] | node_bits[parent_index]
            ancestors[i] = bits
        self._non_fifo_ancestor_bits = ancestors
        self.stats["num_reorder_non_fifo_ancestor_cache_builds"] += 1
        return ancestors

    def _trial_node_coverage(self, changed_nodes, start_values, end_ms):
        phase_starts = {i: start_values[i] for i in changed_nodes if self.phases[i]}
        return self._phase_start_coverage(phase_starts, end_ms)

    def _phase_start_coverage(self, phase_starts, end_ms):
        if not phase_starts:
            return self.coverage
        old = self.shifted({i: self.starts[i] for i in phase_starts})
        new = self.shifted(phase_starts)
        changed_lanes = (bool(old[0]), bool(old[1]))
        if changed_lanes.count(True) == 1:
            lane = changed_lanes.index(True)
            opposite = 1 - lane
            updated = _coverage(new[lane], self.timelines[opposite], end_ms=end_ms)
            previous = _coverage(old[lane], self.timelines[opposite], end_ms=end_ms)
            return [
                current + (after - before)
                for current, after, before in zip(self.coverage, updated, previous, strict=True)
            ]
        old_t, new_t = [_timelines(p) for p in old], [_timelines(p) for p in new]
        # Exclude the old, unshifted suffix from comparisons with a moved prefix.
        # Inclusion-exclusion counts two changed endpoints only once.
        terms = [
            (1, _coverage(new[0], self.timelines[1], end_ms=end_ms)),
            (1, _coverage(new[1], self.timelines[0], end_ms=end_ms)),
            (-1, _coverage(old[0], self.timelines[1], end_ms=end_ms)),
            (-1, _coverage(old[1], self.timelines[0], end_ms=end_ms)),
            (-1, _coverage(new[0], old_t[1])),
            (-1, _coverage(old[0], new_t[1])),
            (1, _coverage(old[0], old_t[1])),
            (1, _coverage(new[0], new_t[1])),
        ]
        return [self.coverage[k] + sum(sign * values[k] for sign, values in terms) for k in range(4)]

    def trial(self, source, target, *, cache_result=True):
        key = (self.state_version, source, target) if cache_result else None
        if key is not None:
            cached = self.trial_cache.get(key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                self.stats["num_trial_cache_hits"] += 1
                return cached
        self.stats["num_trials"] += 1
        if self.clocks[source][self.lane_of[target]] >= self.positions[target]:
            self.stats["num_cycle_rejected"] += 1
            return self._store_trial_result(reorder=False, key=key, value=None)
        if round(self.ends[source] - self.starts[target], 9) <= 0:
            self.stats["num_cv_noop_rejected"] += 1
            return self._store_trial_result(reorder=False, key=key, value=None)
        generation = self._trial_generation + 1
        self._trial_generation = generation
        queued_marks = self._trial_queued_marks
        start_marks = self._trial_start_marks
        start_values = self._trial_start_values
        changed_nodes = self._trial_changed_nodes
        ready = self._trial_ready
        changed_nodes.clear()
        ready.clear()
        queued_marks[target] = generation
        ready.append(self.rank[target])
        num_queued = 1
        end_ms, span = math.inf, self.span
        while ready:
            i = self.order[heapq.heappop(ready)]
            start = 0.0
            for parent in self.parents[i]:
                parent_start = start_values[parent] if start_marks[parent] == generation else self.starts[parent]
                parent_end = parent_start + self.durations[parent]
                start = max(start, parent_end)
            if i == target:
                start = max(start, self.ends[source])
            if round(start - self.starts[i], 9) <= 0:
                continue
            # An added edge cannot shorten the existing downstream critical path.
            if round(start + self.remaining_ms[i] - self.span_limit, 9) > 0:
                self.stats["num_span_rejected"] += 1
                return self._store_trial_result(reorder=False, key=key, value=None)
            start_marks[i] = generation
            start_values[i] = start
            changed_nodes.append(i)
            span = max(span, start + self.durations[i])
            if i in self.joins:
                end_ms = self.ends[i]
                span = self.span + start - self.starts[i]
                self.stats["num_suffix_nodes_skipped"] += len(self.nodes) - self.rank[i] - 1
                break
            for child in self.children[i]:
                if queued_marks[child] != generation:
                    queued_marks[child] = generation
                    num_queued += 1
                    heapq.heappush(ready, self.rank[child])
        self.stats["num_propagated_nodes"] += num_queued
        if not changed_nodes:
            return self._store_trial_result(reorder=False, key=key, value=None)
        coverage = self._trial_node_coverage(changed_nodes, start_values, end_ms)
        score = self._candidate_score(coverage, span)
        if score is None:
            self.stats["num_no_gain_rejected"] += 1
            return self._store_trial_result(reorder=False, key=key, value=None)
        return self._store_trial_result(reorder=False, key=key, value=score)

    def _reorder_trial(
        self,
        lane,
        first,
        proposal,
        *,
        changed_bounds=None,
        window_members=None,
        window_upper_rank=None,
        cache_result=True,
    ):
        key = (self.state_version, lane, first, tuple(proposal)) if cache_result else None
        if key is not None:
            cached = self.reorder_trial_cache.get(key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                self.stats["num_reorder_trial_cache_hits"] += 1
                return cached
        self.stats["num_reorder_trials"] += 1
        if changed_bounds is None:
            # Preserve the full-window reference path for direct callers and
            # equivalence tests. Search supplies exact bounds below.
            left, changed_right = 0, len(proposal) - 1
        else:
            left, changed_right = changed_bounds
        upper = max(self.rank[i] for i in proposal) if window_upper_rank is None else window_upper_rank
        right = bisect_right(self.join_ranks, upper)
        moved = set(proposal) if window_members is None else window_members
        # A join on the other lane may wait for the window's former tail.
        # Do not truncate there if moving that tail could expose later work.
        while right < len(self.join_ranks):
            boundary = self.order[self.join_ranks[right]]
            tail = self.lanes[lane][self.clocks[boundary][lane] - 1]
            if self.lane_of[boundary] == lane or tail not in moved or tail == proposal[-1]:
                break
            right += 1
        stop = self.join_ranks[right] + 1 if right < len(self.join_ranks) else len(self.order)
        patched: dict[int, set[int]] = {}
        previous = proposal[left - 1] if left else self.lanes[lane][first - 1] if first else None
        patch_nodes = list(proposal[left : changed_right + 1])
        following_position = first + changed_right + 1
        if following_position < len(self.lanes[lane]):
            following = (
                proposal[changed_right + 1]
                if changed_right + 1 < len(proposal)
                else self.lanes[lane][following_position]
            )
            patch_nodes.append(following)
        for i in patch_nodes:
            patched[i] = self.base_parents[i] | ({self.gates[i]} if i in self.gates else set())
            if previous is not None:
                patched[i].add(previous)
            previous = i
        patched = {i: parents for i, parents in patched.items() if parents != self.parents[i] and self.rank[i] < stop}
        if not patched:
            self.stats["num_reorder_noop_rejected"] += 1
            return self._store_trial_result(reorder=True, key=key, value=None)
        non_fifo_ancestors = self._non_fifo_ancestors()
        if non_fifo_ancestors is not None:
            node_bits = self._node_bits
            assert node_bits is not None
            for i, parents in patched.items():
                non_fifo = self.base_parents[i] | ({self.gates[i]} if i in self.gates else set())
                if any(non_fifo_ancestors[parent] & node_bits[i] for parent in parents - non_fifo):
                    self.stats["num_reorder_cycle_rejected"] += 1
                    self.stats["num_reorder_non_fifo_cycle_rejected"] += 1
                    return self._store_trial_result(reorder=True, key=key, value=None)
        first_changed = min((self.positions[i] for i in patched), default=len(self.lanes[lane]) + 1)
        generation = self._reorder_generation + 1
        self._reorder_generation = generation
        end_marks = self._reorder_end_marks
        end_values = self._reorder_end_values
        queued_marks = self._reorder_queued_marks
        active_marks = self._reorder_active_marks
        start_values = self._reorder_start_values
        changed_nodes = self._reorder_changed_nodes
        pending = self._reorder_pending
        ready = self._reorder_ready
        stack = self._reorder_stack
        changed_nodes.clear()
        pending.clear()
        pending.extend(patched)
        ready.clear()
        stack.clear()
        num_ends = 0
        # Resolve changed-edge targets and detect cycles before propagating times.
        # All other parent sets retain their original topological order.
        while pending or ready:
            self._reorder_dfs_generation += 1
            dfs_generation = self._reorder_dfs_generation
            stack.append(pending.pop() if pending else ~self.order[heapq.heappop(ready)])
            while stack:
                marker = stack.pop()
                resolved = marker < 0
                i = ~marker if resolved else marker
                if end_marks[i] == generation:
                    continue
                parents = patched.get(i, self.parents[i])
                if not resolved:
                    if active_marks[i] == dfs_generation:
                        self.stats["num_reorder_cycle_rejected"] += 1
                        return self._store_trial_result(reorder=True, key=key, value=None)
                    active_marks[i] = dfs_generation
                    stack.append(~i)
                    # Only descendants of the first changed FIFO position can
                    # change time or enter a new cycle. Parallel prefixes stay fixed.
                    stack.extend(
                        parent
                        for parent in parents
                        if self.clocks[parent][lane] >= first_changed and end_marks[parent] != generation
                    )
                    continue
                start = 0.0
                for parent in parents:
                    parent_end = end_values[parent] if end_marks[parent] == generation else self.ends[parent]
                    start = max(start, parent_end)
                end_marks[i] = generation
                end_values[i] = start + self.durations[i]
                num_ends += 1
                if start != self.starts[i]:
                    start_values[i] = start
                    changed_nodes.append(i)
                    # Added edges already have their targets in pending.
                    for child in self.children[i]:
                        if (
                            self.rank[child] < stop
                            and end_marks[child] != generation
                            and queued_marks[child] != generation
                        ):
                            queued_marks[child] = generation
                            heapq.heappush(ready, self.rank[child])
        self.stats["num_reorder_evaluated_nodes"] += num_ends
        self.stats["num_reorder_nodes_skipped"] += len(self.nodes) - num_ends
        if not changed_nodes:
            self.stats["num_reorder_noop_rejected"] += 1
            return self._store_trial_result(reorder=True, key=key, value=None)
        boundary = self.order[stop - 1]
        boundary_end = end_values[boundary] if end_marks[boundary] == generation else self.ends[boundary]
        span = self.span + boundary_end - self.ends[boundary]
        if round(span - self.span_limit, 9) > 0:
            self.stats["num_reorder_span_rejected"] += 1
            return self._store_trial_result(reorder=True, key=key, value=None)
        coverage = self._trial_node_coverage(changed_nodes, start_values, self.ends[boundary])
        return self._store_trial_result(reorder=True, key=key, value=self._candidate_score(coverage, span))

    def _window_coverage(self, i, start):
        # Permutations often place the same node at the same time against an
        # unchanged opposite timeline. Invalidate on every accepted DAG edit.
        key = (i, start)
        covered = self.window_coverage_cache.get(key)
        if covered is None:
            phases = self.phases[i]
            covered = (
                _coverage(
                    ((start + a, start + b, kind) for a, b, kind in phases),
                    self.timelines[1 - self.lane_of[i]],
                )
                if phases
                else [0.0, 0.0, 0.0, 0.0]
            )
            evicted = _evict_oldest_cache_quarter(self.window_coverage_cache, _COVERAGE_CACHE_LIMIT)
            if evicted:
                self.stats["num_window_coverage_cache_evictions"] += evicted
            self.window_coverage_cache[key] = covered
        else:
            self.stats["num_window_coverage_cache_hits"] += 1
        return covered

    def _window_score(
        self,
        sequence,
        *,
        lane=None,
        first=None,
        changed_bounds=None,
        localized_end=None,
        replayed_ends=None,
    ):
        """Replay the changed portion; accepted permutations still need DAG validation."""
        lane = self.lane_of[sequence[0]] if lane is None else lane
        first = min(self.positions[i] for i in sequence) - 1 if first is None else first
        key = (lane, first, len(sequence))
        if key not in self.window_cache:
            original = self.lanes[lane][first : first + len(sequence)]
            members = set(original)
            internal, external = {}, {}
            prefix: list[tuple[float, ...]] = [(0.0, 0.0, 0.0, 0.0)]
            for i in original:
                parents = self.base_parents[i] | ({self.gates[i]} if i in self.gates else set())
                internal[i] = parents & members
                external[i] = max((self.ends[p] for p in parents - members), default=0.0)
                covered = self._window_coverage(i, self.starts[i])
                prefix.append(tuple(a + b for a, b in zip(prefix[-1], covered, strict=True)))
            self.window_cache[key] = original, internal, external, prefix
        original, internal, external, prefix = self.window_cache[key]
        if changed_bounds is None:
            changed = [p for p, (a, b) in enumerate(zip(original, sequence, strict=True)) if a != b]
            if not changed:
                return tuple(round(value, 9) for value in prefix[-1])
            left, right = changed[0], changed[-1]
        else:
            left, right = changed_bounds
        cursor = self.ends[self.lanes[lane][first + left - 1]] if first + left else 0.0
        ends, coverage = {}, list(prefix[left])
        evaluated_nodes = 0
        skipped_nodes = left
        for position in range(left, len(sequence)):
            i = sequence[position]
            if any(p not in ends and self.positions[p] > first + left for p in internal[i]):
                self.stats["num_window_nodes_evaluated"] += evaluated_nodes
                return None
            start = max(cursor, external[i])
            covered = self._window_coverage(i, start)
            coverage = [a + b for a, b in zip(coverage, covered, strict=True)]
            ends[i] = cursor = start + self.durations[i]
            evaluated_nodes += 1
            # Once FIFO catches up after the permutation, all remaining inputs
            # and external readiness times match the unchanged suffix.
            if position >= right and cursor == self.ends[original[position]]:
                coverage = [a + b - c for a, b, c in zip(coverage, prefix[-1], prefix[position + 1], strict=True)]
                skipped_nodes += len(sequence) - position - 1
                if localized_end is not None:
                    localized_end.append(position)
                break
        self.stats["num_window_nodes_evaluated"] += evaluated_nodes
        self.stats["num_window_nodes_skipped"] += skipped_nodes
        if localized_end and replayed_ends is not None:
            replayed_ends.update(ends)
        return tuple(round(value, 9) for value in coverage)

    def _gate_has_alternate_path(self, source, target):
        """Return whether target already waits for source without its optional gate."""
        if source in self.base_parents[target]:
            return True
        lane = self.lane_of[target]
        position = self.positions[target] - 1
        if position and self.lanes[lane][position - 1] == source:
            return True
        pending = list(self.parents[target] - {source})
        visited = set()
        while pending:
            node = pending.pop()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            pending.extend(self.parents[node])
        return False

    def vector_coverage(self):
        rows = []
        cube_work = [sum((b - a for a, b, kind in phases if kind == "aic"), 0.0) for phases in self.phases]
        cube_prefix = [list(accumulate((cube_work[i] for i in lane), initial=0.0)) for lane in self.lanes]
        dependency_positions = [[self.clocks[i][1 - lane] for i in nodes] for lane, nodes in enumerate(self.lanes)]
        for i, phases in enumerate(self.phases):
            work = cv = vv = 0.0
            for a, b, kind in phases:
                if kind not in {"aiv", "mix_aiv"}:
                    continue
                work += b - a
                other = self.timelines[1 - self.lane_of[i]]
                cv += other["aic"].overlap(self.starts[i] + a, self.starts[i] + b)
                if kind == "aiv":
                    vv += other["aiv"].overlap(self.starts[i] + a, self.starts[i] + b)
            if work:
                remaining = max(0.0, work - cv - vv)
                other_lane = 1 - self.lane_of[i]
                left = self.clocks[i][other_lane]
                right = bisect_left(dependency_positions[other_lane], self.positions[i])
                capacity = max(0.0, cube_prefix[other_lane][right] - cube_prefix[other_lane][left])
                rows.append(
                    {
                        "node_id": node_id(self.nodes[i]),
                        "target": str(self.nodes[i].target),
                        "vector_work_ms": work,
                        "cube_covered_ms": cv,
                        "not_cube_covered_ms": max(0.0, work - cv),
                        "independent_cube_work_ms": capacity,
                        "cube_gap_reason": None
                        if round(work - cv, 9) <= 0
                        else "no_independent_cube_in_final_dag"
                        if capacity == 0
                        else "insufficient_independent_cube_work"
                        if round(work - capacity, 9) > 0
                        else "event_boundary_or_local_search_limit",
                        "vector_covered_ms": vv,
                        "uncovered_ms": remaining,
                        "uncovered_reason": None if round(remaining, 9) == 0 else "remaining_after_legal_local_search",
                    }
                )
        return rows
