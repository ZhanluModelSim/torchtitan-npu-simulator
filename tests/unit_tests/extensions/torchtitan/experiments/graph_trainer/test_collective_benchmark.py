"""Tests for standalone collective benchmarks and CANN cost extraction."""

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.collective_benchmark import (
    _a2a_splits_for_rank,
    _balanced_splits,
    _build_a2a_split_matrix,
    _CollectiveBenchmarkSpec,
    _contiguous_strides,
    _extract_hcom_durations_ms,
    benchmark_collective_with_npu_events,
    benchmark_collectives_with_cann_profiler,
)


@pytest.mark.parametrize(
    ("total", "parts", "expected"),
    [
        (0, 4, [0, 0, 0, 0]),
        (8, 4, [2, 2, 2, 2]),
        (10, 4, [3, 3, 2, 2]),
    ],
)
def test_balanced_splits(total, parts, expected):
    assert _balanced_splits(total, parts) == expected


def test_a2a_split_matrix_is_valid_for_uneven_rank_inputs():
    rows_per_rank = [8, 10, 3, 0]
    matrix = _build_a2a_split_matrix(rows_per_rank)

    for rank, rows in enumerate(rows_per_rank):
        input_splits, output_splits = _a2a_splits_for_rank(matrix, rank)
        assert sum(input_splits) == rows
        assert output_splits == [matrix[source][rank] for source in range(4)]


def test_a2a_split_matrix_requires_square_matrix():
    with pytest.raises(ValueError, match="square"):
        _a2a_splits_for_rank([[1, 2], [3]], rank=0)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ([8, 2048], [2048, 1]),
        ([2, 3, 4], [12, 4, 1]),
        ([0, 2048], [2048, 1]),
    ],
)
def test_contiguous_strides(shape, expected):
    assert _contiguous_strides(shape) == expected


def test_extract_hcom_durations_uses_longest_contained_event():
    marker = "NPU_COMM_BENCH::0::0"
    trace = {
        "traceEvents": [
            {"name": marker, "ph": "X", "ts": 100.0, "dur": 1000.0},
            {"name": "hcom_alltoallv", "ph": "X", "ts": 200.0, "dur": 600.0},
            {"name": "hcom_link", "ph": "X", "ts": 250.0, "dur": 200.0},
            # A barrier from outside the marker must not be attributed to it.
            {"name": "hcom_allreduce", "ph": "X", "ts": 1200.0, "dur": 50.0},
        ]
    }

    assert _extract_hcom_durations_ms(trace, [marker]) == {marker: 0.6}


def test_extract_hcom_durations_associates_async_events_by_issue_order():
    marker0 = "NPU_COMM_BENCH::0::0"
    marker1 = "NPU_COMM_BENCH::0::1"
    trace = [
        {"name": marker0, "ph": "X", "ts": 100.0, "dur": 20.0},
        {"name": marker1, "ph": "X", "ts": 130.0, "dur": 20.0},
        {"name": "hcom_alltoallv_0", "ph": "X", "ts": 300.0, "dur": 10.0},
        {"name": "hcom_alltoallv_1", "ph": "X", "ts": 320.0, "dur": 30.0},
    ]

    assert _extract_hcom_durations_ms(trace, [marker0, marker1]) == {
        marker0: 0.01,
        marker1: 0.03,
    }


@pytest.mark.parametrize(("total", "parts"), [(-1, 2), (1, 0)])
def test_balanced_splits_rejects_invalid_inputs(total, parts):
    with pytest.raises(ValueError):
        _balanced_splits(total, parts)


def test_event_benchmark_uses_npu_a2a_materializer(monkeypatch):
    from types import SimpleNamespace

    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import collective_benchmark

    node = SimpleNamespace(target=torch.ops._c10d_functional.all_to_all_single.default)
    calls = []
    monkeypatch.setattr(
        collective_benchmark,
        "_benchmark_a2a_with_npu_events",
        lambda passed_node, nruns, **kwargs: calls.append((passed_node, nruns, kwargs)) or (2.0, "a2a-event"),
    )

    result = benchmark_collective_with_npu_events(
        node,
        nruns=5,
        uniform_dispatch_rows=24576,
        reverse_uniform_splits=True,
        generic_benchmark=lambda *_args: pytest.fail("used generic benchmark"),
    )

    assert result == (2.0, "a2a-event")
    assert calls == [
        (
            node,
            5,
            {
                "uniform_dispatch_rows": 24576,
                "reverse_uniform_splits": True,
            },
        )
    ]


def test_event_benchmark_delegates_non_a2a_collectives():
    from types import SimpleNamespace

    node = SimpleNamespace(target=object())
    calls = []

    result = benchmark_collective_with_npu_events(
        node,
        nruns=3,
        generic_benchmark=lambda passed_node, nruns: calls.append((passed_node, nruns)) or (1.0, "generic-event"),
    )

    assert result == (1.0, "generic-event")
    assert calls == [(node, 3)]


def test_cann_batch_deduplicates_before_materialization(monkeypatch):
    from torch._inductor.fx_passes import node_runtime_estimation

    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import collective_benchmark

    class Node:
        def __init__(self, name, signature):
            self.name = name
            self.signature = signature

    nodes = [Node("a0", "payload"), Node("a1", "payload"), Node("a2", "score")]
    monkeypatch.setattr(node_runtime_estimation, "can_benchmark_collective", lambda: True)
    monkeypatch.setattr(node_runtime_estimation, "get_cached_runtime", lambda _key: None)
    monkeypatch.setattr(
        collective_benchmark,
        "_collective_preliminary_key",
        lambda node, **_kwargs: (node.signature,),
    )

    described = []

    def describe(node, **_kwargs):
        described.append(node.name)
        return _CollectiveBenchmarkSpec(
            node=node,
            process_group="ep",
            cache_key=node.signature,
            materialize=lambda: pytest.fail("materialized before unique profiling"),
        )

    monkeypatch.setattr(collective_benchmark, "_describe_collective_for_cann", describe)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 1)

    def all_gather(output, value, **_kwargs):
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather)
    profiled = []
    monkeypatch.setattr(
        collective_benchmark,
        "_profile_one_collective_spec",
        lambda spec, **_kwargs: profiled.append(spec.cache_key) or {"payload": 1.0, "score": 2.0}[spec.cache_key],
    )

    result = benchmark_collectives_with_cann_profiler([(node, 12, False) for node in nodes])

    assert described == ["a0", "a2"]
    assert profiled == ["payload", "score"]
    assert result == {nodes[0]: 1.0, nodes[1]: 1.0, nodes[2]: 2.0}
