"""Tests for standalone compute, grouped-MM, and permutation benchmarks."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import compute_benchmark
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.compute_benchmark import (
    ComputeBenchmarker,
    _balanced_group_boundaries,
    _grouped_mm_benchmark_spec,
    _has_concrete_tensor_inputs,
    _is_npu_benchmark_compute_node,
    balanced_routing_indices,
    benchmark_npu_grouped_mm_node,
    benchmark_npu_permutation_node,
)


def _make_binary_node(target, shape=(4, 8)):
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    lhs.meta["val"] = torch.empty(shape, device="meta")
    rhs.meta["val"] = torch.empty((shape[-1], shape[-1]), device="meta")
    return graph.call_function(target, (lhs, rhs))


def _mark_shared_expert_compute(node):
    node.meta["custom"] = {
        "EP": "compute",
        "module_fqn": "layers.1.moe.shared_experts.w1",
    }
    return node


def test_compute_benchmarker_dispatches_supported_node_types(monkeypatch):
    do_bench = object()
    make_calls = []
    calls = []
    monkeypatch.setattr(
        compute_benchmark,
        "make_npu_do_bench",
        lambda: make_calls.append(True) or do_bench,
    )
    monkeypatch.setattr(
        compute_benchmark,
        "benchmark_npu_permutation_node",
        lambda node, backend, **kwargs: calls.append(("permutation", node, backend, kwargs)),
    )
    monkeypatch.setattr(
        compute_benchmark,
        "benchmark_npu_grouped_mm_node",
        lambda node, backend, **kwargs: calls.append(("gmm", node, backend, kwargs)),
    )
    monkeypatch.setattr(
        compute_benchmark,
        "benchmark_npu_compute_node",
        lambda node, backend, **kwargs: calls.append(("compute", node, backend, kwargs)),
    )

    permutation = SimpleNamespace(target="npu.npu_moe_token_permute.default")
    grouped_mm = SimpleNamespace(target="aten._grouped_mm.default")
    compute = SimpleNamespace(target="aten.mm.default")
    benchmarker = ComputeBenchmarker()

    benchmarker.benchmark(permutation, permutation_assumption=(12, 4))
    benchmarker.benchmark(grouped_mm, assumed_tokens=12)
    benchmarker.benchmark(compute)

    assert len(make_calls) == 1
    assert [kind for kind, *_ in calls] == ["permutation", "gmm", "compute"]
    assert all(call[2] is do_bench for call in calls)
    assert calls[0][3]["routed_tokens"] == 12
    assert calls[0][3]["experts"] == 4
    assert calls[1][3]["assumed_tokens"] == 12


def test_matmul_is_npu_benchmark_compute_node():
    node = _mark_shared_expert_compute(_make_binary_node(torch.ops.aten.matmul.default))

    assert _has_concrete_tensor_inputs(node)
    assert _is_npu_benchmark_compute_node(node)


def test_lightweight_floating_point_target_is_selected_by_capability():
    node = _mark_shared_expert_compute(_make_binary_node(torch.ops.aten.add.Tensor))

    assert _is_npu_benchmark_compute_node(node)


def test_unannotated_matmul_is_selected_for_standalone_benchmark():
    node = _make_binary_node(torch.ops.aten.matmul.default)

    assert _is_npu_benchmark_compute_node(node)


def test_static_activation_and_softmax_targets_are_selected():
    targets = (
        torch.ops.aten._log_softmax.default,
        torch.ops.aten._log_softmax_backward_data.default,
        torch.ops.aten._softmax.default,
        torch.ops.aten._softmax_backward_data.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_backward.default,
    )

    for target in targets:
        assert _is_npu_benchmark_compute_node(_make_binary_node(target))


def test_grouped_mm_balanced_boundaries_support_zero_token_experts():
    counts, boundaries = _balanced_group_boundaries(tokens=3, experts=5)

    assert counts == (1, 1, 1, 0, 0)
    assert boundaries == (1, 2, 3, 3, 3)


def test_grouped_mm_balanced_boundaries_support_empty_input():
    counts, boundaries = _balanced_group_boundaries(tokens=0, experts=4)

    assert counts == (0, 0, 0, 0)
    assert boundaries == (0, 0, 0, 0)


def test_grouped_mm_spec_handles_forward_and_weight_gradient_shapes():
    offsets = torch.empty((4,), dtype=torch.int64)
    forward = _grouped_mm_benchmark_spec(
        torch.empty((10, 8)),
        torch.empty((4, 8, 16)),
        offsets,
    )
    wgrad = _grouped_mm_benchmark_spec(
        torch.empty((8, 10)),
        torch.empty((10, 16)),
        offsets,
    )

    assert forward is not None and forward.tokens == 10
    assert wgrad is not None and wgrad.tokens == 10
    assert forward.experts == wgrad.experts == 4
    assert forward.boundaries[-1] == wgrad.boundaries[-1] == 10


def test_grouped_mm_spec_uses_assumed_tokens_for_dynamic_routing_dimension():
    offsets = torch.empty((4,), dtype=torch.int64)
    forward = _grouped_mm_benchmark_spec(
        SimpleNamespace(shape=(None, 8)),
        torch.empty((4, 8, 16)),
        offsets,
        assumed_tokens=12,
    )
    wgrad = _grouped_mm_benchmark_spec(
        SimpleNamespace(shape=(8, None)),
        SimpleNamespace(shape=(None, 16)),
        offsets,
        assumed_tokens=12,
    )

    assert forward is not None and forward.tokens == 12
    assert wgrad is not None and wgrad.tokens == 12
    assert forward.counts == wgrad.counts == (3, 3, 3, 3)


def test_grouped_mm_is_selected_for_benchmark_and_uses_distribution_cache_key():
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    offsets = graph.placeholder("offsets")
    lhs.meta["val"] = torch.empty((7, 8))
    rhs.meta["val"] = torch.empty((4, 8, 16))
    offsets.meta["val"] = torch.empty((4,), dtype=torch.int64)
    node = graph.call_function(
        torch.ops.aten._grouped_mm.default,
        (lhs, rhs, offsets),
    )

    assert _is_npu_benchmark_compute_node(node)
    result = benchmark_npu_grouped_mm_node(node, lambda _callable: 1.25)

    assert result is not None
    elapsed_ms, key = result
    assert elapsed_ms == 1.25
    assert "tokens=7" in key
    assert "experts=4" in key
    assert "distribution=(2, 2, 2, 1)" in key


def test_empty_grouped_mm_returns_zero_without_backend_launch():
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    offsets = graph.placeholder("offsets")
    lhs.meta["val"] = torch.empty((0, 8))
    rhs.meta["val"] = torch.empty((4, 8, 16))
    offsets.meta["val"] = torch.empty((4,), dtype=torch.int64)
    node = graph.call_function(
        torch.ops.aten._grouped_mm.default,
        (lhs, rhs, offsets),
    )

    def should_not_run(_callable):
        raise AssertionError("zero-token GMM must not launch the backend")

    result = benchmark_npu_grouped_mm_node(node, should_not_run)

    assert result is not None and result[0] == 0.0


def test_grouped_mm_benchmark_accepts_offsets_keyword_argument():
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    offsets = graph.placeholder("offsets")
    lhs.meta["val"] = torch.empty((9, 8))
    rhs.meta["val"] = torch.empty((4, 8, 16))
    offsets.meta["val"] = torch.empty((4,), dtype=torch.int64)
    node = graph.call_function(
        torch.ops.aten._grouped_mm.default,
        (lhs, rhs),
        {"offs": offsets},
    )

    result = benchmark_npu_grouped_mm_node(node, lambda _callable: 2.5)

    assert result is not None
    assert result[0] == 2.5
    assert "tokens=9" in result[1]


def test_balanced_routing_preserves_topk_and_balances_experts():
    indices = balanced_routing_indices(11, 3, 8, device="cpu", dtype=torch.int32)
    counts = torch.bincount(indices.flatten().long(), minlength=8)

    assert int(counts.max() - counts.min()) <= 1
    assert all(row.unique().numel() == 3 for row in indices)


@pytest.mark.parametrize(
    ("mode", "dynamic"),
    [
        ("permute", False),
        ("permute", True),
        ("unpermute", False),
        ("weighted_unpermute", False),
        ("weighted_unpermute", True),
        ("wrapper_unpermute", False),
        ("unpermute", True),
    ],
)
def test_permutation_benchmark_inputs_and_cache(monkeypatch, mode, dynamic):
    from torch._inductor.fx_passes import overlap_scheduling

    permute = mode == "permute"
    weighted = mode in {"weighted_unpermute", "wrapper_unpermute"}
    namespace = "torchtitan_npu" if mode == "wrapper_unpermute" else "npu"
    name = "npu_moe_token_permute" if permute else "npu_moe_token_unpermute"
    signature = (
        "Tensor tokens, Tensor indices, int? num_out_tokens=None, bool padded_mode=False"
        if permute
        else "Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None"
    )
    captured = []

    class Target:
        _schema = torch._C.parse_schema(f"{namespace}::{name}({signature}) -> Tensor")

        def __str__(self):
            return f"{namespace}.{name}.default"

        def __call__(self, *args, **kwargs):
            captured.append(args)

    # Model the runtime-produced inverse; verify it is passed through unchanged.
    inverse = torch.arange(12, dtype=torch.int32).flip(0)
    native_calls = []

    def native_permute(tokens, routing):
        native_calls.append(routing)
        assert torch.bincount(routing.flatten().long(), minlength=4).tolist() == [3] * 4
        return tokens, inverse

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_moe_token_permute",
        SimpleNamespace(default=native_permute),
        raising=False,
    )
    tokens = torch.empty_strided((6 if permute else 12, 5), (1, 12))
    indices = (
        torch.empty_strided((6, 2), (1, 6), dtype=torch.int32)
        if permute
        else torch.empty_strided((12,), (2,), dtype=torch.int32)
    )
    probs = torch.empty_strided((6, 2), (1, 6)) if weighted else None
    if dynamic:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        env = ShapeEnv()
        with FakeTensorMode(shape_env=env):
            if permute:
                source_rows = env.create_unbacked_symint()
                tokens = torch.empty_strided((source_rows, 5), (5, 1))
                indices = torch.empty_strided(
                    (source_rows, 2),
                    (2, 1),
                    dtype=torch.int32,
                )
            else:
                routed_rows = env.create_unbacked_symint()
                tokens = torch.empty_strided((routed_rows, 5), (5, 1))
                indices = torch.empty_strided(
                    (routed_rows,),
                    (1,),
                    dtype=torch.int32,
                )
                if weighted:
                    source_rows = env.create_unbacked_symint()
                    probs = torch.empty_strided(
                        (source_rows, 2),
                        (2, 1),
                    )
    args = (tokens, indices) if permute else (tokens, indices, probs)
    monkeypatch.setattr(
        torch._inductor.fx_utils,
        "get_fake_args_kwargs",
        lambda node: (True, args, {}),
    )
    cache = {}
    monkeypatch.setattr(overlap_scheduling, "get_cached_node_time", cache.get)
    monkeypatch.setattr(
        overlap_scheduling,
        "set_cached_node_time",
        lambda key, value: cache.__setitem__(key, value),
    )
    node = SimpleNamespace(op="call_function", target=Target())

    def bench(fn):
        fn()
        return 0.125

    result = benchmark_npu_permutation_node(
        node,
        bench,
        routed_tokens=12,
        experts=4,
    )
    assert result[0] == 0.125
    assert (
        benchmark_npu_permutation_node(
            node,
            bench,
            routed_tokens=12,
            experts=4,
        )
        == result
    )
    assert len(captured) == 1
    actual = captured[0]
    expected_token_shape = (6, 5) if dynamic and permute else ((12, 5) if dynamic else tokens.shape)
    assert actual[0].shape == expected_token_shape
    assert actual[0].stride() == tokens.stride()
    assert actual[1].stride() == indices.stride()
    assert actual[1].dtype == indices.dtype
    if permute:
        assert not native_calls
        assert torch.bincount(actual[1].flatten().long()).tolist() == [3] * 4
    else:
        assert len(native_calls) == 1
        assert torch.equal(actual[1], inverse)
        if weighted:
            assert actual[2].stride() == probs.stride()
            torch.testing.assert_close(actual[2].sum(-1), torch.ones(6))


def test_permute_dropped_tokens_is_not_silently_benchmarked(monkeypatch):
    class Target:
        _schema = torch._C.parse_schema(
            "npu::npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None) -> Tensor"
        )

        def __str__(self):
            return "npu.npu_moe_token_permute.default"

        def __call__(self, *args, **kwargs):
            pytest.fail("unexpected permutation call")

    monkeypatch.setattr(
        torch._inductor.fx_utils,
        "get_fake_args_kwargs",
        lambda node: (
            True,
            (torch.empty(6, 5), torch.empty(6, 2, dtype=torch.int32), 5),
            {},
        ),
    )
    result = benchmark_npu_permutation_node(
        SimpleNamespace(op="call_function", target=Target()),
        lambda fn: pytest.fail("unexpected benchmark"),
        routed_tokens=12,
        experts=4,
    )

    assert result is None
