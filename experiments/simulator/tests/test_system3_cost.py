"""Cost annotation of the system3 workload: library routing plus fallback gaps.

This workspace's simulator carried no cost layer before system3; the module
mirrors the qwen3.5 workspace's ``torchtitan_simulator.cost``.  These tests
pin the system3 multimodal workload's coverage: routed families (matmul/bmm,
grouped GEMM, convolution through operator-performance-models), the declared
equivalent-op fallbacks, and the residual ``cost_unknown`` population (view/
stream/profiler control nodes carry no arithmetic and stay unknown, matching
the established workspace convention).
"""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.cost.hardware import npu_910b3
from torchtitan_simulator.cost.op_cost_model import annotate_workload_costs
from torchtitan_simulator.runner import simulate

_VIEW_CANONICAL = {"view", "transpose", "metadata_view", "stream_wait"}


@pytest.fixture(scope="module")
def _annotated_nodes() -> list:
    workload = simulate(SimulationConfig(config="system3_debugmodel_multimodal", seq_len=32))
    annotate_workload_costs(workload, npu_910b3())
    return [n for template in workload.step_templates.values() for n in template.nodes.values()]


def test_system3_convolution_routes_through_the_library(_annotated_nodes) -> None:
    conv_nodes = [n for n in _annotated_nodes if n.annotations.get("raw_op_type", "").startswith("aten.convolution")]
    assert len(conv_nodes) == 8  # 4 forward stems + 4 backward
    for node in conv_nodes:
        assert "cost_unknown" not in node.annotations
        assert node.annotations.get("flops", 0) > 0
        assert node.annotations.get("memory_traffic_bytes", 0) > 0
        assert node.annotations.get("duration_us", 0) > 0
    forward = [n for n in conv_nodes if n.annotations["raw_op_type"] == "aten.convolution.default"]
    # image stem conv2d: 2 * 1*64*8*8 * (3*8*8) MACs -> 1,572,864 forward FLOPs
    image = next(n for n in forward if tuple(n.inputs[0].shape) == (1, 3, 64, 64))
    assert image.annotations["flops"] == 2 * 64 * 64 * 192


def test_system3_routed_families_are_annotated(_annotated_nodes) -> None:
    nodes = _annotated_nodes
    annotated_by_canonical = Counter(n.op_type for n in nodes if "cost_unknown" not in n.annotations)
    total_by_canonical = Counter(n.op_type for n in nodes)
    # Library-routed cube families
    assert annotated_by_canonical["matmul"] > 0
    assert annotated_by_canonical["bmm"] > 0
    assert annotated_by_canonical["grouped_mm"] == total_by_canonical["grouped_mm"] > 0
    # Every gelu canonical node got a handler (system3 encoders use gelu)
    assert annotated_by_canonical["gelu"] == total_by_canonical["gelu"] > 0


def test_system3_residual_unknowns_are_control_or_view_nodes(_annotated_nodes) -> None:
    nodes = _annotated_nodes
    unknown_raw = Counter(n.annotations.get("raw_op_type", "?") for n in nodes if "cost_unknown" in n.annotations)
    # Profiler markers and stream waits are capture/control noise; the rest
    # are view/metadata rewrites with no arithmetic. All of them stay unknown
    # on the same convention as the qwen3.5 workspace; everything else (the
    # eager system3 math) is annotated.
    assert set(unknown_raw) == {
        # view / metadata rewrites (no arithmetic)
        "aten._unsafe_view.default",
        "aten.alias.default",
        "aten.detach.default",
        "aten.expand.default",
        "aten.new_empty_strided.default",
        "aten.permute.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.dim",
        "aten.t.default",
        "aten.transpose.int",
        "aten.unsqueeze.default",
        "aten.view.default",
        # capture / control noise
        "profiler._record_function_enter_new.default",
        "profiler._record_function_exit._RecordFunction",
        "simulator.stream_wait",
    }


def test_system3_fallback_families_cover_the_eager_math(_annotated_nodes) -> None:
    nodes = _annotated_nodes
    fallback_families = (
        "aten.mul.Tensor",
        "aten.add.Tensor",
        "aten.select_backward.default",
        "aten.sum.dim_IntList",
        "aten.embedding.default",
        "aten.topk.default",
        "aten.sigmoid.default",
    )
    for raw in fallback_families:
        members = [n for n in nodes if n.annotations.get("raw_op_type") == raw]
        assert members, f"{raw} absent from capture"
        assert all("cost_unknown" not in n.annotations for n in members), f"{raw} not annotated"
    # The declared multi-output generalization (values + index streams, split
    # chunks) routes the sort/split spellings the single-output contract used
    # to reject; the GDN2 chunked fusion absorbed the eager constant_pad_nd /
    # masked_fill_ branches, so they no longer appear in the capture.
    for raw in ("aten.sort.stable", "aten.split.Tensor"):
        members = [n for n in nodes if n.annotations.get("raw_op_type") == raw]
        assert members, f"{raw} absent from capture"
        assert all("cost_unknown" not in n.annotations for n in members), f"{raw} not annotated"
    fused = ("aten.constant_pad_nd.default", "aten.masked_fill_.Scalar")
    assert not [n for n in nodes if n.annotations.get("raw_op_type") in fused]
