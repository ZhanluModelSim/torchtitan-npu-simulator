# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cost annotation contracts: shape-derived FLOPs and traffic, nothing else."""

from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.cost import annotate_workload_costs, compute_cost
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.tensor_meta import TensorMeta
from torchtitan_simulator.runner import simulate


def _meta(shape, dtype="bfloat16"):
    return TensorMeta(name="t", shape=tuple(shape), dtype=dtype, device="meta")


def _node(op_type, inputs, outputs):
    return OpNode(
        op_id=0,
        op_type=op_type,
        inputs=list(inputs),
        outputs=list(outputs),
        attrs={},
        predecessors=[],
        successors=[],
    )


def test_mm_flops_and_traffic_match_closed_form():
    # A (128, 512) @ B (512, 256) -> C (128, 256), all BF16.  Nodes key the
    # handler by the normalized op_type ("matmul"), not the raw aten name.
    node = _node("matmul", [_meta((128, 512)), _meta((512, 256))], [_meta((128, 256))])
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    assert cost.flops == 2 * 128 * 256 * 512
    # Read A + read B + write C, once each, at their own dtype width.
    assert cost.traffic_bytes == (128 * 512 + 512 * 256 + 128 * 256) * 2
    assert cost.comm_bytes == 0 and not cost.unknown


def test_mm_traffic_counts_mixed_precision_output_width():
    # FP32 accumulate output: C costs twice the BF16 elements.
    node = _node(
        "matmul",
        [_meta((16, 8)), _meta((8, 4))],
        [_meta((16, 4), dtype="float32")],
    )
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    assert cost.flops == 2 * 16 * 4 * 8
    assert cost.traffic_bytes == (16 * 8 + 8 * 4) * 2 + 16 * 4 * 4


def test_mm_rejects_inconsistent_shapes_instead_of_guessing():
    # K mismatch between the operands: unknown, not a made-up number.
    node = _node("matmul", [_meta((4, 6)), _meta((5, 2))], [_meta((4, 2))])
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    assert cost.unknown and cost.flops == 0


def test_unhandled_ops_stay_unknown():
    node = _node("aten.view.default", [_meta((4, 6))], [_meta((6, 4))])
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    assert cost.unknown


def test_qwen3_5_capture_annotates_every_mm_with_the_lm_head_closed_form():
    workload = simulate(
        SimulationConfig(
            config="qwen3_5_debugmodel",
            seq_len=16,
            world_size=1,
            mixed_precision_param="bfloat16",
        )
    )
    annotate_workload_costs(workload)
    nodes = [n for graph in workload.step_templates.values() for n in graph.nodes.values()]
    mms = [n for n in nodes if n.op_type == "matmul"]
    assert mms and all(n.annotations["flops"] > 0 for n in mms)
    assert all(n.annotations["memory_traffic_bytes"] > 0 for n in mms)
    assert all(n.annotations["raw_op_type"] == "aten.mm.default" for n in mms)

    # The lm_head GEMMs over the full vocab run chunked (chunk of 2 tokens):
    # the forward (2,256)@(256,248320), the weight grad (248320,2)@(2,256) and
    # the input grad (2,248320)@(248320,256) -- each 2*M*N*K with the same
    # M*N*K product, once per chunk x 8 chunks x (fwd + bwd).
    _V, _D, _CHUNK = 248_320, 256, 2
    lm_head_flops = 2 * _CHUNK * _V * _D
    heads = [n for n in mms if any(_V in tensor.shape for tensor in n.inputs + n.outputs)]
    assert len(heads) == 24, "expected 8 chunks x (fwd, dW, dx) lm_head GEMMs"
    for node in heads:
        assert node.annotations["flops"] == lm_head_flops
        assert node.annotations["memory_traffic_bytes"] == (
            sum(tensor_volume_bytes(t.shape, t.dtype) for t in node.inputs + node.outputs)
        )

    # Nothing else was guessed: every non-mm node is explicitly unknown.
    stamped = {n.op_type for n in nodes if "flops" in n.annotations}
    assert stamped == {"matmul"}
    unknown = [n for n in nodes if n.annotations.get("cost_unknown")]
    assert unknown and all("flops" not in n.annotations for n in unknown)


def test_mm_duration_stays_absent_without_a_hardware_profile():
    node = _node("matmul", [_meta((128, 512)), _meta((512, 256))], [_meta((128, 256))])
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    assert cost.duration_us is None
    node2 = _node("matmul", [_meta((128, 512)), _meta((512, 256))], [_meta((128, 256))])
    cost2 = compute_cost(node2.op_type, node2.inputs, node2.outputs, hardware=None)
    assert cost2.duration_us is None


def test_mm_duration_replays_calibration_points_within_measured_band():
    # The utilization curve is calibrated on idle-device microbenchmarks
    # only (square ladder + K=4096 grid, no workload shapes).  Regular
    # shapes replay tightly; deep-K / tiny-M shapes sit below the 1-D FLOPs
    # curve by design and only satisfy the loose limitation envelope
    # documented in cost/hardware.py.
    from torchtitan_simulator.cost.hardware import npu_910b3

    regular = [
        ((4096, 4096), (4096, 4096), 503.6),
        ((1024, 4096), (4096, 4096), 142.6),
        ((260, 1024), (1024, 1024), 22.2),
    ]
    deep_k = [
        ((64, 248320), (248320, 256), 145.4),
        ((64, 256), (256, 248320), 300.8),
    ]
    hardware = npu_910b3()

    def measured_ratio(a_shape, b_shape, measured_us):
        node = _node("matmul", [_meta(a_shape), _meta(b_shape)], [_meta((a_shape[0], b_shape[1]))])
        cost = compute_cost(node.op_type, node.inputs, node.outputs, hardware=hardware)
        assert cost.duration_us is not None and cost.duration_us > 0
        return cost.duration_us / measured_us

    for a_shape, b_shape, measured_us in regular:
        ratio = measured_ratio(a_shape, b_shape, measured_us)
        assert 0.4 < ratio < 2.5, f"{a_shape}x{b_shape}: model {cost.duration_us}us vs {measured_us}us"
    for a_shape, b_shape, measured_us in deep_k:
        ratio = measured_ratio(a_shape, b_shape, measured_us)
        assert 0.15 < ratio < 0.6, f"{a_shape}x{b_shape}: model {cost.duration_us}us vs {measured_us}us"


def test_mm_duration_skips_unsupported_dtypes_but_keeps_work():
    from torchtitan_simulator.cost.hardware import npu_910b3

    node = _node(
        "matmul",
        [_meta((16, 8), dtype="int64"), _meta((8, 4), dtype="int64")],
        [_meta((16, 4), dtype="int64")],
    )
    cost = compute_cost(node.op_type, node.inputs, node.outputs, hardware=npu_910b3())
    assert cost.flops == 2 * 16 * 4 * 8 and not cost.unknown
    assert cost.duration_us is None


def test_capture_with_hardware_annotates_duration_on_every_mm():
    workload = simulate(
        SimulationConfig(
            config="qwen3_5_debugmodel",
            seq_len=16,
            world_size=1,
            mixed_precision_param="bfloat16",
        )
    )
    from torchtitan_simulator.cost.hardware import npu_910b3

    annotate_workload_costs(workload, hardware=npu_910b3())
    nodes = [n for graph in workload.step_templates.values() for n in graph.nodes.values()]
    mms = [n for n in nodes if n.op_type == "matmul"]
    assert mms and all(
        "duration_us" in n.annotations and n.annotations["duration_us"] > 0 and n.annotations["duration_us"] < 1e9
        for n in mms
    )
    # Non-mm nodes carry no duration: only matmul is modeled.
    non_mm_durations = [n for n in nodes if n.op_type != "matmul" and "duration_us" in n.annotations]
    assert not non_mm_durations


def test_capture_without_hardware_writes_no_duration():
    workload = simulate(
        SimulationConfig(
            config="qwen3_5_debugmodel",
            seq_len=16,
            world_size=1,
            mixed_precision_param="bfloat16",
        )
    )
    annotate_workload_costs(workload)
    nodes = [n for graph in workload.step_templates.values() for n in graph.nodes.values()]
    assert all("duration_us" not in n.annotations for n in nodes)
