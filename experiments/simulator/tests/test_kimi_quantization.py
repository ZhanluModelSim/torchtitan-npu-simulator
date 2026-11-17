# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native MXFP8 Kimi expert capture keeps SiTU and all gradient paths."""

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


@pytest.mark.parametrize(
    "world,replicate,tp,cp,ep,rank,ac",
    [
        (1, 1, 1, 1, 1, 0, "none"),
        (4, 1, 1, 1, 2, 0, "full"),
        (8, 1, 2, 1, 2, 0, "selective"),
        (16, 2, 2, 2, 2, 15, "none"),
    ],
    ids=["single", "ep-full-ac", "ep-tp-selective-ac", "all-last"],
)
def test_kimi_native_mxfp8_quantizes_experts_without_changing_situ_or_gradient_coverage(
    world, replicate, tp, cp, ep, rank, ac
):
    workload = simulate(
        SimulationConfig(
            config="kimi_k3_smoketest",
            seq_len=128,
            world_size=world,
            data_parallel_replicate_degree=replicate,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
            expert_parallel_degree=ep,
            logical_rank=rank,
            enable_quantized_training=True,
            activation_checkpoint_mode=ac,
        )
    )
    annotations = workload.iteration.schedule.annotations
    assert validate_workload(workload)["graph_acyclic"]
    assert len(annotations["gradient_validation"]) == 110
    selected = set(annotations["quantization_configuration"]["quantized_parameters"])
    expected = {
        f"layers.{layer}.moe.{suffix}"
        for layer in range(1, 4)
        for suffix in (
            "routed_experts.inner_experts.w1_EFD",
            "routed_experts.inner_experts.w2_EDF",
            "routed_experts.inner_experts.w3_EFD",
            "shared_experts.w1.weight",
            "shared_experts.w2.weight",
            "shared_experts.w3.weight",
        )
    }
    assert selected == expected
    events = annotations["memory_plan"].raw_events
    raw_ops = {event.raw_op_type for event in events}
    assert "npu.npu_quant_matmul.default" in raw_ops
    assert "npu.npu_grouped_matmul.default" in raw_ops
    assert "kimi_sim.situ_glu.default" in raw_ops and "kimi_sim.situ_glu_backward.default" in raw_ops
    assert not any("swiglu" in op.lower() for op in raw_ops)
    forward_situ = [
        event
        for event in events
        if event.raw_op_type == "kimi_sim.situ_glu.default" and event.execution_kind != "recompute"
    ]
    assert len(forward_situ) == 7
    assert any(ref.dtype == "float8_e4m3fn" for event in events for ref in event.outputs)
    if ac != "none":
        assert any(event.execution_kind == "recompute" for event in events)
