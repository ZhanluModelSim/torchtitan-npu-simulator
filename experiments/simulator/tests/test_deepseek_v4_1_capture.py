# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 text-capture contracts: metadata-map adapter, engram bridge."""

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Native deepseek_v4_1_debugmodel geometry, text-only: dim=512, 40 layers,
# vocab=129280, hc_mult=4, 16 experts, FP32 parameters.  The stack carries two
# host-offload Engram tables (one per KV-source layer) with 2,097,152 FP32
# elements in total; production keeps them CPU-resident, so they report the
# host budget, never device bytes (PR #19 review).  40 odd-numel dense
# parameters skew the FSDP2 shards by one element each (shard 0 up,
# shard 1 down); the degree-1 shard needs no padding.
_ENGRAM_NUMEL = 2_097_152
_TOTAL_NUMEL = 430_312_240
_DENSE_NUMEL = _TOTAL_NUMEL - _ENGRAM_NUMEL
_TOTAL_PARAM_TENSORS = 966
_DENSE_GRADIENT_TENSORS = 964
_FSDP_ODD_SKEW = 40


@pytest.mark.parametrize(
    "world,replicate,rank,ac",
    [
        (1, 1, 0, "none"),
        (2, 1, 0, "none"),
        (2, 1, 1, "none"),
        (1, 1, 0, "full"),
    ],
    ids=["single", "fsdp-first", "fsdp-last", "full-ac"],
)
def test_deepseek_v4_1_capture_reaches_iteration_with_metadata_map(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="deepseek_v4_1_debugmodel_text",
            seq_len=128,
            world_size=world,
            logical_rank=rank,
            data_parallel_shard_degree=world // replicate,
            data_parallel_replicate_degree=replicate,
            mixed_precision_param="bfloat16",
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]

    # The metadata-map adapter binds the runner's transitional input seam to
    # the per-backend map the patched decoder builds; the capture reaches a
    # steady-state F/B/O iteration with every dense gradient present.
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _DENSE_GRADIENT_TENSORS
    # Only the sparse host tables are exempt from the dense gradient contract;
    # their gate projections stay ordinary dense parameters.
    assert not any(name.endswith("engram.table.weight") for name in gradients)

    # The stack has exactly two custom-semantics dispatches: the shared NPU
    # MoE token unpermute, once per layer forward (twice under full AC,
    # which replays the forward inside the checkpoint), and the sparse
    # table update boundary, once per host Engram table in the O phase
    # (PR #19 review).
    custom = {
        node.annotations["raw_op_type"]
        for template in workload.step_templates.values()
        for node in template.nodes.values()
        if "tensor_semantics" in node.attrs
    }
    assert custom == {"torchtitan_npu.npu_moe_token_unpermute.default", "dsv41_sim.sparse_adam_update.default"}

    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    unpermute = "torchtitan_npu.npu_moe_token_unpermute.default"
    unpermutes = sum(1 for n in nodes if n.annotations["raw_op_type"] == unpermute)
    assert unpermutes == (80 if ac == "full" else 40)
    sparse_update = "dsv41_sim.sparse_adam_update.default"
    assert sum(1 for n in nodes if n.annotations["raw_op_type"] == sparse_update) == 2

    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # FP32 residency split by production residency (PR #19 review): dense
    # parameters shard under FSDP2 with the odd-numel skew (shard 0 up,
    # shard 1 down), while the host-offload Engram tables stay whole on
    # every rank (EP degree is 1) and report the host budget only.
    if degree == 1:
        local_moment_elements = _DENSE_NUMEL
    elif rank == 0:
        local_moment_elements = _DENSE_NUMEL // 2 + _FSDP_ODD_SKEW
    else:
        local_moment_elements = _DENSE_NUMEL // 2 - _FSDP_ODD_SKEW
    assert memory.persistent_param_bytes == local_moment_elements * 4
    assert memory.host_param_bytes == _ENGRAM_NUMEL * 4
    # Two FP32 AdamW moments per local dense element plus one FP32 step
    # scalar per dense parameter; the host-offload tables' SparseAdam
    # moments (two tables x exp_avg + exp_avg_sq, EP degree 1 keeps them
    # whole per rank) report the host budget, never device bytes.
    assert memory.optimizer_state_bytes == local_moment_elements * 8 + _DENSE_GRADIENT_TENSORS * 4
    assert memory.host_optimizer_state_bytes == _ENGRAM_NUMEL * 8
    layouts = workload.iteration.schedule.annotations["parameter_layouts"]
    assert {name for name in layouts if name.endswith("engram.table.weight")} == {
        name for name, item in layouts.items() if item["residency"] == "host"
    }
    assert not any(item.kind == "host_parameter" for item in memory.tensor_lifetimes)


def test_deepseek_v4_1_multimodal_flavor_is_not_a_capture_preset():
    # The multimodal flavor needs pixel_values the text-only runner does not
    # synthesize, so its vision parameters would fail the gradient manifest
    # (PR #19 review). The capture preset table exposes the text flavor only;
    # the native multimodal factory stays available to native training.
    with pytest.raises(ValueError, match="Unknown simulator preset"):
        simulate(SimulationConfig(config="deepseek_v4_1_debugmodel_multimodal", seq_len=16))
