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
# host-offload Engram tables (one per KV-source layer), which hold 2,097,232
# FP32 elements in total and are excluded from the dense gradient contract.
_ENGRAM_NUMEL = 2_097_232
_TOTAL_NUMEL = 430_312_240
_DENSE_NUMEL = _TOTAL_NUMEL - _ENGRAM_NUMEL
_TOTAL_PARAM_TENSORS = 966
_DENSE_GRADIENT_TENSORS = 964


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

    # The stack has no model-private virtual fusion: the only custom-semantics
    # dispatch is the shared NPU MoE token unpermute, once per layer forward
    # (twice under full AC, which replays the forward inside the checkpoint).
    custom = {
        node.annotations["raw_op_type"]
        for template in workload.step_templates.values()
        for node in template.nodes.values()
        if "tensor_semantics" in node.attrs
    }
    assert custom == {"torchtitan_npu.npu_moe_token_unpermute.default"}

    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    unpermute = "torchtitan_npu.npu_moe_token_unpermute.default"
    unpermutes = sum(1 for n in nodes if n.annotations["raw_op_type"] == unpermute)
    assert unpermutes == (80 if ac == "full" else 40)

    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # FP32 residency. Under FSDP2 each parameter shards independently: the
    # stack has 80 odd-numel dense parameters, so shard 0 rounds up and shard
    # 1 rounds down (an 80-element skew), while the host-offload tables stay
    # whole on every rank (the EP degree is 1, nothing to shard them over).
    # Shard 0 holds the exact half; shard 1 rounds the 80 odd-numel dense
    # parameters down (an 80-element skew).
    local_dense = _DENSE_NUMEL // degree - (80 if degree > 1 and rank > 0 else 0)
    assert memory.persistent_param_bytes == (local_dense + _ENGRAM_NUMEL) * 4
    # Two FP32 AdamW moments per local dense element; the residual covers the
    # host-offload tables' SparseAdam state stubs (dense and sparse parameter
    # groups are accounted separately, and the stubs do not shard).
    assert memory.optimizer_state_bytes == local_dense * 8 + 4496
