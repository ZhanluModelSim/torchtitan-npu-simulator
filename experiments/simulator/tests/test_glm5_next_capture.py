# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash text-capture contracts: frozen indexer, pure-ATen graph."""

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Native glm5_next_debugmodel geometry, text-only: dim=512, 4 layers, vocab,
# hc_mult=4, 8 experts. FP32 parameters; the DSA lightning indexer is frozen
# (requires_grad=False, forward under no_grad), so 7 parameter tensors per
# DSA layer never enter the dense gradient contract. The stack declares no
# virtual fusion: every dispatch is plain ATen/ACLNN semantics.
_TOTAL_NUMEL = 4_882_192
_GRADIENT_TENSORS = 91


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
def test_glm5_next_capture_reaches_iteration(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="glm5_next_debugmodel_text",
            seq_len=216,
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

    # Steady-state F/B/O iteration with every trainable gradient present.
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _GRADIENT_TENSORS
    # The indexer is excluded from FSDP units and carries no gradient.
    assert not any("indexer" in name for name in gradients)

    # v1 is pure ATen capture: no model-private virtual fusion registers
    # tensor_semantics anywhere in the graph.
    custom = {
        node.annotations["raw_op_type"]
        for template in workload.step_templates.values()
        for node in template.nodes.values()
        if "tensor_semantics" in node.attrs
    }
    assert custom == set()

    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # FP32 original-dtype residency. Under FSDP2 each parameter shards
    # independently: odd-numel tensors round shard 0 up and shard 1 down.
    assert (
        memory.persistent_param_bytes
        == {
            1: _TOTAL_NUMEL * 4,
            2: {0: 9_816_240, 1: 9_816_208}[rank],
        }[degree]
    )
    assert (
        memory.optimizer_state_bytes
        == {
            1: 38_850_540,
            2: {0: 19_425_484, 1: 19_425_420}[rank],
        }[degree]
    )
