# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 native DSA capture, frozen state and sparse-key backward dependencies."""

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


@pytest.mark.parametrize(
    "world,rank,tp,cp,ep,ac",
    [
        (1, 0, 1, 1, 1, "none"),
        (2, 1, 1, 1, 1, "none"),
        (1, 0, 1, 1, 1, "full"),
        (1, 0, 1, 1, 1, "selective"),
        (4, 0, 2, 1, 1, "none"),
        (2, 1, 1, 2, 1, "none"),
        (2, 0, 1, 1, 2, "none"),
    ],
    ids=["single", "fsdp-last", "full-ac", "selective-ac", "tp", "cp", "ep"],
)
def test_k3_1_sparse_capture_retains_selection_gradients_and_frozen_memory(world, rank, tp, cp, ep, ac):
    workload = simulate(
        SimulationConfig(
            config="kimi_k3_1_smoketest",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
            expert_parallel_degree=ep,
            activation_checkpoint_mode=ac,
        )
    )

    validation = validate_workload(workload)
    assert validation["graph_acyclic"] and validation["peak_storage_unique"]
    annotations = workload.iteration.schedule.annotations
    nodes = [node for template in workload.step_templates.values() for node in template.nodes.values()]
    names = [node.annotations["raw_op_type"] for node in nodes]
    for raw in (
        "kimi_sim.dsa_indexer.default",
        "kimi_sim.dsa_attention.default",
        "kimi_sim.dsa_attention_grad.default",
    ):
        assert raw in names
    # Selection and selected-key attention are fused in Meta; unrelated top-k
    # or indexing elsewhere in the model is not excluded by name.
    dsa_nodes = [node for node in nodes if node.annotations["raw_op_type"].startswith("kimi_sim.dsa_")]
    assert all("tensor_semantics" in node.attrs for node in dsa_nodes)
    fwd = next(node for node in dsa_nodes if node.annotations["raw_op_type"] == "kimi_sim.dsa_attention.default")
    bwd = next(node for node in dsa_nodes if node.annotations["raw_op_type"] == "kimi_sim.dsa_attention_grad.default")
    assert fwd.inputs[0].shape[-1] == 32 and fwd.outputs[0].shape[-1] == 16
    assert fwd.outputs[1].dtype == "float32"
    assert [t.shape for t in bwd.outputs] == [t.shape for t in bwd.inputs[:3]]
    # Every MLA row uses the selected-key path; dense legacy fusion must not
    # silently replace the sparse computation or its autograd saves.
    assert "kimi_sim.fusion_attention.default" not in names
    gradients = annotations["gradient_validation"]
    assert "layers.3.attention.q_b_proj.weight" in gradients
    assert "layers.3.attention.kv_b_proj.weight" in gradients
    assert not any(".indexer." in name for name in gradients)
    # Added frozen state: 64*32 query + 256*16 key + 256*2 weights + 2*16 norm.
    if world == 1:
        assert annotations["memory_plan"].persistent_param_bytes == (4605048 + 6688) * 4
        assert annotations["memory_plan"].optimizer_state_bytes == (4605048 - 512) * 8 + 110 * 4
