# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real training capture delivers fusion semantics through the FSDP view."""

import json

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize(
    "config,world,tp",
    [("deepseek_v4_smoketest", 4, 1), ("kimi_k3_smoketest", 4, 1), ("kimi_k3_smoketest", 8, 2)],
    ids=["deepseek-scales", "kimi-unweighted", "kimi-tp"],
)
def test_real_training_delivers_custom_semantics_with_local_shapes(config, world, tp, tmp_path):
    workload = simulate(
        SimulationConfig(
            config=config,
            seq_len=128 if config.startswith("deepseek") else 16,
            world_size=world,
            tensor_parallel_degree=tp,
            expert_parallel_degree=2,
            output_dir=str(tmp_path),
        )
    )
    nodes = [node for graph in workload.step_templates.values() for node in graph.nodes.values()]
    custom = [node for node in nodes if "tensor_semantics" in node.attrs]
    assert custom
    assert all(
        "tensor_semantics" not in node.attrs
        for node in nodes
        if node.annotations["raw_op_type"].startswith(("aten.", "npu.", "cann_ops_transformer."))
    )
    source_nodes = {
        node.op_id: node
        for graph in workload.captured_workload.step_templates.values()
        for node in graph.nodes.values()
    }
    for node in custom:
        assert node.attrs["tensor_semantics"] == source_nodes[node.op_id].attrs["tensor_semantics"]
        for key, tensors in (("inputs", node.inputs), ("outputs", node.outputs)):
            assert [item["index"] for item in node.attrs["tensor_semantics"][key]] == list(range(len(tensors)))
            for item in node.attrs["tensor_semantics"][key]:
                assert len(item["axes"]) == len(tensors[item["index"]].shape)
    reroute = next(
        node for node in custom if node.annotations["raw_op_type"] == "torchtitan_npu.npu_moe_re_routing.default"
    )
    roles = [item["role"] for item in reroute.attrs["tensor_semantics"]["inputs"]]
    assert ("rank_major_scales" in roles) == config.startswith("deepseek")
    # Fixed debug presets: DSV4 has 4 experts, Kimi has 8, both sharded by EP=2.
    assert reroute.inputs[1].shape == (2, 2 if config.startswith("deepseek") else 4)
    if config.startswith("kimi"):
        attention = next(
            node for node in custom if node.annotations["raw_op_type"] == "kimi_sim.fusion_attention.default"
        )
        assert attention.attrs["tensor_semantics"]["inputs"][0] == {
            "index": 0,
            "role": "query",
            "axes": ("batch", "head", "q_len", "qk_head_dim"),
        }
        assert attention.inputs[0].shape == (1, 8 // tp, 16, 32)
        assert attention.inputs[2].shape[-1] == 32  # V head width 16 is padded at this boundary.
        (tmp_path / "attention-semantics.json").write_text(
            json.dumps({"shape": attention.inputs[0].shape, **attention.attrs["tensor_semantics"]}, indent=2)
        )
