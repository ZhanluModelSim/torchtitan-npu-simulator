# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi parallel shard geometry and completion dependencies in the captured graph."""

from math import prod

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _split_length(size, degree, rank):
    chunk = (size + degree - 1) // degree
    return max(0, min(chunk, size - rank * chunk))


def _expected_local_shape(name, original, *, world, replicate, tp, ep, rank):
    shape = list(original)
    fsdp, fsdp_rank = world // replicate // tp, (rank // tp) % (world // replicate // tp)
    if ".inner_experts." in name:
        if ep > 1:
            shape[0] = _split_length(shape[0], ep, rank % ep)
            fsdp, fsdp_rank = world // replicate // ep, (rank // ep) % (world // replicate // ep)
        else:
            axis = 2 if name.endswith("w2_EDF") else 1
            shape[axis] = _split_length(shape[axis], tp, rank % tp)
        axis = 1 if fsdp * (ep if ep > 1 else 1) > original[0] else 0
        shape[axis] = _split_length(shape[axis], fsdp, fsdp_rank)
        return tuple(shape)
    tp_axis = None
    if name in {"tok_embeddings.weight", "lm_head.weight"}:
        tp_axis = 0
    elif ".feed_forward." in name or ".shared_experts." in name:
        tp_axis = 1 if name.endswith("w2.weight") else 0
    elif ".attention." in name:
        component = name.split(".attention.", 1)[1]
        if component in {"A_log", "dt_bias"} or "_conv1d.conv.weight" in component:
            tp_axis = 0
        elif component == "o_proj.weight":
            tp_axis = 1
        elif component.split(".")[0] in {
            "q_proj",
            "k_proj",
            "v_proj",
            "f_b_proj",
            "b_proj",
            "g_proj",
            "g_b_proj",
            "q_b_proj",
            "kv_b_proj",
        }:
            tp_axis = 0
    if tp_axis is not None:
        shape[tp_axis] = _split_length(shape[tp_axis], tp, rank % tp)
    shape[0] = _split_length(shape[0], fsdp, fsdp_rank)
    return tuple(shape)


def _ancestors(nodes, target):
    seen, pending = set(), list(nodes[target].predecessors)
    while pending:
        op_id = pending.pop()
        if op_id not in seen:
            seen.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    return seen


@pytest.mark.parametrize(
    "world,replicate,tp,cp,ep,rank,ac",
    [
        (4, 1, 2, 1, 1, 0, "none"),
        (4, 1, 1, 1, 2, 3, "none"),
        (8, 1, 2, 1, 2, 0, "none"),
        (4, 1, 1, 2, 1, 3, "none"),
        (8, 1, 2, 2, 1, 7, "none"),
        (8, 1, 1, 2, 2, 0, "none"),
        (16, 2, 2, 2, 2, 0, "none"),
        (16, 2, 2, 2, 2, 15, "full"),
        (16, 2, 2, 2, 2, 15, "selective"),
        (64, 1, 1, 1, 1, 63, "none"),
    ],
    ids=[
        "tp",
        "ep-last",
        "tp-ep",
        "cp-last",
        "tp-cp-last",
        "cp-ep",
        "all-first",
        "all-last-full-ac",
        "all-last-selective-ac",
        "fsdp-over-expert-count",
    ],
)
def test_kimi_parallel_capture_matches_independent_shards_and_reduces_before_latent_norm(
    world, replicate, tp, cp, ep, rank, ac
):
    workload = simulate(
        SimulationConfig(
            config="kimi_k3_smoketest",
            seq_len=16,
            world_size=world,
            data_parallel_replicate_degree=replicate,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
            expert_parallel_degree=ep,
            logical_rank=rank,
            activation_checkpoint_mode=ac,
        )
    )
    assert validate_workload(workload)["graph_acyclic"]
    annotations = workload.iteration.schedule.annotations
    expected_bytes, frozen_bytes = 0, 0
    for name, item in annotations["parameter_layouts"].items():
        expected = _expected_local_shape(
            name, item["original_shape"], world=world, replicate=replicate, tp=tp, ep=ep, rank=rank
        )
        assert tuple(item["local_shape"]) == expected, name
        expected_bytes += prod(expected) * 4
        if name.startswith("layers.0.self_attention_res."):
            frozen_bytes += prod(expected) * 4
    memory = annotations["memory_plan"]
    assert memory.persistent_param_bytes == expected_bytes
    assert memory.optimizer_state_bytes == 2 * (expected_bytes - frozen_bytes) + 110 * 4
    assert memory.steady_state_peak_active_bytes >= expected_bytes + memory.optimizer_state_bytes + (
        expected_bytes - frozen_bytes
    )
    assert len(annotations["gradient_validation"]) == 110
    events = memory.raw_events
    kda = next(event for event in events if event.raw_op_type == "kimi_sim.chunk_kda.default")
    mla = next(event for event in events if event.raw_op_type == "kimi_sim.fusion_attention.default")
    assert kda.inputs[0].shape == (1, 16, 8 // tp, 32)
    compute_dtype = annotations["precision_configuration"]["mixed_precision_param"]
    assert [ref.dtype for ref in kda.inputs[-2:]] == [compute_dtype, compute_dtype]
    assert mla.inputs[0].shape == (1, 8 // tp, 16, 32)
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    if tp > 1:
        down = [
            event
            for event in events
            if event.module_path.replace("._checkpoint_wrapped_module", "").endswith(
                "layers.1.moe.routed_experts.inner_experts"
            )
            and event.raw_op_type == "aten._grouped_mm.default"
        ]
        norm = next(
            event
            for event in events
            if event.module_path.replace("._checkpoint_wrapped_module", "").endswith("layers.1.moe.norm")
            and event.raw_op_type == "npu.npu_rms_norm.default"
        )
        assert len(down) >= 3
        reductions = [
            event
            for event in annotations["communication_events"]
            if event["comm_dim"] == "tp"
            and event["op_id"] in _ancestors(nodes, norm.op_id)
            and nodes[event["op_id"]].seq_idx > down[2].seq_idx
        ]
        if ep == 1:
            assert reductions, "TP-sharded expert sum must complete before latent RMSNorm"
        assert norm.inputs[0].shape == (1, 16 // cp // tp, 128)
    if cp > 1:
        conv = next(
            event
            for event in events
            if event.module_path.replace("._checkpoint_wrapped_module", "").endswith("layers.0.attention.q_conv1d")
            and event.raw_op_type == "aten.convolution.default"
        )
        gathers = [
            event
            for event in annotations["communication_events"]
            if event["comm_dim"] == "cp" and event["op_id"] in _ancestors(nodes, conv.op_id)
        ]
        assert gathers, "CP full context must reach causal convolution"
