# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 dense parallel shard geometry and completion dependencies in the captured graph."""

from math import prod

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _split_length(size, degree, rank):
    chunk = (size + degree - 1) // degree
    return max(0, min(chunk, size - rank * chunk))


def _expected_local_shape(name, original, *, world, replicate, tp, rank):
    """Independent dense-Qwen3 shard math: TP axis per projection, FSDP on dim 0.

    TP: embeddings/lm_head/wqkv/w1/w3 shard their output dimension (axis 0);
    wo/w2 shard their input dimension (axis 1); all RMSNorm weights
    (attention_norm, ffn_norm, q_norm, k_norm, final norm) replicate.
    FSDP shards every parameter on dim 0 after the TP split.
    """
    shape = list(original)
    fsdp = world // replicate // tp
    fsdp_rank = (rank // tp) % fsdp
    if name in {"tok_embeddings.weight", "lm_head.weight"}:
        tp_axis = 0
    elif ".feed_forward." in name:
        tp_axis = 1 if name.endswith("w2.weight") else 0
    elif name.endswith("attention.qkv_linear.wqkv.weight"):
        tp_axis = 0
    elif name.endswith("attention.wo.weight"):
        tp_axis = 1
    else:
        tp_axis = None
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
    "world,replicate,tp,rank,ac",
    [
        (2, 1, 1, 0, "none"),
        (2, 1, 1, 1, "none"),
        (2, 2, 1, 1, "none"),
        (4, 1, 2, 0, "none"),
        (4, 1, 2, 3, "none"),
        (4, 2, 1, 3, "none"),
        (8, 2, 2, 5, "none"),
        (4, 1, 2, 1, "full"),
        (4, 1, 2, 2, "selective"),
        (8, 1, 1, 7, "none"),
    ],
    ids=[
        "fsdp2-first",
        "fsdp2-last",
        "ddp2",
        "tp2-fsdp2-first",
        "tp2-fsdp2-last",
        "hsdp4-last",
        "hsdp-tp8-mid",
        "tp2-full-ac",
        "tp2-selective-ac",
        "fsdp8-last",
    ],
)
def test_qwen3_parallel_capture_matches_independent_shards_and_fused_kernels(world, replicate, tp, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="qwen3_32b_smoketest",
            seq_len=16,
            world_size=world,
            data_parallel_replicate_degree=replicate,
            tensor_parallel_degree=tp,
            logical_rank=rank,
            activation_checkpoint_mode=ac,
        )
    )
    assert validate_workload(workload)["graph_acyclic"]
    annotations = workload.iteration.schedule.annotations
    expected_bytes = 0
    for name, item in annotations["parameter_layouts"].items():
        expected = _expected_local_shape(
            name, item["original_shape"], world=world, replicate=replicate, tp=tp, rank=rank
        )
        assert tuple(item["local_shape"]) == expected, name
        expected_bytes += prod(expected) * 4
    memory = annotations["memory_plan"]
    # Dense Qwen3 has no frozen parameters: persistent weights, AdamW states
    # (2x) and one float32 step scalar per trainable parameter.
    assert memory.persistent_param_bytes == expected_bytes
    assert memory.optimizer_state_bytes == 2 * expected_bytes + 39 * 4
    assert memory.steady_state_peak_active_bytes >= expected_bytes + memory.optimizer_state_bytes + expected_bytes
    assert len(annotations["gradient_validation"]) == 39

    events = memory.raw_events
    fused = next(event for event in events if event.raw_op_type == "npu.npu_fusion_attention_v3.default")
    # The fused kernel sees the TP-local head count: query keeps 64/tp heads
    # while key/value keep the grouped 8/tp heads.
    assert fused.inputs[0].shape == (16, 64 // tp, 128)
    assert fused.inputs[1].shape == (16, 8 // tp, 128)
    assert fused.inputs[2].shape == (16, 8 // tp, 128)
    fused_grad = next(event for event in events if event.raw_op_type == "npu.npu_fusion_attention_grad_v3.default")
    assert fused_grad.inputs[3].shape == (16, 64 // tp, 128)

    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    if tp > 1:
        def _module(event):
            return event.module_path.replace("._checkpoint_wrapped_module", "")

        # The down projection emits a TP partial sum; its completion must
        # precede the next layer's attention norm under sequence parallelism.
        down = [
            event
            for event in events
            if _module(event).endswith("layers.1.feed_forward.w2") and event.raw_op_type == "aten.mm.default"
        ]
        norm = next(
            event
            for event in events
            if _module(event).endswith("layers.2.attention_norm") and event.raw_op_type == "npu.npu_rms_norm.default"
        )
        assert down
        reductions = [
            event
            for event in annotations["communication_events"]
            if event["comm_dim"] == "tp"
            and event["op_id"] in _ancestors(nodes, norm.op_id)
            and nodes[event["op_id"]].seq_idx > down[0].seq_idx
        ]
        assert reductions, "TP partial sums of the down projection must complete before the next layer norm"
        assert norm.inputs[0].shape == (1, 16 // tp, 5120)
