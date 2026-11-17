# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi virtual fusion capture and native data-parallel memory contracts."""

from collections import Counter

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.compat.kimi import chunk_kda, fusion_attention, situ_glu
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


@pytest.mark.parametrize(
    "world,replicate,rank,ac",
    [
        (1, 1, 0, "none"),
        (2, 1, 0, "none"),
        (2, 1, 1, "none"),
        (4, 2, 0, "none"),
        (1, 1, 0, "full"),
        (1, 1, 0, "selective"),
    ],
    ids=["single", "fsdp-first", "fsdp-last", "hsdp", "full-ac", "selective-ac"],
)
def test_kimi_capture_retains_fused_backward_gradients_and_independent_parameter_bytes(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="kimi_k3_smoketest",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            data_parallel_replicate_degree=replicate,
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    annotations = workload.iteration.schedule.annotations
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    for op in (
        "chunk_kda",
        "chunk_kda_grad",
        "fusion_attention",
        "fusion_attention_grad",
        "situ_glu",
        "situ_glu_backward",
    ):
        assert counts[f"kimi_sim.{op}.default"] > 0, op
    gradients = annotations["gradient_validation"]
    assert len(gradients) == 110
    for key in (
        "layers.0.attention.A_log",
        "layers.0.attention.dt_bias",
        "layers.0.attention.b_proj.weight",
        "layers.3.attention.kv_b_proj.weight",
        "layers.1.moe.router.gate.weight",
        "layers.1.moe.routed_experts.inner_experts.w1_EFD",
        "layers.1.moe.down_proj.weight",
    ):
        assert key in gradients, key
    # Independent projection count for the fixed debug geometry: D=256,
    # H=8, KDA D_h=32; MLA Q/KV rank=64, Q/K=32, V=16;
    # E=8, latent=128, expert F=128, shared=2, vocab=2048.
    d, h, p, f, latent, e = 256, 8, 256, 128, 128, 8
    kda = 5 * d * p + 3 * p * 4 + h + p + d * 32 + 32 * p + d * h + 32
    mla = d * 64 + 64 + 64 * h * 32 + d * (64 + 16) + 64 + 64 * h * 32 + 2 * d * h * 16
    dense = 3 * d * 512
    moe = 3 * e * latent * f + d * e + 2 * d * latent + latent + 3 * d * (2 * f)
    total = 2 * 2048 * d + 3 * d + 4 * 6 * d + 3 * kda + mla + dense + 3 * moe
    assert total == 4605048
    degree = world // replicate
    # Nine attention-residual projections have a singleton leading dimension;
    # native FSDP assigns the entire row to rank 0, leaving rank 1 empty.
    local = total if degree == 1 else total // 2 + (1 if rank == 0 else -1) * 9 * d // 2
    memory = annotations["memory_plan"]
    assert memory.persistent_param_bytes == local * 4
    frozen = 2 * d if degree == 1 else d // 2 + (d if rank == 0 else 0)
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter (including an empty FSDP shard).
    assert memory.optimizer_state_bytes == (local - frozen) * 8 + 110 * 4


@pytest.mark.parametrize(
    "op,backward_name,shapes,attrs",
    [
        (chunk_kda, "chunk_kda_grad", [(1, 8, 2, 4)] * 4 + [(1, 8, 2), (2,), (8,)], (-5.0,)),
        (fusion_attention, "fusion_attention_grad", [(1, 2, 8, 4)] * 3, (0.5,)),
        (situ_glu, "situ_glu_backward", [(4, 8)] * 2, (4.0, 25.0)),
    ],
    ids=["kda", "mla", "situ"],
)
def test_virtual_fusion_saves_actual_inputs_and_preserves_backward_identity_and_dependencies(
    op, backward_name, shapes, attrs
):
    leaves = [torch.empty(shape, device="meta", requires_grad=True) for shape in shapes]
    capture = OpDispatchCapture()
    with capture, AutogradSavedTensorCapture():
        inputs = [leaf.sin() for leaf in leaves]
        output = op(*inputs, *attrs)
        dy = torch.empty_like(output).sin()
        output.backward(dy)
    for leaf, value in zip(leaves, inputs, strict=True):
        assert leaf.grad is not None and leaf.grad.shape == value.shape
        assert leaf.grad.dtype == value.dtype
        assert any(slot.tensor_id == capture.tensor_id(value) for slot in capture.autograd_saved_tensor_events())
    forward_name = str(op._opoverload)
    backward_raw = f"kimi_sim.{backward_name}.default"
    memory = capture.memory_events()
    forward = next(event for event in memory if event.raw_op_type == forward_name)
    backward = next(event for event in memory if event.raw_op_type == backward_raw)
    expected_ids = [capture.tensor_id(value) for value in inputs]
    assert [t.tensor_id for t in forward.inputs] == expected_ids
    # Native saved-tensor unpack may create a new Python Tensor wrapper;
    # its recorded value alias must resolve to the original saved input.
    assert [t.alias_of if t.alias_of is not None else t.tensor_id for t in backward.inputs[:-1]] == expected_ids
    assert [t.storage_key for t in backward.inputs[:-1]] == [t.storage_key for t in forward.inputs]
    assert backward.inputs[-1].tensor_id == capture.tensor_id(dy)
    assert [(t.shape, t.dtype) for t in backward.outputs] == [(t.shape, t.dtype) for t in forward.inputs]
    producers = {event.op_id for event in memory if any(t.tensor_id in expected_ids for t in event.outputs)}
    assert len(producers) == len(inputs)
    nodes = capture.build_nodes()
    ancestors, pending = set(), list(nodes[backward.op_id].predecessors)
    while pending:
        op_id = pending.pop()
        if op_id not in ancestors:
            ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    assert producers <= ancestors


def test_kimi_default_common_overrides_reach_configured_attention_and_dispatch():
    from torchtitan.models.common.linear import Linear
    from torchtitan_simulator.config import build_meta_model, prepare_config
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher

    prepared = prepare_config(SimulationConfig(config="kimi_k3_smoketest"))
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        model = build_meta_model(prepared)
    configured = {fqn for fqn, _, _, _ in prepared.trainer.model_spec.model.traverse(Linear.Config)}
    actual = {fqn for fqn, module in model.named_modules() if isinstance(module, Linear)}
    assert configured == actual
    assert isinstance(model.layers["0"].attention.o_norm.norm, AscRMSNorm)
    assert isinstance(model.layers["3"].attention.q_a_layernorm, AscRMSNorm)
    from torchtitan_npu.models.kimi_k3.model import KimiResidualScoreNorm

    assert isinstance(model.output_attn_res.norm, KimiResidualScoreNorm)
    assert isinstance(model.layers["1"].moe.routed_experts.token_dispatcher, AscAllToAllTokenDispatcher)
    assert not model.layers["1"].moe.routed_experts.token_dispatcher.absorb_router_scores


@pytest.mark.parametrize("param_dtype", ["bfloat16", "float32"])
def test_kimi_normal_fsdp_precision_reaches_kda_and_fp32_gradient_reduction(param_dtype):
    workload = simulate(
        SimulationConfig(
            config="kimi_k3_smoketest",
            seq_len=16,
            world_size=4,
            data_parallel_replicate_degree=2,
            mixed_precision_param=param_dtype,
        )
    )
    annotations = workload.iteration.schedule.annotations
    events = annotations["memory_plan"].raw_events
    kda = next(event for event in events if event.raw_op_type == "kimi_sim.chunk_kda.default")
    assert [ref.dtype for ref in kda.inputs[-2:]] == [param_dtype, param_dtype]
    assert {entry["dtype"] for entry in annotations["gradient_validation"].values()} == {"torch.float32"}
    reductions = [event for event in events if "reduce_scatter" in event.raw_op_type]
    assert reductions
    assert {ref.dtype for event in reductions for ref in event.inputs} == {"float32"}


def test_kimi_base_bfloat16_capture_rejects_mixed_original_parameter_dtypes():
    with pytest.raises(ValueError, match="uniform original parameter dtype"):
        simulate(SimulationConfig(config="kimi_k3_smoketest", training_dtype="bfloat16"))
