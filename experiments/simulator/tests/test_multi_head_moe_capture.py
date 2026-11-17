# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head-MoE capture contracts: faithful SLA2 eager operators and memory plans."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.cost.op_cost_model import annotate_workload_costs
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Debug-model geometry (torchtitan_npu.models.multi_head_moe.config_registry):
# dim=256, vocab=102400, 8 layers (front-2/back-2 dense FFN, MoE in between),
# n_heads=4, head_dim=64, block=64, router proj 64, inter_dim=512,
# MoE heads S=4, head hidden hs=64, Es=16, Ks=4, expert mult 2 (inter=128),
# shared inter 128, FP32 parameters.
_SPARSE_FWD = "multi_head_moe_sim.sla2_sparse_attention.default"
_SPARSE_BWD = "multi_head_moe_sim.sla2_sparse_attention_grad.default"
_LINEAR_FWD = "multi_head_moe_sim.sla2_linear_attention.default"
_LINEAR_BWD = "multi_head_moe_sim.sla2_linear_attention_grad.default"

_D, _V, _HID = 256, 102400, 512
_H, _HD, _PROJ = 4, 64, 64
_HS, _ES, _MULT = 64, 16, 2
_SHARED = 128
_DENSE_LAYER = (
    4 * (_H * _HD * _D)  # to_q/k/v/out
    + 2 * _HD  # norm_q + norm_k
    + 2 * (_PROJ * _HD)  # block router proj_q/proj_k
    + _H  # alpha [H, 1]
    + 2 * _D  # attention_norm + ffn_norm
    + 3 * (_HID * _D)  # dense w1/w2/w3
)
_MOE_LAYER = (
    _DENSE_LAYER
    - 3 * (_HID * _D)
    + _D * (_H * _HS)  # proj_in
    + _H * _HS * _ES  # gate router [S, hs, Es]
    + 3 * (_ES * _H) * (_HS * _MULT) * _HS  # grouped experts w1/w2/w3
    + (_H * _HS) * _D  # proj_out
    + 3 * (_SHARED * _D)  # shared w1/w2/w3
)
_TOTAL_PARAMS = 2 * _V * _D + _D + 4 * _DENSE_LAYER + 4 * _MOE_LAYER
# Dense layer: qkv/out + 2 head norms + 2 router projs + alpha + 2 block norms
# + w1/w2/w3 = 14; MoE layer: 14 attention/norm entries + proj_in + gate
# router + 3 experts + proj_out + 3 shared = 20; plus embeddings, head and
# final norm.
_TOTAL_PARAM_TENSORS = 3 + 4 * 14 + 4 * 20


def test_multi_head_moe_param_formula_matches_debug_geometry():
    assert _TOTAL_PARAMS == 63_395_104
    assert _TOTAL_PARAM_TENSORS == 139


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
def test_multi_head_moe_capture_reaches_iteration_with_sla2_eager_operators(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            data_parallel_replicate_degree=replicate,
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The SLA2 path is the reference's own eager operators; nothing is
    # substituted by virtual fused ops. 8 SLA2 layers. Full AC recomputes the
    # forward inside backward; the default selective policy saves only
    # upstream compute-intensive ops, so those are recomputed there as well.
    # Full AC and the default selective policy both recompute the SLA2 math
    # (only the topk mask output is saved by the selective policy). (The
    # linear branch einsums decompose into permute/bmm under capture;
    # masked_fill, topk and the RoPE complex views are the SLA2-only
    # signatures. Counts span the forward and backward graphs.)
    recompute = 2 if ac in ("full", "selective") else 1
    assert counts["aten.masked_fill.Scalar"] == 8 * recompute + 8
    assert counts["aten.view_as_complex.default"] == 16 * recompute + 16
    assert counts["aten.topk.default"] == 8 * (2 if ac == "full" else 1)
    assert not any(raw.startswith("multi_head_moe_sim.") for raw in counts)
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS
    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # The formulas below assume world is a power of two that divides the
    # parameter shard evenly (no padding imbalance); debug geometry guarantees
    # numel is even at every tested degree.
    local_params = _TOTAL_PARAMS if degree == 1 else _TOTAL_PARAMS // degree
    assert memory.persistent_param_bytes == local_params * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter.
    assert memory.optimizer_state_bytes == local_params * 8 + _TOTAL_PARAM_TENSORS * 4
    # No virtual fusions remain: no node carries declared tensor semantics.
    assert all("tensor_semantics" not in n.attrs for n in nodes)


def test_multi_head_moe_ep2_dispatches_all_to_all_and_reaches_iteration():
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=2,
            logical_rank=0,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 4 MoE layers x (token dispatch + gradient combine) per dispatch pair.
    assert counts["_c10d_functional.all_to_all_single.default"] == 20
    # Collectives carry their logical cross-rank volume: in+out bytes, once,
    # independently recomputed here from the captured tensor facts.
    from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes

    annotate_workload_costs(workload)
    comm = [n for n in nodes if n.op_type in ("all_to_all", "reduce_scatter")]
    assert len(comm) == 20 + 11
    for node in comm:
        assert not node.annotations.get("cost_unknown")
        expected = sum(tensor_volume_bytes(t.shape, t.dtype) for t in (*node.inputs, *node.outputs))
        assert node.annotations["comm_bytes"] == expected
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS


def test_multi_head_moe_rejects_tensor_parallel_at_config_level():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                tensor_parallel_degree=2,
            )
        )


def test_multi_head_moe_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="context parallel"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                context_parallel_degree=2,
            )
        )


def test_multi_head_moe_preset_rejects_mtp_modules():
    from torchtitan_simulator.presets import multi_head_moe_debugmodel

    with pytest.raises(ValueError, match="no MTP modules"):
        multi_head_moe_debugmodel(num_mtp_layers=1)


def _sla2_hardware():
    """Real DV120 profile: 32-bit cube/vec entries match the declared FP32 vec config."""
    from torchtitan_simulator.cost.hardware import dv120_hardware

    return dv120_hardware()


def test_multi_head_moe_sla2_eager_ops_get_declared_costs():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=16))
    annotate_workload_costs(workload, hardware=_sla2_hardware())
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    by_raw = {}
    for node in nodes:
        raw = node.annotations["raw_op_type"]
        if raw in ("aten.mm.default", "aten.masked_fill.Scalar", "aten._softmax.default"):
            by_raw.setdefault(raw, node.annotations.get("duration_us"))
    # The SLA2 eager operators are plain aten ops; the standard handler table
    # prices them (mm via the matmul library roofline, pointwise families by
    # declared equivalent-op counts). (Layout zero-compute ops and small ops
    # outside the handler table stay unknown, as for every model.)
    assert by_raw.get("aten.mm.default") is not None
    assert by_raw.get("aten.masked_fill.Scalar") is not None
    assert by_raw.get("aten._softmax.default") is not None


def test_multi_head_moe_opt_in_fusion_replaces_only_the_two_branch_seams():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=16, enable_sla2_fusion=True))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The fusion scope is exactly the two branch seams: 8 SLA2 layers ->
    # 8 forward + 8 backward virtual ops per branch. Everything around them
    # stays native (topk, sigmoid, RoPE complex views still captured).
    assert counts[_SPARSE_FWD] == 8
    assert counts[_LINEAR_FWD] == 8
    assert counts[_SPARSE_BWD] == 8
    assert counts[_LINEAR_BWD] == 8
    assert counts["aten.topk.default"] == 8
    assert counts["aten.sigmoid.default"] == 16
    assert counts["aten.view_as_complex.default"] == 32
    assert not any(
        raw.startswith("aten.") and "masked_fill" in raw for raw in counts
    )  # the score masking now lives inside the fused boundary

    # The fused ops carry declared tensor semantics; no eager node does.
    custom = [n for n in nodes if "tensor_semantics" in n.attrs]
    assert {n.annotations["raw_op_type"] for n in custom} == {_SPARSE_FWD, _SPARSE_BWD, _LINEAR_FWD, _LINEAR_BWD}

    # Library cost routing: durations appear only with a hardware profile.
    annotate_workload_costs(workload)
    fused = [n for n in nodes if n.annotations["raw_op_type"].startswith("multi_head_moe_sim.sla2_")]
    assert all(node.annotations.get("cost_unknown") for node in fused)
    annotate_workload_costs(workload, hardware=_sla2_hardware())
    assert all(node.annotations.get("duration_us", 0) > 0 for node in fused)


def test_multi_head_moe_seq_len_128_exercises_real_block_selection():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=128))
    annotate_workload_costs(workload, hardware=_sla2_hardware())
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # Two 64-token key blocks per head at seq 128; one topk block-selection
    # per layer forward (8 SLA2 layers), and the sparse branch materializes
    # the full [B, H, T, T] score matrix (forward + gradient fill).
    assert counts["aten.topk.default"] == 8
    score_nodes = [n for n in nodes if n.outputs and tuple(n.outputs[0].shape) == (1, 4, 128, 128)]
    sig = Counter(
        n.annotations["raw_op_type"]
        for n in score_nodes
        if n.annotations["raw_op_type"] in ("aten.div.Tensor", "aten.masked_fill.Scalar", "aten._softmax.default")
    )
    assert sig["aten.div.Tensor"] == 16
    assert sig["aten.masked_fill.Scalar"] == 16
    assert sig["aten._softmax.default"] == 8
    soft_mask = [
        n
        for n in nodes
        if n.annotations["raw_op_type"] == "aten.sigmoid.default" and tuple(n.outputs[0].shape) == (1, 4, 2, 2)
    ]
    assert len(soft_mask) == 8
