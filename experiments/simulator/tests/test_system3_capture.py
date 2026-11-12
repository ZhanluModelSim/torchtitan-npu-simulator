# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""System3 Flash capture contracts: reference eager operators, gradients and memory."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Debug-model geometry (torchtitan_npu.models.system3.config_registry, the
# reference smoke_test config): hidden=128, vocab=512, 12 physical blocks
# looped twice, [ISA, GDN2 x 3] units, 4 experts top-2 + 1 shared in a
# 48-wide latent, Engram on layers 1/5/9 (orders [2], 2 heads x 64 rows x
# 16 dim), mHC hc_mult=2 x branch 1024, FP32 parameters. The multimodal
# flavor adds the four encoders (image 64, audio/point3d/haptic).
_TEXT_PARAMS = 13_886_412
_TEXT_PARAM_TENSORS = 373
_MM_PARAMS = 14_214_060
_MM_PARAM_TENSORS = 440


def test_system3_param_formula_matches_reference_smoke_geometry():
    # The reference smoke_test counts 14,148,524 parameters with a tied LM
    # head; the capture flavor keeps the head untied (+512 x 128). Encoder
    # towers and stem convs match the reference exactly (no conv biases).
    assert _MM_PARAMS == 14_148_524 + 512 * 128
    # The encoder towers over the untied text backbone.
    assert _MM_PARAMS - _TEXT_PARAMS == 327_648


def test_system3_text_capture_reaches_iteration_with_reference_operators():
    workload = simulate(SimulationConfig(config="system3_debugmodel_text", seq_len=16))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The ISA/GDN2/Engram/mHC paths are the reference's own eager operators;
    # nothing is substituted by virtual fused ops. GDN2's per-token delta-rule
    # loop pads k/v/q to the 16-wide state (constant_pad_nd), the ISA indexer
    # masks the compressed grid (masked_fill + topk), the routed experts run
    # three grouped GEMMs per MoE layer, and the router normalizes in fp32.
    assert counts["aten.constant_pad_nd.default"] == 2304
    assert counts["aten.masked_fill_.Scalar"] == 288
    assert counts["aten.topk.default"] == 22
    assert counts["aten._grouped_mm.default"] == 162
    assert counts["aten._softmax.default"] == 24
    assert counts["aten.silu.default"] == 36
    assert counts["aten.embedding.default"] == 13
    # No virtual fusion ops: every node is a native aten/collective op.
    assert not any(raw.startswith(("system3.", "multi_head_moe_sim.", "deepseek_v4.")) for raw in counts)
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TEXT_PARAM_TENSORS
    memory = workload.iteration.schedule.annotations["memory_plan"]
    assert memory.persistent_param_bytes == _TEXT_PARAMS * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter.
    assert memory.optimizer_state_bytes == _TEXT_PARAMS * 8 + _TEXT_PARAM_TENSORS * 4
    assert all("tensor_semantics" not in n.attrs for n in nodes)


def test_system3_full_ac_recomputes_forward_math():
    workload = simulate(
        SimulationConfig(config="system3_debugmodel_text", seq_len=16, activation_checkpoint_mode="full")
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # Full AC recomputes the forward inside backward: the forward-only math
    # doubles (topk/indexer, grouped GEMMs, softmax, router math), while the
    # pad/mask signatures keep their backward-side counts.
    assert counts["aten.topk.default"] == 44
    assert counts["aten._grouped_mm.default"] == 216
    assert counts["aten._softmax.default"] == 48
    assert counts["aten.silu.default"] == 72
    assert counts["aten.embedding.default"] == 25
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TEXT_PARAM_TENSORS


def test_system3_multimodal_capture_includes_encoder_gradients():
    workload = simulate(SimulationConfig(config="system3_debugmodel_multimodal", seq_len=16))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    # 373 backbone + 67 encoder parameter tensors — the reference smoke
    # counts exactly 67 encoder grads as well.
    assert len(gradients) == _MM_PARAM_TENSORS
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The encoders are plain bidirectional blocks (qkv bmm attention +
    # gelu MLP) behind Conv/patch stems; the modality type lookup joins the
    # sequence (backbone lookups + one per modality segment).
    assert counts["aten.embedding.default"] == 17
    assert counts["aten._softmax.default"] == 32  # backbone + 8 encoder blocks
    memory = workload.iteration.schedule.annotations["memory_plan"]
    assert memory.persistent_param_bytes == _MM_PARAMS * 4
    assert memory.optimizer_state_bytes == _MM_PARAMS * 8 + _MM_PARAM_TENSORS * 4


def test_system3_ep2_dispatches_all_to_all_and_reaches_iteration():
    workload = simulate(
        SimulationConfig(
            config="system3_debugmodel_text",
            seq_len=16,
            world_size=2,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 9 MoE layers x (dispatch + combine paths) carry the latent pair per
    # exchange; the routed grouped GEMM count is unchanged (per-rank local
    # experts shrink with EP, the per-layer GEMM count does not).
    assert counts["_c10d_functional.all_to_all_single.default"] == 90
    assert counts["aten._grouped_mm.default"] == 162
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TEXT_PARAM_TENSORS


def test_system3_rejects_tensor_parallel_at_config_level():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        simulate(
            SimulationConfig(
                config="system3_debugmodel_text",
                seq_len=16,
                world_size=2,
                tensor_parallel_degree=2,
            )
        )


def test_system3_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="context parallel"):
        simulate(
            SimulationConfig(
                config="system3_debugmodel_text",
                seq_len=16,
                world_size=2,
                context_parallel_degree=2,
            )
        )


def test_system3_preset_rejects_mtp_modules():
    from torchtitan_npu.models.system3.config_registry import system3_debugmodel_text

    with pytest.raises(ValueError, match="no MTP modules"):
        system3_debugmodel_text(num_mtp_layers=1)
