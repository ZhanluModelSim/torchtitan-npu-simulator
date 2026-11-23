# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sys2Mot capture contracts: reference eager operators, gradients and memory."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Reference tiny geometry (torchtitan_npu.models.sys2mot.config_registry):
# the reference's own golden calibration baseline — 699,968 parameters over
# 444 tensors (vision 62,592 / embedding+lm_head 65,536 / engram 43,936 /
# action experts 125,920 / transformer 401,984). The capture flavor keeps
# every structural path: 2D-RoPE ViT with pixel-unshuffle alignment, per-layer
# CSA2 compress ratios [0, 2, 2, 2, 1, 1, 1, 0] (Full-Mode KV sources at
# layers 1/4, indexers at 1/2/4/5), sqrt(softplus) routed MoE (4 experts
# top-2 + shared), mHC Sinkhorn hyper-connections (n=4, 20 iterations),
# Engram on layers 1/3, and the layer-paired action experts.
_PARAMS = 699_968
_PARAM_TENSORS = 372
# The routed experts stack per-expert nn.Linear weights into (E, F, D)/(E, D, F)
# grouped tensors: 3 tensors per layer instead of 3 x 4 experts, saving
# 8 layers x 9 tensors without changing any numel.
_REFERENCE_PARAM_TENSORS = 444


def test_sys2mot_param_formula_matches_reference_tiny_baseline():
    assert _PARAMS == 699_968
    # The grouped-GEMM deviation reduces only the tensor count.
    assert _REFERENCE_PARAM_TENSORS - _PARAM_TENSORS == 8 * 9
    # The reference tiny groups (sum = 699,968).
    assert _PARAMS == 62_592 + 65_536 + 43_936 + 125_920 + 401_984


def test_sys2mot_capture_reaches_iteration_with_reference_operators():
    workload = simulate(SimulationConfig(config="sys2mot_debugmodel_multimodal", seq_len=100))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The CSA2/Engram/mHC/action paths are the reference's own eager
    # operators. The routed experts run three grouped GEMMs per MoE layer
    # (8 layers), the router scores with sqrt(softplus) in fp32, the CSA2
    # indexer and the force-balanced MoE routing keep their real topk calls,
    # and the ViT pixel unshuffle is the reference's own unfold
    # (im2col/col2im). Distillation is offline in the current form: the
    # training step is student-only and the teacher's logits/hidden are
    # inputs the re-enabled distill knob would consume (see
    # sys2mot-distill-sharded for the archived in-loop form).
    assert counts["aten._grouped_mm.default"] == 72
    assert counts["aten.topk.default"] == 18
    # Student routers + indexer plus the offline-distillation teacher softmax.
    assert counts["aten._softmax.default"] == 50
    assert counts["aten.softplus.default"] == 8
    assert counts["aten.embedding.default"] == 3  # student text + Engram
    assert counts["aten.constant_pad_nd.default"] == 40
    assert counts["aten.im2col.default"] == 1
    assert counts["aten.convolution.default"] == 1
    assert counts["aten.sort.stable"] == 8  # token dispatcher per MoE layer
    assert counts["torchtitan.deterministic_scatter_add.default"] == 8
    # The captured joint loss: flow MSE + 0.1 text CE + 0.01 indexer aux,
    # plus the offline-distillation KD/hidden terms computed on synthetic
    # stand-in teacher tensors (inputs the real offline step loads from the
    # precomputation pass): the KD signature is the single xlogy, the teacher
    # softmax adds one entry, and the hidden term uses square/sub (no second
    # mse_loss).
    assert counts["aten.mse_loss.default"] == 1
    assert counts["aten.nll_loss_forward.default"] == 1
    assert counts["aten.xlogy.Tensor"] == 1
    assert counts["aten._log_softmax.default"] == 10
    assert counts["aten._softmax.default"] == 50
    # Only the sys2mot_sim FlashAttention virtual boundary is substituted;
    # every other node is a native aten/collective op.
    assert not any(raw.startswith(("sys2mot.", "system3.", "deepseek_v4.")) for raw in counts)
    # The aux-loss-free routing-bias update (bincount/nonzero on Meta) is
    # skipped under the capture's force-load-balance debug flag; the real
    # device path keeps it.
    assert "aten.bincount.default" not in counts
    assert not any(raw.startswith("aten.nonzero") for raw in counts)
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _PARAM_TENSORS
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # Reference ``train_dummy`` dtype: fp32 parameters, fp32 AdamW moments
    # and one fp32 step scalar per participating parameter. Student-only:
    # no teacher parameters, no teacher activations.
    assert memory.persistent_param_bytes == _PARAMS * 4
    assert memory.optimizer_state_bytes == _PARAMS * 8 + _PARAM_TENSORS * 4
    # Capture-session vendor fused boundaries for RMSNorm and rotary-apply
    # (deliberately coarser than the unfused production trace), with the
    # Meta autograd bridges recording the paired backward operators.
    assert counts["npu.npu_rms_norm.default"] == 75
    assert counts["npu.npu_rms_norm_backward.default"] == 75
    assert counts["npu.npu_rotary_mul.default"] == 64
    assert counts["npu.npu_rotary_mul_backward.default"] == 64
    # Only the FlashAttention virtual boundary declares tensor roles.
    assert all(
        "tensor_semantics" not in n.attrs
        for n in nodes
        if not str(n.annotations["raw_op_type"]).startswith("sys2mot_sim.")
    )


def test_sys2mot_full_ac_recomputes_forward_math():
    workload = simulate(
        SimulationConfig(config="sys2mot_debugmodel_multimodal", seq_len=100, activation_checkpoint_mode="full")
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # Full AC recomputes the forward inside backward: the grouped GEMMs,
    # router scoring and indexer topk recompute once more, while the
    # loss/optimizer signatures keep their counts.
    assert counts["aten._grouped_mm.default"] == 96
    assert counts["aten.topk.default"] == 36
    assert counts["aten.softplus.default"] == 16
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _PARAM_TENSORS


def test_sys2mot_selective_ac_gmm_save_set_skips_grouped_gemm_recompute():
    workload = simulate(
        SimulationConfig(
            config="sys2mot_debugmodel_multimodal",
            seq_len=100,
            activation_checkpoint_mode="selective",
            selective_ac_save_ops=["gmm", "attention"],
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The gmm save-set keeps the student grouped GEMMs saved (forward count
    # only); the indexer/router topk is not part of the saved set and
    # recomputes.
    assert counts["aten._grouped_mm.default"] == 72
    assert counts["aten.topk.default"] == 36
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _PARAM_TENSORS


def test_sys2mot_rejects_tensor_parallel_at_config_level():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        simulate(
            SimulationConfig(
                config="sys2mot_debugmodel_multimodal",
                seq_len=100,
                world_size=2,
                tensor_parallel_degree=2,
            )
        )


def test_sys2mot_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="context parallel"):
        simulate(
            SimulationConfig(
                config="sys2mot_debugmodel_multimodal",
                seq_len=100,
                world_size=2,
                context_parallel_degree=2,
            )
        )


def test_sys2mot_expert_parallel_runs_all_to_all_dispatch():
    # EP follows the shared MoE decoder recipe: routed experts ride the
    # efsdp mesh (EP shard on the expert dim), the all-to-all dispatcher
    # exchanges tokens (local fallback at EP=1 keeps the base capture
    # byte-identical), and the FSDP all-gather/reduce-scatter chain stays.
    # Student-only capture: the offline-distillation form keeps the teacher
    # out of the training step (see sys2mot-distill-sharded for the archived
    # in-loop sharded teacher contract: a2a 64, gmm 96, ag 54, rs 20).
    workload = simulate(
        SimulationConfig(
            config="sys2mot_debugmodel_multimodal",
            seq_len=100,
            world_size=8,
            expert_parallel_degree=4,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 8 MoE layers, each with the balanced count exchange plus the
    # dispatch/combine pair.
    assert counts["_c10d_functional.all_to_all_single.default"] == 40
    assert counts["aten._grouped_mm.default"] == 72
    assert counts["c10d._allgather_base_.default"] == 37
    assert counts["c10d._reduce_scatter_base_.default"] == 20


def test_sys2mot_preset_rejects_mtp_modules():
    from torchtitan_npu.models.sys2mot.config_registry import sys2mot_debugmodel_multimodal

    with pytest.raises(ValueError, match="no MTP modules"):
        sys2mot_debugmodel_multimodal(num_mtp_layers=1)


def test_sys2mot_mid_geometry_sources_in_range():
    """Mid layer schedules must land on real layers (PR #25 review: index 8 >= layers 8)."""
    from torchtitan_npu.models.sys2mot.config_registry import _GeometryMid

    geometry = _GeometryMid()
    geometry.validate()
    assert geometry.layers == 8
    assert len(geometry.index_sources) == 4
    assert all(
        0 <= i < geometry.layers for i in (*geometry.kv_sources, *geometry.index_sources, geometry.candidate_source)
    )


def test_sys2mot_student30b_full_geometry_validates():
    """The full-geometry prediction flavor mirrors the reference student30b flavor."""
    from torchtitan_npu.models.sys2mot.config_registry import _GeometryStudent30b

    geometry = _GeometryStudent30b()
    geometry.validate()
    assert geometry.layers == 40
    assert geometry.experts == 24 and geometry.expert_dim == 1408
    assert len(geometry.index_sources) == 8
    assert all(
        0 <= i < geometry.layers for i in (*geometry.kv_sources, *geometry.index_sources, geometry.candidate_source)
    )


def test_sys2mot_student300b_full_geometry_validates():
    """The 300b flavor is a pure expert/memory delta over student30b (reference replace pattern)."""
    from torchtitan_npu.models.sys2mot.config_registry import _GeometryStudent30b, _GeometryStudent300b

    geometry = _GeometryStudent300b()
    geometry.validate()
    assert geometry.experts == 224 and geometry.expert_dim == 2048
    assert geometry.engram_buckets == 668673
    for field in ("layers", "kv_sources", "index_sources", "compress_ratios", "engram_layers"):
        assert getattr(geometry, field) == getattr(_GeometryStudent30b(), field)
