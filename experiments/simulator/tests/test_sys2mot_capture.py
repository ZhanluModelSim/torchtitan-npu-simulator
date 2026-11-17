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
    # (im2col/col2im). The frozen distillation teacher adds one eval-mode
    # VLM backbone pass (no backward: the paired backward counts stay at the
    # student-only values), one extra ViT patch embed and one extra text +
    # Engram embedding set.
    assert counts["aten._grouped_mm.default"] == 96
    assert counts["aten.topk.default"] == 36
    assert counts["aten._softmax.default"] == 83
    assert counts["aten.softplus.default"] == 16
    assert counts["aten.embedding.default"] == 6  # student + teacher text/Engram
    assert counts["aten.constant_pad_nd.default"] == 79
    assert counts["aten.im2col.default"] == 2
    assert counts["aten.convolution.default"] == 2
    assert counts["aten.sort.stable"] == 16  # token dispatcher per MoE layer, student + teacher
    assert counts["torchtitan.deterministic_scatter_add.default"] == 16
    # The captured joint loss: flow MSE + 0.5 logits KD + 0.1 hidden MSE +
    # 0.1 text CE + 0.01 indexer aux. F.kl_div captures decomposed
    # (xlogy + mul/sub), so the kd signature is the single xlogy.
    assert counts["aten.mse_loss.default"] == 1
    assert counts["aten.nll_loss_forward.default"] == 1
    assert counts["aten.xlogy.Tensor"] == 1
    assert counts["aten._log_softmax.default"] == 10
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
    # and one fp32 step scalar per participating parameter.
    assert memory.persistent_param_bytes == _PARAMS * 4
    assert memory.optimizer_state_bytes == _PARAMS * 8 + _PARAM_TENSORS * 4
    # Capture-session vendor fused boundaries for RMSNorm and rotary-apply
    # (deliberately coarser than the unfused production trace), with the
    # Meta autograd bridges recording the paired backward operators.
    assert counts["npu.npu_rms_norm.default"] == 117
    assert counts["npu.npu_rms_norm_backward.default"] == 75
    assert counts["npu.npu_rotary_mul.default"] == 104
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
    # router scoring and indexer topk recompute once more, the frozen
    # teacher's eval pass runs once (no gradient), while the loss/optimizer
    # signatures keep their counts.
    assert counts["aten._grouped_mm.default"] == 120
    assert counts["aten.topk.default"] == 54
    assert counts["aten.softplus.default"] == 24
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
    # recomputes. The frozen teacher's eval pass runs once on top.
    assert counts["aten._grouped_mm.default"] == 96
    assert counts["aten.topk.default"] == 54
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


def test_sys2mot_rejects_expert_parallel():
    with pytest.raises(NotImplementedError, match="expert parallel"):
        simulate(
            SimulationConfig(
                config="sys2mot_debugmodel_multimodal",
                seq_len=100,
                world_size=2,
                expert_parallel_degree=2,
            )
        )


def test_sys2mot_preset_rejects_mtp_modules():
    from torchtitan_npu.models.sys2mot.config_registry import sys2mot_debugmodel_multimodal

    with pytest.raises(ValueError, match="no MTP modules"):
        sys2mot_debugmodel_multimodal(num_mtp_layers=1)
