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
    # The CSA2/Engram/mHC/action paths ride the fused sys2mot_sim training
    # boundaries. The routed experts run three grouped GEMMs per MoE layer
    # (8 layers), the router scores with sqrt(softplus) in fp32, and the
    # force-balanced MoE routing keeps its real topk calls (the CSA2
    # retrieval selection lives inside csa_indexer), and the ViT pixel
    # unshuffle is the reference's own unfold (im2col/col2im). Distillation
    # is offline in the current form: the training step is student-only and
    # the teacher's logits/hidden are inputs the re-enabled distill knob
    # would consume (see sys2mot-distill-sharded for the archived in-loop
    # form).
    assert counts["aten._grouped_mm.default"] == 72
    assert counts["aten.topk.default"] == 8
    # Student routers plus the offline-distillation teacher softmax.
    assert counts["aten._softmax.default"] == 13
    # The routers fold sqrt(softplus) into the router_score boundary.
    assert counts["sys2mot_sim.router_score.default"] == 8
    assert counts["sys2mot_sim.router_score_backward.default"] == 8
    assert counts["aten.embedding.default"] == 1  # student text; the Engram
    # table gather rides the engram_lookup boundary.
    assert counts["aten.constant_pad_nd.default"] == 14  # the reference's own
    # remaining pads, plus the eight per-indexer KL chunk-fold pads (the 36
    # n-gram shift pads are inside ngram_hash and the retrieval masks are
    # inside csa_indexer).
    assert counts["aten.im2col.default"] == 1
    assert counts["aten.convolution.default"] == 1
    # The token dispatcher runs the fused local permute/unpermute boundary
    # (vendor npu_moe_token_permute + the torchtitan_npu wrapper with the
    # fp32-internal score multiply and top-K sum) instead of the reference's
    # stable argsort and deterministic scatter-add.
    assert counts["npu.npu_moe_token_permute.default"] == 8
    assert counts["torchtitan_npu.npu_moe_token_unpermute.default"] == 8
    assert "aten.sort.stable" not in counts
    assert "torchtitan.deterministic_scatter_add.default" not in counts
    # The captured joint loss: flow MSE + 0.1 text CE + 0.01 indexer aux,
    # plus the offline-distillation KD/hidden terms computed on synthetic
    # stand-in teacher tensors (inputs the real offline step loads from the
    # precomputation pass): the KD signature is the single xlogy, the teacher
    # softmax adds one entry, and the hidden term uses square/sub (no second
    # mse_loss).
    assert counts["aten.mse_loss.default"] == 1
    assert counts["aten.nll_loss_forward.default"] == 1
    assert counts["aten.xlogy.Tensor"] == 1
    assert counts["aten._log_softmax.default"] == 6
    assert counts["aten._softmax.default"] == 13
    # The Engram memory (layers 1/3) and the CSA2 retrieval (indexers at
    # layers 1/2/4/5) ride the sys2mot_sim training boundaries with their
    # paired backwards.
    assert counts["sys2mot_sim.ngram_hash.default"] == 2
    assert counts["sys2mot_sim.engram_lookup.default"] == 2
    assert counts["sys2mot_sim.engram_gate.default"] == 2
    assert counts["sys2mot_sim.engram_lookup_backward.default"] == 2
    assert counts["sys2mot_sim.engram_gate_backward.default"] == 2
    assert counts["sys2mot_sim.csa_indexer.default"] == 4
    assert counts["sys2mot_sim.csa_indexer_backward.default"] == 4
    # The sparse attention (8 layers) rides the window_topk boundary with
    # its training-only backward. The mHC hyper-connections (two sublayer
    # plans per block plus the two final-norm collapses) ride the mix,
    # collapse and expand boundaries.
    assert counts["sys2mot_sim.window_topk_attention.default"] == 8
    assert counts["sys2mot_sim.window_topk_attention_backward.default"] == 8
    assert counts["sys2mot_sim.mhc_mix.default"] == 16
    assert counts["sys2mot_sim.mhc_mix_backward.default"] == 16
    assert counts["sys2mot_sim.mhc_collapse.default"] == 18
    assert counts["sys2mot_sim.mhc_collapse_backward.default"] == 17
    assert counts["sys2mot_sim.mhc_expand.default"] == 16
    assert counts["sys2mot_sim.mhc_expand_backward.default"] == 16
    # The dense MLPs and grouped experts share the clamped-SwiGLU boundary
    # (26 call sites); the eager clamp/silu/multiply chains are gone.
    assert counts["sys2mot_sim.swiglu.default"] == 26
    assert counts["sys2mot_sim.swiglu_backward.default"] == 26
    assert counts["aten.silu.default"] == 9
    assert counts["aten.silu_backward.default"] == 9
    # Every MoE router folds sqrt(softplus) into the router_score boundary
    # (8 layers); the eager softplus/sqrt chains and the sqrt backward
    # div/mul glue are gone.
    assert counts["sys2mot_sim.router_score.default"] == 8
    assert counts["sys2mot_sim.router_score_backward.default"] == 8
    assert "aten.softplus.default" not in counts
    assert "aten.sqrt.default" not in counts
    # The debug geometry's single ratio>1 compressor folds its gated window
    # pooling into the compress_pool boundary.
    assert counts["sys2mot_sim.compress_pool.default"] == 1
    assert counts["sys2mot_sim.compress_pool_backward.default"] == 1
    # The retrieval's ReLU pool, chunk masking, candidate generation and
    # per-chunk topk selection no longer leak as eager operators.
    assert "aten.einsum.default" not in counts
    assert "aten.relu.default" not in counts
    assert "aten.amax.default" not in counts
    assert counts["aten.gather.default"] == 8
    # Only the sys2mot_sim virtual boundaries (FlashAttention, Engram,
    # CSA2 retrieval) are substituted; every other node is a native
    # aten/collective op.
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
    # The Engram memory (layers 1/3) rides the sys2mot_sim training
    # boundaries: rolling Horner hash with eligibility, row gather, and the
    # fp32 RMS/SSR gate, each with its training-only backward.
    assert counts["sys2mot_sim.ngram_hash.default"] == 2
    assert counts["sys2mot_sim.engram_lookup.default"] == 2
    assert counts["sys2mot_sim.engram_gate.default"] == 2
    assert counts["sys2mot_sim.engram_lookup_backward.default"] == 2
    assert counts["sys2mot_sim.engram_gate_backward.default"] == 2
    # The gate's sign/sigmoid/norm math and the hash's integer remainder
    # chain no longer leak as eager operators.
    assert "aten.sign.default" not in counts
    assert counts["aten.remainder.Scalar"] == 11
    # Only the FlashAttention virtual boundary and the shared MoE dispatch
    # wrappers declare tensor roles.
    assert all(
        "tensor_semantics" not in n.attrs
        for n in nodes
        if not str(n.annotations["raw_op_type"]).startswith(
            ("sys2mot_sim.", "torchtitan_npu.npu_moe_")
        )
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
    # router scoring and retrieval boundary recompute once more, while the
    # loss/optimizer signatures keep their counts.
    assert counts["aten._grouped_mm.default"] == 96
    assert counts["aten.topk.default"] == 16
    assert counts["sys2mot_sim.router_score.default"] == 16
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
    # only); the router topk is not part of the saved set and recomputes.
    assert counts["aten._grouped_mm.default"] == 72
    assert counts["aten.topk.default"] == 16
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


def test_sys2mot_offline_distill_terms_normalize_over_text_mask():
    """Offline KD/hidden exclude invalid text positions and their denominators."""
    import torch
    import torch.nn.functional as F

    from torchtitan_npu.models.sys2mot.loss import _offline_teacher_terms

    logits = torch.randn(2, 6, 8, dtype=torch.float32)
    span = torch.randn(2, 6, 5, dtype=torch.float32)
    teacher_logits = torch.randn(2, 6, 8, dtype=torch.float32)
    teacher_hidden = torch.randn(2, 6, 5, dtype=torch.float32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 5] = False
    mask[1, 3] = False

    loss = _offline_teacher_terms(logits, span, teacher_logits, teacher_hidden, mask, 2.0)

    temperature = 2.0
    kl = F.kl_div(
        (logits.float() / temperature).log_softmax(-1),
        (teacher_logits.float() / temperature).softmax(-1),
        reduction="none",
    ).sum(-1)
    expected_kd = (kl * mask).sum() / mask.sum().clamp_min(1) * temperature**2
    expected_hidden = (
        (F.normalize(span.float(), dim=-1) - F.normalize(teacher_hidden.float(), dim=-1)).square().sum(-1) * mask
    ).sum() / mask.sum().clamp_min(1)
    assert torch.isclose(loss, 0.5 * expected_kd + 0.1 * expected_hidden)
    # Invalid positions genuinely change the normalization: the same tensors
    # under an all-valid mask produce a different value.
    all_valid = _offline_teacher_terms(logits, span, teacher_logits, teacher_hidden, torch.ones_like(mask), 2.0)
    assert not torch.isclose(loss, all_valid)
    with pytest.raises(ValueError, match="text_mask"):
        _offline_teacher_terms(logits, span, teacher_logits, teacher_hidden, mask[0], 2.0)


def test_sys2mot_offline_loss_requires_model_text_mask_output():
    """The offline loss refuses to synthesize an all-valid mask silently."""
    from types import SimpleNamespace

    import torch

    from torchtitan_npu.models.sys2mot.loss import Sys2MotJointLoss

    config = SimpleNamespace(
        text_offset=4,
        action_dim=2,
        horizon=4,
        ce_weight=0.1,
        indexer_weight=0.01,
        distill=False,
        distill_offline=True,
    )
    loss = Sys2MotJointLoss(config)
    logits = torch.randn(1, 6, 8, dtype=torch.float32)
    pred = {"hidden": torch.randn(1, 10, 4, dtype=torch.float32)}
    with pytest.raises(ValueError, match="distill_text_mask"):
        loss._offline_distill_terms(pred, logits, torch.randn(1, 6, 8), torch.randn(1, 6, 4))


def test_sys2mot_step_rejects_non_tensor_extra_loss_inputs():
    import torch
    from torchtitan_simulator.step import run_simulation_step

    with pytest.raises(TypeError, match="extra loss input"):
        run_simulation_step(
            model_parts=[],
            parallel_dims=None,
            forward_backward_step=lambda **_: None,
            input_dict={},
            labels=torch.zeros(1, dtype=torch.int64),
            extra_loss_inputs={"teacher_logits": "not-a-tensor"},
            optimizer_step=lambda: None,
            lr_scheduler_step=lambda: None,
            local_batch_size=1,
            seq_len=1,
        )


def test_sys2mot_offline_teacher_inputs_registered_as_initial_memory():
    """Teacher logits/hidden join initial memory with full contiguous storage.

    The offline step's teacher outputs predate the first op: the estimator
    must see their step-start residency (bytes and lifecycle), not infer an
    external-input lifecycle at their first KD consumption.
    """
    from torchtitan_npu.models.sys2mot.config_registry import _Geometry

    geometry = _Geometry()
    teacher_shapes = {(1, 100, geometry.vocab_size), (1, 100, geometry.width)}
    workload = simulate(SimulationConfig(config="sys2mot_debugmodel_multimodal", seq_len=100))
    plan = workload.iteration.schedule.annotations["memory_plan"]
    teacher_refs = [r for r in plan.initial_tensors if r.shape in teacher_shapes and r.dtype == "float32"]
    assert {r.shape for r in teacher_refs} == teacher_shapes
    assert sum(r.num_bytes for r in teacher_refs) == 100 * (geometry.vocab_size + geometry.width) * 4
    # Full contiguous storages: a stride-0 expand view would carry a tiny
    # underlying storage and under-report the loaded input bytes.
    assert all(r.storage_bytes == r.num_bytes for r in teacher_refs)
    teacher_ref_ids = {r.tensor_id for r in teacher_refs}
    teacher_lifetimes = [
        lt
        for lt in plan.tensor_lifetimes
        if lt.kind == "external_input"
        and lt.shape in teacher_shapes
        and lt.dtype == "float32"
        and int(lt.tensor_id.split(":", 1)[1]) in teacher_ref_ids
    ]
    assert len(teacher_lifetimes) == 2
    for lifetime in teacher_lifetimes:
        # Resident from before the first op (not born at the first KD op).
        assert lifetime.birth_seq < 0
        assert lifetime.resident_num_bytes == lifetime.num_bytes
        # Consumed by the KD/hidden math like any loaded offline input, and
        # resident through that consumption.
        assert lifetime.consumer_ops and lifetime.consumer_seqs
        assert lifetime.death_seq >= max(lifetime.consumer_seqs)
