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

from torchtitan_npu.models.system3.model import GDN2Attention

# Debug-model geometry (torchtitan_npu.models.system3.config_registry, the
# reference smoke_test config): hidden=128, vocab=512, 12 physical blocks
# looped twice, [ISA, GDN2 x 3] units, 4 experts top-2 + 1 shared in a
# 48-wide latent, Engram on layers 1/5/9 (orders [2], 2 heads x 64 rows x
# 16 dim — each module executes twice per pass with the block loop), mHC
# hc_mult=2 x branch 1024, FP32 parameters. The multimodal flavor adds the
# four encoders (image 64, audio/point3d/haptic).
# Engram runs the official host-table paradigm: the three hash tables are
# CPU-resident (device "cpu" in the parameter layouts, excluded from the
# device persistent bytes and the dense gradient contract), each pass
# gathers rows on the host and moves them over as visible transfers, EP>1
# routes the lookups through the production all-to-all exchange, and the
# tables train through owner-local sparse gradients fed to SparseAdam.
_TEXT_PARAMS = 13_886_412
_TEXT_PARAM_TENSORS = 373
_ENGRAM_TABLE_PARAMS = 3 * 2 * 64 * 16
_TEXT_DENSE_PARAMS = _TEXT_PARAMS - _ENGRAM_TABLE_PARAMS
_TEXT_DENSE_TENSORS = 370
_MM_PARAMS = 14_214_060
_MM_PARAM_TENSORS = 440
_MM_DENSE_PARAMS = _MM_PARAMS - _ENGRAM_TABLE_PARAMS
_MM_DENSE_TENSORS = 437


def _table_host_transfers(nodes):
    """The visible rows H2D transfers of the host hash tables."""
    return [
        n
        for n in nodes
        if n.annotations["raw_op_type"] == "aten._to_copy.default"
        and n.annotations.get("module_path", "").endswith("engram.hash_tables.0")
    ]


def test_system3_param_formula_matches_reference_smoke_geometry():
    # The reference smoke_test counts 14,148,524 parameters with a tied LM
    # head; the capture flavor keeps the head untied (+512 x 128). Encoder
    # towers and stem convs match the reference exactly (no conv biases).
    assert _MM_PARAMS == 14_148_524 + 512 * 128
    # The encoder towers over the untied text backbone.
    assert _MM_PARAMS - _TEXT_PARAMS == 327_648
    # The dense contract excludes exactly the three host tables.
    assert _TEXT_DENSE_PARAMS == _TEXT_PARAMS - _ENGRAM_TABLE_PARAMS
    assert _TEXT_DENSE_TENSORS == _TEXT_PARAM_TENSORS - 3
    assert _MM_DENSE_PARAMS == _MM_PARAMS - _ENGRAM_TABLE_PARAMS
    assert _MM_DENSE_TENSORS == _MM_PARAM_TENSORS - 3


def test_system3_text_capture_reaches_iteration_with_reference_operators():
    workload = simulate(SimulationConfig(config="system3_debugmodel_text", seq_len=16))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The compatibility layer installs the system3 virtual fusions and the
    # npu_rms_norm recipe: the per-token delta-rule loop collapses into
    # ``system3_sim.gdn2_chunked`` (absorbing the eager pad/mask branches),
    # the MoE FFN activation collapses into ``system3_sim.swiglu_clamp``
    # (absorbing silu/gelu), the ISA top-K selection and the selected-key
    # attention core collapse into ``system3_sim.isa_indexer`` /
    # ``isa_sparse_attention`` (absorbing the indexer topk/softmax and the
    # gather/score/combine einsums), and the reference RMSNorm math
    # dispatches as ``npu_rms_norm``.  The remaining eager operators (MoE
    # routing topk, grouped GEMMs, router softmax) keep their counts.
    assert counts["system3_sim.gdn2_chunked.default"] == 18
    assert counts["system3_sim.gdn2_chunked_backward.default"] == 18
    assert counts["system3_sim.swiglu_clamp.default"] == 36
    assert counts["system3_sim.swiglu_clamp_backward.default"] == 36
    assert counts["system3_sim.isa_indexer.default"] == 4
    assert counts["system3_sim.isa_indexer_grad.default"] == 4
    assert counts["system3_sim.isa_sparse_attention.default"] == 6
    assert counts["system3_sim.isa_sparse_attention_grad.default"] == 6
    assert counts["npu.npu_rms_norm.default"] == 243
    assert counts["npu.npu_rms_norm_backward.default"] == 243
    assert counts["aten.constant_pad_nd.default"] == 0
    assert counts["aten.masked_fill_.Scalar"] == 0
    assert counts["aten.silu.default"] == 0
    assert counts["aten.topk.default"] == 18
    assert counts["aten._grouped_mm.default"] == 162
    assert counts["aten._softmax.default"] == 18
    # The tok embedding is the only dense lookup left: the six per-pass
    # Engram table executions no longer dispatch per-head F.embedding; each
    # one gathers its rows on the host instead, leaving a host-scratch
    # ``empty`` and a visible rows H2D ``_to_copy`` per execution.
    assert counts["aten.embedding.default"] == 1
    assert counts["aten.embedding_dense_backward.default"] == 1
    assert counts["aten.empty.memory_format"] == 6
    assert len(_table_host_transfers(nodes)) == 6
    # No foreign virtual fusions: the only virtual ops are this workspace's
    # system3_sim family, and they are the only nodes carrying
    # ``tensor_semantics`` schema attributes.
    assert not any(raw.startswith(("multi_head_moe_sim.", "deepseek_v4.")) for raw in counts)
    semantic_nodes = [n for n in nodes if "tensor_semantics" in n.attrs]
    assert {n.annotations["raw_op_type"] for n in semantic_nodes} == {
        "system3_sim.gdn2_chunked.default",
        "system3_sim.gdn2_chunked_backward.default",
        "system3_sim.swiglu_clamp.default",
        "system3_sim.swiglu_clamp_backward.default",
        "system3_sim.isa_indexer.default",
        "system3_sim.isa_indexer_grad.default",
        "system3_sim.isa_sparse_attention.default",
        "system3_sim.isa_sparse_attention_grad.default",
    }
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    # Only the dense parameters carry the gradient contract; the three host
    # tables reach SparseAdam through their pending sparse gradients.
    assert len(gradients) == _TEXT_DENSE_TENSORS
    assert not any("hash_tables" in name for name in gradients)
    groups = workload.iteration.schedule.annotations["optimizer_parameter_groups"]
    sparse_groups = [group for group in groups if group["algorithm"] == "SparseAdam"]
    assert len(sparse_groups) == 1
    assert len(sparse_groups[0]["parameter_names"]) == 3
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # The CPU tables are not device-persistent; only the dense parameters
    # hold device residency.
    assert memory.persistent_param_bytes == _TEXT_DENSE_PARAMS * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating dense parameter; the host tables get no
    # dense moments (SparseAdam skips grad-less tables at capture).
    assert memory.optimizer_state_bytes == _TEXT_DENSE_PARAMS * 8 + _TEXT_DENSE_TENSORS * 4


def test_system3_full_ac_recomputes_forward_math():
    workload = simulate(
        SimulationConfig(config="system3_debugmodel_text", seq_len=16, activation_checkpoint_mode="full")
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # Full AC recomputes the forward inside backward: the forward-only math
    # doubles (topk/indexer, grouped GEMMs, softmax, router math, and the
    # virtual fusion forward stems), while the backward-side counts keep
    # their single-pass population.
    assert counts["system3_sim.gdn2_chunked.default"] == 36
    assert counts["system3_sim.gdn2_chunked_backward.default"] == 18
    assert counts["system3_sim.swiglu_clamp.default"] == 72
    assert counts["system3_sim.swiglu_clamp_backward.default"] == 36
    assert counts["system3_sim.isa_indexer.default"] == 8
    assert counts["system3_sim.isa_indexer_grad.default"] == 4
    assert counts["system3_sim.isa_sparse_attention.default"] == 12
    assert counts["system3_sim.isa_sparse_attention_grad.default"] == 6
    assert counts["npu.npu_rms_norm.default"] == 485
    assert counts["npu.npu_rms_norm_backward.default"] == 243
    assert counts["aten.topk.default"] == 36
    assert counts["aten._grouped_mm.default"] == 216
    assert counts["aten._softmax.default"] == 36
    # Full AC replays the Engram host lookups with the recompute: the
    # tok embedding stays single (outside the checkpointed blocks).
    assert counts["aten.embedding.default"] == 1
    assert counts["aten.empty.memory_format"] == 12
    assert len(_table_host_transfers(nodes)) == 12
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TEXT_DENSE_TENSORS


def test_system3_multimodal_capture_includes_encoder_gradients():
    workload = simulate(SimulationConfig(config="system3_debugmodel_multimodal", seq_len=16))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    # 370 dense backbone + 67 encoder parameter tensors — the reference smoke
    # counts exactly 67 encoder grads as well; the host tables stay outside
    # the dense gradient contract.
    assert len(gradients) == _MM_DENSE_TENSORS
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The encoders are plain bidirectional blocks (qkv bmm attention + gelu
    # MLP) behind Conv/patch stems; the modality type lookup joins the
    # sequence (backbone lookups + one per modality segment).  The encoder
    # attention runs as ``npu_fusion_attention`` (8 forward + 8 gradient
    # stems), and the backbone ISA units fuse into the virtual selection and
    # selected-key boundaries, so the softmax count keeps only the eager
    # router math.
    assert counts["npu.npu_fusion_attention.default"] == 8
    assert counts["npu.npu_fusion_attention_grad.default"] == 8
    assert counts["system3_sim.isa_indexer.default"] == 4
    assert counts["system3_sim.isa_sparse_attention.default"] == 6
    # tok embedding + one modality-segment lookup per encoder; the Engram
    # per-head lookups are host gathers now.
    assert counts["aten.embedding.default"] == 5
    assert counts["aten._softmax.default"] == 18  # eager router math only
    assert counts["aten.gelu.default"] == 13  # encoder MLP activations stay eager
    assert counts["aten.empty.memory_format"] == 6
    assert len(_table_host_transfers(nodes)) == 6
    memory = workload.iteration.schedule.annotations["memory_plan"]
    assert memory.persistent_param_bytes == _MM_DENSE_PARAMS * 4
    assert memory.optimizer_state_bytes == _MM_DENSE_PARAMS * 8 + _MM_DENSE_TENSORS * 4


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
    # experts shrink with EP, the per-layer GEMM count does not). The Engram
    # lookups join the routed path: 6 table executions x the production
    # counts/ids/rows exchange triple adds 18 more all-to-alls.
    assert counts["_c10d_functional.all_to_all_single.default"] == 108
    engram_a2a = [
        n
        for n in nodes
        if n.annotations["raw_op_type"] == "_c10d_functional.all_to_all_single.default"
        and "engram" in n.annotations.get("module_path", "")
    ]
    assert len(engram_a2a) == 18
    assert counts["aten._grouped_mm.default"] == 162
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TEXT_DENSE_TENSORS
    # The tables shard their rows over EP: 128 flat rows / 2 = 64 local rows
    # on each owner, CPU-resident, outside FSDP.
    layouts = workload.iteration.schedule.annotations["parameter_layouts"]
    table_layouts = {name: layout for name, layout in layouts.items() if "hash_tables" in name}
    assert len(table_layouts) == 3
    assert all(layout["device"] == "cpu" for layout in table_layouts.values())
    assert all(tuple(layout["local_shape"]) == (64, 16) for layout in table_layouts.values())


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


# ---------------------------------------------------------------------------
# Review regressions (PR #24): mask propagation, virtual-op value geometry
# and the shared fusion-attention fake contracts.
# ---------------------------------------------------------------------------


def _built_debugmodel():
    from torchtitan_npu.models.system3.config_registry import system3_configs

    return system3_configs["debugmodel_text"]().build()


def test_gdn2_block_call_forwards_padding_mask():
    """GDN2 layers own the block's attention slot directly; the uniform block
    call must land the mask in GDN2Attention.attention_mask (not kv_bufs)."""
    import torch

    model = _built_debugmodel()
    block = next(mod for name, mod in model.named_modules() if name.endswith("layers.2"))
    gdn2 = block.attention
    assert isinstance(gdn2, GDN2Attention)

    captured = {}
    original = gdn2.forward

    def spy(hidden_states, kv_bufs=None, attention_mask=None):
        captured["kv_bufs"] = kv_bufs
        captured["attention_mask"] = attention_mask
        return original(hidden_states, kv_bufs=kv_bufs, attention_mask=attention_mask)

    gdn2.forward = spy
    mask = torch.ones(2, 16, dtype=torch.bool)
    block(torch.zeros(2, 16, 128), None, None, mask)
    assert captured["attention_mask"] is mask
    assert captured["kv_bufs"] is None


def test_gdn2_delta_rule_scan_gates_state_writes_by_padding_mask():
    """Leading padding must not write the delta-rule state: with the mask,
    valid-position reads equal the outputs of the truncated sequence; without
    it, the garbage writes pollute the state and the reads drift."""
    import torch

    model = _built_debugmodel()
    gdn2 = next(mod for name, mod in model.named_modules() if name.endswith("layers.1.attention"))

    torch.manual_seed(3)
    b, s, lead = 1, 16, 4
    hidden = torch.randn(b, s, 128)
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[:, :lead] = False

    def scan(sequence_length, sequence_mask):
        h = hidden[:, s - sequence_length :]
        q = gdn2.q_proj(h).view(b, sequence_length, gdn2.H, gdn2.d_k)
        k = gdn2.k_proj(h).view(b, sequence_length, gdn2.H, gdn2.d_k)
        v = gdn2.v_proj(h).view(b, sequence_length, gdn2.H, gdn2.d_v)
        alpha = torch.sigmoid(gdn2.alpha_gate(h).view(b, sequence_length, gdn2.H, gdn2.d_k))
        beta = gdn2.beta_gate(h).view(b, sequence_length, 2, gdn2.H, gdn2.d_state)
        da = torch.sigmoid(gdn2.da_norm(gdn2.da_proj(h)).view(b, sequence_length, gdn2.H, gdn2.d_a).mean(dim=-1))
        return gdn2._delta_rule_scan(q, k, v, alpha, beta[:, :, 0], beta[:, :, 1], da, sequence_mask)

    truncated = scan(s - lead, None)
    masked = scan(s, mask)
    unmasked = scan(s, None)

    assert torch.equal(masked[:, lead:], truncated)
    assert not torch.equal(unmasked[:, lead:], truncated)


def test_isa_sparse_attention_meta_output_follows_value_width():
    """The selected-key attention boundary declares value_head_dim outputs;
    the fake must allocate (b, s, heads, v_width), not the query width."""
    import torch
    from torchtitan_simulator.compat.system3_fusions import _isa_sparse_attention_meta

    b, s, heads, qk, value = 1, 8, 2, 8, 16
    q = torch.empty(b, s, heads, qk, dtype=torch.bfloat16, device="meta")
    k = torch.empty(b, s // 2, heads, qk, dtype=torch.bfloat16, device="meta")
    v = torch.empty(b, s // 2, heads, value, dtype=torch.bfloat16, device="meta")
    indices = torch.zeros(b, s, 4, dtype=torch.int64, device="meta")
    bias = torch.zeros(b, s, 4, dtype=torch.bfloat16, device="meta")

    out, lse = _isa_sparse_attention_meta(q, k, v, indices, bias, None, 0.25, 50.0, 2)
    assert out.shape == (b, s, heads, value)
    assert lse.shape == (b, heads, s)


def test_fusion_attention_grad_meta_follows_operand_geometry():
    """The shared BSND fusion-attention backward fake must mirror each
    operand's own shape (GQA heads / differing widths), not the query shape."""
    import torch
    from torchtitan_simulator.compat.attention import (
        _npu_fusion_attention_grad_meta,
        _npu_fusion_attention_meta,
    )

    q = torch.empty(1, 8, 4, 16, dtype=torch.bfloat16, device="meta")
    k = torch.empty(1, 8, 2, 16, dtype=torch.bfloat16, device="meta")
    v = torch.empty(1, 8, 2, 32, dtype=torch.bfloat16, device="meta")
    dy = torch.empty(1, 8, 4, 16, dtype=torch.bfloat16, device="meta")

    out = _npu_fusion_attention_meta(q, k, v, 4, "BSND")[0]
    assert out.shape == q.shape

    dq, dk, dv = _npu_fusion_attention_grad_meta(q, k, v, dy, 4, "BSND")[:3]
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
