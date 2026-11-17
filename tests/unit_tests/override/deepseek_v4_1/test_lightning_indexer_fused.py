# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused V4.1 LightningIndexer: selection parity, teacher edge, dispatch.

The NPU kernels are replaced by eager math with their exact contract
(document-local indices, ``-1``/``-inf`` padding, the SLIKG closed form
``dI = Z * Y - p``), so these tests pin the wiring: the fused selector matches the
reference path's selection and scores, the teacher reaches SLIKG on the ``topk_scores``
edge, and every layer -- pool-configured ones included -- runs the fused path.

**The indexer only learns when both overrides are enabled together.** ``sparse_attn.asc``
produces the teacher (SMLAG's backward is the only source of it) and
``lightning_indexer.asc`` consumes it (SLIKG is the only consumer). With the selector
override alone the indexer's selection feeds the attention as indices, which carry no
gradient, so its parameters stay frozen; that is why nothing here checks an "eager
indexer gradient" -- with the distillation loss removed there is no longer a path to one.
"""

import dataclasses
import importlib
from itertools import pairwise

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.config import derive

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import layer_metadata, preprocess
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import (
    AscSelector,
)
from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import build_fused_metadata

_TOKENS = torch.cat([torch.arange(64), torch.arange(63, -1, -1)]).unsqueeze(0)
_POSITIONS = torch.cat([torch.arange(64), torch.arange(64)]).unsqueeze(0)

# The tiny debug layer's own geometry, at the widths ``_tiny_model`` builds it with.  The port's
# records are sized from the override's arguments and it has no inline operator path, so every
# fused layer a test drives and the metadata that layer reads must state the *same* numbers: the
# lookup and the record's guard compare them, and a mismatch is an error rather than a fallback.
_TINY_GEOMETRY = dict(num_heads=2, head_dim=8, index_topk=32, index_num_heads=2, index_head_dim=4)
_TINY_LAYER_GEOMETRY = AicpuGeometry(window_size=128, **_TINY_GEOMETRY)


# ---------------------------------------------------------------------------
# Eager stand-ins for the NPU kernels, following the verified op contracts.
# ---------------------------------------------------------------------------


def _metadata(bounds: list[int], *, ratio: int = 2, geometry: AicpuGeometry | None = None):
    """Build one geometry's fused metadata for a packed row of these document boundaries.

    The positions that produce the boundaries are rebuilt here, so the metadata always comes from
    the same builders the attention configs call; one ``arange`` per document is what a packed
    loader emits.  The row comes from the reference backend's builder at this entry's ratio --
    ``cu_seq_k`` is folded there -- and the entry adds this geometry's frames and records, which
    is what the fused selector and core read.  The default geometry is the tiny layer's own, which
    is what the fused stacks below are built with; a test that drives another shape states it.
    """
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)
    reference = CompressedSparseInnerAttention2.Config(window_size=128, softmax_scale=0.5, compress_ratio=ratio)
    return build_fused_metadata(
        reference.build_attention_metadata(positions),
        ratio=ratio,
        geometry=_TINY_LAYER_GEOMETRY if geometry is None else geometry,
    )


def _fake_li_metadata(*args, **kwargs):
    return torch.zeros(1024, dtype=torch.int32)


def _fake_slig_metadata(*args, **kwargs):
    return torch.zeros(64, dtype=torch.int32)


def _teacher_sources(index_source_layers: tuple[int, ...]) -> tuple[int, ...]:
    """The layers whose selection the teacher edge trains -- one SLIKG call each.

    Every index source owns a selection, and a selection only reaches SLIKG through the
    layer that owns it, so the sources are exactly the calls.  The count therefore comes
    from the layer table rather than from the run.
    """
    return tuple(index_source_layers)


def _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio):
    """The relaxed index score over the packed batch, and the visibility that masks it.

    The reference path's exact expressions: a per-doc contraction or a doc-sliced top-k
    rounds and tie-breaks differently (relu zeroes tie at the top-k boundary), which the
    parity tests must not mistake for a wiring difference.  ``starts`` is where each row's
    document begins on the compressed axis, which is the origin of its document-local
    coordinates.
    """
    total = q.shape[0]
    scores_blhn = torch.einsum("blhd,bnd->blhn", q.unsqueeze(0), k.squeeze(1).unsqueeze(0))
    scores_bln = (scores_blhn.relu() * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)
    cu_q = cu_seqlens_q.to(torch.long).tolist()
    cu_k = cu_seqlens_k.to(torch.long).tolist()
    visible = torch.zeros_like(scores_bln, dtype=torch.bool)
    starts = torch.zeros(total, dtype=torch.long)
    for doc in range(len(cu_q) - 1):
        q0, q1 = cu_q[doc], cu_q[doc + 1]
        k0, k1 = cu_k[doc], cu_k[doc + 1]
        token_pos = torch.arange(q1 - q0)
        visible[0, q0:q1, k0:k1] = torch.arange(k1 - k0).view(1, -1) < ((token_pos + 1) // cmp_ratio).view(-1, 1)
        starts[q0:q1] = k0
    return scores_bln.masked_fill(~visible, float("-inf")), visible, starts


def _select(relaxed, reachable, starts, topk):
    """Doc-local top-k of the relaxed score, ``-1``/``-inf`` padded.

    The picks come back in document-local coordinates, the kernels' contract, and the
    unreachable ones as ``-1``.
    """
    total = relaxed.size(1)
    indices = torch.full((total, 1, topk), -1, dtype=torch.int32)
    values = torch.full((total, 1, topk), float("-inf"))
    pick = min(topk, relaxed.size(-1))
    selected = relaxed.topk(pick, dim=-1, sorted=False).indices
    selected = selected.topk(pick, dim=-1, largest=False, sorted=True).values[0]
    ok = reachable[0].gather(-1, selected)
    local = torch.where(ok, selected - starts.unsqueeze(-1), torch.full_like(selected, -1))
    indices[:, 0, :pick] = local.to(torch.int32)
    values[:, 0, :pick] = relaxed[0].gather(-1, selected)
    return indices, values


def _fake_lightning_indexer(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    **_,
):
    """The BF16 ``lightning_indexer`` selection contract."""
    relaxed, visible, starts = _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio)
    return _select(relaxed, visible, starts, topk)


def _fake_slig(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    **_,
):
    """The SLIKG backward: the closed form dI = Z * Y - p of the reference loss."""
    # The real Function's backward runs grad-disabled; the stand-in rebuilds
    # the eager graph internally, so it opts back in.
    with torch.enable_grad():
        q_e = q.detach().float().requires_grad_(True)
        k_e = k.detach().float().requires_grad_(True)
        w_e = w.detach().requires_grad_(True)
        si = sparse_indices.squeeze(1).to(torch.long)
        p = attn_softmax_l1_norm.squeeze(1)
        lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        offsets = torch.repeat_interleave(cu_seqlens_k[:-1].long(), lengths)
        sel = (si + offsets[:, None]).clamp_min(0)
        valid = (si >= 0).float()
        k_sel = k_e.squeeze(1)[sel]
        logits = torch.einsum("thd,tkd->tkh", q_e, k_sel).relu() * w_e.unsqueeze(1)
        i_sel = logits.sum(-1) * valid
        i_masked = torch.where(valid > 0, i_sel, torch.full_like(i_sel, float("-inf")))
        z = p.sum(-1)
        loss = (z * torch.logsumexp(i_masked, -1).nan_to_num(0.0) - (p * i_sel).sum(-1)).sum()
        dq, dk, dw = torch.autograd.grad(loss, [q_e, k_e, w_e])
        softmax_out = torch.softmax(i_masked, -1).nan_to_num(0.0)
    return dq.to(q.dtype), dk.to(k.dtype), dw, softmax_out.unsqueeze(1)


@pytest.fixture(autouse=True)
def _fake_kernels(monkeypatch):
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer_metadata", _fake_li_metadata, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", _fake_lightning_indexer, raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        _fake_slig_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", _fake_slig, raising=False
    )


def _tiny_model(monkeypatch, *, fused: bool):
    """The registered debug flavor at CPU-sized widths, optionally with the LI override."""
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    if fused:
        # ``fused`` means the fused stack, not one of its halves: the teacher edge only
        # exists when both sides are fused, which the model itself enforces.  Ratio-0
        # layers are left on the reference core because they have no compressed stream.
        from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

        for layer in config.layers:
            if layer.attention.inner_attention.compress_ratio == 0:
                continue
            layer.attention.indexer.selector = derive(layer.attention.indexer.selector, AscSelector.Config)
            # State the geometry the way the recipe's override arguments do: this stack runs the
            # tiny widths, and the records its metadata builder produces are sized from these.
            layer.attention.inner_attention = derive(
                layer.attention.inner_attention, AscV41SparseAttention.Config, **_TINY_GEOMETRY
            )
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    for module in model.modules():
        if isinstance(module, AscSelector):
            module.num_global_queries = torch.tensor(float(_TOKENS.numel()))
    return model


def _run_forward_backward(model):
    _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
    output = model(_TOKENS, **kwargs)
    output.sum().backward()
    return output


def test_fused_selection_matches_the_reference_selector(monkeypatch):
    """Per-layer parity: same valid index sets, same scores, kernel-shaped outputs."""
    eager = _tiny_model(monkeypatch, fused=False)
    fused = _tiny_model(monkeypatch, fused=True)
    preprocess(eager, _TOKENS, _TOKENS, _POSITIONS)
    # The reference selector reads that row; the fused one reads the entry its own port built
    # for the same row, which is what carries the frames and the records the kernels take.
    metadata = _metadata([0, _POSITIONS.shape[-1]], ratio=2)

    eager_indexer = eager.layers["2"].attention.indexer.selector
    fused_indexer = fused.layers["2"].attention.indexer.selector
    torch.manual_seed(7)
    b, l = 1, 128
    hi, di = eager_indexer.num_index_heads, eager_indexer.index_head_dim
    idx_q = torch.randn(b, l, hi, di, requires_grad=True)
    idx_k = torch.randn(b, 64, di, requires_grad=True)
    weights = torch.randn(b, l, hi, requires_grad=True)

    ei, es, _ = eager_indexer.forward(idx_q, idx_k, weights, metadata, candidates_BL1C=None)
    fi, fs, _ = fused_indexer.forward(idx_q, idx_k, weights, metadata, candidates_BL1C=None)

    # int32, the dtype the kernels speak -- the LI op emits it and SMLAG/SLIKG take it,
    # so it is never widened.  The reference selector's int64 is a ``torch.topk`` artefact,
    # and ``gather_attn`` takes either dtype, so nothing is cast on that side.
    assert fi.dtype == torch.int32
    # The fused carrier is a fabricated FP32 buffer of the selection's shape.  Its
    # *contents* are meaningless by construction -- the teacher rides its gradient, not
    # its values -- so only shape, dtype and the graph edge are asserted here.
    assert fs.dtype == torch.float32
    assert es.dtype == fs.dtype == torch.float32
    assert fs.shape == fi.shape
    # Same reachable selection per row.  Both selectors number each query's own document's
    # entries -- the fused one because that is what the kernel grids, the eager one because
    # that is what ``gather_attn`` reads -- so the comparison is direct.
    for t in range(l):
        eager_row = {int(j) for j in ei[0, t].tolist() if j >= 0}
        fused_row = {int(j) for j in fi[0, t].tolist() if j >= 0}
        assert fused_row == eager_row, t

    # A pool smaller than index_topk: the fused kernel keeps its own K and reports the
    # unreachable slots as ``-1``, so it is wider than the eager selection.  The
    # reachable sets still have to agree, and the wider padding must stay inert.
    small_metadata = _metadata([0, 16, 32])
    idx_q2 = torch.randn(1, 32, hi, di, requires_grad=True)
    idx_k2 = torch.randn(1, 16, di, requires_grad=True)
    weights2 = torch.randn(1, 32, hi, requires_grad=True)
    ei2, _es2, _ = eager_indexer.forward(idx_q2, idx_k2, weights2, small_metadata, candidates_BL1C=None)
    fi2, fs2, _ = fused_indexer.forward(idx_q2, idx_k2, weights2, small_metadata, candidates_BL1C=None)
    assert fi2.shape[-1] == fused_indexer.index_topk == 32
    assert ei2.shape[-1] == 16  # the eager selector clamps to the pool
    assert fs2.dtype == torch.float32
    # Same document-local numbering as above.
    for t in range(32):
        fused_t = {int(j) for j in fi2[0, t].tolist() if j >= 0}
        eager_t = {int(j) for j in ei2[0, t].tolist() if j >= 0}
        assert fused_t == eager_t, t


@pytest.mark.parametrize("num_global_queries", [128, 512], ids=["one-row", "four-rows"])
def test_teacher_uses_global_query_count(monkeypatch, num_global_queries):
    """All compression ratios share the step denominator; invalid candidates stay zero."""
    seen = {"calls": 0, "teacher": [], "padding": [], "tokens": [], "emitters": 0}
    constant = 1024.0

    def fake_slig(q, k, w, sparse_indices, attn_softmax_l1_norm, **kw):
        seen["teacher"].append(attn_softmax_l1_norm.detach().clone())
        seen["padding"].append(sparse_indices < 0)
        seen["tokens"].append(q.shape[0])
        seen["calls"] += 1
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(w).float(),
            torch.zeros_like(attn_softmax_l1_norm),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", fake_slig)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def fake_smlag(q, grad, *args, **kw):
        indices = kw["cmp_sparse_indices"]
        teacher = torch.full_like(indices, constant, dtype=torch.float32) if indices is not None else None
        if teacher is not None:
            seen["emitters"] += 1
        shared = kw["cmp_kv"]
        return (
            grad,
            torch.zeros_like(kw["ori_kv"]),
            torch.zeros_like(shared) if shared is not None else None,
            torch.zeros_like(kw["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", fake_smlag)
    model = _tiny_model(monkeypatch, fused=True)
    for module in model.modules():
        if isinstance(module, AscSelector):
            module.num_global_queries = torch.tensor(float(num_global_queries))
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    assert seen["calls"], "SLIKG must be reached for the indexer to train"
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    # One SLIKG call per index source: the override trains each source's selection, and the
    # debug flavor shares the flash topology's tables, so the count comes from there and not
    # from a number read back out of this run.
    sources = _teacher_sources(registry.V41_FULL_INDEX_SOURCE_LAYERS)
    assert seen["calls"] == len(sources), (seen["calls"], sources)
    assert len(set(seen["tokens"])) == 1, set(seen["tokens"])

    for teacher, padding in zip(seen["teacher"], seen["padding"], strict=True):
        tokens = seen["tokens"][0]
        assert tokens == _TOKENS.shape[1], (tokens, _TOKENS.shape[1])
        assert constant > num_global_queries
        # The ``-1`` slots are SMLAG's junk zeroed out on the way here; the unpadded ones
        # are what the division acts on.
        torch.testing.assert_close(
            teacher.masked_select(padding),
            torch.zeros(int(padding.sum())),
            rtol=0,
            atol=0,
        )
        value = teacher.masked_select(~padding).max().item()
        # Shared index sources accumulate one normalized teacher per contributing layer.
        contributors = value * num_global_queries / constant
        assert abs(contributors - round(contributors)) < 1e-6, (
            f"teacher is not a multiple of constant/num_global_queries ({constant}/{num_global_queries}): {value}"
        )
        # No source can carry more contributions than there are layers emitting a teacher, and the
        # split across sources is the layer table's business -- a per-call count of SLIKG calls, or
        # a sum over them, would only encode today's topology (some layers feed two sources).
        assert 1 <= round(contributors) <= seen["emitters"], (
            f"the teacher implies {round(contributors)} contributors but only {seen['emitters']} "
            f"layers emitted one, so the seqlen division ({tokens}) is missing"
        )
        expected = torch.full_like(teacher, constant * round(contributors) / num_global_queries).masked_fill(
            padding, 0.0
        )
        torch.testing.assert_close(teacher, expected)


def test_the_teacher_carrier_is_on_the_operands_device(monkeypatch):
    """The carrier must not land on the default device.

    A Function's edges are its Tensor arguments, and the engine checks each returned
    gradient's device against its input's.  The teacher is emitted by
    ``sparse_flash_mla_grad`` wherever its operands live, so a carrier built on the
    default device fails backward on the NPU with

        Function _SparseMLABackward returned an invalid gradient at index 5 -
        expected device cpu but got npu:0

    (index 5 = the carrier, once the non-Tensor arguments are dropped from the edge
    list).  ``meta`` stands in for "not the default device" here, since the CPU harness
    has no second device to hand.
    """
    from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as li

    device = torch.device("meta")
    seen = {}

    def fake_kernel(q, k, w, topk, **kwargs):
        total = q.shape[0]
        indices = torch.zeros(total, 1, topk, dtype=torch.int32, device=device)
        values = torch.zeros(total, 1, topk, dtype=torch.float32, device=device)
        return indices, values

    def fake_metadata(*args, **kwargs):
        return torch.zeros(8, dtype=torch.int32, device=device)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", fake_kernel)
    # Both of the selector's AI_CPU records are faked: this test is about the device their
    # consumer lands on, not about the operators that produce them.
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer_metadata", fake_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad_metadata", fake_metadata)

    original = li._LightningIndexerTND.forward

    def capture(ctx, *args):
        out = original(ctx, *args)
        seen["indices"] = out[0].device
        seen["carrier"] = out[1].device
        return out

    monkeypatch.setattr(li._LightningIndexerTND, "forward", staticmethod(capture))

    selector = _tiny_model(monkeypatch, fused=True).layers["2"].attention.indexer.selector
    heads, head_dim = selector.num_index_heads, selector.index_head_dim
    selector.forward(
        torch.zeros(1, 8, heads, head_dim, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 4, head_dim, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 8, heads, dtype=torch.float32, device=device),
        _metadata([0, 8], ratio=1),
        candidates_BL1C=None,
    )

    assert seen["indices"] == device, seen["indices"]
    assert seen["carrier"] == device, (
        f"the carrier must share the selection's device, got {seen['carrier']} against {seen['indices']}"
    )


def test_both_kernels_receive_the_layout_their_contract_requires(monkeypatch):
    """The two boundaries where layout has to be translated, and the one where it must not.

    ``layout_q="TND"`` means a ``[T, N2, K]`` selection at SMLA, while the model carries it
    as ``[B, L, 1, K]``.  SLIKG is stricter: its tiling requires ``attn_softmax_l1_norm`` to
    match ``sparse_indices`` exactly.  Both are silent in the CPU fakes -- they accept any
    shape -- so this pins them here, since a wrong rank only surfaces on device.
    """
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as sa

    seen = {}
    real_apply = sa._SparseMLA.apply

    def spy_apply(q, swa_k, cmp_k, topk_indices, sinks, masks, scale, ratio, window, tscore, wants):
        seen["smla_indices"] = tuple(topk_indices.shape)
        seen["smla_carrier"] = None if tscore is None else tuple(tscore.shape)
        return real_apply(q, swa_k, cmp_k, topk_indices, sinks, masks, scale, ratio, window, tscore, wants)

    def fake_slig(q, k, w, sparse_indices, attn_softmax_l1_norm, **kw):
        seen["slig_indices"] = tuple(sparse_indices.shape)
        seen["slig_teacher"] = tuple(attn_softmax_l1_norm.shape)
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(w).float(),
            torch.zeros_like(attn_softmax_l1_norm),
        )

    monkeypatch.setattr(sa._SparseMLA, "apply", staticmethod(spy_apply))
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", fake_slig)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def fake_smlag(q, grad, *args, **kw):
        indices = kw["cmp_sparse_indices"]
        teacher = (indices >= 0).float() * 0.01 if indices is not None else None
        shared = kw["cmp_kv"]
        return (
            grad,
            torch.zeros_like(kw["ori_kv"]),
            torch.zeros_like(shared) if shared is not None else None,
            torch.zeros_like(kw["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", fake_smlag)
    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    seq_len = _TOKENS.shape[1]
    topk = model.layers["2"].attention.indexer.index_topk
    # Model-facing ``[B, L, 1, K]`` becomes the kernel's ``[T, N2, K]`` at both SMLA inputs.
    assert seen["smla_indices"] == seen["smla_carrier"] == (seq_len, 1, topk)
    # And SLIKG gets the teacher in exactly the selection's shape -- that equality is what
    # its tiling checks.
    assert seen["slig_indices"] == seen["slig_teacher"] == (seq_len, 1, topk)


def test_the_selection_is_sorted_with_padding_last(monkeypatch):
    """The model contract is one position order with the padding at the tail.

    The kernel's own order is unspecified, so the port re-sorts.  The direction is ours --
    neither the kernel schema nor the operator docs fix one, and the reference happens to
    sort ascending -- and is descending, because keying the sort on ``-index`` pushes
    ``-1`` behind every valid entry for free.

    The fake is made to emit a deliberately unsorted order, because a fake that
    pre-sorts cannot see whether the port sorts at all.

    Both the indices and the scores must move together: the scores are the carrier whose
    gradient is the teacher, and the teacher is per slot.
    """
    real_fake = _fake_lightning_indexer

    def unsorted_fake(*args, **kw):
        indices, values = real_fake(*args, **kw)
        return indices.flip(-1).contiguous(), values.flip(-1).contiguous()

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", unsorted_fake)
    model = _tiny_model(monkeypatch, fused=True)
    _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
    metadata = layer_metadata(model.layers["2"], kwargs)

    selector = model.layers["2"].attention.indexer.selector
    # requires_grad on the operands, as in a real layer: an autograd Function only gives
    # its outputs a grad_fn when one of its inputs needs a gradient, and the carrier's edge
    # is the whole point of the second output.
    q = torch.randn(1, 128, selector.num_index_heads, selector.index_head_dim, requires_grad=True)
    k = torch.randn(1, 64, selector.index_head_dim, requires_grad=True)
    w = torch.randn(1, 128, selector.num_index_heads, requires_grad=True)
    indices, scores, _ = selector.forward(q, k, w, metadata, candidates_BL1C=None)

    checked = 0
    for t in range(indices.shape[1]):
        row = indices[0, t]
        valid = row[row >= 0]
        if valid.numel() < 2:
            continue
        # Descending positions, and no valid entry after the first padding.
        assert torch.all(valid[1:] < valid[:-1]), (t, valid[:8].tolist())
        first_pad = int((row < 0).nonzero()[0]) if bool((row < 0).any()) else row.numel()
        assert torch.all(row[first_pad:] < 0), t
        checked += 1
    assert checked > 0, "the batch must contain rows with a real selection"
    # The carrier is a fabricated buffer of the selection's shape, produced by the
    # autograd Function so that the teacher's gradient has a path back to the indexer.
    # Its values are read by neither side, so nothing is asserted about them.
    assert scores.shape == indices.shape
    assert scores.requires_grad


def test_lightning_indexer_asc_stack_applies_without_claim_conflicts(monkeypatch):
    """The fused node override composes with the blanket norm/rope stacks.

    ``lightning_indexer.asc`` swaps only the parameterless ``Selector`` node, which
    owns no nested norm or rope, so the blanket ``common.rms_norm.asc`` and
    ``common.rope.asc_complex`` apply alongside it with disjoint claims: the
    indexer's ``k_norm`` and ``rope`` are covered like every other site.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.models.common.rope import ComplexRoPE

    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.rope import AscComplexRoPE

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    model = registry.model_registry("deepseek_v4_1_debugmodel").model

    apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.rms_norm.asc",
                "torchtitan_npu.override.common.rope.asc_complex",
                "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            ]
        ),
        model,
    )

    indexer = model.layers[2].attention.indexer
    assert isinstance(indexer.selector, AscSelector.Config)
    assert isinstance(indexer.rope, ComplexRoPE.Config)
    assert isinstance(model.layers[0].attention.q_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention.rope, AscComplexRoPE.Config)


def test_full_model_smla_trains_the_indexer_through_the_teacher_edge(monkeypatch):
    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def backward(q, grad, *args, **kw):
        indices = kw["cmp_sparse_indices"]
        teacher = (indices >= 0).float() * 0.01 if indices is not None else None
        shared = kw["cmp_kv"]
        return (
            grad,
            torch.zeros_like(kw["ori_kv"]),
            torch.zeros_like(shared) if shared is not None else None,
            torch.zeros_like(kw["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)
    # The indexer is trained only through the teacher edge, so a finite gradient on
    # every indexer parameter is what proves the teacher reached SLIKG -- on every
    # layer, the pool group included.
    indexer_params = [(n, p) for n, p in model.named_parameters() if ".indexer." in n]
    assert indexer_params, "the model must own indexer parameters"
    for name, parameter in indexer_params:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_both_fused_ports_build_the_kernel_stack(monkeypatch):
    """With both ports derived, the built model carries the kernels rather than the reference.

    The half-fused cases are rejected one stage earlier, on the override request -- see
    ``test_a_half_fused_teacher_pair_is_rejected_through_override_imports``.
    """
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")

    def config(*, fused_selector: bool, fused_core: bool):
        cfg = registry.model_registry("deepseek_v4_1_debugmodel").model
        for layer in cfg.layers:
            # The pair only exists where there is a compressed stream.
            if layer.attention.inner_attention.compress_ratio == 0:
                continue
            if fused_selector:
                layer.attention.indexer.selector = derive(layer.attention.indexer.selector, AscSelector.Config)
            if fused_core:
                layer.attention.inner_attention = derive(
                    layer.attention.inner_attention, AscV41SparseAttention.Config, **_TINY_GEOMETRY
                )
        return cfg

    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config(fused_selector=True, fused_core=True))
    assert isinstance(model.layers["2"].attention.inner_attention, AscV41SparseAttention)
    assert isinstance(model.layers["2"].attention.indexer.selector, AscSelector)


@pytest.mark.parametrize(
    "imports",
    [
        ["torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc"],
        ["torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc"],
    ],
    ids=["selector-only", "core-only"],
)
def test_a_half_fused_teacher_pair_is_rejected_through_override_imports(imports):
    """The rejection fires on the override request, which is the stage that can see it."""
    from torchtitan.config.override import OverrideConfig

    recipes = importlib.import_module("torchtitan_npu.models.deepseek_v4_1.config_registry")
    job = recipes.deepseek_v4_1_debugmodel_text()
    job.override = OverrideConfig(imports=imports)
    with pytest.raises(ValueError, match="must be enabled together"):
        job.model_spec.model.update_from_config(config=job)


def test_both_ports_requested_pass_the_pair_gate():
    """Both requested is the shipped configuration, so the same stage lets it through."""
    from torchtitan.config.override import OverrideConfig

    recipes = importlib.import_module("torchtitan_npu.models.deepseek_v4_1.config_registry")
    job = recipes.deepseek_v4_1_debugmodel_text()
    # The sequence length is part of a CP-capable config: torchtitan's rope budget is checked
    # inside update_from_config, alongside the gate this test is about.
    job.training = dataclasses.replace(job.training, seq_len=512)
    job.override = OverrideConfig(
        imports=[
            "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
        ]
    )
    job.model_spec.model.update_from_config(config=job)


def test_the_teacher_sources_are_the_index_source_layers():
    """The count the seqlen test leans on, pinned against the shipped table.

    It is the whole reason that test can name an expected teacher value instead of reading
    one back out of the run, so it is checked here rather than only inside a forward.
    """
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    sources = _teacher_sources(registry.V41_FULL_INDEX_SOURCE_LAYERS)
    assert sources == (2, 8, 14, 20, 24, 28, 32, 36)
    # The debug flavor the seqlen test builds shares this topology, so the frame it measures
    # has this many teacher-carrying layers.
    assert len(sources) == 8


def test_global_query_scaling_matches_explicit_distillation_gradient(monkeypatch):
    selector = _tiny_model(monkeypatch, fused=True).layers["2"].attention.indexer.selector
    generator = torch.Generator().manual_seed(42)
    q = torch.rand(1, 8, selector.num_index_heads, selector.index_head_dim, generator=generator, requires_grad=True)
    k = torch.rand(1, 4, selector.index_head_dim, generator=generator, requires_grad=True)
    w = torch.rand(1, 8, selector.num_index_heads, generator=generator, requires_grad=True)
    selector.num_global_queries = torch.tensor(32.0)
    indices, scores, _ = selector(q, k, w, _metadata([0, 8], ratio=2), candidates_BL1C=None)
    teacher = (indices >= 0).float() * 0.25
    # A later forward's denominator must not change this graph's backward.
    selector.num_global_queries = torch.tensor(64.0)
    scores.backward(teacher)

    qr, kr, wr = [x.detach().clone().requires_grad_() for x in (q, k, w)]
    selected = kr.squeeze(0)[indices.squeeze(0).clamp_min(0)].unsqueeze(0)
    logits = (torch.einsum("blhd,blkd->blhk", qr, selected).relu() * wr.unsqueeze(-1)).sum(dim=2)
    logits = logits.masked_fill(indices < 0, -1e9)
    loss = -(teacher * logits.log_softmax(dim=-1)).sum() / 32
    expected = torch.autograd.grad(loss, (qr, kr, wr))
    for actual, reference in zip((q.grad, k.grad, w.grad), expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_indexer_backward_requires_global_query_count(monkeypatch):
    selector = _tiny_model(monkeypatch, fused=True).layers["2"].attention.indexer.selector
    selector.num_global_queries = None
    q = torch.ones(1, 8, selector.num_index_heads, selector.index_head_dim, requires_grad=True)
    k = torch.ones(1, 4, selector.index_head_dim, requires_grad=True)
    w = torch.ones(1, 8, selector.num_index_heads, requires_grad=True)
    _, scores, _ = selector(q, k, w, _metadata([0, 8], ratio=2), candidates_BL1C=None)
    with pytest.raises(RuntimeError, match=r"Set Selector\.num_global_queries"):
        scores.backward(torch.ones_like(scores))
