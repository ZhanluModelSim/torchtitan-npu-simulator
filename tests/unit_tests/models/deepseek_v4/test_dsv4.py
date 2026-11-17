"""DSV4 host-logic CPU tests: the AscendC metadata layer, the packed layout,
the compressor/indexer/attention numerics, and the ported DSA semantics.

The AscendC override host wiring (``test_asc_metadata_*``) and the model-dir
contract (``test_layout_*``, ``test_compressor_*``, ``test_attention_*``,
``test_indexer_*``, ``test_ported_*``) all import against the real
torchtitan checkout with the plugin's patches applied; only the
``cann_ops_transformer`` surface is faked (the conftest's recorder).

Numerics verification criterion: a match is only accepted when it is
*bitwise*, or the residual sits at the fp32 rounding floor of the outputs
(at most a few ulps of the output scale) — a floating-point artifact, not
an arithmetic difference.  The compressor and the indexer are bitwise; the
reference attention core vs the eager golden stays within 1.8 ulps (the golden
is exact-fp32 by design, so its own f32/f64 delta is zero and the remaining
residual is the flex path's fp32 rounding).

Every comparison runs single-threaded (``_reproducible_numerics`` below): the
pool gradient's backward accumulates duplicate contributions in a
thread-scheduling dependent order, which makes the residual a sample of a
distribution rather than a number, and that distribution crosses this file's
4-ulp budget often enough to flake.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

DIM, HD, RD = 8, 16, 4
_GRAD_ULPS = 4
# Two documents longer than one compression block, with different tails (at
# ratio 128 neither is a whole number of blocks).
_GRADIENT_DOCS = (200, 133)
_EPS32 = 1.1920929e-7  # fp32 machine epsilon


@pytest.fixture(autouse=True)
def _reproducible_numerics():
    """Pin the intra-op thread count so the rounding-floor residuals are reproducible.

    The core and the golden both differentiate the *same* compressed-KV pool, and
    both read it through advanced indexing (the golden's ``kv[batch, index, :]``,
    attn-gym's ``kv[b, h, indices]``).  The pool's rows are read by many query
    rows, so that gather's backward is an accumulate scatter (ATen lowers it to
    ``index_put_`` with ``accumulate=True``), and its CPU kernel sums the
    duplicate contributions in a thread-scheduling dependent order: measured in
    isolation, ``index_put_(accumulate=True)`` and both advanced-index backwards
    give different last bits from run to run at 16 threads with a non-uniform
    upstream gradient.

    So with more than one intra-op thread each consumer drifts against *itself*
    by up to ~3.5 ulps of ``cmp_k``'s scale, and the core-vs-golden residual
    wanders between ~2.3 and ~5.2 ulps against this file's 4-ulp budget (an
    in-process 100-rep loop of the comparison below: median 3.46, max 5.19, 25
    runs over budget — only ``cmp_k`` moves; ``out``, ``q``, ``swa_k`` and
    ``sink`` are bitwise constant).  Single-threaded every residual is bitwise
    reproducible, with measured worsts of 1.8 (out), 1.1 (q), 1.4 (swa_k), 1.2
    (cmp_k) and 0.0 (sink) ulps.  ``torch.use_deterministic_algorithms(True)``
    does not help (torch does not call these kernels nondeterministic), and
    neither does pinning ``MKL_DYNAMIC``/``OMP_NUM_THREADS`` to 16.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class IdentityRoPE(nn.Module):
    """Identity rope — the compressor-side counterpart of ``manual_compress``.

    The pooling reference applies NO rope, so the compressor's
    ``self.rope(...)`` step must be a no-op for the comparison to be exact.
    The real rope numerics are covered by the rope patch's own on-device
    verification.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        dim: int = 1
        original_max_position_embeddings: int = 1
        rope_theta: float = 1.0

    def __init__(self, config=None):
        super().__init__()

    def forward(self, x, key=None, positions=None, *, inverse=False):
        if key is None:
            return x
        return x, x


class BuildCfg:
    def __init__(self, fn):
        self._fn = fn

    def build(self):
        return self._fn()


class RMS(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
                x.dtype
            )
            * self.weight
        )


def make_linear(n, m):
    lin = nn.Linear(n, m, bias=False)
    nn.init.normal_(lin.weight, std=0.02)
    return lin


# B=1, S=64; docs A(10) B(17) C(20) D(17)
_CU = torch.tensor([0, 10, 27, 47, 64], dtype=torch.int32)


def _build_metadata(dsv4):
    """The 64-token four-document layout (ratios 1/4/128) as the model builds it."""
    v = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=17, max_k=17)
    return dsv4.metadata.build_compressed_varlen_metadata(v, (1, 4, 128))


def _pack_container(dsv4, pooled, plan):
    """The plan-driven container pack (the dispatcher's ``select``)."""
    disp = dsv4.token_dispatcher.CPTokenDispatcher(
        dsv4.token_dispatcher.CPTokenDispatcher.Config()
    )
    return disp.select(pooled, plan)


def build_compressor(dsv4, ratio):
    coff = 2 if ratio == 4 else 1
    comp = dsv4.compressor.Compressor(
        dsv4.compressor.Compressor.Config(
            rope=BuildCfg(lambda: IdentityRoPE()),
            head_dim=HD,
            rope_head_dim=RD,
            compress_ratio=ratio,
            wkv=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            wgate=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            norm=BuildCfg(lambda: RMS(HD)),
        )
    )
    nn.init.trunc_normal_(comp.ape, std=0.02)
    return comp


def manual_compress(x, md, comp, ratio):
    """Independent per-document pooling reference (no rope)."""
    plan = md.plans[ratio]
    flat_x = x.flatten(0, 1).float()
    out = torch.zeros(md.batch_size * (md.seq_len // ratio), HD)
    for doc_id, (qs, qe) in enumerate(
        zip(
            md.varlen.cu_seq_q[:-1].tolist(),
            md.varlen.cu_seq_q[1:].tolist(),
            strict=True,
        )
    ):
        length = qe - qs
        c_len = length // ratio
        if c_len == 0:
            continue
        tokens = flat_x[qs : qs + c_len * ratio].view(c_len, ratio, DIM)
        kv = F.linear(tokens, comp.wkv.weight.float())
        score = F.linear(tokens, comp.wgate.weight.float()) + comp.ape
        if ratio == 4:
            c_start = plan.cu_seqlens_cmp_k[doc_id].item()
            c_end = plan.cu_seqlens_cmp_k[doc_id + 1].item()
            valid = (torch.arange(c_end - c_start) != 0).view(-1, 1, 1)
            prev_idx = (torch.arange(c_end - c_start) - 1).clamp_min(0)
            kv_prev = torch.where(
                valid, kv[prev_idx, :, :HD], torch.zeros_like(kv[:, :, :HD])
            )
            score_prev = torch.where(
                valid,
                score[prev_idx, :, :HD],
                torch.full_like(score[:, :, :HD], float("-inf")),
            )
            kv = torch.cat([kv_prev, kv[:, :, HD:]], dim=1)
            score = torch.cat([score_prev, score[:, :, HD:]], dim=1)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = comp.norm(kv.to(x.dtype))
        start = plan.cu_seqlens_cmp_k[doc_id].item()
        out[start : start + c_len] = kv
    return out.view(md.batch_size, md.seq_len // ratio, HD)


def _pool_blocks_where(
    comp, block_tokens, block_positions, overlap_valid, dtype, ratio
):
    """The pre-refactor pooling math (two ``where`` masks) — the bitwise
    reference for the scatter-masking implementation (exact-zero softmax
    weights make the two forms identical)."""
    kv = comp.wkv(block_tokens)
    score = comp.wgate(block_tokens) + comp.ape
    if ratio == 4:
        head_dim = comp.head_dim
        n = kv.shape[0]
        overlap_kv = kv.new_zeros(n, 2 * ratio, head_dim)
        overlap_score = score.new_full((n, 2 * ratio, head_dim), float("-inf"))
        overlap_kv[:, ratio:] = kv[:, :, head_dim:]
        overlap_score[:, ratio:] = score[:, :, head_dim:]
        prev = (torch.arange(n) - 1).clamp_min(0)
        valid = overlap_valid.view(-1, 1, 1)
        overlap_kv[:, :ratio] = torch.where(
            valid, kv[prev, :, :head_dim], overlap_kv[:, :ratio]
        )
        overlap_score[:, :ratio] = torch.where(
            valid, score[prev, :, :head_dim], overlap_score[:, :ratio]
        )
        kv, score = overlap_kv, overlap_score
    kv = (kv * score.softmax(dim=1)).sum(dim=1)
    kv = comp.norm(kv.to(dtype))
    nope_dim = comp.head_dim - comp.rope_head_dim
    kv_nope, kv_rope = torch.split(kv, [nope_dim, comp.rope_head_dim], dim=-1)
    kv_rope = (
        comp.rope(
            kv_rope.unsqueeze(0).unsqueeze(2), positions=block_positions.unsqueeze(0)
        )
        .squeeze(0)
        .squeeze(1)
    )
    return torch.cat([kv_nope, kv_rope], dim=-1)


def _build_varlen(doc_lens):
    cu = torch.tensor(
        [0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    return VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=17, max_k=17)


def _run_extension(dsv4, doc_lens, ratios, **shape):
    """The model-built metadata + the AscendC metadata extension."""
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
        AscMetadataExtension,
        AscLightningIndexerMetadata,
    )

    v = _build_varlen(doc_lens)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, ratios)
    md = AscLightningIndexerMetadata(
        AscLightningIndexerMetadata.Config(
            window_size=shape.get("window_size", 128),
            index_n_heads=shape.get("index_n_heads", 8),
            index_head_dim=shape.get("index_head_dim", 128),
            index_topk=shape.get("index_topk", 512),
        )
    )(md)
    ext = AscMetadataExtension(
        AscMetadataExtension.Config(
            window_size=shape.get("window_size", 128),
            num_heads=shape.get("num_heads", 16),
            head_dim=shape.get("head_dim", 512),
            index_n_heads=shape.get("index_n_heads", 8),
            index_head_dim=shape.get("index_head_dim", 128),
            index_topk=shape.get("index_topk", 512),
        )
    )
    return ext(md)


@pytest.fixture
def inputs(dsv4):
    torch.manual_seed(0)
    md = _build_metadata(dsv4)
    x = torch.randn(1, 64, DIM)
    pos = torch.arange(64).unsqueeze(0)
    return dsv4, md, x, pos


# ---------------------------------------------------------------------------
# The AscendC metadata layer (handler + kernel fills)
# ---------------------------------------------------------------------------


def test_asc_metadata_zero_block_batch_rejected_before_any_asc_call(dsv4):
    dsv4.cann_ops.calls.clear()
    with pytest.raises(ValueError, match="no complete compression block"):
        _run_extension(dsv4, (60, 60, 60, 76), (1, 4, 128))  # ratio-128: no blocks
    assert not dsv4.cann_ops.calls, (
        "AscendC metadata must not be computed for an invalid batch"
    )


def test_asc_metadata_smla_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    assert md.batch_size == 1 and md.seq_len == 1024
    smla = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_metadata"]
    assert len(smla) == 3  # ratios 1, 4, 128
    by_ratio = {c[2]["cmp_ratio"]: c for c in smla}
    assert by_ratio[4][1] == (16, 1, 512)
    assert by_ratio[4][2]["cmp_topk"] == 512 and by_ratio[4][2]["has_cmp_kv"] is True
    assert by_ratio[128][2]["cmp_topk"] == 0 and by_ratio[128][2]["has_cmp_kv"] is True
    assert by_ratio[1][2]["has_cmp_kv"] is False
    assert torch.equal(
        by_ratio[4][2]["cu_seqlens_cmp_kv"], md.plans[4].cu_seqlens_cmp_k
    )
    assert torch.equal(by_ratio[4][2]["cmp_residual_kv"], md.plans[4].block_remainder)


def test_asc_metadata_grad_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    grad = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_grad_metadata"]
    assert len(grad) == 3
    assert all(
        "ori_topk_length" not in c[2] and "cmp_topk_length" not in c[2] for c in grad
    )


def test_asc_metadata_li_and_slig_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    li = [c for c in dsv4.cann_ops.calls if c[0] == "lightning_indexer_metadata"]
    slig = [
        c
        for c in dsv4.cann_ops.calls
        if c[0] == "sparse_lightning_indexer_kl_loss_grad_metadata"
    ]
    assert len(li) == 1 and len(slig) == 1
    assert li[0][1][:3] == (8, 1, 128), li[0][1]
    assert li[0][1][3] == 512
    assert slig[0][1] == (8, 1, 128), slig[0][1]
    assert slig[0][2]["topk"] == 512  # keyword-only in the binding schema
    assert md.asc_plans[4].li_metadata is not None
    assert md.asc_plans[4].slig_metadata is not None
    assert (
        md.asc_plans[128].li_metadata is None
        and md.asc_plans[128].slig_metadata is None
    )


def test_asc_metadata_slim_contract(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    assert not hasattr(md, "reference")
    assert md.plans[4].cu_seqlens_cmp_k.shape[0] == 5
    assert md.plans[4].block_remainder.shape[0] == 4
    assert md.plans[4].gather_indices.numel() == 1024
    assert md.asc_plans[4].smla_metadata is not None
    assert md.asc_plans[128].smla_metadata is not None
    assert md.asc_plans[1].smla_metadata is not None


def test_asc_metadata_ratio128_only_model(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (512, 512), (128,))
    assert "lightning_indexer" not in [c[0] for c in dsv4.cann_ops.calls]
    assert md.asc_plans[128].li_metadata is None


# ---------------------------------------------------------------------------
# The packed layout (kernel contract + reference tier)
# ---------------------------------------------------------------------------


def test_layout_container_shape(dsv4):
    md = _build_metadata(dsv4)
    assert (md.batch_size, md.seq_len) == (1, 64)


def test_layout_cu_seqs_and_remainders(dsv4):
    p4 = _build_metadata(dsv4).plans[4]
    assert torch.equal(
        p4.cu_seqlens_cmp_k, torch.tensor([0, 2, 6, 11, 15], dtype=torch.int32)
    )
    assert torch.equal(
        p4.block_remainder, torch.tensor([2, 1, 0, 1], dtype=torch.int32)
    )


def test_layout_host_cached_lengths(dsv4):
    """Metadata builders cache scalar lengths before entering compiled layers.

    The cached values must agree with the device-side cumulative-sequence
    tensors: ``seq_len`` is the total token count and each compressed plan
    records its document-wise complete-block count.  This guards the host
    cache introduced to remove repeated ``.item()`` synchronizations.
    """
    varlen = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=17, max_k=17)
    md = dsv4.metadata.build_compressed_varlen_metadata(varlen, (1, 4, 128))
    assert md.seq_len_host == 64
    assert md.seq_len == 64
    assert md.plans[1].n_cmp_blocks_host is None
    assert md.plans[4].n_cmp_blocks_host == 15
    assert md.plans[128].n_cmp_blocks_host == 0


def test_layout_gather_indices(dsv4):
    p4 = _build_metadata(dsv4).plans[4]
    assert torch.equal(
        p4.gather_indices,
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
            ],
            dtype=torch.int64,
        ),
    )


def test_compressor_numerics_ratio4(inputs):
    """C4A pooling vs the independent manual reference.

    The residual must sit at the fp32 rounding floor: the two paths project
    with different matmul batching (all blocks at once vs per document), so
    the CPU kernel's shape-dependent reduction order yields 1-ulp residuals
    (the documented batch-size artifact; the same-size calls are
    deterministic).  ``_assert_rounding_floor`` accepts that — bitwise would
    only hold on shape-invariant kernels; the suite-wide 4-ulp budget keeps
    a 2.2x margin over the measured worst (at most 1.8 ulps, single-threaded)
    while still flagging any multi-ulp arithmetic regression.
    """
    dsv4, md, x, _pos = inputs
    comp4 = build_compressor(dsv4, 4)
    pooled = comp4(x, md)
    cmp4 = _pack_container(dsv4, pooled, md.plans[4])
    assert cmp4.shape == (1, 16, 16)
    ref = manual_compress(x, md, comp4, 4)
    _assert_rounding_floor((cmp4 - ref).abs(), ref.abs().max().item())


def test_compressor_numerics_ratio128(inputs):
    """C128A with zero complete blocks — shape/emptiness only (no numerics)."""
    dsv4, md, x, _pos = inputs
    comp128 = build_compressor(dsv4, 128)
    cmp128 = comp128(x, md)
    assert cmp128.shape == (0, 16)
    assert cmp128.numel() == 0


@pytest.mark.parametrize(
    "case,ratio",
    [
        pytest.param(c, r, id=f"{c[0]}-r{r}")
        for c in [
            ("tiny", (5, 2, 9, 1, 7, 4, 3, 8, 2, 6, 5, 3, 4, 9, 4)),
            ("single", (7, 1)),
            ("single-full", (8,)),
            ("r-multiples", (8, 12, 16, 4)),
            ("r-boundary", (9, 13, 17, 1)),
            ("zero-block", (3, 5, 9, 7)),
            ("plain", (10, 17, 20, 17)),
        ]
        for r in (4, 128)
    ],
)
def test_compressor_overlap_scatter_bitwise(dsv4, case, ratio):
    """The scatter-masked overlap (``first_indices``) is bitwise-identical
    to the old two-``where`` form: the masked borrowed rows have exactly
    zero softmax weight, so their kv content never contributes."""
    name, docs = case
    v = _build_varlen(docs)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, (ratio,))
    plan = md.plans[ratio]
    comp = build_compressor(dsv4, ratio)
    torch.manual_seed(0)
    x = torch.randn(1, sum(docs), DIM)
    pooled = comp(x, md)
    if pooled.shape[0] == 0:
        return
    bt = x.flatten(0, 1)[plan.gather_indices].reshape(pooled.shape[0], ratio, -1)
    bids = torch.arange(pooled.shape[0])
    seq_ids = torch.searchsorted(plan.cu_seqlens_cmp_k[1:], bids, right=True)
    overlap_valid = (bids - plan.cu_seqlens_cmp_k[seq_ids]) != 0
    old = _pool_blocks_where(
        comp, bt, plan.block_positions, overlap_valid, torch.float32, ratio
    )
    assert torch.equal(pooled, old), name


# ---------------------------------------------------------------------------
# Attention numerics (the gather core vs the eager golden)
# ---------------------------------------------------------------------------


def _assert_rounding_floor(diff, scale, *, ulps: int = 4):
    """The residual must sit at the fp32 rounding floor of the outputs
    (<= ``ulps`` ulps of the output scale) — floating-point error only, not
    arithmetic (an arithmetic difference would exceed it by orders of
    magnitude).  ``ulps=4`` is the suite-wide budget; the measured worsts are
    1.8 (attention outputs) and 1.2 (attention gradients) ulps, reproducible
    because the comparisons run single-threaded."""
    assert scale > 0
    assert diff.max().item() <= ulps * _EPS32 * scale, (diff.max().item(), scale)


def _build_attn_pair(dsv4, ratio, topk, *, indexer_loss_coeff: float = 1.0):
    core = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=topk,
        )
    )
    golden = dsv4.golden.GoldenCompressedSparseInnerAttention(
        dsv4.golden.GoldenCompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=topk,
            indexer_loss_coeff=indexer_loss_coeff,
        )
    )
    return core, golden


def _assert_attn_matches_rounding_floor(dsv4, md, x, pos, ratio, topk):
    torch.manual_seed(0)
    q = torch.randn(1, 64, 3, HD)
    swa_k = torch.randn(1, 64, HD)
    sink = torch.randn(3)
    cmp = None
    aq = ak = aw = None
    if ratio == 4:
        aq = torch.randn(1, 64, 4, 8)
        ak = torch.randn(1, 16, 8)
        aw = torch.randn(1, 64, 4)
        cmp = _pack_container(dsv4, build_compressor(dsv4, 4)(x, md), md.plans[4])
    elif ratio > 1:
        cmp = _pack_container(dsv4, build_compressor(dsv4, 128)(x, md), md.plans[128])
    core, golden = _build_attn_pair(dsv4, ratio, topk)
    with torch.no_grad():
        # Both sides read the same packed selection, in its own numbering.
        sparse = None
        if ratio == 4:
            sparse = dsv4.compressor.select_compressed_slots(
                aq, ak, aw, md.varlen.cu_seq_q, md.plans[4].cu_seqlens_cmp_k, topk
            )
        out = core(q, swa_k, cmp, attn_sink=sink, sparse_indices=sparse, attention_masks=md)
        gout = golden(q, swa_k, cmp, attn_sink=sink, sparse_indices=sparse, attention_masks=md)
    assert out.shape == (1, 64, 3, HD) and torch.isfinite(out).all()
    _assert_rounding_floor((out - gout).abs(), gout.abs().max().item())


@pytest.mark.parametrize("ratio,topk", [(1, 0), (4, 8), (128, 0)])
def test_attention_numerics_ratio(inputs, ratio, topk):
    """Reference core vs the eager golden (SWA / CSA / HCA) — the residual
    must sit at the fp32 rounding floor (bitwise-adjacent), proving the
    difference is floating-point only, not arithmetic."""
    dsv4, md, x, pos = inputs
    _assert_attn_matches_rounding_floor(dsv4, md, x, pos, ratio, topk)


def test_attention_numerics_ratio4_zero_blocks(dsv4):
    """The ratio-4 reference path with zero complete blocks — shape/finite
    (no numerics to compare; the golden is not exercised)."""
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=5, max_k=5)
    md_s = dsv4.metadata.build_compressed_varlen_metadata(v, (4,))
    comp_s = build_compressor(dsv4, 4)
    cmp_s = _pack_container(dsv4, comp_s(torch.randn(1, 8, DIM), md_s), md_s.plans[4])
    assert cmp_s.shape == (1, 2, 16)
    core_s = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=4,
            softmax_scale=0.25,
            index_topk=4,
        )
    )
    with torch.no_grad():
        sparse_s = dsv4.compressor.select_compressed_slots(
            torch.randn(1, 8, 4, 8),
            torch.randn(1, 2, 8),
            torch.randn(1, 8, 4),
            md_s.varlen.cu_seq_q,
            md_s.plans[4].cu_seqlens_cmp_k,
            4,
        )
        out_s = core_s(
            torch.randn(1, 8, 3, HD),
            torch.randn(1, 8, HD),
            cmp_s,
            attn_sink=torch.randn(3),
            sparse_indices=sparse_s,
            attention_masks=md_s,
        )
    # The first document has no complete group, so all of its picks are padding; the second
    # has one, and the rest of its width is padding.
    assert (sparse_s[0, :3] == -1).all()
    assert (sparse_s[0, 3:] >= 0).any()
    assert out_s.shape == (1, 8, 3, HD) and torch.isfinite(out_s).all()


def _gradient_case(dsv4, ratio=4):
    """The gradient comparison's inputs: one pool, one selection, one draw per consumer.

    One input set is built for both consumers, so any difference in the outputs
    or the gradients is arithmetic, not a different draw.  Raises if the pool is
    empty, i.e. if the case stops exercising the compressed path.
    """
    torch.manual_seed(0)
    docs = _GRADIENT_DOCS
    md = dsv4.metadata.build_compressed_varlen_metadata(_build_varlen(docs), (4, 128))
    seqlen = sum(docs)
    sink = torch.randn(3)
    pool = _pack_container(
        dsv4, build_compressor(dsv4, ratio)(torch.randn(1, seqlen, DIM), md), md.plans[ratio]
    )
    assert pool.shape[1] > 0, "the fixture must exercise a non-empty compressed pool"

    sparse = None
    if ratio == 4:
        n_cmp = int(md.plans[4].cu_seqlens_cmp_k[-1])
        sparse = dsv4.compressor.select_compressed_slots(
            torch.randn(1, seqlen, 4, 8),
            torch.randn(1, n_cmp, 8),
            torch.randn(1, seqlen, 4),
            md.varlen.cu_seq_q,
            md.plans[4].cu_seqlens_cmp_k,
            8,
        )

    base = (
        torch.randn(1, seqlen, 3, HD),
        torch.randn(1, seqlen, HD),
        pool.detach().clone(),
        sink,
    )
    core, golden = _build_attn_pair(dsv4, ratio, 8 if ratio == 4 else 0, indexer_loss_coeff=0.0)
    return md, sparse, base, core, golden


def _consumer_grads(consumer, sparse, md, base):
    """One differentiated pass; each consumer gets its own differentiable copy."""
    q, swa_k, cmp_k, attn_sink = (t.detach().clone().requires_grad_() for t in base)
    out = consumer(q, swa_k, cmp_k, attn_sink=attn_sink, sparse_indices=sparse, attention_masks=md)
    upstream = torch.linspace(-1, 1, out.numel()).reshape_as(out)
    (out * upstream).sum().backward()
    return out, (q.grad, swa_k.grad, cmp_k.grad, attn_sink.grad)


@pytest.mark.parametrize("ratio", [4, 128])
def test_attention_gradients_match_the_golden(dsv4, ratio):
    """Two documents longer than 128 tokens with different tails, differentiated end to end.

    The compressed pool is non-empty at both ratios, the golden's own LI auxiliary loss is off
    so both passes carry the same objective, and the q / swa_k / cmp_k gradients are compared in
    addition to the outputs (the case above only compared outputs, under ``no_grad``, on a
    64-token row whose ratio-128 pool is empty)."""
    md, sparse, base, core, golden = _gradient_case(dsv4, ratio)
    seqlen = base[0].shape[1]
    outs, grads = [], []
    for consumer in (core, golden):
        out, grad = _consumer_grads(consumer, sparse, md, base)
        outs.append(out)
        grads.append(grad)

    core_out, golden_out = outs
    assert core_out.shape == golden_out.shape == (1, seqlen, 3, HD)
    _assert_rounding_floor((core_out - golden_out).abs(), golden_out.abs().max().item())
    for name, core_grad, golden_grad in zip(("q", "swa_k", "cmp_k", "sink"), *grads):
        assert core_grad is not None and golden_grad is not None, name
        _assert_rounding_floor(
            (core_grad - golden_grad).abs(), golden_grad.abs().max().item(), ulps=_GRAD_ULPS
        )


def test_attention_pool_gradient_is_reproducible(dsv4):
    """The same consumer, run twice on identical inputs, must give the same pool gradient.

    This is what the single-threaded fixture buys, and the reason it exists: the
    pool gradient is the backward of the gather that reads the pool, an
    accumulate scatter whose contributions are summed in a thread-scheduling
    dependent order, so with more than one intra-op thread even one consumer
    disagrees with itself by up to ~3.5 ulps of ``cmp_k``'s scale — and the
    core-vs-golden residual the test above asserts on then wanders between ~2.3
    and ~5.2 ulps against its 4-ulp budget.  The thread-count assertion keeps
    this guard deterministic; without it the two runs only *usually* differ.
    """
    assert torch.get_num_threads() == 1, "the module fixture must pin the intra-op threads"
    md, sparse, base, core, _golden = _gradient_case(dsv4)
    _out, (_q_grad, _swa_k_grad, first, _sink_grad) = _consumer_grads(core, sparse, md, base)
    _out, (_q_grad, _swa_k_grad, second, _sink_grad) = _consumer_grads(core, sparse, md, base)

    assert torch.equal(first, second), (
        "the pool gradient must be reproducible; a thread-dependent accumulate "
        "scatter in its backward makes the rounding-floor comparison above flaky"
    )


def test_reference_consumers_reject_context_parallel_frames(dsv4):
    """The default core and the golden refuse context-parallel frames.

    Under CP the query offsets describe this rank's chunks while the compressed pool is the
    gathered row, so a reference consumer would read boundaries that are not there.  The
    reference tier used to refuse the shape on the consumers' behalf; now that the default
    extension is the identity, the consumers refuse it themselves, and the fused kernels keep
    taking the frames through their own metadata."""
    from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan

    cu = torch.tensor([0, 8], dtype=torch.int32)
    # the shard path's builder wants the global contract, i.e. both offsets as one object
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=8, max_k=8)
    cp_meta, plans, window = build_cp_plan(
        v, None, rank=1, cp_size=2, shard_len=4, window_size=4, ratios=[4]
    )
    assert not torch.equal(cp_meta.cu_seq_q, cp_meta.cu_seq_k)
    md = dsv4.metadata.CompressedVarlenMetadata(varlen=cp_meta, plans=plans, window=window)
    seqlen = int(cp_meta.cu_seq_q[-1])
    core, golden = _build_attn_pair(dsv4, 4, 4)

    for consumer in (core, golden):
        with pytest.raises(ValueError, match="context parallel"):
            consumer(
                torch.randn(1, seqlen, 3, HD),
                torch.randn(1, seqlen, HD),
                None,
                attn_sink=torch.randn(3),
                attention_masks=md,
            )
