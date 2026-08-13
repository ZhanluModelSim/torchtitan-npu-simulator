"""DSV4 host-logic CPU tests: the CANN metadata layer, the packed layout,
the compressor/indexer/attention numerics, and the ported DSA semantics.

The CANN override host wiring (``test_cann_metadata_*``) and the model-dir
contract (``test_layout_*``, ``test_compressor_*``, ``test_attention_*``,
``test_indexer_*``, ``test_ported_*``) all import against the real
torchtitan checkout with the plugin's patches applied; only the
``cann_ops_transformer`` surface is faked (the conftest's recorder).

Numerics verification criterion: a match is only accepted when it is
*bitwise*, or the residual sits at the fp32 rounding floor of the outputs
(at most a few ulps of the output scale) — a floating-point artifact, not
an arithmetic difference.  The compressor and the indexer are bitwise; the
reference attention core vs the eager golden is ~1 ulp (the golden is
exact-fp32 by design, so its own f32/f64 delta is zero and the remaining
residual is the flex path's fp32 rounding).
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

DIM, HD, RD = 8, 16, 4
_EPS32 = 1.1920929e-7  # fp32 machine epsilon


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


def _build_layout(dsv4, window_size=8, block_size=(8, 8)):
    """The 64-token four-document layout (ratios 1/4/128) with the
    reference tier applied (the default ``metadata_extension``)."""
    v = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=17, max_k=17)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, (1, 4, 128))
    return dsv4.reference.ReferenceMetadataExtension(
        dsv4.reference.ReferenceMetadataExtension.Config(
            window_size=window_size, block_size=block_size
        )
    )(md)


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
    """The model-built metadata + the CANN metadata extension."""
    from torchtitan_npu.override.deepseek_v4.sparse_attn.cann import (
        CANNMetadataExtension,
    )

    v = _build_varlen(doc_lens)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, ratios)
    ext = CANNMetadataExtension(
        CANNMetadataExtension.Config(
            window_size=shape.get("window_size", 128),
            num_heads=shape.get("num_heads", 16),
            head_dim=shape.get("head_dim", 512),
            index_n_heads=shape.get("index_n_heads", 8),
            index_head_dim=shape.get("index_head_dim", 128),
            index_topk=shape.get("index_topk", 512),
        )
    )
    return ext(md)


def window_idxs(window_size, bsz, seqlen, device):
    window = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (base - window + 1).clamp(0) + torch.arange(window, device=device)
    idxs = torch.where(idxs > base, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def compress_idxs(ratio, bsz, seqlen, device, offset):
    compress_len = seqlen // ratio
    if compress_len == 0:
        return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)
    idxs = torch.arange(compress_len, device=device).repeat(seqlen, 1)
    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
    causal_limit = causal_limit // ratio
    idxs = torch.where(idxs >= causal_limit, -1, idxs + offset)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def old_indexer_topk(bsz, seqlen, ratio, topk, device, seed):
    g = torch.Generator(device).manual_seed(seed)
    scores = torch.randn(bsz, seqlen, seqlen // ratio, generator=g, device=device)
    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    mask = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
    mask = mask >= causal_limit
    scores = scores + torch.where(mask, torch.finfo(torch.float32).min, 0)
    _, topk_idxs = scores.topk(min(topk, seqlen // ratio), dim=-1)
    topk_idxs = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)
    return topk_idxs


def old_attended(topk_idxs, sink_idx):
    out = []
    for b in range(topk_idxs.size(0)):
        for q in range(topk_idxs.size(1)):
            s = {int(i) for i in topk_idxs[b, q] if i >= 0}
            s.add(sink_idx)
            out.append(s)
    return out


def new_attended(block_mask, seqlen, n_cmp):
    kv_len = seqlen + n_cmp + 1
    B = block_mask.kv_num_blocks.size(0)
    out = []
    for b in range(B):
        q_idx = torch.arange(seqlen).unsqueeze(0).unsqueeze(-1)
        kv_idx = torch.arange(kv_len).unsqueeze(0).unsqueeze(-2)
        m = block_mask.mask_mod(torch.tensor(b), torch.tensor(0), q_idx, kv_idx)
        m = m[0]
        for q in range(seqlen):
            out.append(set(m[q].nonzero().flatten().tolist()))
    return out


@pytest.fixture(scope="module")
def inputs(dsv4):
    torch.manual_seed(0)
    md = _build_layout(dsv4)
    x = torch.randn(1, 64, DIM)
    pos = torch.arange(64).unsqueeze(0)
    return dsv4, md, x, pos


# ---------------------------------------------------------------------------
# The CANN metadata layer (handler + kernel fills)
# ---------------------------------------------------------------------------


def test_cann_metadata_zero_block_batch_rejected_before_any_cann_call(dsv4):
    dsv4.cann_ops.calls.clear()
    with pytest.raises(ValueError, match="no complete compression block"):
        _run_extension(dsv4, (60, 60, 60, 76), (1, 4, 128))  # ratio-128: no blocks
    assert not dsv4.cann_ops.calls, (
        "CANN metadata must not be computed for an invalid batch"
    )


def test_cann_metadata_smla_fills(dsv4):
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


def test_cann_metadata_grad_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    grad = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_grad_metadata"]
    assert len(grad) == 3
    assert all(
        "ori_topk_length" not in c[2] and "cmp_topk_length" not in c[2] for c in grad
    )


def test_cann_metadata_li_and_slig_fills(dsv4):
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
    assert md.cann_plans[4].li_metadata is not None
    assert md.cann_plans[4].slig_metadata is not None
    assert (
        md.cann_plans[128].li_metadata is None
        and md.cann_plans[128].slig_metadata is None
    )


def test_cann_metadata_slim_contract(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    assert not hasattr(md, "reference")
    assert md.plans[4].cu_seqlens_cmp_k.shape[0] == 5
    assert md.plans[4].block_remainder.shape[0] == 4
    assert md.plans[4].gather_indices.numel() == 1024
    assert md.cann_plans[4].smla_metadata is not None
    assert md.cann_plans[128].smla_metadata is not None
    assert md.cann_plans[1].smla_metadata is not None


def test_cann_metadata_ratio128_only_model(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (512, 512), (128,))
    assert "lightning_indexer" not in [c[0] for c in dsv4.cann_ops.calls]
    assert md.cann_plans[128].li_metadata is None


# ---------------------------------------------------------------------------
# The packed layout (kernel contract + reference tier)
# ---------------------------------------------------------------------------


def test_layout_container_shape(dsv4):
    md = _build_layout(dsv4, window_size=16)
    assert (md.batch_size, md.seq_len) == (1, 64)


def test_layout_cu_seqs_and_remainders(dsv4):
    p4 = _build_layout(dsv4, window_size=16).plans[4]
    assert torch.equal(
        p4.cu_seqlens_cmp_k, torch.tensor([0, 2, 6, 11, 15], dtype=torch.int32)
    )
    assert torch.equal(
        p4.block_remainder, torch.tensor([2, 1, 0, 1], dtype=torch.int32)
    )


def test_layout_gather_indices(dsv4):
    p4 = _build_layout(dsv4, window_size=16).plans[4]
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


def test_layout_container_slots(dsv4):
    r4 = _build_layout(dsv4, window_size=16).reference.ratios[4]
    # container width 16, doc blocks at slots 0-1, 2-5, 6-10, 11-14; slot 15 unused
    assert r4.doc_of_block[0, 15] == -1 and r4.block_local[0, 15] == -1


def test_layout_dense_mask(dsv4):
    dm = _build_layout(dsv4, window_size=16).reference.ratios[4].dense_mask[0, 0]
    assert bool(dm[8, 0]) and bool(dm[8, 1]) and not bool(dm[8, 2])
    assert bool(dm[3, 0]) and not bool(dm[3, 1])
    assert not dm[12, 2:].any()
    assert bool(dm[16, 2]) and not dm[16, 3]
    assert not dm[:10, 2:].any() and not dm[10:27, :2].any()
    assert bool(dm[30, 6]) and not dm[30, 11]


def test_layout_doc_of_token_and_pos(dsv4):
    md = _build_layout(dsv4, window_size=16)
    assert torch.equal(
        md.reference.doc_of_token[0, :10], torch.zeros(10, dtype=torch.int32)
    )
    assert torch.equal(
        md.reference.doc_of_token[0, 10:27], torch.ones(17, dtype=torch.int32)
    )
    assert torch.equal(
        md.reference.pos_in_doc[0],
        torch.cat(
            [torch.arange(10), torch.arange(17), torch.arange(20), torch.arange(17)]
        ).to(torch.int32),
    )


def test_layout_static_blocks(dsv4):
    md = _build_layout(dsv4, window_size=16)
    r4 = md.reference.ratios[4]
    sb4 = r4.static_blocks[0, 0]  # [nQ, nKV], kv_len = 64+16+1 = 81 -> 11 blocks
    assert sb4.shape == (8, 11)
    # q block 0 (tokens 0..7): window kv blocks {0}; sink at 80 -> block 10
    assert bool(sb4[0, 0]) and not sb4[0, 1:8].any() and bool(sb4[0, 10])
    # q block 1 (8..15): window blocks {(8-16+1)//8=0 .. 15//8=1} = {0,1}
    assert bool(sb4[1, 0]) and bool(sb4[1, 1]) and not sb4[1, 2:8].any()
    # HCA full cmp range listed: blocks 8..9 (seqlen//8=8 .. (64+15)//8=9)
    assert bool(sb4[0, 8]) and bool(sb4[0, 9])

    sb1 = md.reference.ratios[1].static_blocks[0, 0]
    assert sb1.shape == (8, 9)  # kv_len = 65 -> 9 blocks; no cmp region
    assert bool(sb1[0, 0]) and bool(sb1[0, 8]) and not sb1[0, 1:8].any()


def test_layout_container_round_trip(dsv4):
    p4 = _build_layout(dsv4, window_size=16).plans[4]
    x = torch.arange(64).view(1, 64).float()
    block_tokens = x.flatten()[p4.gather_indices].reshape(15, 4)
    container = torch.zeros(16, dtype=x.dtype)
    container[:15] = block_tokens.sum(dim=1)
    assert container[15] == 0
    assert torch.equal(container[:15], block_tokens.sum(dim=1))


def test_layout_ratio128_empty(dsv4):
    md = _build_layout(dsv4, window_size=16)
    p128 = md.plans[128]
    r128 = md.reference.ratios[128]
    assert p128.gather_indices.numel() == 0 and (r128.doc_of_block < 0).all()
    assert r128.dense_mask.shape == (1, 1, 64, 0)
    assert r128.static_blocks.shape[0] == 1  # kv_len = 65 -> window+sink only


def test_layout_cu_q_eq_cu_k_guard(dsv4):
    """The plain-stream guard: the common build accepts only contiguous
    documents (``cu_seq_q == cu_seq_k``) — context-parallel plans come
    from ``build_cp_plan``.  The reference tier keeps its own guard for
    CP-shaped metadata."""
    bad = VarlenMetadata(
        cu_seq_q=_CU,
        cu_seq_k=torch.tensor([0, 5, 27, 47, 64]),
        max_q=17,
        max_k=27,
    )
    with pytest.raises(ValueError):
        dsv4.metadata.build_compressed_varlen_metadata(bad, (4,))
    # the reference tier still rejects CP-shaped metadata outright
    md = dsv4.metadata.CompressedVarlenMetadata(
        varlen=bad,
        plans={
            1: dsv4.metadata.CompressedBlockLayout(
                cu_seqlens_cmp_k=None, block_remainder=None, gather_indices=None
            )
        },
    )
    with pytest.raises(ValueError):
        dsv4.reference.ReferenceMetadataExtension(
            dsv4.reference.ReferenceMetadataExtension.Config(window_size=16)
        )(md)


# ---------------------------------------------------------------------------
# Compressor / indexer numerics
# ---------------------------------------------------------------------------


def test_compressor_numerics_ratio4(inputs):
    """C4A pooling vs the independent manual reference.

    The residual must sit at the fp32 rounding floor: the two paths project
    with different matmul batching (all blocks at once vs per document), so
    the CPU kernel's shape-dependent reduction order yields 1-ulp residuals
    (the documented batch-size artifact; the same-size calls are
    deterministic).  ``_assert_rounding_floor`` accepts that — bitwise would
    only hold on shape-invariant kernels; the suite-wide 4-ulp budget keeps
    a 4x margin over the measured worst (~1 ulp across seeds and thread
    counts) while still flagging any multi-ulp arithmetic regression.
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


def test_indexer_numerics(inputs):
    dsv4, md, _, _ = inputs
    torch.manual_seed(0)
    idx_q = torch.randn(1, 64, 4, 8)
    idx_k = torch.randn(1, 16, 8)
    idx_w = torch.randn(1, 64, 4)
    dense = md.reference.ratios[4].dense_mask
    topk_idx, full_scores = dsv4.compressor.Indexer.select(
        idx_q, idx_k, idx_w, dense, 8
    )
    assert topk_idx.shape == (1, 64, 8)
    valid = topk_idx >= 0
    sel = topk_idx.clamp_min(0)
    assert torch.equal(
        dense.squeeze(1).gather(-1, sel)[valid].bool(),
        torch.ones(valid.sum(), dtype=torch.bool),
    ), "all selected slots must be attendable"
    scores = torch.einsum("bshd,btd->bsht", idx_q.float(), idx_k.float())
    scores = scores.relu_() * idx_w.float().unsqueeze(-1)
    scores = scores.sum(dim=2)
    masked = scores.where(dense.squeeze(1), float("-inf"))
    m_scores, m_idx = masked.topk(8, dim=-1)
    assert torch.equal(topk_idx, m_idx.where(m_scores.isfinite(), -1))
    assert torch.equal(full_scores, masked)


# ---------------------------------------------------------------------------
# Attention numerics (reference core vs the eager golden)
# ---------------------------------------------------------------------------


def _assert_rounding_floor(diff, scale, *, ulps: int = 4):
    """The residual must sit at the fp32 rounding floor of the outputs
    (<= ``ulps`` ulps of the output scale) — floating-point error only, not
    arithmetic (an arithmetic difference would exceed it by orders of
    magnitude).  ``ulps=4`` is the suite-wide budget: measured worsts are
    ~1 ulp (compressor batching artifact) and ~2.5 ulps (attention core vs
    golden) across seeds and thread counts."""
    assert scale > 0
    assert diff.max().item() <= ulps * _EPS32 * scale, (diff.max().item(), scale)


def _build_attn_pair(dsv4, ratio, topk):
    core = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=topk,
            block_size=(8, 8),
        )
    )
    golden = dsv4.golden.GoldenCompressedSparseInnerAttention(
        dsv4.golden.GoldenCompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=topk,
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
        out = core(q, swa_k, cmp, aq, ak, aw, attn_sink=sink, attention_masks=md)
        gout = golden(q, swa_k, cmp, aq, ak, aw, attn_sink=sink, attention_masks=md)
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
    md_s = dsv4.reference.ReferenceMetadataExtension(
        dsv4.reference.ReferenceMetadataExtension.Config(
            window_size=8, block_size=(8, 8)
        )
    )(dsv4.metadata.build_compressed_varlen_metadata(v, (4,)))
    comp_s = build_compressor(dsv4, 4)
    cmp_s = _pack_container(dsv4, comp_s(torch.randn(1, 8, DIM), md_s), md_s.plans[4])
    assert cmp_s.shape == (1, 2, 16)
    core_s = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=4,
            softmax_scale=0.25,
            index_topk=4,
            block_size=(8, 8),
        )
    )
    with torch.no_grad():
        out_s = core_s(
            torch.randn(1, 8, 3, HD),
            torch.randn(1, 8, HD),
            cmp_s,
            torch.randn(1, 8, 4, 8),
            torch.randn(1, 2, 8),
            torch.randn(1, 8, 4),
            attn_sink=torch.randn(3),
            attention_masks=md_s,
        )
    assert out_s.shape == (1, 8, 3, HD) and torch.isfinite(out_s).all()


# ---------------------------------------------------------------------------
# Ported upstream DSA semantics (the mask vs the old index formulation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,topk", [(0, 0), (1, 0), (4, 16), (128, 0)])
def test_ported_attended_sets_match_old_formulation(dsv4, ratio, topk):
    torch.manual_seed(0)
    device = torch.device("cpu")
    bsz, seqlen, window_size = 1, 512, 16
    meta_ratio = max(ratio, 1)  # metadata supports 1/4/128; ratio 0 == 1
    n_cmp = seqlen // ratio if ratio > 1 else 0
    sink_idx = seqlen + n_cmp
    if ratio == 4:
        topk_sel = old_indexer_topk(bsz, seqlen, ratio, topk, device, 7)
        compress = torch.where(topk_sel < 0, -1, topk_sel + seqlen)
    elif ratio > 1:
        topk_sel = None
        compress = compress_idxs(ratio, bsz, seqlen, device, seqlen)
    else:
        topk_sel = None
        compress = torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)
    win = window_idxs(window_size, bsz, seqlen, device)
    topk_idxs = torch.cat([win, compress], dim=-1) if compress.size(-1) else win

    cu = torch.tensor([0, seqlen], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=seqlen, max_k=seqlen)
    md = dsv4.reference.ReferenceMetadataExtension(
        dsv4.reference.ReferenceMetadataExtension.Config(
            window_size=window_size, block_size=128
        )
    )(dsv4.metadata.build_compressed_varlen_metadata(v, (meta_ratio,)))
    dsa = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            block_size=128,
            window_size=window_size,
            compress_ratio=meta_ratio,
            softmax_scale=0.1,
            index_topk=16,
        )
    )
    bm = dsa._build_varlen_block_mask(md, topk_sel, n_cmp, device)
    assert isinstance(bm, BlockMask)
    expected = old_attended(topk_idxs, sink_idx)
    actual = new_attended(bm, seqlen, n_cmp)
    assert expected == actual, f"attended sets differ for ratio={ratio}"


def test_ported_indexer_topk_matches_old(dsv4):
    torch.manual_seed(0)
    device = torch.device("cpu")
    bsz, seqlen, ratio, topk = 1, 512, 4, 16
    n_cmp = seqlen // ratio
    g = torch.Generator(device).manual_seed(11)
    idx_q = torch.randn(bsz, seqlen, 8, 32, generator=g, device=device)
    idx_k = torch.randn(bsz, n_cmp, 32, generator=g, device=device)
    idx_w = torch.randn(bsz, seqlen, 8, generator=g, device=device)

    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    dense = (
        (
            torch.arange(n_cmp, device=device).view(1, 1, -1)
            < causal_limit.view(1, -1, 1)
        )
        .unsqueeze(1)
        .expand(bsz, 1, seqlen, n_cmp)
    )
    selected, _ = dsv4.compressor.Indexer.select(idx_q, idx_k, idx_w, dense, topk)

    scores = torch.einsum("bshd,btd->bsht", idx_q, idx_k)
    scores = scores.relu_() * idx_w.unsqueeze(-1)
    scores = scores.sum(dim=2)
    mask = torch.arange(n_cmp, device=device).repeat(seqlen, 1) >= causal_limit
    scores = scores + torch.where(mask, torch.finfo(idx_q.dtype).min, 0)
    _, topk_idxs = scores.topk(min(topk, n_cmp), dim=-1)
    old = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)

    assert selected.shape == (bsz, seqlen, topk)
    assert torch.equal(selected.masked_fill(old < 0, -1), old)
