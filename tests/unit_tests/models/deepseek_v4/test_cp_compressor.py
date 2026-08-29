"""CPU experiment: compressor semantics under CP (plain sharding / headtail LB).

Oracle: an unsliced global ``VarlenMetadata`` + activations -> the real
``Compressor`` produces the per-doc pooled compressed keys (ground truth).

Per rank: derive ``CPVarlenMetadata`` from the global metadata, build the
*boundary borrow plan* (the missing predecessor-block tokens that make
straddled blocks and cross-boundary overlaps computable locally), simulate
the all-to-all (pure tensor slicing from the global stream), augment the
rank-local stream, compress locally, and assemble the per-segment packed
stream from a *compressed-level gather*: a global (doc, block) -> keys
table collected from every rank's local blocks (ownership = the rank whose
fragment contains the block's start), which replaces the oracle's per-doc
blocks in the assembly.  The oracle's doc blocks remain the ground truth
for the final assertion only.

Also exercises the ``CPDispatcher`` prototype (the MoE EP dispatcher
analogue of the design doc's pack/unpack): ``pack`` builds the per-segment
end-aligned ori ranges (window-preceding tokens + local rows) by slicing
the global stream at the segment's K-range window positions (the simulated
window-token all-to-all); ``unpack`` strips the rank-local rows (the Q
side), slices the per-segment ``swa_k`` ori ranges, and repacks the
per-segment compressed streams (``cmp_k`` / ``idx_k``) from the gather.
"""

import pytest
import torch
import torch.nn as nn
from test_dsv4 import IdentityRoPE
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata

torch.manual_seed(0)
DIM, HD, RD = 8, 16, 4
RATIO = 4


def _ht(seq, cp):
    return _HeadTailLoadBalancer(seq, cp, "cpu")


@pytest.fixture(scope="module")
def dsv4_globals(dsv4):
    """Install the stub harness and bind the model-dir modules the helpers
    reference as globals (deferred out of collection time)."""
    globals()["meta_mod"] = dsv4.metadata
    globals()["comp_mod"] = dsv4.compressor
    globals()["CPVarlenMetadata"] = dsv4.CPVarlenMetadata
    return None


class BuildCfg:
    def __init__(self, fn):
        self._fn = fn

    def build(self):
        return self._fn()


class RMS(nn.Module):
    def __init__(self, n, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n, dtype=dtype))

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
                x.dtype
            )
            * self.weight
        )


def make_linear(n, m, dtype=torch.float32):
    lin = nn.Linear(n, m, bias=False, dtype=dtype)
    nn.init.normal_(lin.weight, std=0.02)
    return lin


def build_compressor(dtype=torch.float32, ratio=RATIO):
    coff = 2 if ratio == 4 else 1
    comp = comp_mod.Compressor(
        comp_mod.Compressor.Config(
            rope=BuildCfg(lambda: IdentityRoPE()),
            head_dim=HD,
            rope_head_dim=RD,
            compress_ratio=ratio,
            wkv=BuildCfg(lambda: make_linear(DIM, coff * HD, dtype)),
            wgate=BuildCfg(lambda: make_linear(DIM, coff * HD, dtype)),
            norm=BuildCfg(lambda: RMS(HD, dtype=dtype)),
        )
    )
    nn.init.trunc_normal_(comp.ape, std=0.02)
    return comp


class FakeMesh:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size
        self.ndim = 1

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def compress_blocks(
    comp, block_tokens, block_positions, overlap_valid, dtype, ratio=RATIO
):
    """Mirror of the Compressor's pooling math (identical ops)."""
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


def rank_plan(cp_meta, shard_len, rank, perm=None, ratio=RATIO, doc_table=None):
    """Per-segment borrow plan for one rank, derived from CPVarlenMetadata only.

    ``doc_start`` of segment s = k_global_gather_indices[cu_seq_k[s]] (the
    first K position of the segment's causal range); ``p0`` (the fragment's
    doc-relative start) = seqlen_k - seg_len (the Q fragment is the tail of
    its K range).  ``doc_table`` maps each doc-start value to its
    (doc_index, doc_len) — the doc structure derived from the metadata set
    alone (see ``derive_doc_table``) — deciding whether the fragment-end
    straddle block exists (the document must continue past the fragment).

    Returns a list of segments, each with:
      prepend_global: [r] gather positions of the predecessor block (empty
                      when the segment starts at a doc start).
      local_global:   [seg_len] gather positions of the segment's own tokens.
      blocks:         doc-local block indices to compute: [b_first-1 (the
                      prepend borrow source, stripped), b_first..b_last] and,
                      when the fragment ends mid-block inside the document,
                      the fragment-end straddle block b_last+1 (its key is
                      owned by this fragment, the start-owner).
      block_positions: doc-relative starts of the computed blocks.
      overlap_valid:  per computed block (first plan block -> False).
      tail_len:       tokens of the straddle block beyond the fragment end
                      (0 when none), borrowed from the next fragment.
      doc:            document index of the segment.

    The segment's K range in ``k_global_gather_indices`` is doc-ordered (the
    restore composes the inverse load-balancer permutation), so the fragment
    is the last ``seg_len`` entries of the range and the predecessor block's
    tokens are its contiguous ``[(b_first-1)r, b_first*r)`` slice.  All
    positions are gather positions into the (possibly permuted) stream.
    """
    base = rank * shard_len
    # local indices into the (possibly permuted) rank stream; the segment
    # structure from CPVarlenMetadata is already over this stream.
    local_global = base + torch.arange(shard_len)
    cu_q = cp_meta.cu_seq_q
    cu_k = cp_meta.cu_seq_k
    kgather = cp_meta.k_global_gather_indices
    segments = []
    for s in range(len(cu_q) - 1):
        qs, qe = int(cu_q[s]), int(cu_q[s + 1])
        seg_len = qe - qs
        if seg_len == 0:
            continue
        seqlen_k = int(cu_k[s + 1]) - int(cu_k[s])
        p0 = seqlen_k - seg_len
        toks = local_global[qs:qe]
        k_start = int(cu_k[s])
        doc_start = int(kgather[k_start])
        if doc_table is None:
            doc = doc_start
            doc_len = None
        else:
            doc = doc_start
            doc_len = doc_table[doc_start]
        b_first = (p0 + ratio - 1) // ratio
        b_last = (p0 + seg_len) // ratio - 1
        prepend_global = torch.empty((0,), dtype=torch.int64)
        blocks = []
        positions = []
        tail_len = 0
        if b_first <= b_last:
            if b_first > 0:
                prepend_global = kgather[
                    k_start + (b_first - 1) * ratio : k_start + b_first * ratio
                ]
                blocks.append(b_first - 1)
                positions.append((b_first - 1) * ratio)
            blocks += list(range(b_first, b_last + 1))
            positions += [b * ratio for b in range(b_first, b_last + 1)]
        # Fragment-end straddle block (b_last + 1): the block containing the
        # fragment's last token when the fragment ends mid-block inside the
        # document, with the fragment as its start-owner.  Its overlap
        # predecessor must be locally computable: either the fragment has
        # complete blocks (the predecessor b_last is complete here), the
        # straddle is the document's first block (no borrow), or the
        # predecessor's head is borrowed (case C — the predecessor becomes a
        # stripped borrow-source block, like the prepend).  The tokens beyond
        # the fragment end (tail_len) are always borrowed.
        straddle_idx = (p0 + seg_len) // ratio
        straddle_start = straddle_idx * ratio
        pred_head_rel = None
        if (
            (p0 + seg_len) % ratio != 0
            and straddle_start >= p0
            and (doc_table is None or (straddle_idx + 1) * ratio <= doc_len)
        ):
            if straddle_idx > 0 and b_first > b_last:
                # case C: borrow the predecessor's head [pred_start, p0)
                pred_head_rel = ((straddle_idx - 1) * ratio, p0)
                blocks.append(straddle_idx - 1)
                positions.append((straddle_idx - 1) * ratio)
            tail_len = (straddle_idx + 1) * ratio - (p0 + seg_len)
            blocks.append(straddle_idx)
            positions.append(straddle_idx * ratio)
        # (the pred_head_global positions are filled by the caller)
        # The first plan block never borrows: it is either the doc's first
        # block (b_first == 0, no prepend) or the discarded borrow source
        # (the prepend block).  The rest borrow their in-plan predecessor.
        overlap = [False] + [True] * (len(blocks) - 1)
        segments.append(
            {
                "prepend_global": prepend_global,
                "local_global": toks,
                "blocks": blocks,
                "block_positions": positions,
                "overlap_valid": overlap,
                "p0": p0,
                "tail_len": tail_len,
                "doc": doc,
                "pred_head_rel": pred_head_rel,
            }
        )
    return segments


def compress_rank(comp, x_perm, plan, dtype, ratio=RATIO):
    """Compress one rank's blocks: augmented stream -> local packed keys.

    ``x_perm`` is the rank's view of the (possibly load-balanced) stream;
    all plan positions are gather positions into it.
    """
    augmented = []
    for seg in plan:
        if seg["prepend_global"].numel():
            augmented.append(x_perm[seg["prepend_global"]])
        if seg.get("pred_head_global") is not None:
            _, p0 = seg["pred_head_rel"]
            pred_end = seg["blocks"][-1] * ratio
            head_len = pred_end - p0
            augmented.append(
                torch.cat(
                    [
                        x_perm[seg["pred_head_global"]],
                        x_perm[seg["local_global"][:head_len]],
                    ]
                )
            )
        augmented.append(x_perm[seg["local_global"]])
        if seg["tail_len"]:
            augmented.append(x_perm[seg["tail_global"]])
    aug = torch.cat(augmented, dim=0)
    packed = []
    offset = 0
    for seg in plan:
        local_len = seg["local_global"].numel()
        tail_len = seg["tail_len"]
        prep = seg["prepend_global"].numel() > 0
        pred = seg.get("pred_head_global") is not None
        nb = (1 if prep else 0) + (1 if pred else 0)
        if not seg["blocks"]:
            offset += nb * ratio + local_len + tail_len
            continue
        n_blocks = len(seg["blocks"])
        p0 = seg["p0"]
        local_off = offset + nb * ratio
        indices = []
        if nb:
            indices += list(range(offset, offset + ratio))
        for b in seg["blocks"][nb:]:
            first = b * ratio - p0
            indices += list(range(local_off + first, local_off + first + ratio))
        offset += nb * ratio + local_len + tail_len
        block_tokens = aug[torch.tensor(indices)].reshape(n_blocks, ratio, -1)
        keys = compress_blocks(
            comp,
            block_tokens,
            torch.tensor(seg["block_positions"], dtype=torch.int32),
            torch.tensor(seg["overlap_valid"]),
            dtype,
            ratio,
        )
        if nb:
            keys = keys[nb:]  # drop the discarded borrow-source block(s)
        packed.append(keys)
    return torch.cat(packed, dim=0) if packed else torch.empty((0, HD), dtype=dtype)


def verify_ori_ranges(cp_meta, shard_len, rank, n_win=128):
    """Verify the window-based ori packing against the kernel's formulas.

    Per segment, the delivered ori range is end-aligned: oriLen =
    seg_len + p0 - win_start_rel with win_start_rel = max(p0 - (n_win-1), 0),
    and its positions are the tail of the segment's causal K range.  The
    kernel (end-aligned coordinate) computes oriMaskLeft/Right per query;
    the mapped document window must equal [seg_start + pos - (n_win-1),
    seg_start + pos] within the range.
    """
    cu_q, cu_k, kg = cp_meta.cu_seq_q, cp_meta.cu_seq_k, cp_meta.k_global_gather_indices
    for s in range(len(cu_q) - 1):
        seg_len = int(cu_q[s + 1]) - int(cu_q[s])
        if seg_len == 0:
            continue
        seqlen_k = int(cu_k[s + 1]) - int(cu_k[s])
        p0 = seqlen_k - seg_len
        win_start_rel = max(p0 - (n_win - 1), 0)
        ori_len = seg_len + p0 - win_start_rel
        ks = int(cu_k[s])
        rng = kg[ks + win_start_rel : ks + seqlen_k]
        assert rng.numel() == ori_len, (s, rng.numel(), ori_len)
        # the kernel's end-aligned window per query, in doc-relative coords
        for pos in range(seg_len):
            left = max(ori_len - seg_len + pos - (n_win - 1), 0)
            right = ori_len - seg_len + pos
            assert win_start_rel + left == max(p0 + pos - (n_win - 1), win_start_rel), (
                s,
                pos,
            )
            assert win_start_rel + right == p0 + pos, (s, pos)


def derive_doc_table(infos):
    """Doc structure (doc index, length per doc-start) from the metadata set.

    Every document's first segment starts at its doc start, and every
    segment's K range starts at its doc's start, so the distinct values of
    ``kgather[cu_seq_k[s]]`` over ALL ranks' segments are exactly the doc
    starts.  Sorting them yields the doc indices; a doc's length is the
    distance to the next doc start (for the last doc: the maximum
    ``seqlen_k`` among its segments).  No global metadata is involved — the
    table is a pure function of the per-rank ``CPVarlenMetadata`` set.
    """
    doc_len = {}
    for info in infos:
        cu_k = info["cp_meta"].cu_seq_k
        kg = info["cp_meta"].k_global_gather_indices
        for s in range(len(cu_k) - 1):
            seqlen_k = int(cu_k[s + 1]) - int(cu_k[s])
            if seqlen_k == 0:
                continue
            doc_start = int(kg[int(cu_k[s])])
            doc_len[doc_start] = max(doc_len.get(doc_start, 0), seqlen_k)
    # A doc's length is the maximum seqlen_k among its segments: the segment
    # containing the doc's last token has seqlen_k = doc_len, and every
    # segment's K range is [doc_start, its last token] (full prefix).  The
    # doc identity is the doc start's *permuted* position — the same value
    # in every rank's metadata, so (doc, block) keys agree across ranks.
    return dict(doc_len)


class CPDispatcher:
    """Prototype of the design doc's §8 pack/unpack dispatcher.

    Mirrors the MoE EP dispatcher shape (dispatch -> pack, combine ->
    unpack).  ``pack`` performs the window-token all-to-all — simulated here
    by slicing the global (possibly load-balanced) stream at the segment's
    K-range window positions — and assembles the per-segment end-aligned
    ori ranges ``[window-preceding | local]``; ``unpack`` strips the
    rank-local rows (the Q side), slices the per-segment ``swa_k`` ori
    ranges, and repacks the per-segment compressed streams (``cmp_k`` /
    ``idx_k``) delivered by the compressed-level gather, together with the
    kernel prefix-sum tensors.
    """

    def __init__(self, cp_meta, seg_plan, global_x, ratio, n_win=128):
        self.cp_meta = cp_meta
        self.seg_plan = seg_plan
        self.global_x = global_x
        self.ratio = ratio
        self.n_win = n_win

    def pack(self):
        """Assemble ``packed_x`` = cat over segments of the ori ranges.

        Per segment: oriLen = seg_len + p0 - win_start_rel with
        win_start_rel = max(p0 - (n_win - 1), 0); the range positions are
        ``kgather[cu_k[s] + win_start_rel : cu_k[s+1]]`` — the tail of the
        segment's causal K range, end-aligned by construction.  The
        window-preceding rows (positions < the local fragment's start) are
        other ranks' tokens; the local rows are the segment's own.  Returns
        ``(packed_x, seg_meta)`` where seg_meta carries the per-segment
        layout (start / ori_len / seg_len / p0 / win_start_rel / k_s).
        """
        cu_k = self.cp_meta.cu_seq_k
        kg = self.cp_meta.k_global_gather_indices
        parts = []
        seg_meta = []
        total = 0
        for s, seg in enumerate(self.seg_plan):
            seg_len = seg["local_global"].numel()
            p0 = seg["p0"]
            win_start_rel = max(p0 - (self.n_win - 1), 0)
            ori_len = seg_len + p0 - win_start_rel
            k_start = int(cu_k[s])
            positions = kg[k_start + win_start_rel : k_start + p0 + seg_len]
            parts.append(self.global_x[positions])
            seg_meta.append(
                {
                    "start": total,
                    "ori_len": ori_len,
                    "seg_len": seg_len,
                    "p0": p0,
                    "win_start_rel": win_start_rel,
                    "k_s": (p0 + seg_len) // self.ratio,
                    "seqlen_k": p0 + seg_len,
                }
            )
            total += ori_len
        packed_x = torch.cat(parts, dim=0) if parts else self.global_x[:0]
        return packed_x, seg_meta

    def unpack(self, packed_x, seg_meta, cmp_streams=None, idx_streams=None):
        """The six clean outputs + the kernel prefix-sum tensors.

        - ``q_rows``: the per-segment local rows (the tail ``seg_len`` rows
          of each ori range) — the Q-side strip; the plan's local offsets.
        - ``swa_k``: the per-segment ori ranges, end-aligned
          (oriLen = seg_len + p0 - win_start_rel).
        - ``cmp_k`` / ``idx_k``: the per-segment packed compressed streams
          from the compressed-level gather (step 1), concatenated.
        - the ``cu_seqlens_*`` tensors the kernels consume.
        """
        q_parts, swa_parts = [], []
        cu_q, cu_ori, cu_cmp = [0], [0], [0]
        residuals = []
        for m in seg_meta:
            st, ori_len, seg_len = m["start"], m["ori_len"], m["seg_len"]
            swa_parts.append(packed_x[st : st + ori_len])
            q_parts.append(packed_x[st + ori_len - seg_len : st + ori_len])
            cu_q.append(cu_q[-1] + seg_len)
            cu_ori.append(cu_ori[-1] + ori_len)
            cu_cmp.append(cu_cmp[-1] + m["k_s"])
            residuals.append(m["seqlen_k"] % self.ratio)
        dtype = packed_x.dtype
        return {
            "q_rows": torch.cat(q_parts, dim=0)
            if q_parts
            else packed_x.new_empty((0, packed_x.shape[1])),
            "swa_k": torch.cat(swa_parts, dim=0)
            if swa_parts
            else packed_x.new_empty((0, packed_x.shape[1])),
            "cmp_k": torch.cat(cmp_streams, dim=0)
            if cmp_streams
            else torch.empty((0, HD), dtype=dtype),
            "idx_k": torch.cat(idx_streams, dim=0)
            if idx_streams
            else torch.empty((0, HD), dtype=dtype),
            "cu_seqlens_q": torch.tensor(cu_q, dtype=torch.int32),
            "cu_seqlens_ori_kv": torch.tensor(cu_ori, dtype=torch.int32),
            "cu_seqlens_cmp_kv": torch.tensor(cu_cmp, dtype=torch.int32),
            "residual": torch.tensor(residuals, dtype=torch.int32),
        }


def verify_dispatcher(
    info,
    rank,
    x_perm,
    x_flat,
    cu,
    comp,
    comp_idx,
    q_proj,
    idx_q_proj,
    idx_w_proj,
    doc_blocks,
    idx_doc_blocks,
    cmp_table,
    idx_table,
    ratio,
    dtype,
    shard_len,
    failures,
    orig_start,
):
    """Dispatcher round-trip vs the oracle (tensor/layout semantics).

    - ``pack``: per-segment ori ranges; the local tail must be the rank's
      own stream rows and the whole range must equal the oracle's doc
      tokens (end-aligned window), so ``swa_k`` = wkv over the same window
      tokens as the oracle path (exact — same module, same rows).
    - ``unpack``: the local-row strip (q / idx_q / idx_w projections equal
      the direct local-row projections), the ``swa_k`` ori ranges, the
      per-segment ``cmp_k`` / ``idx_k`` streams (must equal the oracle's
      doc blocks re-packed per segment), and the kernel cumsums.
    """
    cp_meta, seg_plan, doc_map = info["cp_meta"], info["seg_plan"], info["doc_map"]
    disp = CPDispatcher(cp_meta, seg_plan, x_perm, ratio)
    packed_x, seg_meta = disp.pack()
    cu_k = cp_meta.cu_seq_k.tolist()
    cmp_segs, idx_segs, exp_cmp, exp_idx = [], [], [], []
    for s, seg in enumerate(seg_plan):
        doc = doc_map[s]
        k_s = (cu_k[s + 1] - cu_k[s]) // ratio
        if k_s:
            cmp_segs.append(torch.stack([cmp_table[(doc, b)] for b in range(k_s)]))
            idx_segs.append(torch.stack([idx_table[(doc, b)] for b in range(k_s)]))
        else:
            cmp_segs.append(torch.empty((0, HD), dtype=dtype))
            idx_segs.append(torch.empty((0, HD), dtype=dtype))
        exp_cmp.append(doc_blocks[doc][:k_s])
        exp_idx.append(idx_doc_blocks[doc][:k_s])
    out = disp.unpack(packed_x, seg_meta, cmp_segs, idx_segs)
    tol = (1e-12, 1e-12) if dtype == torch.float64 else (1e-6, 1e-6)

    # Q-side strip: the unpacked local rows are the rank's own stream
    local = x_perm[rank * shard_len : (rank + 1) * shard_len]
    if not torch.equal(out["q_rows"], local):
        failures.append((rank, "disp", "q strip"))
    # q / idx_q / idx_w over the local rows == the direct projections
    for tag, proj in (("q", q_proj), ("idx_q", idx_q_proj), ("idx_w", idx_w_proj)):
        if not torch.allclose(
            proj(out["q_rows"]), proj(local), atol=tol[0], rtol=tol[1]
        ):
            failures.append((rank, "disp", f"{tag} proj"))
    # swa_k: per-segment ori ranges == the oracle's end-aligned window
    # tokens; the wkv projection is exact over the same rows
    for s, m in enumerate(seg_meta):
        seg = seg_plan[s]
        doc = doc_map[s]
        ori_tokens = packed_x[m["start"] : m["start"] + m["ori_len"]]
        o_start = orig_start[doc]
        oracle_tokens = x_flat[
            o_start + m["win_start_rel"] : o_start + seg["p0"] + m["seg_len"]
        ]
        if not torch.equal(ori_tokens, oracle_tokens):
            failures.append((rank, s, "disp swa tokens"))
        if not torch.allclose(
            comp.wkv(ori_tokens), comp.wkv(oracle_tokens), atol=tol[0], rtol=tol[1]
        ):
            failures.append((rank, s, "disp wkv"))
    # cmp_k / idx_k: per-segment packed streams == the oracle re-packed
    for s in range(len(seg_plan)):
        if not torch.allclose(cmp_segs[s], exp_cmp[s], atol=tol[0], rtol=tol[1]):
            failures.append((rank, s, "disp cmp"))
        if not torch.allclose(idx_segs[s], exp_idx[s], atol=tol[0], rtol=tol[1]):
            failures.append((rank, s, "disp idx"))
    # unpack plumbing: the TND streams and the kernel cumsums
    all_cmp = (
        torch.cat(cmp_segs, dim=0) if cmp_segs else torch.empty((0, HD), dtype=dtype)
    )
    all_idx = (
        torch.cat(idx_segs, dim=0) if idx_segs else torch.empty((0, HD), dtype=dtype)
    )
    if not torch.equal(out["cmp_k"], all_cmp):
        failures.append((rank, "disp", "cmp stream"))
    if not torch.equal(out["idx_k"], all_idx):
        failures.append((rank, "disp", "idx stream"))
    exp_cu_q = torch.tensor(
        [0] + [m["seg_len"] for m in seg_meta], dtype=torch.int32
    ).cumsum(0)
    exp_cu_ori = torch.tensor(
        [0] + [m["ori_len"] for m in seg_meta], dtype=torch.int32
    ).cumsum(0)
    exp_cu_cmp = torch.tensor(
        [0] + [m["k_s"] for m in seg_meta], dtype=torch.int32
    ).cumsum(0)
    if not torch.equal(out["cu_seqlens_q"], exp_cu_q):
        failures.append((rank, "disp", "cu_q"))
    if not torch.equal(out["cu_seqlens_ori_kv"], exp_cu_ori):
        failures.append((rank, "disp", "cu_ori"))
    if not torch.equal(out["cu_seqlens_cmp_kv"], exp_cu_cmp):
        failures.append((rank, "disp", "cu_cmp"))
    if out["cu_seqlens_ori_kv"][-1].item() != packed_x.shape[0]:
        failures.append((rank, "disp", "cu_ori total"))
    if out["cu_seqlens_cmp_kv"][-1].item() != (
        all_cmp.shape[0] if all_cmp.numel() else 0
    ):
        failures.append((rank, "disp", "cu_cmp total"))
    exp_res = (
        torch.tensor(
            [int(cu_k[s + 1]) - int(cu_k[s]) for s in range(len(seg_plan))],
            dtype=torch.int32,
        )
        % ratio
    )
    if not torch.equal(out["residual"], exp_res):
        failures.append((rank, "disp", "residual"))


def run_case(name, docs, cp_size, lb, dtype=torch.float32, ratio=RATIO):
    """docs: doc lengths; lb: None or a load balancer instance.

    ``ratio`` selects the compressor variant: 4 = C4A (overlapping), 128 =
    C128A (non-overlapping).  The oracle is the real ``Compressor`` for
    float32; for float64 the real forward hard-casts to float32 internally,
    so the oracle is ``compress_blocks`` on the full stream.
    """
    seq_len = sum(docs)
    cu = torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=max(docs), max_k=max(docs))
    md = meta_mod.build_compressed_varlen_metadata(v, (ratio,))
    plan = md.plans[ratio]
    comp = build_compressor(dtype, ratio)
    comp_idx = build_compressor(dtype, ratio)  # indexer mirror (own modules)
    q_proj = make_linear(DIM, HD, dtype)
    idx_q_proj = make_linear(DIM, HD, dtype)
    idx_w_proj = make_linear(DIM, HD, dtype)
    x = torch.randn(1, seq_len, DIM, dtype=dtype)
    x_flat = x.flatten(0, 1)
    n_blocks = int(plan.cu_seqlens_cmp_k[-1])
    if dtype == torch.float32:
        with torch.no_grad():
            pooled = comp(x, md)
        packed_oracle = pooled  # doc-major blocks
    else:
        if n_blocks:
            bt_full = x_flat[plan.gather_indices].reshape(n_blocks, ratio, -1)
            bids = torch.arange(n_blocks)
            seq_ids = torch.searchsorted(plan.cu_seqlens_cmp_k[1:], bids, right=True)
            block_local = bids - plan.cu_seqlens_cmp_k[seq_ids]
            packed_oracle = compress_blocks(
                comp,
                bt_full,
                (block_local * ratio).to(torch.int32),
                block_local != 0,
                dtype,
                ratio,
            )
        else:
            packed_oracle = torch.empty((0, HD), dtype=dtype)
    if lb is None:
        perm = None
    else:
        perm = lb._generate_indices(restore=False).reshape(-1)
        assert perm is not None and perm.numel() == seq_len
    x_perm = x_flat[perm] if perm is not None else x_flat
    # per-doc oracle blocks, keyed by the doc start's *permuted* position —
    # the doc identity the scheme derives from the metadata
    cu_b = plan.cu_seqlens_cmp_k.tolist()
    doc_blocks = {}
    for d in range(len(docs)):
        ps = int((perm == cu[d]).nonzero()[0]) if perm is not None else int(cu[d])
        doc_blocks[ps] = packed_oracle[cu_b[d] : cu_b[d + 1]]
    # indexer oracle: the same plan with the indexer's own compressor modules
    if n_blocks:
        bt_full = x_flat[plan.gather_indices].reshape(n_blocks, ratio, -1)
        bids = torch.arange(n_blocks)
        seq_ids = torch.searchsorted(plan.cu_seqlens_cmp_k[1:], bids, right=True)
        block_local = bids - plan.cu_seqlens_cmp_k[seq_ids]
        idx_packed_oracle = compress_blocks(
            comp_idx,
            bt_full,
            (block_local * ratio).to(torch.int32),
            block_local != 0,
            dtype,
            ratio,
        )
    else:
        idx_packed_oracle = torch.empty((0, HD), dtype=dtype)
    idx_doc_blocks = {}
    orig_start = {}
    for d in range(len(docs)):
        ps = int((perm == cu[d]).nonzero()[0]) if perm is not None else int(cu[d])
        idx_doc_blocks[ps] = idx_packed_oracle[cu_b[d] : cu_b[d + 1]]
        orig_start[ps] = int(cu[d])

    shard_len = seq_len // cp_size
    failures = []
    borrow_stats = []
    maxdiffs = []
    tol = (1e-12, 1e-12) if dtype == torch.float64 else (1e-6, 1e-6)
    # ---- pass A (all ranks): plans, local compression, and the global
    # ---- (doc, block) -> keys tables (compressed-level gather)
    per_rank = []
    cmp_table = {}
    idx_table = {}
    per_rank = []
    for rank in range(cp_size):
        cp_meta = CPVarlenMetadata.from_global(
            v, FakeMesh(rank, cp_size), 1, seq_len, lb
        )
        per_rank.append({"cp_meta": cp_meta})
    # the doc structure from the metadata set alone (no global cu/perm)
    doc_table = derive_doc_table(per_rank)
    for rank, info in enumerate(per_rank):
        cp_meta = info["cp_meta"]
        seg_plan = rank_plan(cp_meta, shard_len, rank, ratio=ratio, doc_table=doc_table)
        verify_ori_ranges(cp_meta, shard_len, rank)
        borrow = sum(
            int(s["prepend_global"].numel())
            + int(
                s.get("pred_head_global").numel()
                if s.get("pred_head_global") is not None
                else 0
            )
            + int(s["tail_len"])
            for s in seg_plan
        )
        borrow_stats.append(borrow)
        info["seg_plan"] = seg_plan
        info["doc_map"] = [s["doc"] for s in seg_plan]
    # ---- fill the tail positions (the fragment-end straddle borrow): the
    # ---- doc's widest segment covers every doc-relative offset; map the
    # ---- tail offsets through its kgather slice
    widest = {}
    for info in per_rank:
        cu_k = info["cp_meta"].cu_seq_k
        for s, doc in enumerate(info["doc_map"]):
            seqlen_k = int(cu_k[s + 1]) - int(cu_k[s])
            if doc not in widest or seqlen_k > widest[doc][0]:
                widest[doc] = (
                    seqlen_k,
                    int(cu_k[s]),
                    info["cp_meta"].k_global_gather_indices,
                )
    for info in per_rank:
        for seg in info["seg_plan"]:
            if seg["tail_len"]:
                _, wk, wkg = widest[seg["doc"]]
                q0 = seg["p0"] + seg["local_global"].numel()
                seg["tail_global"] = wkg[wk + q0 : wk + q0 + seg["tail_len"]]
                assert seg["tail_global"].numel() == seg["tail_len"]
            if seg.get("pred_head_rel") is not None:
                ps, pe = seg["pred_head_rel"]
                _, wk, wkg = widest[seg["doc"]]
                seg["pred_head_global"] = wkg[wk + ps : wk + pe]
                assert seg["pred_head_global"].numel() == pe - ps
    # ---- pass A2: local compression and the global (doc, block) tables
    for info in per_rank:
        seg_plan = info["seg_plan"]
        doc_map = info["doc_map"]
        local_keys = compress_rank(comp, x_perm, seg_plan, x.dtype, ratio)
        idx_local_keys = compress_rank(comp_idx, x_perm, seg_plan, x.dtype, ratio)
        # ownership: block b of a doc is computed by the rank whose
        # fragment contains its start (implicit in the plan's blocks)
        local_idx = 0
        for seg, doc in zip(seg_plan, doc_map, strict=True):
            prep = seg["prepend_global"].numel() > 0
            pred = seg.get("pred_head_global") is not None
            nb = (1 if prep else 0) + (1 if pred else 0)
            count = len(seg["blocks"]) - nb
            if count > 0:
                first_local = seg["blocks"][nb]
                for i, b in enumerate(range(first_local, first_local + count)):
                    cmp_table[(doc, b)] = local_keys[local_idx + i]
                    idx_table[(doc, b)] = idx_local_keys[local_idx + i]
            local_idx += count
    # ---- pass B + C (all ranks): assemble the per-segment packed streams
    # ---- from the gather and run the Dispatcher round-trip
    for rank in range(cp_size):
        info = per_rank[rank]
        cp_meta, seg_plan, doc_map = info["cp_meta"], info["seg_plan"], info["doc_map"]
        cu_k = cp_meta.cu_seq_k.tolist()
        assembled = []
        for s, seg in enumerate(seg_plan):
            doc = doc_map[s]
            k_s = (cu_k[s + 1] - cu_k[s]) // ratio
            expected = doc_blocks[doc][:k_s]
            got = (
                torch.stack([cmp_table[(doc, b)] for b in range(k_s)])
                if k_s
                else torch.empty((0, HD), dtype=dtype)
            )
            if not torch.allclose(got, expected, atol=tol[0], rtol=tol[1]):
                failures.append((rank, s, seg["blocks"], "mismatch"))
            if got.numel() and expected.numel():
                maxdiffs.append((got - expected).abs().max().item())
            assembled.append(got)
        if assembled:
            per_rank_stream = torch.cat(assembled, dim=0)
            # the full per-rank packed stream == the oracle's per-doc blocks re-packed per segment
            expected_all = torch.cat(
                [
                    doc_blocks[doc_map[s]][: (cu_k[s + 1] - cu_k[s]) // ratio]
                    for s, seg in enumerate(seg_plan)
                ],
                dim=0,
            )
            if not torch.allclose(
                per_rank_stream, expected_all, atol=tol[0], rtol=tol[1]
            ):
                failures.append((rank, "stream", "mismatch"))
        # ---- Dispatcher pack/unpack round-trip vs the oracle
        verify_dispatcher(
            info,
            rank,
            x_perm,
            x_flat,
            cu,
            comp,
            comp_idx,
            q_proj,
            idx_q_proj,
            idx_w_proj,
            doc_blocks,
            idx_doc_blocks,
            cmp_table,
            idx_table,
            ratio,
            dtype,
            shard_len,
            failures,
            orig_start,
        )
    status = "OK" if not failures else f"FAIL {failures[:3]}"
    tag = "f64" if dtype == torch.float64 else "f32"
    md_ = max(maxdiffs) if maxdiffs else 0.0
    print(
        f"{name:12s} cp={cp_size} {tag} {status}  maxdiff={md_:.3e}  borrow={borrow_stats}"
    )
    return not failures


_DETERMINISTIC_CASES = [
    # ---- baseline geometries (rounds <= 14)
    ("plain2", (10, 17, 20, 17), 2, None),
    ("plain4", (10, 17, 20, 17), 4, None),
    # doc start exactly at a shard boundary (no borrow)
    ("docstart", (10, 22, 32), 2, None),
    # pure-remainder fragment (no complete block -> all gathered)
    ("zero-block", (10, 8, 17, 5, 24), 4, None),
    # lengths not multiple of the ratio: 133/122/744/281 (122 < 128 -> no
    # C128A blocks), seq_len 1280
    ("irreg", (133, 122, 744, 281), 2, None),
    ("irreg4", (133, 122, 744, 281), 4, None),
    # ---- round 15: stress matrix (all f32 + f64, plain + headtail)
    # cp=8, shard 502 % 4 = 2: every shard boundary cuts through a block
    ("cp8", (401, 913, 1003, 1699), 8, None),
    ("cp8-ht", (401, 913, 1003, 1699), 8, _ht(4016, 8)),
    # seq 8184: shard 2046 % 4 = 2 and headtail chunk 1023 % 4 = 3 —
    # CP boundaries cut through blocks at every scale; ~2000 blocks at
    # r=4; a 27-token sub-128 doc
    ("long", (307, 1511, 27, 4017, 1019, 1303), 4, None),
    ("long-ht", (307, 1511, 27, 4017, 1019, 1303), 4, _ht(8184, 4)),
    # seq 8176, cp=8: shard 1022 % 4 = 2, chunks 511 % 4 = 3 — max
    # fragmentation (16 chunks)
    ("long8", (301, 1507, 27, 4017, 1021, 1303), 8, None),
    ("long8-ht", (301, 1507, 27, 4017, 1021, 1303), 8, _ht(8176, 8)),
    # seq 8192: shard 2048 and chunk 1024 block-aligned — straddles only
    # at doc boundaries (the plan must degrade to zero shard traffic)
    ("aligned-shards", (307, 1511, 29, 4021, 1021, 1303), 4, None),
    ("aligned-shards-ht", (307, 1511, 29, 4021, 1021, 1303), 4, _ht(8192, 4)),
    # 15 docs of 1..9 tokens: zero-block fragments, case-B doc-start
    # straddles, case-C predecessor borrows, doc-boundary exclusion
    # almost everywhere (shard 18 % 4 = 2, chunks 9 % 4 = 1)
    ("tiny-docs", (5, 2, 9, 1, 7, 4, 3, 8, 2, 6, 5, 3, 4, 9, 4), 4, None),
    ("tiny-docs-ht", (5, 2, 9, 1, 7, 4, 3, 8, 2, 6, 5, 3, 4, 9, 4), 4, _ht(72, 4)),
    # r-multiples (aligned doc starts), r*k+1 (remainder-1 straddles),
    # r-1 docs
    ("aligned", (32, 96, 129, 63, 33, 7), 4, None),
    ("aligned-ht", (32, 96, 129, 63, 33, 7), 4, _ht(360, 4)),
    # a doc ends exactly at a shard boundary mid-block: the partial
    # block must be computed by no one
    ("boundary-midblock", (34, 30, 64), 2, None),
    ("boundary-midblock-ht", (34, 30, 64), 2, _ht(128, 2)),
    # degenerate single-rank sanity at scale (no borrows at all)
    ("cp1", (1000, 2000, 1496), 1, None),
]
# headtail load balancer: cp2 and cp4, longer non-aligned sequences
for _cp in (2, 4):
    for _docs in ((37, 41, 63, 115), (97, 131, 149, 135)):  # 256 / 512
        _DETERMINISTIC_CASES.append(
            (f"headtail{_cp}-{sum(_docs)}", _docs, _cp, _ht(sum(_docs), _cp))
        )
    _DETERMINISTIC_CASES.append(
        (f"headtail{_cp}-irreg", (133, 122, 744, 281), _cp, _ht(1280, _cp))
    )


def _deterministic_params():
    params = []
    for name, docs, cp, lb in _DETERMINISTIC_CASES:
        for ratio in (4, 128):  # C4A (overlapping) and C128A (non-overlapping)
            for dtype in (torch.float32, torch.float64):
                params.append(
                    pytest.param(
                        (name, docs, cp, lb, dtype, ratio),
                        id=f"{name}-r{ratio}-{'f32' if dtype == torch.float32 else 'f64'}",
                    )
                )
    return params


@pytest.mark.parametrize("case", _deterministic_params())
def test_cp_case(dsv4_globals, case):
    """The full CP pipeline for one deterministic geometry: plan, borrows,
    local compression, the compressed-level gather, the per-segment
    assembly vs the oracle, and the Dispatcher round-trip."""
    name, docs, cp, lb, dtype, ratio = case
    assert run_case(f"{name}-r{ratio}", docs, cp, lb, dtype, ratio)


def _sweep_params():
    """Seeded randomized stress sweep (round 15) — fixed configs.

    Each config: cp in {2,4,8}, target seq in {2048,4096,8192} adjusted to
    the divisibility rule (plain: seq % cp == 0; headtail: seq % 2cp == 0),
    4..12 docs whose lengths mix tiny (1..16) and huge (500..3000) spans,
    ~half with the headtail load balancer.
    """
    params = []
    rng = torch.Generator().manual_seed(0)
    made = 0
    attempts = 0
    while made < 10 and attempts < 40:
        attempts += 1
        cp = int(torch.randint(2, 9, (1,), generator=rng).item())
        lb = None if torch.rand(1, generator=rng).item() < 0.5 else "ht"
        div = (2 * cp) if lb else cp
        lcm = torch.lcm(torch.tensor(div), torch.tensor(8)).item()  # seq % 8 == 0 too
        target = int(
            torch.tensor([2048, 4096, 8192])[
                int(torch.randint(0, 3, (1,), generator=rng))
            ]
        )
        seq = (target // lcm) * lcm
        n_docs = int(torch.randint(4, 13, (1,), generator=rng).item())
        docs = []
        for _ in range(n_docs - 1):
            if torch.rand(1, generator=rng).item() < 0.4:
                docs.append(int(torch.randint(1, 17, (1,), generator=rng).item()))
            else:
                docs.append(int(torch.randint(1, 2501, (1,), generator=rng).item()))
        if sum(docs) >= seq:  # the last doc must stay >= 1
            continue
        docs.append(seq - sum(docs))  # the remainder becomes one huge doc
        assert sum(docs) == seq, (seq, sum(docs))
        lb_inst = _ht(seq, cp) if lb else None
        name = f"sweep{made}-cp{cp}-{'ht' if lb else 'plain'}-{seq}"
        for ratio in (4, 128):
            for dtype in (torch.float32, torch.float64):
                params.append(
                    pytest.param(
                        (name, tuple(docs), cp, lb_inst, dtype, ratio),
                        id=f"{name}-r{ratio}-{'f32' if dtype == torch.float32 else 'f64'}",
                    )
                )
        made += 1
    assert made == 10, (made, attempts)
    return params


@pytest.mark.parametrize("case", _sweep_params())
def test_cp_sweep_case(dsv4_globals, case):
    """One seeded randomized sweep config through the full pipeline."""
    name, docs, cp, lb, dtype, ratio = case
    assert run_case(f"{name}-r{ratio}", docs, cp, lb, dtype, ratio)


@pytest.mark.parametrize(
    "docs,ratio",
    [
        pytest.param(d, r, id=f"docs{sum(d)}-r{r}")
        for d in ((10, 17, 20, 17), (133, 122, 744, 281), (37, 41, 63, 115))
        for r in (4, 128)
    ],
)
def test_degeneracy(dsv4_globals, docs, ratio):
    """Plain ``build_kernel_layout`` vs the cp=1 ``build_cp_plan``: the CP
    plan degenerates bitwise to the plain contract — the permanent
    no-drift guard between the two plan builders."""
    from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan

    seq = sum(docs)
    cu = torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=max(docs), max_k=max(docs))
    plain = meta_mod.build_kernel_layout(v, (ratio,))
    _cp_meta, cp1_plans, _window = build_cp_plan(
        v,
        None,
        rank=0,
        cp_size=1,
        shard_len=seq,
        window_size=8,
        ratios=[ratio],
    )
    cp1 = cp1_plans[ratio]
    assert torch.equal(plain[ratio].cu_seqlens_cmp_k, cp1.cu_seqlens_cmp_k)
    assert torch.equal(plain[ratio].block_remainder, cp1.block_remainder)
    # at cp=1 every row is local and the assembly order is the doc-major
    # order, so the gather indices and the contract match bitwise
    assert torch.equal(plain[ratio].gather_indices, cp1.gather_indices)
    assert torch.equal(plain[ratio].block_positions, cp1.block_positions)
    assert torch.equal(plain[ratio].first_indices, cp1.first_indices)
