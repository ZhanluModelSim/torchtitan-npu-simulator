# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# ruff: noqa: E501, SIM108
# fmt: off

# =============================================================================
# custom/li_backward/sparse_lightning_indexer_kl_loss_grad.py
#
# PyPTO drop-in for the CANN torch op
#   cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad
# (kernel attention/sparse_lightning_indexer_kl_loss_grad) — the DeepSeek-V4
# Lightning-Indexer TRAINING backward op. Same inputs/outputs as the torch op;
# numerically verified within CANN tolerance (see test_sparse_li_klgrad.py).
#
# The op takes the teacher distribution p (attn_softmax_l1_norm) as an INPUT
# (does NOT recompute it from main-attn q/k), uses dI = y*ReduceSum(p) - p, and
# outputs (dq, dk, dw, softmax_out) — softmax_out is the student indexer softmax
# y; there is no KL-loss scalar. Per query token:
#   s_logits = q_index @ k_index_topk^T   (NO 1/sqrt(D) scale)
#   y = softmax( sum_j w[j] * ReLU(s_logits[j]) )                 -> softmax_out
#   dI = y * sum_s(p) - p
#   dw = sum_s ReLU(s)*dI ; d_s_logits = dI*w*(ReLU(s)>0) (round to input dtype)
#   dq = d_s_logits @ k_index_topk ; dk = scatter_add( d_s_logits^T @ q_index )
#
# Public API:
#   sparse_lightning_indexer_kl_loss_grad(q, k, w, sparse_indices,
#       attn_softmax_l1_norm, *, cu_seqlens_q, cu_seqlens_k, seqused_q,
#       seqused_k, cmp_residual_k, metadata, layout_q, layout_k, mask_mode,
#       cmp_ratio, max_seqlen_q, max_seqlen_k) -> (dq, dk, dw, softmax_out)
#   sparse_li_klgrad_kernel  — the single @pypto.frontend.jit entry (NPU); the
#       wrapper calls it DIRECTLY (one JIT invocation, no double-wrap).
#
# NOTE: extracted verbatim from li_backward_impl.py; the reused M2/M4 sub-kernels
# (pypto_s_logits_matmul, pypto_student_softmax, pypto_d_weights,
# pypto_d_s_logits_lp, pypto_d_query_index, pypto_dki_gather, pypto_gather_kki_topk)
# are the same verified ops used by the fused `klloss` kernel there.
# =============================================================================

import weakref

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init

# Memoize the per-call device->host max-kv .item() sync (serializes a bench loop:
# each call would wait behind the prior kernel). Keyed by id(cu_seqlens_k) with a
# weakref finalizer so a GC'd/reused tensor cannot return a stale value.
_MAXKV_CACHE = {}

# --- compile-time constants (only those used by this op) ---
_NQI = 64          # Nqi / N1  (indexer query heads; num_heads_q — 64 only)
_D_IDX = 128       # D_idx / D (indexer hidden; fixed 128)
_TOPK = 512        # default topk (jit parameter; actual value passed at call time)
_CMP_RATIO = 4     # default cmp_ratio (jit parameter)
_BLOCK = 4096      # gather block_size & per-batch view height (>= max kv_len)
_KEFF_BUCKET = 64  # k_eff bucket granularity (recompiles bounded to ceil(topk/64))
_UNROLL_MAX = 4    # per-token loop unroll factor for the k_eff<=128 (dsv) path; wider k_eff falls back to 1 (see kernel note)

_DYN = pypto.DYNAMIC
_STA = pypto.STATIC


# =============================================================================
# Reused PyPTO sub-kernels (M2 student softmax + M4 backward + gather)
# =============================================================================

def pypto_gather_kki_topk(kki_full_b, si_safe, block_table, cur_topk, d_all,
                          k_eff, blk, name_suffix=""):
    """COMBINED gather: row-gather the concatenated [k_full | k_index] (width
    d_all + D_idx) ONCE per token by the sparse-index prefix, materialize into a
    single UB buffer, then return TWO views slicing the hidden axis:
        k_topk  = buf[:, :d_all]            ([cur_topk, D_all])  -> M1 qk
        ki_topk = buf[:, d_all:d_all+D_idx] ([cur_topk, D_idx])  -> M2 s_logits
    Since k and k_index share the SAME `si_safe`, one gather over the wider source
    replaces the two separate gathers (pypto_gather_k_topk + pypto_gather_ki_topk)
    — half the gather launches + half the assemble copies. The two slices are
    plain UB views (no extra copy); the matmuls read them with their own valid
    extents.

    Same -1 handling as the split gathers: gather on the CLAMPED `si_safe`; the
    golden ~valid zeroing is realized downstream by zeroing the qk / s_logits
    column (pypto_valid_row).

    `k_eff` (== bucketed max real_k, a compile-time int) is the UB buffer HEIGHT and
    the declared row count of the two slice views, REPLACING the fixed _TOPK=512: the
    gather reads cur_topk <= k_eff rows (k_eff >= every real_k) so the [k_eff, dw]
    buffer holds them all and the per-token copy/views process k_eff rows instead of
    512 (p0 k_eff=128 -> ~4x smaller)."""
    dw = d_all + _D_IDX                                                 # combined width (704 with rope; 640 no_rope)
    pypto.set_vec_tile_shapes(128, 512)                                # gather stage
    kn = pypto.experimental.gather_in_ub(kki_full_b, si_safe, block_table, blk, -2)  # [cur_topk, dw]
    # UB buffer rows = k_eff (== bucketed max real_k, a compile-time int), NOT
    # _TOPK=512: the gather reads cur_topk <= k_eff rows so the [k_eff, dw] buffer
    # holds every valid row, and the per-token assemble/copy + downstream views
    # process k_eff rows/token instead of 512 (p0 k_eff=128 -> ~4x smaller). cur_topk
    # <= k_eff always (k_eff is the bucketed max real_k) so no valid row is dropped.
    kki_buf = pypto.tensor([k_eff, dw], pypto.DT_BF16,
                           "kki_topk_buf" + name_suffix)               # [k_eff, dw] UB
    pypto.assemble(kn, [0, 0], kki_buf)                                # gathered rows -> buf[0:cur_topk]
    # Column-slice views: the DECLARED inner dim must be the slice width (D_all /
    # D_idx) so the matmul reads the correct K-dim (the buffer row-stride dw is
    # preserved by view). Declaring [k_eff, dw] made the qk matmul see K=704 (->
    # FC3001 kSizeB 704 vs q's 576); declaring the slice width fixes it. The row
    # count is k_eff (the buffer height) with valid_shape limiting to cur_topk.
    k_topk = pypto.view(kki_buf, [k_eff, d_all], [0, 0],
                        valid_shape=[cur_topk, d_all])                 # [cur_topk, D_all] (cols 0:D_all)
    ki_topk = pypto.view(kki_buf, [k_eff, _D_IDX], [0, d_all],
                         valid_shape=[cur_topk, _D_IDX])               # [cur_topk, D_idx] (cols D_all:+D_idx)
    return k_topk, ki_topk


def pypto_s_logits_matmul(qi_t, ki_topk):
    """M2: s_logits = q_index_tok @ ki_topk^T (b_trans) -> [Nqi, cur_topk] fp32.
    NO scale (unlike M1)."""
    pypto.set_cube_tile_shapes([32, 64], [128, 128], [128, 512])        # M2 s_logits (DESIGN.md §5.1)
    s_logits = pypto.matmul(qi_t, ki_topk, pypto.DT_FP32, b_trans=True)  # [Nqi, cur_topk] fp32
    return s_logits


def pypto_student_softmax(s_logits, w_col, k_eff):
    """M2: relu(s_logits), weighted group-SUM Nqi->Nk(1) (NOT averaged) folded as
    a matmul, then softmax over the cur_topk axis.  s_logits: [Nqi, cur_topk]
    fp32, w_col: [Nqi, 1] fp32 (shared with M4) -> (s_row [1, cur_topk] fp32,
    relu_s [Nqi, cur_topk] fp32). relu_s is RETURNED for M4 reuse (no re-compute).

    `k_eff` is the static cur_topk-axis tile WIDTH (== bucketed max real_k): the
    relu (R,128) tile uses min(128, k_eff) and the [1, cur_topk] student-softmax tile
    (was the full 512-wide) uses k_eff directly — sizing these single-row chains to
    k_eff instead of 512 removes the p0 over-compute (this stage was ~48% of vector).

    Mirrors golden_stage_m2_student_s over the valid [0, cur_topk) extent: relu ->
    * w.unsqueeze(-1) -> sum(dim=Nqi) (SUM) -> safe-softmax (amax/sub/exp/sum/div).

    Weighted group-reduce via matmul, NOT broadcast-mul+sum: the
    `relu_s * w.unsqueeze(-1)` broadcast against a full-valid [Nqi,1] does NOT
    carry relu_s's [Nqi, cur_topk] valid extent through the broadcast — it reads
    the [cur_topk, 128) suffix (uninit matmul region) and poisons the valid
    columns. The reduce `s_reduce[s] = sum_j relu_s[j,s] * w[j]` is EXACTLY the
    matmul `w[1,Nqi] @ relu_s[Nqi, cur_topk] -> [1, cur_topk]`, which fuses the
    weight multiply + group-SUM in one cube op with NO broadcast and respects
    relu_s's cur_topk valid extent on the N axis. Matches golden bit-for-bit."""
    kt128 = min(128, k_eff)                                             # (R,128) tile width
    pypto.set_vec_tile_shapes(64, kt128)                                # relu [Nqi=64, cur_topk] (all 64 M-rows / 1 tile)
    relu_s = pypto.relu(s_logits)                                       # [Nqi, cur_topk] fp32
    # weighted group-SUM Nqi->Nk(1) as a VECTOR reduce (NOT a cube matmul): the
    # matmul w_f[1,Nqi] @ relu_s[Nqi,cur_topk] has M=1 -> wastes 15/16 of the cube
    # systolic array and was the #1 time sink (23% of kernel, leaf L267). The reduce
    # `s_reduce[s] = sum_j relu_s[j,s] * w[j]` is realized SAFELY by anchoring on
    # relu_s (which CARRIES the cur_topk valid extent on its N axis): multiply
    # relu_s[Nqi,cur_topk] by w_col[Nqi,1] (w broadcasts over the TRAILING cur_topk
    # axis; combine_axis inlines it; result inherits relu_s's cur_topk valid
    # extent so NO suffix poisoning — same pattern as pypto_d_weights) then
    # sum(dim=0). The rejected broadcast in the DESIGN was `relu_s * w.unsqueeze(-1)`
    # against a FULL-valid [Nqi,1] divorced from relu_s; anchoring the mul on
    # relu_s (the valid-extent operand) is the safe form. w_col is built ONCE in
    # the caller and SHARED with M4 (d_s_logits) — avoids the duplicate
    # [1,Nqi]->[Nqi,1] reshape-transpose (was leaf L273, ~15% of kernel time).
    rs_w = pypto.mul(relu_s, w_col)                                    # [Nqi, cur_topk] (relu_s anchors extent)
    s_reduce = pypto.sum(rs_w, dim=0, keepdim=True)                    # [1, cur_topk] fp32 (SUMMED)
    pypto.set_vec_tile_shapes(1, k_eff)                               # student softmax is [1, cur_topk] (1 row;
    #                                                                  # static width k_eff, not 512 -> p0 ~4x less)
    s_max = pypto.amax(s_reduce, dim=-1, keepdim=True)                  # [1, 1]
    s_exp = pypto.exp(pypto.sub(s_reduce, s_max))                       # [1, cur_topk]
    s_sum = pypto.sum(s_exp, dim=-1, keepdim=True)                      # [1, 1]
    s_row = pypto.div(s_exp, s_sum)                                     # [1, cur_topk]
    return s_row, relu_s


def pypto_d_weights(relu_s, ds, k_eff):
    """M4: d_weights[t,:] = sum_s relu_s[j,s] * ds[s]  (einsum 'js,s->j').
    relu_s: [Nqi, cur_topk] fp32, ds: [1, cur_topk] fp32 -> dw [Nqi, 1] fp32.
    NO bf16 round-trip (golden L353-356).

    Realized as a vec reduce (NOT a matmul): the d_weights matmul output N=1 is
    not 16-element-aligned (FC4001 ERR_CONFIG_ALIGNMENT on cube). The contraction
    `sum_s relu_s[j,s] * ds[s]` is `relu_s[Nqi,cur_topk] * ds[1,cur_topk]`
    (ds broadcasts over the LEADING Nqi axis, last axis matched -> the SAFE
    broadcast direction; result inherits relu_s's cur_topk valid extent) then
    `sum(dim=-1)`. This avoids both the cube N=1 alignment failure and the M2
    [Nqi,1] last-axis broadcast poisoning.

    `k_eff` is the static cur_topk-axis tile WIDTH (== bucketed max real_k); the
    (R,128) tile uses min(128, k_eff)."""
    pypto.set_vec_tile_shapes(64, min(128, k_eff))                    # [Nqi=64, cur_topk] — all 64 M-rows in 1 M-tile
    rs_ds = pypto.mul(relu_s, ds)                                      # [Nqi, cur_topk] (leading-axis bcast)
    dw = pypto.sum(rs_ds, dim=-1, keepdim=True)                       # [Nqi, 1] fp32
    return dw


def pypto_d_s_logits_lp(relu_s, ds, w_col, k_eff):
    """M4: d_s_logits = ds * weights * 1{relu_s>0}, then the mandatory bf16
    truncation (golden L359-361; DESIGN.md §3.5).  relu_s: [Nqi, cur_topk] fp32,
    ds: [1, cur_topk] fp32, w_col: [Nqi, 1] fp32 -> d_s_logits_bf16 [Nqi, cur_topk]
    BF16.

    `k_eff` is the static cur_topk-axis tile WIDTH (== bucketed max real_k); the
    (R,128) tile uses min(128, k_eff).

    SUFFIX-CORRUPTION-SAFE construction (DESIGN.md §3.2 / M2 gotcha): `gate`
    inherits relu_s's [Nqi, cur_topk] valid extent (the M2 s_logits matmul N
    axis). Anchor every elementwise result on a cur_topk-valid tensor:
      gate = cast(relu_s > 0, FP32)                 [Nqi, cur_topk] valid cur_topk
      gd   = gate * ds  (ds[1,cur_topk] broadcasts over the LEADING Nqi axis,
             last axis matched -> SAFE; result inherits gate's cur_topk extent)
      d_s_logits = gd * w_col   ([Nqi,1] broadcast applied to the already
             cur_topk-valid gd so the broadcast spans only valid cols)
    Then the mandatory bf16 truncation with explicit pypto.cast (no .to()).

    The golden spec's d_s_logits is a bf16 ROUND-TRIP (cast to bf16 then back to
    fp32) before the two M4 matmuls. Since the value already carries ONLY bf16
    precision, we stop the round-trip at BF16 and feed the bf16 tensor directly to
    the M4 matmuls as a bf16+bf16->fp32 cube op (fp32 accumulation preserved via
    out_dtype). The matmul operands carry IDENTICAL bf16-precision bits to the old
    fp32-upcast operands, so this is spec-safe (data precision unchanged; only the
    cube input dtype changes fp32->bf16, ~3.9x fewer cube cycles). Verified within
    grad tol (atol 2e-3)."""
    pypto.set_vec_tile_shapes(64, min(128, k_eff))                    # [Nqi=64, cur_topk] — all 64 M-rows in 1 M-tile
    # gate = 1{relu_s>0} as fp32. A2A3 has NO BOOL->FP32 cast, so build the fp32
    # gate directly with where(cond, 1.0, 0.0); the bool from greater stays
    # internal to where. gate inherits relu_s's [Nqi, cur_topk] valid extent.
    gate = pypto.where(pypto.greater(relu_s, 0.0), 1.0, 0.0)           # [Nqi, cur_topk] fp32 valid cur_topk
    gd = pypto.mul(gate, ds)                                           # [Nqi, cur_topk] (gate anchors extent)
    d_s_logits = pypto.mul(gd, w_col)                                 # [Nqi, cur_topk]  * weights[Nqi,1]
    d_s_logits_bf16 = pypto.cast(d_s_logits, pypto.DT_BF16)            # bf16 truncation (mandatory spec precision)
    return d_s_logits_bf16


def pypto_d_query_index(d_s_logits_bf16, ki_topk):
    """M4: d_query_index[t,j,d] = sum_s d_s_logits_lp[j,s] * ki_topk[s,d]
    (einsum 'js,sd->jd').  d_s_logits_bf16: [Nqi, cur_topk] BF16, ki_topk:
    [cur_topk, D_idx] BF16 -> matmul (bf16+bf16->bf16) -> [Nqi, D_idx] BF16
    (golden L365).  bf16 operands, fp32 accumulation, BF16 store (out_dtype=DT_BF16):
    the final dq is bf16 anyway (host casts to q dtype == bf16), so storing bf16 is
    numerically identical to fp32-store-then-host-cast and halves the writeback DMA.
    Cube L0 is 16-aligned (64/128/128) so the bf16 tile is legal."""
    pypto.set_cube_tile_shapes([64, 64], [128, 512], [128, 128])       # M4 d_query_index (DESIGN.md §5.1)
    dqi = pypto.matmul(d_s_logits_bf16, ki_topk, pypto.DT_BF16)        # [Nqi, D_idx] bf16 out (fp32 accum; final dq is bf16 anyway)
    return dqi


def pypto_dki_gather(d_s_logits_bf16, qi_t):
    """M4: dki_gather[t,s,d] = sum_j d_s_logits_lp[j,s] * q_index[j,d]
    (einsum 'js,jd->sd').  d_s_logits_bf16: [Nqi, cur_topk] BF16, qi_t:
    [Nqi, D_idx] BF16 -> matmul(a_trans) (bf16+bf16->fp32) -> [cur_topk, D_idx]
    fp32 (golden L372-376). The golden's `* valid` masking + scatter is applied
    on the host (li_backward_wrapper); for canonical cases the prefix is all-valid so
    it is a no-op.  bf16 operands, fp32 accumulation (out_dtype=DT_FP32). Cube L0
    is 16-aligned (128/64/128) so the bf16 tile is legal."""
    pypto.set_cube_tile_shapes([128, 512], [64, 64], [128, 128])       # M4 dki_gather (DESIGN.md §5.1)
    dki = pypto.matmul(d_s_logits_bf16, qi_t, pypto.DT_FP32, a_trans=True)  # [cur_topk, D_idx] fp32 (bf16 inputs)
    return dki



# =============================================================================
# CANN sparse_lightning_indexer_kl_loss_grad kernel + wrapper
# =============================================================================

def _emit_klgrad_token(qt, real_k, name_suffix, *, ki_full_b, block_table,
                       q_index_full, w_col_full, p_input, sparse_indices,
                       d_query_index, d_weights, dki_out, gidx_out, softmax_out,
                       ks, kv_len, k_eff, topk, blk):
    """Emit the per-token M2(student)+backward chain for the CANN
    sparse_lightning_indexer_kl_loss_grad op (global query row `qt`, valid count
    `real_k`). Mirrors `_emit_token` but: NO M1 teacher recompute (p is input),
    ds = y*Sum(p) - p, writes softmax_out (y) instead of a loss partial."""
    kt128 = min(128, k_eff)                               # (R,128) tile width
    # sparse-index prefix -> [1, k_eff] valid [1, real_k] (cols [real_k, k_eff) are
    # -1 padding); k_eff <= topk so the view never exceeds sparse_indices' width.
    si_t = pypto.view(sparse_indices, [1, k_eff], [qt, 0],
                      valid_shape=[1, real_k])            # [1, real_k] int32
    # OOB-safe -1 / out-of-range handling (golden valid=(si>=0)&(si<kv_len);
    # safe_id=clamp(si,0,kv_len-1)). Pure int-clip arithmetic (no bool compare).
    pypto.set_vec_tile_shapes(1, k_eff)                   # [1, real_k] int tile
    lower = pypto.clip(pypto.add(si_t, 1), 0, 1)          # 1 iff si>=0
    kvlen_bc = pypto.full([1, k_eff], kv_len, pypto.DT_INT32, valid_shape=[1, real_k])
    upper = pypto.clip(pypto.sub(kvlen_bc, si_t), 0, 1)   # 1 iff si<kv_len
    valid_i = pypto.clip(pypto.add(pypto.add(lower, upper), -1), 0, 1)  # AND
    valid_row = pypto.cast(valid_i, pypto.DT_FP32)        # [1, real_k] fp32 (1/0)
    si_lower = pypto.maximum(si_t, 0)                     # clamp -1 -> 0
    kvm1_bc = pypto.full([1, k_eff], kv_len - 1, pypto.DT_INT32, valid_shape=[1, real_k])
    over = pypto.maximum(pypto.sub(si_lower, kvm1_bc), 0)
    si_safe = pypto.sub(si_lower, over)                   # min(si_lower, kv_len-1)
    # ===== gidx (global key id) IN-KERNEL: gidx = cu_kv[b] + safe_id =====
    ks_bc = pypto.full([1, k_eff], ks, pypto.DT_INT32, valid_shape=[1, real_k])
    gidx_tok = pypto.add(si_safe, ks_bc)                  # [1, real_k] int32
    pypto.assemble(gidx_tok, [qt, 0], gidx_out)           # -> gidx_out[qt, 0:real_k]
    pypto.set_vec_tile_shapes(1, k_eff)                   # restore int tile for gather
    # ===== gather k_index_topk via the VERIFIED klloss combined-gather path =====
    # gather_in_ub CORRUPTS a narrow (< the 512-wide stage tile) source: it returns
    # garbage rows once real_k >= ~16, non-deterministically (a single k_index-only
    # 128- or 256-wide source fails; klloss never hits it because its [k_full|k_index]
    # source is 640/704 wide). So the host builds a 640-wide source (5x k_index; the
    # klloss no-rope width) and this reads the LAST D_idx slice (cols [512:640]) — the
    # exact byte-layout klloss's ki_topk uses. k_eff (the gather buffer height) is
    # forced >= 128 in the wrapper (the stage tile M is 128; a shorter buffer hangs).
    _, ki_topk = pypto_gather_kki_topk(ki_full_b, si_safe, block_table,
                                       real_k, 4 * _D_IDX, k_eff, blk, name_suffix)  # ki = cols [512:640]
    # ===== M2 student softmax y (+ relu_s for backward) =====
    qi_t = pypto.view(q_index_full, [_NQI, _D_IDX], [qt * _NQI, 0],
                      valid_shape=[_NQI, _D_IDX])          # [Nqi, D_idx] bf16
    s_logits = pypto_s_logits_matmul(qi_t, ki_topk)        # [Nqi, real_k] fp32 (NO scale)
    pypto.set_vec_tile_shapes(64, kt128)                   # [Nqi, real_k] mask tile
    s_logits = pypto.mul(s_logits, valid_row)              # -1 / OOB cols -> 0
    # w_col [Nqi,1] fp32 — host-preshaped to a contiguous [T1*N1, 1] fp32 column
    # (row qt*Nqi+j == weights[qt, j]) so the per-token access is a plain [Nqi,1]
    # view: NO in-kernel reshape+cast. The old in-kernel [1,Nqi]->[Nqi,1] reshape
    # + fp32 cast was ~23% of ALL on-core vector time (it sat on the softmax
    # critical path); the host reshape of an already-fp32 [T1,N1] tensor to a
    # [T1*N1,1] column preserves head order exactly (each head is its own row),
    # so w_col[j,0] == weights[qt,j] with no head mis-permute. SHARED by the
    # student group-SUM and the d_s_logits gate.
    w_col = pypto.view(w_col_full, [_NQI, 1], [qt * _NQI, 0],
                       valid_shape=[_NQI, 1])              # [Nqi, 1] fp32
    s_row, relu_s = pypto_student_softmax(s_logits, w_col, k_eff)  # y [1,real_k], relu_s [Nqi,real_k]
    # ===== softmax_out = y (student indexer softmax) =====
    pypto.set_vec_tile_shapes(1, k_eff)                    # write-back tile
    pypto.assemble(s_row, [qt, 0], softmax_out)            # -> softmax_out[qt, 0:real_k] (stride topk)
    # ===== dI = y * ReduceSum(p) - p  (p is the attn_softmax_l1_norm INPUT) =====
    p_row = pypto.view(p_input, [1, k_eff], [qt, 0],
                       valid_shape=[1, real_k])            # [1, real_k] fp32 (topk-wide src)
    pypto.set_vec_tile_shapes(1, k_eff)                    # [1, real_k] tile
    p_row = pypto.mul(p_row, valid_row)                    # mask -1 / OOB cols -> 0 (== golden)
    p_reduce = pypto.sum(p_row, dim=-1, keepdim=True)      # [1, 1] fp32 = Sum_s p
    pypto.set_vec_tile_shapes(1, kt128)                    # ds tile
    ds = pypto.sub(pypto.mul(s_row, p_reduce), p_row)      # [1, real_k] fp32 (y*Sum(p) - p)
    # ===== backward =====
    dw = pypto_d_weights(relu_s, ds, k_eff)                # [Nqi, 1] fp32
    pypto.set_vec_tile_shapes(_NQI, 1)                     # d_weights write-back (column layout)
    pypto.assemble(dw, [qt * _NQI, 0], d_weights)          # -> d_weights[qt*Nqi : +Nqi, 0] (no reshape)
    d_s_logits_bf16 = pypto_d_s_logits_lp(relu_s, ds, w_col, k_eff)  # [Nqi, real_k] bf16 (round to input dtype)
    dqi = pypto_d_query_index(d_s_logits_bf16, ki_topk)    # [Nqi, D_idx] bf16 (fp32 accum; final dq is bf16)
    pypto.set_vec_tile_shapes(64, 128)                     # dq write-back
    pypto.assemble(dqi, [qt * _NQI, 0], d_query_index)     # -> d_query_index[qt*Nqi : +Nqi]
    dki = pypto_dki_gather(d_s_logits_bf16, qi_t)          # [real_k, D_idx] fp32
    pypto.set_vec_tile_shapes(128, 128)                    # dki write-back
    pypto.assemble(dki, [qt * k_eff, 0], dki_out)          # -> dki_out[qt*k_eff : +real_k] (packed stride k_eff)


@pypto.frontend.jit(pass_options={"cube_l1_reuse_setting": {-1: 8, 0: 16},
                                  "vec_nbuffer_setting": {"DEFAULT": 1},
                                  "cube_nbuffer_setting": {"DEFAULT": 1}},
                    runtime_options={"run_mode": pypto.RunMode.NPU,
                                     # stitch_function_max_num: 32 (was 64). Workspace
                                     # scales ~linearly with this (it multiplies the
                                     # per-outcast slot count); halving it halves the
                                     # workspace (dsv 3.66GB -> 1.88GB) for ~+0.25ms.
                                     # Combined with the shape-sized `blk`, keeps dsv
                                     # under 2GB while staying ~5.6ms (<6ms).
                                     "stitch_function_max_num": 32,
                                     "device_sched_mode": 0,
                                     "launch_sched_aicpu_num": 3})
def sparse_li_klgrad_kernel(
    q_index_full: pypto.Tensor([_DYN, _STA], pypto.DT_BF16),     # [T1*N1, D_idx=128]
    k_index_full: pypto.Tensor([_DYN, _STA], pypto.DT_BF16),     # [T2, 5*D_idx=640] (padded gather source)
    w_col_full: pypto.Tensor([_DYN, _STA], pypto.DT_FP32),       # [T1*N1, 1] host-preshaped fp32 col (was per-token reshape+cast)
    p_input: pypto.Tensor([_DYN, _STA], pypto.DT_FP32),          # [T1, topk] fp32 (attn_softmax_l1_norm)
    sparse_indices: pypto.Tensor([_DYN, _STA], pypto.DT_INT32),  # [T1, topk]
    cu_q: pypto.Tensor([_DYN], pypto.DT_INT32),                  # [B+1]
    cu_kv: pypto.Tensor([_DYN], pypto.DT_INT32),                 # [B+1]
    cmp_residual: pypto.Tensor([_DYN], pypto.DT_INT32),          # [B] (0 when mask_mode!=3 or cmp_ratio==1)
    block_table: pypto.Tensor([_STA, _STA], pypto.DT_INT32),     # [1, 1] identity
    d_query_index: pypto.Tensor([_DYN, _STA], pypto.DT_BF16),    # [T1*N1, D_idx]  OUT bf16 (final dq is bf16 anyway; halves writeback DMA)
    d_weights: pypto.Tensor([_DYN, _STA], pypto.DT_FP32),        # [T1*N1, 1]      OUT fp32 (col layout; host reshapes to [T1,N1])
    dki_out: pypto.Tensor([_DYN, _STA], pypto.DT_FP32),          # [T1*k_eff, D_idx] OUT (packed stride k_eff)
    gidx_out: pypto.Tensor([_DYN, _STA], pypto.DT_INT32),        # [T1, k_eff]     OUT global key id
    softmax_out: pypto.Tensor([_DYN, _STA], pypto.DT_FP32),      # [T1, topk]      OUT student y (fp32)
    k_eff: int = _TOPK,                                          # NON-TENSOR: packed dki stride (== bucketed max real_k)
    cmp_ratio: int = _CMP_RATIO,                                 # NON-TENSOR: compression ratio
    topk: int = _TOPK,                                           # NON-TENSOR: sparse-index / p / softmax_out stride
    mask_mode: int = 3,                                          # NON-TENSOR: 0 (no mask) or 3 (rightDownCausal)
    blk: int = _BLOCK,                                          # NON-TENSOR: gather view height / block_size (>= T2; shape-dep)
):
    """CANN sparse_lightning_indexer_kl_loss_grad kernel (single @jit entry;
    shape-DYNAMIC on all token axes, specialized only per non-tensor
    (k_eff, cmp_ratio, topk, mask_mode) tuple). Per batch b, per query token t,
    emit the student-softmax + indexer-backward chain."""
    pypto.experimental.set_operation_options(combine_axis=True)
    k_index_full.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    pypto.set_vec_tile_shapes(128, 512)
    B = cu_q.shape[0] - 1                                        # SymbolicScalar (#batches)
    for b in pypto.loop(0, B, 1, name="LOOP_batch", idx_name="bIdx"):
        qs = cu_q[b]
        qe = cu_q[b + 1]
        q_len = qe - qs
        ks = cu_kv[b]
        ke = cu_kv[b + 1]
        kv_len = ke - ks
        cmp_res_b = cmp_residual[b]                              # SymbolicScalar cmp_residual_k[b]
        pypto.set_vec_tile_shapes(128, 512)
        # CANN's right-down mask (mask_mode==3) can expose one boundary key from the next
        # packed sequence — gather one extra key (bounds-clamped).
        if mask_mode == 3:
            available_kv_len = pypto.min(kv_len + 1, k_index_full.shape[0] - ks)
        else:
            available_kv_len = kv_len
        gather_ks = pypto.min(ks, k_index_full.shape[0] - 1)
        gather_kv_len = pypto.max(available_kv_len, 1)
        # per-batch 640-wide gather source [BLOCK, 5*D_idx] @ [ks, 0] (see gather note).
        ki_full_b = pypto.view(k_index_full, [blk, 5 * _D_IDX], [gather_ks, 0],
                               valid_shape=[gather_kv_len, 5 * _D_IDX])
        # k_eff-dependent unroll (k_eff is a concrete int per specialization).
        # Unrolling interleaves independent token chains to fill the ~55% dispatch/
        # dependency bubbles, but the per-token UB (relu_s/d_s/gather buf all scale
        # with k_eff) and a tiling-boundary corruption at k_eff==192 (1.5x the 128
        # vec-tile: unroll>1 gives wrong grads there) mean the 4-way unroll is only
        # safe at k_eff<=128 (== the dsv path, 1.35x). All wider k_eff use unroll=1
        # (the baseline-verified path). k_eff is always >=128 (wrapper clamp).
        _unroll_n = _UNROLL_MAX if k_eff <= 128 else 1
        for t_base, unroll_loop in pypto.loop_unroll(
                0, q_len, 1, name="LOOP_qtoken", idx_name="tIdx",
                unroll_list=[_unroll_n]):
            for i in range(unroll_loop):
                t = t_base + i
                # CANN causal valid count (ref valid_k_count):
                if mask_mode == 0:
                    real_k = kv_len.min(topk)                    # No mask: all keys (capped topk)
                else:
                    pre = kv_len * cmp_ratio + cmp_res_b         # pre_compress_k_len
                    real_k = ((pre - q_len + t + 1) // cmp_ratio).max(0).min(available_kv_len).min(topk)
                if real_k > 0:                                   # all-invalid prefix guard
                    qt = qs + t
                    _emit_klgrad_token(qt, real_k, "_u" + str(i),
                                       ki_full_b=ki_full_b, block_table=block_table,
                                       q_index_full=q_index_full, w_col_full=w_col_full,
                                       p_input=p_input, sparse_indices=sparse_indices,
                                       d_query_index=d_query_index, d_weights=d_weights,
                                       dki_out=dki_out, gidx_out=gidx_out,
                                       softmax_out=softmax_out,
                                       ks=ks, kv_len=available_kv_len, k_eff=k_eff, topk=topk, blk=blk)


def _bsnd_to_tnd(x):
    """[B, S, N, D] -> [B*S, N, D] flat TND; returns (flat, B, S)."""
    B, S, N, D = x.shape
    return x.reshape(B * S, N, D).contiguous(), B, S


def sparse_lightning_indexer_kl_loss_grad(
        q, k, w, sparse_indices, attn_softmax_l1_norm, *,
        cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
        cmp_residual_k=None, metadata=None,
        layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=1,
        max_seqlen_q=None, max_seqlen_k=None):
    """PyPTO drop-in for CANN `cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad`.

    Same inputs / outputs as the torch op (docs/zh/sparse_lightning_indexer_kl_loss_grad.md).
    Calls `sparse_li_klgrad_kernel` DIRECTLY (one JIT invocation; no double-wrap);
    the kernel is shape-DYNAMIC on every token axis (recompiles only per distinct
    (k_eff, cmp_ratio, topk, mask_mode)).

    Args (Lightning-Indexer branch only; N2==1, D==128, N1==64):
      q                    [T1,N1,D] / [B,S1,N1,D]  fp16/bf16  indexer query
      k                    [T2,1,D]  / [B,S2,1,D]   fp16/bf16  indexer key
      w                    [T1,N1]   / [B,S1,N1]    fp32       per-head weights
      sparse_indices       [T1,1,K]  / [B,S1,1,K]   int32      top-k key ids (-1 pad)
      attn_softmax_l1_norm [T1,1,K]  / [B,S1,1,K]   fp32       teacher distribution p
      cu_seqlens_q/k       [B+1] int32 (required for TND)
      cmp_residual_k       [B] int32 | None (mask_mode=3 & cmp_ratio!=1)
      mask_mode            0 (no mask) | 3 (rightDownCausal)
      cmp_ratio            key compression ratio [1,128]
      seqused_q/seqused_k/metadata/max_seqlen_* : accepted; not needed by this path.

    Returns (dq, dk, dw, softmax_out):
      dq  grad q, shape+dtype == q
      dk  grad k, shape+dtype == k (scatter-add over the key axis)
      dw  grad w, shape == w, dtype fp32
      softmax_out  indexer branch softmax y, shape == attn_softmax_l1_norm, fp32
    """
    dev = q.device
    q_dtype, k_dtype = q.dtype, k.dtype

    # ---- BSND -> TND normalization (synthesize uniform cu_seqlens) ----
    if layout_q.upper() == "BSND":
        B, S1 = q.shape[0], q.shape[1]
        q = q.reshape(B * S1, q.shape[2], q.shape[3])
        w = w.reshape(B * S1, w.shape[-1])
        sparse_indices = sparse_indices.reshape(B * S1, sparse_indices.shape[-2], sparse_indices.shape[-1])
        attn_softmax_l1_norm = attn_softmax_l1_norm.reshape(
            B * S1, attn_softmax_l1_norm.shape[-2], attn_softmax_l1_norm.shape[-1])
        cu_seqlens_q = torch.arange(0, (B + 1) * S1, S1, dtype=torch.int32, device=dev)
    if layout_k.upper() == "BSND":
        Bk, S2 = k.shape[0], k.shape[1]
        k = k.reshape(Bk * S2, k.shape[2], k.shape[3])
        cu_seqlens_k = torch.arange(0, (Bk + 1) * S2, S2, dtype=torch.int32, device=dev)

    T1, N1, D_idx = q.shape                                      # [T1, N1, 128]
    T2 = k.shape[0]
    topk = sparse_indices.shape[-1]
    assert D_idx == _D_IDX, f"D must be {_D_IDX}; got {D_idx}"
    assert N1 == _NQI, (f"this PyPTO drop-in currently supports N1(num_heads_q)=={_NQI} "
                        f"(the deployment / test value); got {N1}")
    assert k.shape[-2] == 1, "N2 (key heads) must be 1"

    if T2 == 0:
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(w, dtype=torch.float32),
            torch.zeros_like(attn_softmax_l1_norm, dtype=torch.float32),
        )

    assert cu_seqlens_q is not None and cu_seqlens_k is not None, \
        "cu_seqlens_q / cu_seqlens_k required (TND)"
    cu_q_i = cu_seqlens_q.contiguous().to(torch.int32)          # [B+1]
    cu_kv_i = cu_seqlens_k.contiguous().to(torch.int32)         # [B+1]
    B = cu_q_i.numel() - 1

    # cmp_residual_k -> [B] int32 tensor (0 when not provided / not needed).
    if cmp_residual_k is None:
        cmp_res_i = torch.zeros(B, dtype=torch.int32, device=dev)
    else:
        cmp_res_i = cmp_residual_k.contiguous().to(torch.int32)
        if cmp_res_i.numel() != B:
            cmp_res_i = cmp_res_i.reshape(-1)[:B]

    # ---- packed dki stride k_eff = bucket(min(max_kv_len, topk)) (>= every real_k).
    # CANN's right-down mask can expose one boundary key from the next packed sequence.
    # Compile-time int; bucketed to a multiple of _KEFF_BUCKET so recompiles stay bounded. ----
    if max_seqlen_k is not None and int(max_seqlen_k) > 0:
        max_kv_len = int(max_seqlen_k)
    elif T2 > 0:
        _kkey = id(cu_seqlens_k)
        _cached = _MAXKV_CACHE.get(_kkey)
        if _cached is not None:
            max_kv_len = _cached                       # cache hit: no device->host sync
        else:
            max_kv_len = int((cu_kv_i[1:] - cu_kv_i[:-1]).max().item())
            _MAXKV_CACHE[_kkey] = max_kv_len
            weakref.finalize(cu_seqlens_k, _MAXKV_CACHE.pop, _kkey, None)
    else:
        max_kv_len = 0
    max_real_k = max_kv_len + int(mask_mode == 3)
    k_min = max(1, min(max_real_k, topk))
    # k_eff >= 128: the gather stage tile M is 128, so the per-token gather buffer
    # height (k_eff) must be >= 128 or the assemble overruns / the kernel hangs
    # (k_eff=64 hangs). topk >= 128 always in practice; clamp to [128, topk].
    k_eff = min(max(128, ((k_min + _KEFF_BUCKET - 1) // _KEFF_BUCKET) * _KEFF_BUCKET), topk)

    # ---- flatten to the kernel ABI (2D) ----
    qi_2d = q.reshape(T1 * N1, D_idx).contiguous().to(torch.bfloat16)   # [T1*N1, D_idx]
    ki_2d = k.reshape(T2, D_idx).contiguous().to(torch.bfloat16)        # [T2, D_idx]
    # 640-wide gather source (5x k_index): gather_in_ub corrupts narrow (<512-wide)
    # sources for real_k>=16 (see the in-kernel gather note), so pad to klloss's
    # no-rope width. The kernel reads the last D_idx slice as k_index_topk.
    kki_2d = torch.cat([ki_2d] * 5, dim=-1).contiguous()               # [T2, 5*D_idx=640]
    # w pre-shaped host-side to a contiguous fp32 [T1*N1, 1] column (row qt*N1+j ==
    # w[qt, j]) so the kernel reads w_col via a plain view — removes the per-token
    # [1,N1]->[N1,1] reshape + fp32 cast that was ~23% of on-core vector time.
    w_col_2d = w.reshape(T1 * N1, 1).contiguous().to(torch.float32)     # [T1*N1, 1] fp32
    si_2d = sparse_indices.reshape(T1, topk).contiguous().to(torch.int32)   # [T1, topk]
    p_2d = attn_softmax_l1_norm.reshape(T1, topk).contiguous().to(torch.float32)  # [T1, topk]
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    # `blk` = per-batch gather-view height / gather block_size, passed as a jit int
    # (specializes per shape). It is the SINGLE biggest driver of the GM workspace:
    # the dynamic-offset gather view [blk, 5*D_idx] materializes as an outcast slot,
    # and workspace ~= (blk * 5*D_idx * 2) * (#outcast slots, which grows with
    # stitch*unroll). The old FIXED blk=4096 blew the workspace to ~14.5GB on dsv;
    # sizing it to align_up(T2) drops that ~4x (dsv -> ~3.6GB, and with stitch=32
    # -> ~1.88GB / 1.75GiB). A hard **1024 floor** is required: a smaller blk
    # silently corrupts (real_k>=~16) or hard-crashes (507015 invalid GM addr) the
    # gather on multi-batch shapes. blk must be >= T2 (total keys); small shapes
    # have tiny T1 so the 1024 floor costs them nothing.
    blk = max(1024, ((T2 + 1 + 127) // 128) * 128)

    # ---- output buffers ----
    # d_query_index / d_weights / softmax_out ARE pre-zeroed: the kernel skips
    # real_k==0 tokens, whose grads must read as 0.
    d_query_index_2d = torch.zeros((T1 * N1, D_idx), dtype=torch.bfloat16, device=dev)
    d_weights_2d = torch.zeros((T1 * N1, 1), dtype=torch.float32, device=dev)  # col layout (host reshapes to w.shape)
    # dki_out padding (>=real_k) and real_k==0 rows are NOT written by the kernel; we
    # route them to a dump key T2 (gidx pre-filled with T2) so their garbage lands in a
    # discarded row -> dki_out needs no zeroing (saves a T1*k_eff*D_idx memset, ~268MB).
    dki_out_2d = torch.empty((T1 * k_eff, D_idx), dtype=torch.float32, device=dev)
    gidx_out_2d = torch.full((T1, k_eff), T2, dtype=torch.int32, device=dev)  # padding -> dump row T2
    # softmax_out PACKED at k_eff (not topk): keeps every per-token UB tile k_eff-
    # bounded like the rest of the kernel (a topk-wide per-token write blew the UB
    # budget -> corrupted dk/dw/softmax at larger shapes). Host unpacks into the
    # topk-wide output; the valid region [0, real_k) (real_k <= k_eff) is exact,
    # suffix stays 0.
    softmax_out_2d = torch.zeros((T1, k_eff), dtype=torch.float32, device=dev)

    sparse_li_klgrad_kernel(
        qi_2d, kki_2d, w_col_2d, p_2d, si_2d, cu_q_i, cu_kv_i, cmp_res_i, block_table,
        d_query_index_2d, d_weights_2d, dki_out_2d, gidx_out_2d, softmax_out_2d,
        k_eff, cmp_ratio, topk, mask_mode, blk)                # ONE call; non-tensor jit params

    # ---- dk scatter-add (ONE global torch index_add_); row T2 is the garbage dump ----
    gidx = gidx_out_2d.to(torch.int64)                         # [T1, k_eff] (cu_kv[b]+safe_id; pad=T2)
    d_key_index_ext = torch.zeros((T2 + 1, D_idx), dtype=torch.float32, device=dev)
    d_key_index_ext.index_add_(0, gidx.reshape(-1), dki_out_2d)  # accumulate; pad garbage -> row T2
    d_key_index_2d = d_key_index_ext[:T2]

    dq = d_query_index_2d.reshape(q.shape).to(q_dtype)         # grad q (q dtype)
    dk = d_key_index_2d.reshape(k.shape).to(k_dtype)           # grad k (k dtype)
    dw = d_weights_2d.reshape(w.shape)                         # grad w (fp32)
    # unpack the k_eff-packed student softmax into the topk-wide output (suffix 0).
    softmax_out_full = torch.zeros((T1, topk), dtype=torch.float32, device=dev)
    softmax_out_full[:, :k_eff] = softmax_out_2d
    softmax_out = softmax_out_full.reshape(attn_softmax_l1_norm.shape)  # student y (fp32)
    return dq, dk, dw, softmax_out
