"""Real CP dispatch code: planner + dispatcher, verified on CPU.

The single-process tests run the real ``build_cp_plan`` /
``CPTokenDispatcher`` (with the mock exchange override) /
``Compressor`` CP branch against the oracle semantics of
``test_cp_compressor`` (the experiment), with the collectives' semantics
reproduced call-indexed.  The multi-process test runs the same code over a
real gloo process group via ``cp_dispatch_worker.py`` — the actual
``torch.distributed`` path (the ``spmd_types`` collectives are the NPU
path, on-device; the portable gloo emulation lives in the worker).

Also exercises the AscendC handler's CP path (with the fake ``cann_ops``
recorder) and the ``CPAttention`` flow with a recording inner core.
"""

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import test_cp_compressor as tcc
import torch
import torch.nn as nn
from test_cp_compressor import (
    FakeMesh,
    _ht,
    build_compressor,
    compress_blocks,
    make_linear,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import compressor as comp_mod
from torchtitan_npu.models.deepseek_v4 import metadata as meta_mod
from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp_mod
from torchtitan_npu.models.deepseek_v4.attention import Attention
from torchtitan_npu.models.deepseek_v4.token_dispatcher import (
    build_cp_plan,
    segment_structure,
)

tcc.comp_mod = comp_mod

torch.manual_seed(0)
DIM, HD, RD = 8, 16, 4
_REPO = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# The mock transport (single-process faithful stand-in)
# ---------------------------------------------------------------------------


class MockRegistry:
    """Shared per-rank payload + send-splits store for the single-process
    mock (the splits let the transport locate each sender's group for the
    receiver — the information the production all_to_all derives from the
    real input splits)."""

    def __init__(self, world: int):
        self.payloads: list[torch.Tensor | None] = [None] * world
        self.send_splits: list[list[int] | None] = [None] * world

    def set(self, rank: int, tensor: torch.Tensor, splits: list[int]) -> None:
        self.payloads[rank] = tensor
        self.send_splits[rank] = splits


class MockTransport:
    """Reproduces the real collectives' semantics on pre-registered payloads.

    ``regs`` is a call-indexed list of registries: ``all_to_all`` consumes
    the next registry (the window exchange, then the block exchanges), and
    ``all_gather`` the next (the compressed gather)."""

    def __init__(self, rank: int, world: int, regs: list[MockRegistry]):
        self.rank, self.world = rank, world
        self._regs = regs
        self._a2a = 0
        self._ag = 0

    def all_to_all(self, x, in_splits, out_splits):
        del x, in_splits
        reg = self._regs[min(self._a2a, len(self._regs) - 1)]
        self._a2a += 1
        payloads = [reg.payloads[s] for s in range(self.world)]
        # each sender's group for me sits at its own send-splits cumsum
        starts = [sum(reg.send_splits[s][: self.rank]) for s in range(self.world)]
        return torch.cat(
            [
                payloads[s][starts[s] : starts[s] + out_splits[s]]
                for s in range(self.world)
            ],
            dim=0,
        )

    def all_gather(self, x):
        reg = self._regs[min(self._a2a + self._ag, len(self._regs) - 1)]
        self._ag += 1
        return list(reg.payloads)


class _MockDispatcher(cp_mod.CPTokenDispatcher):
    """Test dispatcher whose exchanges go through the MockTransport (the
    production ``_all_to_all`` override seam)."""

    @dataclass(kw_only=True, slots=True)
    class Config(cp_mod.CPTokenDispatcher.Config):
        pass

    def __init__(self, config, *, mock=None):
        super().__init__(config)
        self.mock = mock

    def _all_to_all(self, x, in_splits, out_splits):
        return self.mock.all_to_all(x, in_splits, out_splits)


def _doc_table(v, restore):
    """Doc lengths from the global context (keyed by the permuted doc
    starts, like the plan's segment identity)."""
    cu = v.cu_seq_q.tolist()
    return {
        int(restore[cu[d]].item()): cu[d + 1] - cu[d]
        for d in range(len(cu) - 1)
        if cu[d + 1] > cu[d]
    }


# ---------------------------------------------------------------------------
def make_cp_metas(docs, cp, lb=None):
    seq_len = sum(docs)
    cu = torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=max(docs), max_k=max(docs))
    from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
        CPVarlenMetadata,
    )

    metas = [
        CPVarlenMetadata.from_global(v, FakeMesh(r, cp), 1, seq_len, lb)
        for r in range(cp)
    ]
    return v, metas, seq_len // cp


def _oracle_blocks(v, x_flat, docs, ratio, comp, perm=None):
    """The oracle's per-doc packed blocks (the no-CP compressor on the full
    stream), keyed by the doc start's permuted position."""
    md = meta_mod.build_compressed_varlen_metadata(v, (ratio,))
    plan = md.plans[ratio]
    n_blocks = int(plan.cu_seqlens_cmp_k[-1])
    if n_blocks:
        bt = x_flat[plan.gather_indices].reshape(n_blocks, ratio, -1)
        bids = torch.arange(n_blocks)
        seq_ids = torch.searchsorted(plan.cu_seqlens_cmp_k[1:], bids, right=True)
        block_local = bids - plan.cu_seqlens_cmp_k[seq_ids]
        packed = compress_blocks(
            comp,
            bt,
            (block_local * ratio).to(torch.int32),
            block_local != 0,
            torch.float32,
            ratio,
        )
    else:
        packed = torch.empty((0, HD), dtype=torch.float32)
    cu_b = plan.cu_seqlens_cmp_k.tolist()
    doc_blocks = {}
    for d in range(len(docs)):
        ps = (
            int((perm == v.cu_seq_q[d]).nonzero()[0])
            if perm is not None
            else int(v.cu_seq_q[d])
        )
        doc_blocks[ps] = packed[cu_b[d] : cu_b[d + 1]]
    return doc_blocks


_CASES = [
    ("plain2", (10, 17, 20, 17), 2, None),
    ("plain4", (10, 17, 20, 17), 4, None),
    ("ht2", (37, 41, 63, 115), 2, _ht(256, 2)),
    ("ht4", (37, 41, 63, 115), 4, _ht(256, 4)),
    ("irreg", (133, 122, 744, 281), 2, None),
    ("irreg-ht", (133, 122, 744, 281), 2, _ht(1280, 2)),
    ("tiny-ht", (5, 2, 9, 1, 7, 4, 3, 8, 2, 6, 5, 3, 4, 9, 4), 4, _ht(72, 4)),
    ("docstart", (10, 22, 32), 2, None),
    ("zero-block", (10, 8, 17, 5, 24), 4, None),
]


@pytest.fixture(scope="module")
def dsv4_globals(dsv4):
    globals()["meta_mod"] = dsv4.metadata
    globals()["comp_mod"] = dsv4.compressor
    return None


def _register_window_exchange(contexts, windows, payload_all):
    """Register every rank's grouped payloads for the window exchange (the
    rank-local rows of the exchanged stream — the post-RoPE ``swa_k`` in
    the attention flow; raw rows are fine in the dispatcher-semantics
    tests since the gather is payload-agnostic)."""
    reg = MockRegistry(len(contexts))
    for rr in range(len(contexts)):
        ex = windows[rr].exchange
        reg.set(rr, payload_all[rr][ex.send_indices], ex.send_splits)
    return reg


def _register_block_exchange(contexts, plan_dicts, payload_all, ratio):
    """Register every rank's grouped payloads for ONE block exchange (kv or
    score — the projected rows the compressor gathers)."""
    reg = MockRegistry(len(contexts))
    for rr in range(len(contexts)):
        ex = plan_dicts[rr][ratio].exchange
        reg.set(rr, payload_all[rr][ex.send_indices], ex.send_splits)
    return reg


def build_plans(v, lb, *, cp_size, shard_len, window_size, ratios):
    """The real plan for every rank, derived purely from the global context
    (no communication, no rounds): returns ``(contexts, plan_dicts,
    windows)`` — the ``_MockDispatcher`` (mock wired by the callers), the
    per-ratio ``plans[ratio]`` dict, and the window plan per rank."""
    contexts, plan_dicts, windows = [], [], []
    for r in range(cp_size):
        _cp_meta, plans, window = build_cp_plan(
            v,
            lb,
            rank=r,
            cp_size=cp_size,
            shard_len=shard_len,
            window_size=window_size,
            ratios=list(ratios),
        )
        contexts.append(_MockDispatcher(cp_mod.CPTokenDispatcher.Config()))
        plan_dicts.append(plans)
        windows.append(window)
    return contexts, plan_dicts, windows


# ---------------------------------------------------------------------------
# 1. The real plan vs the experiment's rank_plan + kernel contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in _CASES])
def test_plan_matches_experiment(dsv4_globals, case):
    name, docs, cp, lb = case
    for ratio in (4, 128):
        _v, cp_metas, shard_len = make_cp_metas(docs, cp, lb)
        contexts, plan_dicts, windows = build_plans(
            _v,
            lb,
            cp_size=cp,
            shard_len=shard_len,
            window_size=8,
            ratios=[ratio],
        )
        from test_cp_compressor import rank_plan

        perm = (
            lb._generate_indices(restore=False).reshape(-1)
            if lb is not None
            else torch.arange(sum(docs))
        )
        doc_len = _doc_table(_v, torch.argsort(perm))
        for r, (ctx, plans, window) in enumerate(
            zip(contexts, plan_dicts, windows, strict=True)
        ):
            exp_segs = rank_plan(
                cp_metas[r], shard_len, r, ratio=ratio, doc_table=doc_len
            )
            rplan = plans[ratio]
            bps = rplan.block_positions.tolist()
            # The plan's blocks (borrow sources included) match the
            # experiment's plan blocks.
            assert [bps[i] // ratio for i in range(len(bps))] == [
                b for s in exp_segs for b in s["blocks"]
            ], (name, r, ratio)
            # ``compressed_rows`` select exactly the experiment's kept blocks
            # (the borrow-source blocks are stripped per segment).
            exp_kept = []
            for s in exp_segs:
                nb = int(s["prepend_global"].numel() > 0) + int(
                    bool(s["pred_head_rel"])
                )
                exp_kept += s["blocks"][nb:]
            assert [bps[i] // ratio for i in rplan.compressed_rows.tolist()] == exp_kept, (
                name,
                r,
                ratio,
            )
            segs = segment_structure(cp_metas[r])
            # the kernel tensors match the direct per-segment derivation
            exp_cu = torch.tensor(
                [0, *[s[2] // ratio for s in segs]], dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            exp_rem = torch.tensor([s[2] % ratio for s in segs], dtype=torch.int32)
            assert torch.equal(rplan.cu_seqlens_cmp_k, exp_cu), (name, r, ratio)
            assert torch.equal(rplan.block_remainder, exp_rem), (name, r, ratio)
            # the CP gather indices cover the pooled rows (the dispatcher
            # round-trip below proves their order)
            assert rplan.gather_indices.numel() // ratio == len(bps)
            assert rplan.cmp_k_global_gather_indices is not None
            # the window plan: the per-segment win lengths (window_size=8)
            # and the assembly covering the whole ori stream
            exp_ori = torch.tensor(
                [0, *[s[1] + s[3] - max(s[3] - 7, 0) for s in segs]],
                dtype=torch.int32,
            ).cumsum(0, dtype=torch.int32)
            assert torch.equal(window.cu_seqlens_ori_kv, exp_ori), (name, r)
            assert window.gather_indices.numel() == int(
                window.cu_seqlens_ori_kv[-1]
            ), (
                name,
                r,
            )


# ---------------------------------------------------------------------------
# 2. The dispatcher round-trip vs the oracle (window pack, borrows, gather)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in _CASES])
def test_dispatcher_vs_oracle(dsv4_globals, case):
    name, docs, cp, lb = case
    for ratio in (4, 128):
        v, cp_metas, shard_len = make_cp_metas(docs, cp, lb)
        seq_len = sum(docs)
        perm = lb._generate_indices(restore=False).reshape(-1) if lb else None
        x = torch.randn(1, seq_len, DIM)
        x_flat = x.flatten(0, 1)
        x_perm = x_flat[perm] if perm is not None else x_flat
        comp = build_compressor(torch.float32, ratio)
        doc_blocks = _oracle_blocks(v, x_flat, docs, ratio, comp, perm)
        contexts, plan_dicts, windows = build_plans(
            v,
            lb,
            cp_size=cp,
            shard_len=shard_len,
            window_size=8,
            ratios=[ratio],
        )
        segs_all = [segment_structure(m) for m in cp_metas]
        out_width = max(plan_dicts[r][ratio].out_width for r in range(cp))

        # ---- per-rank: the real flow — the window borrow (the packed ori
        # ---- stream), the compressor with its own kv/score block packs,
        # ---- and the plan-driven container packing (select)
        local_kept = []
        for r in range(cp):
            x_local = x_perm[r * shard_len : (r + 1) * shard_len]
            # the window exchange (payload-agnostic in the semantics test)
            win_reg = _register_window_exchange(
                contexts,
                windows,
                [x_perm[rr * shard_len : (rr + 1) * shard_len] for rr in range(cp)],
            )
            contexts[r].mock = MockTransport(r, cp, [win_reg])
            swa = contexts[r].gather(x_local.view(1, shard_len, DIM), windows[r])
            assert swa.shape[1] == int(windows[r].cu_seqlens_ori_kv[-1]), (
                name,
                r,
                ratio,
            )
            # the compressor's own dispatcher: the two block packs (kv,
            # score — the projected rows the exchange carries)
            comp_disp = _MockDispatcher(cp_mod.CPTokenDispatcher.Config())
            kv_all = [
                comp.wkv(x_perm[rr * shard_len : (rr + 1) * shard_len])
                for rr in range(cp)
            ]
            sc_all = [
                comp.wgate(x_perm[rr * shard_len : (rr + 1) * shard_len])
                for rr in range(cp)
            ]
            comp_disp.mock = MockTransport(
                r,
                cp,
                [
                    _register_block_exchange(contexts, plan_dicts, kv_all, ratio),
                    _register_block_exchange(contexts, plan_dicts, sc_all, ratio),
                ],
            )
            comp.token_dispatcher = comp_disp
            md = _slim_cp_metadata(plan_dicts[r])
            pooled = comp(x_local.view(1, shard_len, DIM), md)
            container = contexts[r].select(pooled, plan_dicts[r][ratio])
            n_kept = int(plan_dicts[r][ratio].compressed_rows.numel())
            assert container.shape == (1, int(plan_dicts[r][ratio].out_width), HD)
            local_kept.append(container.flatten(0, 1)[:n_kept])
            assert local_kept[-1].shape[0] == n_kept

        # ---- the compressed-level gather (the ShardingConfig semantics):
        # ---- the padded containers all-gathered and assembled per segment
        # ---- with the plan's cmp_k_global_gather_indices
        for r in range(cp):
            rplan = plan_dicts[r][ratio]
            # every rank's padded container: the kept blocks in the leading
            # slots, zero-padded to the uniform width (a valid S(1) shard —
            # the concat is the all-gather output)
            containers = [
                torch.cat(
                    [
                        local_kept[rr],
                        local_kept[rr].new_zeros(
                            (out_width - local_kept[rr].shape[0], HD)
                        ),
                    ]
                )
                for rr in range(cp)
            ]
            assert all(c.shape[0] == out_width for c in containers)
            gathered = torch.cat(containers, dim=0)
            cmp_k = gathered[rplan.cmp_k_global_gather_indices]
            exp = torch.cat(
                [
                    doc_blocks[seg[0]][: seg[2] // ratio]
                    if seg[2] // ratio
                    else torch.empty((0, HD), dtype=torch.float32)
                    for seg in segs_all[r]
                ]
            )
            assert torch.allclose(cmp_k, exp, atol=1e-6, rtol=1e-6), (name, r, ratio)
            assert int(rplan.out_width) == out_width

        # ---- the window slice: per-segment end-aligned ori ranges ----
        for r in range(cp):
            x_local = x_perm[r * shard_len : (r + 1) * shard_len]
            win_reg = _register_window_exchange(
                contexts,
                windows,
                [x_perm[rr * shard_len : (rr + 1) * shard_len] for rr in range(cp)],
            )
            contexts[r].mock = MockTransport(r, cp, [win_reg])
            swa = (
                contexts[r]
                .gather(x_local.view(1, shard_len, DIM), windows[r])
                .flatten(0, 1)
            )
            cu_o = windows[r].cu_seqlens_ori_kv.tolist()
            for s, seg in enumerate(segs_all[r]):
                o_start = int(perm[seg[0]]) if perm is not None else seg[0]
                win_start_rel = max(seg[3] - 7, 0)
                got = swa[cu_o[s] : cu_o[s + 1]]
                exp_o = x_flat[o_start + win_start_rel : o_start + seg[3] + seg[1]]
                assert torch.equal(got, exp_o), (name, r, ratio, s)


# ---------------------------------------------------------------------------
# 3. The real Compressor CP branch (augmented stream + plan fields)
# ---------------------------------------------------------------------------


def _slim_cp_metadata(plans, window=None):
    """A minimal kernel-contract holder exposing the per-ratio block plans
    at ``plans[ratio]`` and the window plan at ``window`` (the same access
    points the AscendC handler builds in CP mode)."""

    class _Slim:
        pass

    md = _Slim()
    md.batch_size = 1
    md.plans = dict(plans)
    md.window = window
    return md


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in _CASES])
def test_compressor_cp_branch(dsv4_globals, case):
    name, docs, cp, lb = case
    for ratio in (4, 128):
        v, cp_metas, shard_len = make_cp_metas(docs, cp, lb)
        seq_len = sum(docs)
        perm = lb._generate_indices(restore=False).reshape(-1) if lb else None
        x = torch.randn(1, seq_len, DIM)
        x_flat = x.flatten(0, 1)
        x_perm = x_flat[perm] if perm is not None else x_flat
        comp = build_compressor(torch.float32, ratio)
        doc_blocks = _oracle_blocks(v, x_flat, docs, ratio, comp, perm)
        contexts, plan_dicts, _windows = build_plans(
            v,
            lb,
            cp_size=cp,
            shard_len=shard_len,
            window_size=8,
            ratios=[ratio],
        )
        from test_cp_compressor import rank_plan

        perm = lb._generate_indices(restore=False).reshape(-1) if lb else None
        restore = torch.argsort(perm) if perm is not None else torch.arange(seq_len)
        doc_len = _doc_table(v, restore)

        for r in range(cp):
            x_local = x_perm[r * shard_len : (r + 1) * shard_len]
            # the compressor's own dispatcher: the two block packs (kv,
            # score — the projected rows the exchange carries)
            comp_disp = _MockDispatcher(cp_mod.CPTokenDispatcher.Config())
            kv_all = [
                comp.wkv(x_perm[rr * shard_len : (rr + 1) * shard_len])
                for rr in range(cp)
            ]
            sc_all = [
                comp.wgate(x_perm[rr * shard_len : (rr + 1) * shard_len])
                for rr in range(cp)
            ]
            comp_disp.mock = MockTransport(
                r,
                cp,
                [
                    _register_block_exchange(contexts, plan_dicts, kv_all, ratio),
                    _register_block_exchange(contexts, plan_dicts, sc_all, ratio),
                ],
            )
            comp.token_dispatcher = comp_disp
            md = _slim_cp_metadata(plan_dicts[r])
            pooled = comp(x_local.view(1, shard_len, DIM), md)
            rplan = plan_dicts[r][ratio]
            n_plan = rplan.gather_indices.numel() // ratio
            assert pooled.shape == (n_plan, HD), (name, r, ratio)
            # Only the kept blocks carry full overlap chains and must match
            # the oracle; the stripped borrow-source blocks' keys are masked
            # (by design) and never enter the gather.
            kept = pooled[rplan.compressed_rows]
            exp_segs = rank_plan(
                cp_metas[r], shard_len, r, ratio=ratio, doc_table=doc_len
            )
            exp_parts = []
            for s in exp_segs:
                nb = int(s["prepend_global"].numel() > 0) + int(
                    bool(s["pred_head_rel"])
                )
                if len(s["blocks"]) > nb:
                    exp_parts.append(doc_blocks[s["doc"]][s["blocks"][nb:]])
            exp = (
                torch.cat(exp_parts)
                if exp_parts
                else torch.empty((0, HD), dtype=torch.float32)
            )
            assert torch.allclose(kept, exp, atol=1e-6, rtol=1e-6), (name, r, ratio)
            assert rplan.compressed_rows.numel() == kept.shape[0]


# ---------------------------------------------------------------------------
# 4. The AscendC handler's CP path (fake cann_ops recorder, mocked transport)
# ---------------------------------------------------------------------------


def test_asc_extension_cp_metadata(dsv4_globals, dsv4):
    """The AscendC metadata extension over the model-built CP metadata (the
    pure global-context plan derivation — no rounds, no transport)."""
    ratio = 4
    docs = (1000, 1000, 1000, 1000)
    lb = _ht(4000, 2)
    _v, cp_metas, shard_len = make_cp_metas(docs, 2, lb)
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
        AscMetadataExtension,
    )

    for r in range(2):
        dsv4.cann_ops.calls.clear()
        cp_meta, plans, window = cp_mod.build_cp_plan(
            _v,
            lb,
            rank=r,
            cp_size=2,
            shard_len=shard_len,
            window_size=128,
            ratios=[1, ratio, 128],
        )
        md = AscMetadataExtension(
            AscMetadataExtension.Config(
                window_size=128,
                num_heads=16,
                head_dim=512,
                index_n_heads=8,
                index_head_dim=128,
                index_topk=512,
            )
        )(
            dsv4.metadata.CompressedVarlenMetadata(
                varlen=cp_meta, plans=plans, window=window
            )
        )
        assert md.batch_size == 1 and md.seq_len == shard_len
        assert md.window is not None
        # the window plan's packed-ori cumsum drives the kernel tensors
        smla = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_metadata"]
        assert len(smla) == 3
        for k, c in {c[2]["cmp_ratio"]: c for c in smla}.items():
            assert torch.equal(c[2]["cu_seqlens_ori_kv"], md.window.cu_seqlens_ori_kv), k
        segs = segment_structure(cp_metas[r])
        for k in (4, 128):
            exp_cu = torch.tensor(
                [0, *[s[2] // k for s in segs]], dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            exp_rem = torch.tensor([s[2] % k for s in segs], dtype=torch.int32)
            assert torch.equal(md.plans[k].cu_seqlens_cmp_k, exp_cu)
            assert torch.equal(md.plans[k].block_remainder, exp_rem)
            # The full per-ratio plan (part 1 + part 2) is exposed at
            # ``plans[ratio]`` — the same access point as the no-CP layout.
            assert md.plans[k].first_indices is not None
            assert md.plans[k].block_positions is not None
            assert md.plans[k].compressed_rows is not None
            assert md.plans[k].cmp_k_global_gather_indices is not None
            assert md.plans[k].gather_indices is not None
            assert md.plans[k].exchange is not None


# ---------------------------------------------------------------------------
# 5. CPAttention end-to-end (recording inner core)
# ---------------------------------------------------------------------------


class _RecorderCore(nn.Module):
    """Inner-core stand-in recording its positional contract."""

    @dataclass(kw_only=True, slots=True)
    class Config:
        def build(self):
            return _RecorderCore()

    def forward(self, *args, **kwargs):
        self.last = (args, kwargs)
        return args[0]


class _RopeStub(nn.Module):
    """The attention's rope contract: ``(q, k)`` out of ``(x, key)``."""

    def forward(self, x, key=None, positions=None, *, inverse=False):
        if key is None:
            return x
        return x, key


def make_rope_config():
    class _Cfg:
        def build(self):
            return _RopeStub()

    return _Cfg()


def make_linear_cfg(n, m):
    class _Cfg:
        def build(self):
            return make_linear(n, m, torch.float32)

    return _Cfg()


def make_rms_cfg(n):
    class _Cfg:
        def build(self):
            return nn.Identity()

    return _Cfg()


def make_batched_cfg(d, r, o):
    class _Cfg:
        def build(self):
            class _B(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.randn(2, r, d))

                def forward(self, x):
                    return torch.einsum("bsgd,grd->bsgr", x, self.weight)

            return _B()

    return _Cfg()


def make_compressor_cfg(ratio):
    class _Cfg:
        def build(self):
            return build_compressor(torch.float32, ratio)

    return _Cfg()


def make_indexer_cfg(dim, n_heads, hd, rd, ratio):
    class _Cfg:
        def build(self):
            class _Idx(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.num_index_heads = n_heads
                    self.head_dim = hd
                    self.rope_head_dim = rd
                    self.softmax_scale = hd**-0.5
                    from test_dsv4 import IdentityRoPE

                    self.rope = IdentityRoPE()
                    self.wq_b = make_linear(4, n_heads * hd)  # qr (q_lora_rank)
                    self.weights_proj = make_linear(dim, n_heads)
                    self.compressor = build_compressor(torch.float32, ratio)

                def _rotate_activation(self, x):
                    return x

                def forward(self, x, qr, *, positions, attention_masks):
                    bsz, seqlen, _ = qr.size()
                    rd = self.rope_head_dim
                    idx_q = self.wq_b(qr)
                    idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
                    q_nope, q_rope = torch.split(
                        idx_q, [self.head_dim - rd, rd], dim=-1
                    )
                    q_rope = self.rope(q_rope, positions=positions)
                    idx_q = torch.cat([q_nope, q_rope], dim=-1)
                    idx_q = self._rotate_activation(idx_q)
                    idx_k = self.compressor(x, attention_masks)
                    idx_k = self._rotate_activation(idx_k)
                    idx_w = self.weights_proj(x) * (
                        self.softmax_scale * self.num_index_heads**-0.5
                    )
                    return idx_q, idx_k, idx_w

            return _Idx()

    return _Cfg()


def test_cp_attention_flow(dsv4_globals, dsv4):
    ratio = 4
    docs = (10, 17, 20, 17)
    _v, _cp_metas, shard_len = make_cp_metas(docs, 2, None)
    contexts, plan_dicts, windows = build_plans(
        _v,
        None,
        cp_size=2,
        shard_len=shard_len,
        window_size=8,
        ratios=[ratio],
    )
    x = torch.randn(1, 64, DIM)
    x_flat = x.flatten(0, 1)

    cfg = Attention.Config(
        n_heads=2,
        head_dim=HD,
        rope_head_dim=RD,
        q_lora_rank=4,
        n_groups=2,
        compress_ratio=ratio,
        norm_eps=1e-6,
        rope=make_rope_config(),
        token_dispatcher=_MockDispatcher.Config(),
        wq_a=make_linear_cfg(DIM, 4),
        q_norm=make_rms_cfg(4),
        wq_b=make_linear_cfg(4, 2 * HD),
        wkv=make_linear_cfg(DIM, HD),
        kv_norm=make_rms_cfg(HD),
        wo_a=make_batched_cfg(HD, 4, 2),
        wo_b=make_linear_cfg(8, DIM),
        compressor=make_compressor_cfg(ratio),
        indexer=make_indexer_cfg(DIM, 4, HD, RD, ratio),
        inner_attention=_RecorderCore.Config(),
    )
    attn = Attention(cfg)

    mds = []
    for r in range(2):
        mds.append(_slim_cp_metadata(plan_dicts[r], windows[r]))

    # stage 1: the compressors over the local stream with their own pack
    # exchanges -> the kept blocks via the plan-driven packing
    kept = {}
    for rr in range(2):
        xl = x_flat[rr * shard_len : (rr + 1) * shard_len]
        for comp in (attn.compressor, attn.indexer.compressor):
            kv_all = [
                comp.wkv(x_flat[i * shard_len : (i + 1) * shard_len]) for i in range(2)
            ]
            sc_all = [
                comp.wgate(x_flat[i * shard_len : (i + 1) * shard_len]) for i in range(2)
            ]
            disp = _MockDispatcher(cp_mod.CPTokenDispatcher.Config())
            disp.mock = MockTransport(
                rr,
                2,
                [
                    _register_block_exchange(contexts, plan_dicts, kv_all, ratio),
                    _register_block_exchange(contexts, plan_dicts, sc_all, ratio),
                ],
            )
            comp.token_dispatcher = disp
        md = _slim_cp_metadata(plan_dicts[rr], windows[rr])
        cmp_k = contexts[rr].select(
            attn.compressor(xl.view(1, shard_len, DIM), md), plan_dicts[rr][ratio]
        )
        idx_k = contexts[rr].select(
            attn.indexer.compressor(xl.detach().view(1, shard_len, DIM), md),
            plan_dicts[rr][ratio],
        )
        n_kept = int(plan_dicts[rr][ratio].compressed_rows.numel())
        kept[rr] = (cmp_k.flatten(0, 1)[:n_kept], idx_k.flatten(0, 1)[:n_kept])

    for r in range(2):
        # the dispatcher mocks: the forward's collective call order is the
        # window exchange (the local post-RoPE swa_k rows), then the
        # indexer compressor's kv/score packs, then the attention
        # compressor's kv/score packs
        swa_all = [
            attn.kv_norm(attn.wkv(x_flat[i * shard_len : (i + 1) * shard_len]))
            for i in range(2)
        ]
        regs = [_register_window_exchange(contexts, windows, swa_all)]
        for comp in (attn.indexer.compressor, attn.compressor):
            kv_all = [
                comp.wkv(x_flat[i * shard_len : (i + 1) * shard_len]) for i in range(2)
            ]
            sc_all = [
                comp.wgate(x_flat[i * shard_len : (i + 1) * shard_len]) for i in range(2)
            ]
            regs.append(_register_block_exchange(contexts, plan_dicts, kv_all, ratio))
            regs.append(_register_block_exchange(contexts, plan_dicts, sc_all, ratio))
        # each dispatcher gets its own mock whose registry list covers
        # exactly its own collective calls (the call counters are
        # per-transport): the window for the attention's, the two packs
        # per compressor
        attn.token_dispatcher.mock = MockTransport(r, 2, regs[0:1])
        attn.indexer.compressor.token_dispatcher.mock = MockTransport(r, 2, regs[1:3])
        attn.compressor.token_dispatcher.mock = MockTransport(r, 2, regs[3:5])
        md = mds[r]
        x_local = x_flat[r * shard_len : (r + 1) * shard_len]
        positions = torch.arange(r * shard_len, (r + 1) * shard_len).unsqueeze(0)
        out = attn(x_local.view(1, shard_len, DIM), md, positions)
        assert out.shape == (1, shard_len, DIM)
        core = attn.inner_attention
        args, kwargs = core.last
        q, swa_k, cmp_k, idx_q, idx_k, idx_w = args[:6]
        assert kwargs["attn_sink"].shape == (2,)
        assert q.shape == (1, shard_len, 2, HD)
        assert swa_k.shape == (1, int(windows[r].cu_seqlens_ori_kv[-1]), HD)
        # the core receives the padded containers (the ShardingConfig
        # all-gather + the per-segment assembly happen at/inside the core)
        assert cmp_k.shape == (1, int(plan_dicts[r][ratio].out_width), HD)
        assert idx_q.shape == (1, shard_len, 4, HD)
        assert idx_k.shape == cmp_k.shape
        # idx_w is the local rows (the indexer sees x, not the augmented
        # stream — no strip)
        assert idx_w.shape == (1, shard_len, 4)
        # the containers' leading slots = the oracle's kept blocks
        kept_flat = kept[r][0]
        assert torch.allclose(
            cmp_k.flatten(0, 1)[: kept_flat.shape[0]], kept_flat, atol=1e-6, rtol=1e-6
        )
        assert torch.all(cmp_k.flatten(0, 1)[kept_flat.shape[0] :] == 0)


# ---------------------------------------------------------------------------
# 6. The real transport over gloo (multi-process)
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_multiprocess_gloo(tmp_path):
    """The real build_cp_plan + dispatcher over a real gloo process group.

    Spawns ``cp_dispatch_worker.py`` per rank (subprocesses, real
    ``torch.distributed`` all-gather); each worker verifies its plan and
    per-layer dispatch against the oracle and writes a result file.
    """
    python = sys.executable
    worker = _REPO / "tests" / "unit_tests" / "models" / "deepseek_v4" / "cp_dispatch_worker.py"
    assert worker.exists()
    port = _free_port()
    env = dict(os.environ)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    # The worker is launched as a standalone script, so it does not inherit
    # the pytest process's sys.path. Propagate the test directories
    # explicitly; this also works when PYTHONSAFEPATH is enabled.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(_REPO / "tests" / "unit_tests" / "models" / "deepseek_v4"),
                str(_REPO),
                env.get("PYTHONPATH", ""),
            ],
        )
    )
    outdir = tmp_path
    procs = []
    for r in range(2):
        env_r = dict(env)
        env_r["RANK"] = str(r)
        env_r["OUT"] = str(outdir / f"rank{r}.json")
        procs.append(
            subprocess.Popen(
                [python, str(worker)],
                env=env_r,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
    for p in procs:
        out, _ = p.communicate(timeout=600)
        assert p.returncode == 0, out.decode()
    for r in range(2):
        with open(outdir / f"rank{r}.json") as f:
            rep = json.load(f)
        assert rep["ok"] is True, (r, rep)
