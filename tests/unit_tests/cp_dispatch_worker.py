"""Multi-process gloo worker: the real CP plan + dispatcher on real dist.

Run per rank (``RANK`` env) with ``MASTER_ADDR`` / ``MASTER_PORT`` set;
initializes a gloo process group and exercises the pure
``build_cp_plan`` (derived from the global context — no plan-time
communication), the window gather (the post-RoPE ``swa_k`` rows), the
compressors (each gathering its projected kv/score rows internally via
``gather``), the plan-driven container packing (the
dispatcher's ``select``), and the compressed-level gather
assembled with ``cmp_k_global_gather_indices`` — over real ``torch.distributed``
collectives via the portable gloo exchange emulation (the
``_PortableTransport`` below, moved out of the production code; the
``spmd_types`` collectives are the NPU path), verifying every rank's
outputs against the global-stream oracle.  Writes a JSON report to
``OUT``.
"""

import json
import os
import sys

sys.path.insert(0, "/home/developer/workspace/torchtitan-npu/tests/unit_tests")
sys.path.insert(0, "/home/developer/workspace/torchtitan-npu")

import torch

torch.manual_seed(0)

import test_cp_compressor as tcc
from test_cp_compressor import (
    FakeMesh,
    _ht,
    build_compressor,
    compress_blocks,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import compressor as comp_mod
from torchtitan_npu.models.deepseek_v4 import metadata as meta_mod
from torchtitan_npu.models.deepseek_v4.token_dispatcher import (
    CPTokenDispatcher,
    build_cp_plan,
    segment_structure,
)
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)

tcc.comp_mod = comp_mod

DIM, HD, RD = 8, 16, 4
DOCS = (37, 41, 63, 115)
CP = 2
RATIOS = (4, 128)

# The global facts (identical in every rank's process).
_cu = torch.tensor([0, *torch.tensor(DOCS).cumsum(0).tolist()], dtype=torch.int32)
_seq_len = sum(DOCS)
_v = VarlenMetadata(cu_seq_q=_cu, cu_seq_k=_cu, max_q=max(DOCS), max_k=max(DOCS))
_lb = _ht(_seq_len, CP)
_perm = _lb._generate_indices(restore=False).reshape(-1)
_shard_len = _seq_len // CP
_x_flat = torch.randn(1, _seq_len, DIM).flatten(0, 1)
_x_perm = _x_flat[_perm]


class _PortableTransport:
    """The gloo-capable exchange emulation (moved from the production
    transport): gloo cannot run uneven ``all_to_all``, so the exchange is
    emulated as an all-gather of the rank-grouped payloads + local
    extraction of each sender's group (the all_to_all output layout)."""

    def __init__(self, group):
        self._group = group
        self.rank = torch.distributed.get_rank(group)
        self.world = torch.distributed.get_world_size(group)

    def all_gather(self, x: torch.Tensor) -> list[torch.Tensor]:
        sizes = [
            torch.zeros((), dtype=torch.int64, device=x.device)
            for _ in range(self.world)
        ]
        torch.distributed.all_gather(
            sizes, torch.tensor(x.shape[0], device=x.device), group=self._group
        )
        size_list = [int(s.item()) for s in sizes]
        max_len = max(size_list)
        padded = x.new_zeros((max_len, *x.shape[1:]))
        padded[: x.shape[0]] = x
        out = x.new_zeros((self.world * max_len, *x.shape[1:]))
        with torch.no_grad():
            torch.distributed.all_gather_into_tensor(out, padded, group=self._group)
        return [
            out[r * max_len : (r + 1) * max_len][: size_list[r]]
            for r in range(self.world)
        ]

    def all_to_all(self, x, in_splits, out_splits):
        union = self.all_gather(x)
        # each sender's group for me sits at its own send-splits cumsum —
        # gather every rank's input splits to reconstruct the starts
        splits_list = [None] * self.world
        torch.distributed.all_gather_object(splits_list, in_splits, group=self._group)
        starts = [sum(splits_list[s][: self.rank]) for s in range(self.world)]
        return torch.cat(
            [
                union[s][starts[s] : starts[s] + out_splits[s]]
                for s in range(self.world)
            ],
            dim=0,
        )


class _GlooDispatcher(CPTokenDispatcher):
    """Test dispatcher whose exchanges go through the portable gloo
    emulation (the production ``_all_to_all`` override seam)."""

    def __init__(self, config, *, group):
        super().__init__(config)
        self._portable = _PortableTransport(group)

    def _all_to_all(self, x, in_splits, out_splits):
        return self._portable.all_to_all(x, in_splits, out_splits)


def _slim_metadata(plans, window):
    """A minimal kernel-contract holder (the AscendC handler's slim shape)."""

    class _Slim:
        pass

    md = _Slim()
    md.batch_size = 1
    md.plans = plans
    md.window = window
    return md


def main():
    rank = int(os.environ["RANK"])
    out_path = os.environ["OUT"]
    failures: list[str] = []
    try:
        import torch.distributed as dist

        dist.init_process_group("gloo", rank=rank, world_size=CP)
        group = dist.distributed_c10d._get_default_group()
        run(rank, group, failures)
        dist.destroy_process_group()
    except Exception as e:
        import traceback

        failures.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    with open(out_path, "w") as f:
        json.dump({"rank": rank, "ok": not failures, "failures": failures}, f)
    if failures:
        raise SystemExit(1)


def run(rank: int, group, failures: list[str]) -> None:
    for ratio in RATIOS:
        cp_meta = CPVarlenMetadata.from_global(_v, FakeMesh(rank, CP), 1, _seq_len, _lb)
        derived, plans, window = build_cp_plan(
            _v,
            _lb,
            rank=rank,
            cp_size=CP,
            shard_len=_shard_len,
            window_size=8,
            ratios=[ratio],
        )
        # the derived rank-local varlen equals from_global's output (the
        # plan reuses the shard path's builder directly)
        for name, a, b in (
            ("cu_q", derived.cu_seq_q, cp_meta.cu_seq_q),
            ("cu_k", derived.cu_seq_k, cp_meta.cu_seq_k),
            ("kg", derived.k_global_gather_indices, cp_meta.k_global_gather_indices),
        ):
            if not torch.equal(a, b):
                failures.append(f"r{rank} r{ratio}: derived {name} mismatch")
                return
        context = _GlooDispatcher(CPTokenDispatcher.Config(), group=group)
        verify_plan(rank, plans, window, cp_meta, ratio, failures)
        verify_dispatch(rank, context, plans, window, cp_meta, ratio, failures)
        if failures:
            return
        print(f"[r{rank}] ratio {ratio}: OK", flush=True)


def verify_plan(rank, plans, window, cp_meta, ratio, failures):
    rplan = plans[ratio]
    # the kernel tensors match the direct per-segment derivation
    segs = segment_structure(cp_meta)
    exp_cu = torch.tensor(
        [0, *[s[2] // ratio for s in segs]], dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)
    exp_rem = torch.tensor([s[2] % ratio for s in segs], dtype=torch.int32)
    if not torch.equal(rplan.cu_seqlens_cmp_k, exp_cu):
        failures.append(f"r{rank} r{ratio}: cu_seqs mismatch")
    if not torch.equal(rplan.block_remainder, exp_rem):
        failures.append(f"r{rank} r{ratio}: residuals mismatch")
    if rplan.gather_indices.numel() % ratio != 0:
        failures.append(f"r{rank} r{ratio}: gather length mismatch")
    if rplan.cmp_k_global_gather_indices is None:
        failures.append(f"r{rank} r{ratio}: cmp_k_global_gather_indices missing")
    if window is None:
        failures.append(f"r{rank} r{ratio}: window plan missing")
        return
    if int(window.gather_indices.numel()) != int(window.cu_seqlens_ori_kv[-1]):
        failures.append(f"r{rank} r{ratio}: window gather length mismatch")


def verify_dispatch(rank, context, plans, window, cp_meta, ratio, failures):
    comp = build_compressor(torch.float32, ratio)
    segs = segment_structure(cp_meta)
    x_local = _x_perm[rank * _shard_len : (rank + 1) * _shard_len]
    x_local = x_local.view(1, _shard_len, DIM)
    rplan = plans[ratio]

    # ---- the window gather: the packed ori stream vs the global stream ----
    cu_o = window.cu_seqlens_ori_kv.tolist()
    swa = context.gather(x_local, window).flatten(0, 1)
    for s, seg in enumerate(segs):
        o_start = int(_perm[seg[0]])
        win_start_rel = max(seg[3] - 7, 0)
        got = swa[cu_o[s] : cu_o[s + 1]]
        exp_o = _x_flat[o_start + win_start_rel : o_start + seg[3] + seg[1]]
        if not torch.equal(got, exp_o):
            failures.append(f"r{rank} r{ratio}: window seg{s} mismatch")
            return

    # ---- the compressor with its own kv/score packs + the plan-driven
    # ---- container packing (the exchange payloads are the projected rows
    # ---- the portable transport carries automatically)
    comp.token_dispatcher = context
    md = _slim_metadata(plans, window)
    pooled = comp(x_local, md)
    container = context.select(pooled, rplan)
    n_kept = int(rplan.compressed_rows.numel())
    local_kept = container.flatten(0, 1)[:n_kept]
    if local_kept.shape[0] != n_kept:
        failures.append(f"r{rank} r{ratio}: kept count mismatch")
        return

    # ---- the compressed-level gather (the ShardingConfig semantics):
    # ---- the padded containers all-gathered and assembled per segment
    out_width = int(rplan.out_width)
    if container.shape != (1, out_width, HD):
        failures.append(f"r{rank} r{ratio}: container shape mismatch")
        return
    gathered = context._portable.all_gather(container.flatten(0, 1))
    cmp_stream = torch.cat(gathered, dim=0)[rplan.cmp_k_global_gather_indices]
    doc_blocks = oracle_blocks(comp, ratio)
    exp = torch.cat(
        [
            doc_blocks[seg[0]][: seg[2] // ratio]
            if seg[2] // ratio
            else torch.empty((0, HD), dtype=torch.float32)
            for seg in segs
        ]
    )
    if not torch.allclose(cmp_stream, exp, atol=1e-6, rtol=1e-6):
        failures.append(f"r{rank} r{ratio}: cmp stream mismatch")


_oracle = {}


def oracle_blocks(comp, ratio):
    """The no-CP oracle: per-doc packed blocks from the full stream."""
    if ratio in _oracle:
        return _oracle[ratio]
    md = meta_mod.build_compressed_varlen_metadata(_v, (ratio,))
    plan = md.plans[ratio]
    n_blocks = int(plan.cu_seqlens_cmp_k[-1])
    if n_blocks:
        bt = _x_flat[plan.gather_indices].reshape(n_blocks, ratio, -1)
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
    for d in range(len(DOCS)):
        ps = int((_perm == _cu[d]).nonzero()[0])
        doc_blocks[ps] = packed[cu_b[d] : cu_b[d + 1]]
    _oracle[ratio] = doc_blocks
    return doc_blocks


if __name__ == "__main__":
    main()
