# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallel for DeepSeek-V4: the plan and the token dispatcher.

Two halves live here (the single home of all DSV4 CP logic):

1. **The plan builder** — a pure derivation from the global context: the
   pre-shard document structure (the model's ``get_attention_masks``
   result) plus the load-balancer permutation.  Every rank derives every
   rank's rank-local shard metadata in-frame via
   ``CPVarlenMetadata.from_global`` (the shard path's own builder), so
   there is **no plan-time communication** at all.  The whole derivation
   is expressed in the documents' **permuted slices** (``docs``, keyed by
   the segment's doc identity): every row a plan consumes — window
   ``[win_start, q0)`` or plan blocks ``[A, B)`` — is a plain slice of
   one document's slice.

   The plan has two independent parts:

   - the **window plan** (``WindowPlan``, ratio-independent): the per-
     segment sliding-window rows — the exchange routing, the packed ori
     stream's ``gather_indices``, and the packed cumsum
     ``cu_seqlens_ori_kv``.  The Attention gathers the **post-RoPE**
     ``swa_k`` rows through it;
   - the **block plans** (``CompressedBlockLayout`` at ``plans[ratio]``,
     per ratio > 1): the per-segment ratio-aligned plan-block region
     ``[A, B)`` (``_block_range`` — the borrow-source blocks included) —
     the exchange routing + the pooled order's ``gather_indices`` (the
     assembly into ``cat([x_local, recv])``), the directly derived
     compressor contract (``block_positions`` = the doc-relative block
     starts, ``first_indices`` = the per-segment pooled starts), and the
     container-packing / compressed-gather fields.  Each Compressor
     gathers its **projected kv/score** rows through it.

2. **The dispatcher** — the ``BaseEPTokenDispatcher``-shaped mechanism,
   a plain ``Configurable``: one instance on the ``Attention`` (the
   window gather of ``swa_k``) and one per ``Compressor`` (the block
   gather of kv/score), all wired to the CP mesh by their owners'
   ``parallelize``.  The two plan-driven ops are ``gather`` (a local
   gather without an exchange — the plan's ``gather_indices`` over the
   local stream; a remote gather + permute with one — the exchange plus
   the same ``gather_indices`` over ``cat([x_local, recv])``) and
   ``select`` (the container packing).  The exchanges are uneven
   all-to-alls over the CP process group (``spmd_types``); ``_all_to_all``
   is the override seam — the CPU tests subclass and replace it with
   their mock / gloo-capable exchange, so no portable fallback lives in
   production code.

The model's ``build_attention_masks`` calls ``build_cp_plan`` and returns
``CompressedVarlenMetadata`` (the varlen + the per-ratio block plans + the
``window`` plan); the per-layer forward then runs the window gather, the
per-compressor block gathers, and the declarative all-gather of the padded
containers (the core's ``ShardingConfig`` ``cp: S(1) -> R``), assembled per
segment with each ratio plan's ``cmp_k_global_gather_indices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.config import Configurable
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata

from .metadata import CompressedBlockLayout

__all__ = [
    "CPTokenDispatcher",
    "ExchangePlan",
    "WindowPlan",
    "build_cp_plan",
    "segment_structure",
]


class _RankMesh:
    """``CPVarlenMetadata.from_global``'s ``DeviceMesh`` contract for a fixed
    rank (the pure plan derives every rank's shard metadata in-frame)."""

    ndim = 1

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


# ---------------------------------------------------------------------------
# The plan builder (pure host math, per batch, no communication)
# ---------------------------------------------------------------------------


def segment_structure(cp_meta) -> list[tuple[int, int, int, int]]:
    """Per non-empty segment: ``(doc_start, seg_len, seqlen_k, p0)``.

    ``doc_start = kgather[cu_seq_k[s]]`` is the document identity
    (identical across ranks, permuted coordinates — the key of the
    document's permuted slice in ``docs``); ``p0 = seqlen_k - seg_len`` is
    the fragment's doc-relative start.
    """
    cu_q = cp_meta.cu_seq_q.cpu().tolist()
    cu_k = cp_meta.cu_seq_k.cpu().tolist()
    kg = cp_meta.k_global_gather_indices.cpu().tolist()
    segs: list[tuple[int, int, int, int]] = []
    for s in range(len(cu_q) - 1):
        seg_len = cu_q[s + 1] - cu_q[s]
        if seg_len == 0:
            continue
        seqlen_k = cu_k[s + 1] - cu_k[s]
        segs.append((kg[cu_k[s]], seg_len, seqlen_k, seqlen_k - seg_len))
    return segs


def _window_range(seg: tuple[int, int, int, int], window_size: int) -> tuple[int, int]:
    """``(win_start, win_len)`` of the segment's window rows — ratio-
    independent: the doc-relative range ``[max(p0 - (window - 1), 0),
    p0 + seg_len)``."""
    win_start = max(seg[3] - (window_size - 1), 0)
    return win_start, seg[1] + seg[3] - win_start


def _block_range(seg: tuple[int, int, int, int], doc_len: int, *, ratio: int) -> tuple[int, int, int]:
    """``(A, B, strip)`` of the segment's ratio-aligned plan-block region.

    ``[A, B)`` covers the segment's complete blocks plus their borrow
    source: ``A`` is the borrow-source block start (ratio-aligned), ``B``
    the straddle end (``q0`` without a straddle — the fragment-end block
    that ends mid-block inside the document and starts in the fragment),
    and ``strip`` counts the leading borrow-source blocks the container
    drops.  Requires ``ratio > 1``.
    """
    p0, seg_len = seg[3], seg[1]
    q0 = p0 + seg_len
    straddle_idx = q0 // ratio
    straddle = q0 % ratio != 0 and straddle_idx * ratio >= p0 and (straddle_idx + 1) * ratio <= doc_len
    b_first = (p0 + ratio - 1) // ratio
    b_last = q0 // ratio - 1
    if b_first <= b_last:
        # Case A/B: the prepend block (b_first - 1) completes the overlap
        # chain of the local blocks.
        A = (b_first - 1) * ratio if b_first > 0 else 0
        B = (straddle_idx + 1) * ratio if straddle else q0
        strip = 1 if b_first > 0 else 0
    elif straddle and straddle_idx > 0:
        # Case C: the pred block is the straddle block's overlap
        # predecessor (a stripped borrow source).
        A = (straddle_idx - 1) * ratio
        B = (straddle_idx + 1) * ratio
        strip = 1
    elif straddle:
        # The document's first block straddles the fragment (p0 == 0): it
        # is the fragment's only plan block, kept with no borrow.
        A = 0
        B = (straddle_idx + 1) * ratio
        strip = 0
    else:
        # No plan blocks: the sub-range is empty.
        A = q0
        B = q0
        strip = 0
    return A, B, strip


def _tensor(vals: list[int], device) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int64, device=device)


def _routing_geometry(
    foreign_all: list[list[int]], *, shard_len: int, cp_size: int
) -> list[tuple[list[int], list[int], list[int], list[int]]]:
    """The alltoallv routing of one exchange.

    ``foreign_all[r]`` = rank r's foreign positions (permuted-stream
    coordinates, in its per-segment receive order).  Returns per rank
    ``(send_indices, send_splits, recv_splits, recv_offsets)``: the send
    payload rows grouped by receiver (receiver order), the per-receiver /
    per-sender split sizes, and each receive position's flat offset in
    the all_to_all output layout (cat over senders of [my rows from that
    sender]).
    """
    routing: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for r in range(cp_size):
        send_indices: list[int] = []
        send_splits = [0] * cp_size
        for j in range(cp_size):
            for p in foreign_all[j]:
                if p // shard_len == r:
                    send_indices.append(p % shard_len)
                    send_splits[j] += 1
        recv_splits = [0] * cp_size
        for p in foreign_all[r]:
            recv_splits[p // shard_len] += 1
        starts = [0]
        for n in recv_splits:
            starts.append(starts[-1] + n)
        seen = [0] * cp_size
        recv_offsets: list[int] = []
        for p in foreign_all[r]:
            o = p // shard_len
            recv_offsets.append(starts[o] + seen[o])
            seen[o] += 1
        routing.append((send_indices, send_splits, recv_splits, recv_offsets))
    return routing


def _row_order(rows: list[int], *, rank: int, shard_len: int) -> tuple[list[int], int]:
    """The row order of one packed stream and its unique-receive count.

    ``rows`` are the stream's rows in packed order (permuted
    coordinates); local rows map to their local offset, foreign rows to
    ``shard_len + receive slot`` — the first-appearance order matching the
    routing's ``recv_offsets``.
    """
    order: list[int] = []
    recv_of: dict[int, int] = {}
    cursor = 0
    for pos in rows:
        if pos // shard_len == rank:
            order.append(pos - rank * shard_len)
        else:
            if pos not in recv_of:
                recv_of[pos] = cursor
                cursor += 1
            order.append(shard_len + recv_of[pos])
    return order, cursor


def _build_exchange_plan(row, device) -> ExchangePlan:
    """One rank's routing row as the tensor/list mix the collective APIs
    need: the splits stay host lists (built per batch) so the per-layer
    exchange never syncs."""
    send, splits, recv, off = row
    return ExchangePlan(
        send_indices=_tensor(send, device),
        send_splits=splits,
        recv_splits=recv,
        recv_offsets=_tensor(off, device),
    )


def _container_slots(segs_all, seg_blocks_all, *, ratio: int):
    """The ``(doc_start, block) -> container slot`` map and the uniform
    container width ``max_kept``.

    A rank's kept blocks fill the leading slots of its padded container
    (per segment the plan blocks after the strip — the borrow-source
    blocks are dropped); ownership follows the start-owner rule (the
    first rank claiming a block).  The container width is uniform
    (``max_kept``) so every rank's container is a valid ``S(1)`` shard of
    the all-gathered ``[cp * max_kept, D]``, and a block's slot is
    ``owner * max_kept + local_offset``.
    """
    local: dict[tuple[int, int], tuple[int, int]] = {}
    max_kept = 0
    for rr, (segs, blocks) in enumerate(zip(segs_all, seg_blocks_all, strict=True)):
        off = 0
        for seg, (A, block_end, strip) in zip(segs, blocks, strict=True):
            for b in range(A // ratio + strip, block_end // ratio):
                local.setdefault((seg[0], b), (rr, off))
                off += 1
        max_kept = max(max_kept, off)
    slots = {(doc, b): owner * max_kept + off for (doc, b), (owner, off) in local.items()}
    return slots, max_kept


def _assemble_window_plan(
    segs: list[tuple[int, int, int, int]],
    docs: dict[int, list[int]],
    routing_row,
    *,
    rank: int,
    shard_len: int,
    window_size: int,
    device,
) -> WindowPlan:
    """The ratio-independent window plan: the exchange + the packed ori
    stream's ``gather_indices`` (``cu_seqlens_ori_kv`` bounds the
    per-segment window rows)."""
    win_rows: list[int] = []
    ori_lens: list[int] = []
    for seg in segs:
        win_start, win_len = _window_range(seg, window_size)
        ori_lens.append(win_len)
        win_rows += docs[seg[0]][win_start : win_start + win_len]
    # The receive slots number the stream-wide foreign order (matching the
    # routing's recv_offsets), so the gather indices run once over all rows.
    win_order, _ = _row_order(win_rows, rank=rank, shard_len=shard_len)
    cu_ori = torch.tensor([0, *ori_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    return WindowPlan(
        exchange=_build_exchange_plan(routing_row, device),
        gather_indices=_tensor(win_order, device),
        cu_seqlens_ori_kv=cu_ori,
    )


def _assemble_block_plan(
    segs: list[tuple[int, int, int, int]],
    segs_all: list[list[tuple[int, int, int, int]]],
    docs: dict[int, list[int]],
    seg_blocks_all: list[list[tuple[int, int, int]]],
    routing_row,
    *,
    ratio: int,
    rank: int,
    shard_len: int,
    device,
) -> CompressedBlockLayout:
    """One ratio's block plan from the pure global-context derivation.

    ``seg_blocks_all[r]`` holds rank r's per-segment ``(A, block_end,
    strip)`` scalars.  Part 1 (the unified compressor/kernel contract) is
    derived directly over the plan blocks; the ``gather_indices`` order
    the pooled stream the exchange produces (``cat([x_local, recv])``)."""
    my_blocks = seg_blocks_all[rank]
    rows: list[int] = []
    for seg, (A, block_end, _strip) in zip(segs, my_blocks, strict=True):
        rows += docs[seg[0]][A:block_end]
    block_total = len(rows)
    order, n_foreign = _row_order(rows, rank=rank, shard_len=shard_len)
    # The plan blocks of one rank never overlap, so every foreign row is
    # received exactly once (the recv_offsets length is the receive count).
    assert n_foreign == len(routing_row[3]), (n_foreign, len(routing_row[3]))

    # The compressor contract, derived directly over the plan blocks.
    pos_parts: list[torch.Tensor] = []
    first: list[int] = []
    compressed_rows: list[int] = []
    pool_start = 0
    for seg, (A, block_end, strip) in zip(segs, my_blocks, strict=True):
        # Segments without complete blocks have ``block_end <= A`` (the
        # no-plan-block case: ``A == B == q0`` with ``block_end < q0``).
        if block_end <= A:
            continue
        cnt = (block_end - A) // ratio
        pos_parts.append(torch.arange(A, block_end, ratio, dtype=torch.int32, device=device))
        first.append(pool_start)
        compressed_rows += list(range(pool_start + strip, pool_start + cnt))
        pool_start += cnt
    block_positions = torch.cat(pos_parts) if pos_parts else torch.empty((0,), dtype=torch.int32, device=device)
    # The packed kernel tensors (the real causal-prefix counts).
    cu_cmp = [seg[2] // ratio for seg in segs]
    rem = [seg[2] % ratio for seg in segs]
    cu_cmp_t = torch.tensor([0, *cu_cmp], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    # ---- compressed-level gather: ownership + assembly ----
    slots, max_kept = _container_slots(segs_all, seg_blocks_all, ratio=ratio)
    cmp_k_global_gather_indices = [slots[(seg[0], b)] for seg in segs for b in range(seg[2] // ratio)]
    return CompressedBlockLayout(
        cu_seqlens_cmp_k=cu_cmp_t,
        block_remainder=torch.tensor(rem, dtype=torch.int32, device=device),
        gather_indices=_tensor(order, device),
        block_positions=block_positions,
        first_indices=_tensor(first, device),
        exchange=_build_exchange_plan(routing_row, device),
        compressed_rows=_tensor(compressed_rows, device),
        out_width=max_kept,
        cmp_k_global_gather_indices=_tensor(cmp_k_global_gather_indices, device),
    )


def build_cp_plan(
    global_varlen,
    load_balancer,
    *,
    rank: int,
    cp_size: int,
    shard_len: int,
    window_size: int,
    ratios: list[int],
) -> tuple[CPVarlenMetadata, dict[int, CompressedBlockLayout], WindowPlan]:
    """The pure per-rank plan derivation from the global context (no
    communication): every rank's rank-local varlen is derived in-frame via
    ``CPVarlenMetadata.from_global`` (the shard path's own builder), so the
    rank-local varlen, the per-ratio block plans, and the window plan all
    fall out of the pre-shard document structure + the load-balancer
    permutation.  The model's ``build_attention_masks`` calls this; the
    dispatchers are wired by their owners' ``parallelize``.
    """
    global_cu = global_varlen.cu_seq_q
    device = global_cu.device
    seq_len = int(global_cu[-1].item())
    rearrange = (
        load_balancer._generate_indices(restore=False).reshape(-1)
        if load_balancer is not None
        else torch.arange(seq_len, device=device)
    )
    restore = torch.argsort(rearrange)
    varlens = [
        CPVarlenMetadata.from_global(
            # The ``_RankMesh`` shim implements the ``DeviceMesh`` contract
            # (``size`` / ``get_local_rank``) the pure derivation needs.
            global_varlen,
            _RankMesh(cp_size, r),  # pyrefly: ignore [bad-argument-type]
            1,
            seq_len,
            load_balancer,
        )
        for r in range(cp_size)
    ]
    segs_all = [segment_structure(v) for v in varlens]
    my_segs = segs_all[rank]
    # The full permuted document slices, keyed by the segment's doc
    # identity (the permuted doc start): every row a plan consumes — the
    # window range or the plan blocks — is a slice of one document's slice.
    docs: dict[int, list[int]] = {}
    for d in range(len(global_cu) - 1):
        d0, d1 = int(global_cu[d].item()), int(global_cu[d + 1].item())
        if d1 > d0:
            docs[int(restore[d0].item())] = restore[d0:d1].tolist()

    # ---- the window plan (ratio-independent) ----
    win_foreign: list[list[int]] = [[] for _ in range(cp_size)]
    for r in range(cp_size):
        for seg in segs_all[r]:
            win_start, win_len = _window_range(seg, window_size)
            doc = docs[seg[0]]
            win_foreign[r] += [p for p in doc[win_start : win_start + win_len] if p // shard_len != r]
    routing = _routing_geometry(win_foreign, shard_len=shard_len, cp_size=cp_size)
    window = _assemble_window_plan(
        my_segs,
        docs,
        routing[rank],
        rank=rank,
        shard_len=shard_len,
        window_size=window_size,
        device=device,
    )

    # ---- the per-ratio block plans ----
    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in ratios:
        if ratio == 1:
            plans[1] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
            continue
        block_foreign: list[list[int]] = [[] for _ in range(cp_size)]
        seg_blocks_all: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
        for r in range(cp_size):
            for seg in segs_all[r]:
                A, B, strip = _block_range(seg, len(docs[seg[0]]), ratio=ratio)
                block_end = (B // ratio) * ratio
                seg_blocks_all[r].append((A, block_end, strip))
                block_foreign[r] += [p for p in docs[seg[0]][A:block_end] if p // shard_len != r]
        routing = _routing_geometry(block_foreign, shard_len=shard_len, cp_size=cp_size)
        plans[ratio] = _assemble_block_plan(
            my_segs,
            segs_all,
            docs,
            seg_blocks_all,
            routing[rank],
            ratio=ratio,
            rank=rank,
            shard_len=shard_len,
            device=device,
        )
    return varlens[rank], plans, window


# ---------------------------------------------------------------------------
# The dispatcher (the BaseEPTokenDispatcher mirror)
# ---------------------------------------------------------------------------
#
# A plain ``Configurable`` (no learnable state): one instance on the
# ``Attention`` (the window gather of the post-RoPE ``swa_k`` rows) and one
# per ``Compressor`` (the block gather of the projected kv/score rows),
# wired to the CP mesh by their owners' ``parallelize``.  The two
# plan-driven ops: ``gather`` (a local gather without an exchange — the
# plan's ``gather_indices`` over the local stream; a remote gather +
# permute with one — the exchange plus the same ``gather_indices`` over
# ``cat([x_local, recv])``) and ``select`` (the pooled-key container
# packing).  The compressed-level gather of the padded containers is
# declarative (the core's ``ShardingConfig`` ``cp: S(1) -> R``), not a
# dispatcher op.


@dataclass(kw_only=True, slots=True)
class ExchangePlan:
    """The alltoallv routing of one exchange (the EP TokenDispatcher form).

    The send payload is ``x_local[send_indices]`` — the rank's rows grouped
    by receiver (receiver order), so the exchange is a native all_to_all
    with ``send_splits`` / ``recv_splits``.  ``recv_offsets[k]`` is the flat
    offset of the k-th foreign receive position in the exchange output
    (cat over senders of [my rows from that sender]).

    The splits are plain host lists, built once per batch: the collective
    APIs need Python ints and the per-layer exchange must never call
    ``.tolist()`` (a D2H sync per layer per step).  Only the payload rows
    and the receive offsets ride as tensors.
    """

    send_indices: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    recv_offsets: torch.Tensor


@dataclass(kw_only=True, slots=True)
class WindowPlan:
    """The sliding-window plan (ratio-independent, CP only).

    Describes the Attention's ``swa_k`` gather: the exchange routing of
    the per-segment window rows ``[win_start, q0)``, the packed ori
    stream's ``gather_indices`` (indices into
    ``cat([x_local, exchange_recv_rows])``), and the packed-ori cumsum
    (``cu_seqlens_ori_kv``) the kernels consume.  Both the window rows and
    this plan are ratio-independent — one object per rank.
    """

    exchange: ExchangePlan
    gather_indices: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor


class CPTokenDispatcher(Configurable):
    """The DSV4 CP token dispatcher (the ``BaseEPTokenDispatcher`` mirror):
    a plain ``Configurable`` — not an ``nn.Module`` — with no learnable
    parameters or buffers.  One instance per consumer: the ``Attention``
    (the window gather of ``swa_k``) and each ``Compressor`` (the block
    gather of kv/score).  The CP mesh is installed once by ``wire_meshes``
    (from the owner's ``parallelize``).  ``gather`` is self-guarding:
    without a plan it is the identity, without an exchange it is a plain
    local gather — the forward path never special-cases context parallel.

    The CPU tests subclass and override ``_all_to_all`` with their mock /
    gloo-capable exchange — no portable fallback lives in production code.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        self.cp_mesh = None
        self.rank: int | None = None
        self.cp_size: int | None = None

    def wire_meshes(self, *, cp_mesh=None):
        """Install the CP mesh (mirrors ``BaseEPTokenDispatcher.wire_meshes``;
        called once from the owner's ``parallelize``)."""
        self.cp_mesh = cp_mesh
        self.rank = cp_mesh.get_local_rank() if cp_mesh is not None else None
        self.cp_size = cp_mesh.size() if cp_mesh is not None else None

    def _all_to_all(self, x, in_splits, out_splits):
        """The uneven all-to-all (mirrors ``AllToAllTokenDispatcher``'s
        transport selection: ``spmd_types`` eager, the native
        ``all_to_all_single`` under compile/tracing or non-spmd backends)."""
        mesh = self.cp_mesh
        assert mesh is not None, "CPTokenDispatcher must be wired to a CP mesh before an exchange"
        if (
            torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
        ) or get_spmd_backend() != "spmd_types":
            return all_to_all_single(x, out_splits, in_splits, group=mesh.get_group())
        return spmd.all_to_all(
            x,
            mesh.get_group(),
            src=spmd.V,
            dst=spmd.V,
            input_split_sizes=in_splits,
            output_split_sizes=out_splits,
        )

    def gather(self, x: torch.Tensor, plan: CompressedBlockLayout | WindowPlan) -> torch.Tensor:
        """The plan-driven row gather.

        ``plan`` is the ``WindowPlan`` (the swa path — the post-RoPE
        ``swa_k`` rows) or the ``CompressedBlockLayout`` (the compressor
        path — the projected kv/score rows).  Without an exchange the
        plan's ``gather_indices`` are a plain local gather over the
        stream (the identity without a plan — the non-CP window path);
        with one, the exchange gathers the foreign rows and the same
        ``gather_indices`` order ``cat([x_local, recv])`` into the pooled
        stream — a remote gather + permute.  Always returns
        ``[1, N, D]``.
        """
        if plan is None or plan.exchange is None:
            if plan is not None and plan.gather_indices is not None:
                return x.flatten(0, 1)[plan.gather_indices].view(1, -1, *x.shape[2:])
            return x
        ex = plan.exchange
        rows = self._all_to_all(
            x.flatten(0, 1)[ex.send_indices],
            ex.send_splits,
            ex.recv_splits,
        )
        aug = torch.cat([x.flatten(0, 1), rows[ex.recv_offsets]], dim=0)[plan.gather_indices]
        return aug.view(1, -1, *x.shape[2:])

    def select(self, x: torch.Tensor, plan: CompressedBlockLayout) -> torch.Tensor:
        """The plan-indexed container packing: selects the pooled stream's
        ``compressed_rows`` (the kept blocks) and zero-pads them into the
        uniform container grid ``[1, out_width, D]``."""
        # Flatten only the batch+sequence of the 3-D tensors; the 2-D
        # pooled streams select directly.
        x2 = x.flatten(0, 1) if x.ndim > 2 else x
        out = x2 if plan.compressed_rows is None else x2[plan.compressed_rows]
        out_width = plan.out_width
        assert out_width is not None, "select requires the container width"
        container = x.new_zeros((1, out_width, x.shape[-1]))
        if out.shape[0]:
            container[0, : out.shape[0]] = out
        return container
