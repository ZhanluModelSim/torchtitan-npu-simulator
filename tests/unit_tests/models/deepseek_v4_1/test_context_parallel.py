# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 context-parallel query plan: the movement, the frames, and their invariants.

The layout is one packed row's coordinate change, so every property worth pinning is
checkable without a device and without a process group: which rows a rank ends up holding,
that the movement is a permutation of the slab, that the receive side's closed form agrees
with the row-by-row derivation, and that the frames say the queries are the chunks while the
KV streams stay the row.

The collectives are emulated in process, from the plan's own tables -- the same tables the
forward hands to ``all_to_all_single`` -- because a plan is what this file is about; the
module's own call into the collective is a thin wrapper over them.
"""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.context_parallel import (
    ExchangeMetadata,
    QueryExchange,
    cp_plan,
)
from torchtitan_npu.models.deepseek_v4_1.model import (
    DeepSeekV41Metadata,
    DeepSeekV41Model,
)
from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
    build_fused_metadata,
    shard_asc_v41_metadata,
)

# One geometry per entry, so the frame tests state the ratio they exercise; the alignment the
# launcher pads to is ``cp_size * this ratio`` per entry (the stack-wide lcm is the dataloader's).
_RATIO = 2


def _positions(*documents: int) -> torch.Tensor:
    """One packed row whose documents have the given lengths, positions resetting."""
    flat: list[int] = []
    for length in documents:
        flat += list(range(length))
    return torch.tensor([flat])


def _bounds(cu: list[int]) -> torch.Tensor:
    return torch.tensor(cu, dtype=torch.int32)


# The port's record geometry, at the widths these CPU tests run.  Only the frames are read
# here, but a build always produces the records, so the numbers are stated rather than omitted.
_GEOMETRY = AicpuGeometry(window_size=8, num_heads=2, head_dim=4, index_topk=4, index_num_heads=4, index_head_dim=4)


def _row(positions: torch.Tensor, *, ratio: int = _RATIO) -> DeepSeekV41Metadata:
    """The row's own metadata, from the builder every V4.1 attention backend calls.

    ``cu_seq_k`` is the backend's own ratio applied to the row's boundaries, so the row is built
    at the ratio of the entry that consumes it -- the fold happens once, where the ratio is known,
    and no consumer divides a boundary set it was handed.
    """
    config = CompressedSparseInnerAttention2.Config(window_size=8, softmax_scale=0.5, compress_ratio=ratio)
    return config.build_attention_metadata(positions)


def _masks(positions: torch.Tensor, cp_size: int = 1, cp_rank: int = 0) -> DeepSeekV41Metadata:
    """The row's frames, or this rank's after the backend's CP transform.

    ``seq_len`` is the **slab** width, which is what the framework hands every rank; the
    transform is given the pre-shard row because that is the geometry it derives the slab
    from.
    """
    # The records a build always produces are incidental here -- these tests read the frames --
    # so the geometry is a stated small layer rather than an omitted one.
    entry = build_fused_metadata(_row(positions), ratio=_RATIO, geometry=_GEOMETRY)
    if cp_size == 1:
        return entry
    return shard_asc_v41_metadata(
        entry,
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch_size=1,
        seq_len=positions.numel() // cp_size,
    )


def _oracle_splits(cu: list[int], cp_size: int, seq_len: int) -> torch.Tensor:
    """``M[source][dest]`` by counting every slab row's destination, the slow obvious way."""
    columns = []
    bounds = _bounds(cu)
    chunk = (bounds[1:] - bounds[:-1]) // cp_size
    for source in range(cp_size):
        rows = torch.arange(source * seq_len, (source + 1) * seq_len)
        document = torch.bucketize(rows, bounds, right=True) - 1
        columns.append(torch.bincount((rows - bounds[document]) // chunk[document], minlength=cp_size))
    return torch.stack(columns)


def _oracle_held_rows(cu: list[int], cp_size: int, rank: int) -> list[int]:
    """Rank ``rank``'s rows: chunk ``rank`` of every document, in document order."""
    rows: list[int] = []
    for start, end in pairwise(cu):
        chunk = (end - start) // cp_size
        rows += range(start + rank * chunk, start + (rank + 1) * chunk)
    return rows


def _oracle_positions(cu: list[int], cp_size: int, rank: int) -> list[int]:
    bounds = _bounds(cu)
    held = torch.tensor(_oracle_held_rows(cu, cp_size, rank))
    return (held - bounds[torch.bucketize(held, bounds, right=True) - 1]).tolist()


_CASES = [
    ((0, 8, 24), 2),
    ((0, 16, 48), 4),
    ((0, 8, 24, 48), 2),
    ((0, 64), 8),
    ((0, 8, 32, 40), 4),
    ((0, 16, 48, 80), 8),
    ((0, 32, 64, 96), 4),
    ((0, 8, 16, 24, 32), 4),
    ((0, 4096, 8192, 12288), 8),
]


def test_the_bounds_are_read_off_the_row():
    """The boundaries come from the position resets, which is where every frame starts."""
    for cu, _ in _CASES:
        row = _row(_positions(*[e - s for s, e in pairwise(cu)]))
        assert row.cu_seq_q.tolist() == list(cu)
        # The pool grid is the same backend's ratio applied once: the row carries it too, so the
        # reference core and every fused port address one set of boundaries.
        assert row.cu_seq_k.tolist() == [bound // _RATIO for bound in cu]


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_plan_matches_the_row(cu, cp_size):
    """Both split vectors and the positions, against a derivation that counts every row.

    The receive side is the one worth arguing about: it is a closed form over the documents'
    chunk intervals rather than a histogram of the rows this rank receives, so it is checked
    against the row-by-row matrix and not against itself.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    matrix = _oracle_splits(cu, cp_size, seq_len)
    assert matrix.sum().item() == cu[-1], "the splits must partition the row"
    for rank in range(cp_size):
        plan = cp_plan(_bounds(cu), cp_size, rank, seq_len)
        assert list(plan.send_split) == matrix[rank].tolist()
        assert list(plan.recv_split) == matrix[:, rank].tolist()
        assert plan.positions_q.tolist() == _oracle_positions(cu, cp_size, rank)


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_movement_is_a_permutation_of_the_slab(cu, cp_size):
    """Every rank presents one slab and receives one slab, addressed row by row.

    ``send_index`` being a permutation is what makes the reverse movement the inverse rather
    than the adjoint (nothing is copied, so nothing accumulates), and it is also what makes
    the receiver's buffer the held rows' own order -- which is why there is no scatter index.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    chunk = (torch.tensor(cu[1:]) - torch.tensor(cu[:-1])) // cp_size
    for rank in range(cp_size):
        plan = cp_plan(_bounds(cu), cp_size, rank, seq_len)
        assert sorted(plan.send_index.tolist()) == list(range(seq_len))
        assert sum(plan.send_split) == seq_len
        assert sum(plan.recv_split) == seq_len
        # A document's chunk is one interval and spans at most two slabs; the closed form
        # leans on that, so the property is pinned rather than assumed.
        start = torch.tensor(cu[:-1]) + rank * chunk
        span = (start + chunk - 1) // seq_len - start // seq_len + 1
        assert int(span.max()) <= 2


@pytest.mark.parametrize(
    "cu, cp_size, why",
    [
        ((0, 24, 32), 16, "a length that is not a multiple of cp_size"),
        ((0, 22, 32), 4, "a length that is not a multiple of cp_size"),
        ((0, 32, 32, 32), 8, "a zero-length document"),
    ],
)
def test_the_plan_rejects_a_row_it_cannot_grid(cu, cp_size, why):
    """A row the launcher should never have produced fails here rather than downstream.

    ``// cp_size`` on a ragged length is silent, and a zero-length chunk turns the receive
    side's interval arithmetic into an out-of-range index, so both are checked up front.
    """
    with pytest.raises(ValueError, match="positive multiple of cp_size"):
        cp_plan(_bounds(list(cu)), cp_size, 0, max(1, list(cu)[-1] // cp_size))


@pytest.mark.parametrize(
    "documents, cp_size",
    [((48, 20), 4), ((16, 12), 4), ((32, 24), 8)],
    ids=[
        "20 is not a multiple of cp_size * ratio = 8",
        "12 is not a multiple of 8",
        "24 is not a multiple of cp_size * ratio = 16",
    ],
)
def test_the_sharded_path_checks_the_launcher_contract(documents, cp_size):
    """A row the launcher should never have produced fails here, per document.

    ``cp_size * compress_ratio`` is the alignment one entry is padded to (the stack-wide lcm is
    the dataloader's); without it a chunk is ragged (rows land in the wrong place) and a compressed
    prefix is a partial group (the kernels grid an axis the pooling did not produce).  Both failures
    are silent downstream, so the check is a precondition rather than a diagnostic.
    """
    with pytest.raises(ValueError, match="document alignment"):
        _masks(_positions(*documents), cp_size, cp_size - 1)


def test_the_exchange_carries_the_movement_and_the_positions():
    """The container holds nothing a consumer could work out for itself."""
    plan = cp_plan(_bounds([0, 8, 24]), 2, 1, 12)
    assert isinstance(plan, ExchangeMetadata)
    assert set(plan.__slots__) == {"send_index", "send_split", "recv_split", "positions_q"}
    assert plan.send_index.dtype == torch.int64 and plan.positions_q.dtype == torch.int64


def test_the_exchange_is_the_identity_without_a_plan():
    """No plan means no movement, so the same call serves the unsharded forward.

    The early return is what keeps the call sites branch-free and lets ``cp = 1`` run without a
    process group at all; an unwired exchange with a plan is still an error rather than a silent
    fall back onto the default group.
    """
    exchange = QueryExchange(QueryExchange.Config())
    tensor = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    assert exchange.group is None
    assert exchange.permute(tensor, None) is tensor
    assert exchange.unpermute(tensor, None) is tensor
    with pytest.raises(RuntimeError, match="parallelize"):
        exchange.permute(tensor, cp_plan(_bounds([0, 8, 24]), 2, 0, 12))


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_movement_round_trips_the_whole_row(cu, cp_size):
    """Moving the query side out and back is the identity, row for row.

    Emulated in process: each rank presents ``slab[send_index]``, the slices are routed to
    their peer's inbox in peer order, the receiver takes the concatenation as its own order
    (no scatter index), and the reverse swaps the split vectors and scatters through
    ``send_index``.  Every rank's present is taken from **its own** slab, which is the
    property that makes the tables usable at all.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    plans = [cp_plan(_bounds(cu), cp_size, rank, seq_len) for rank in range(cp_size)]
    slabs = [torch.arange(r * seq_len, (r + 1) * seq_len, dtype=torch.float32).reshape(-1, 1) for r in range(cp_size)]
    held = _emulate(slabs, plans)
    for rank in range(cp_size):
        assert held[rank].reshape(-1).tolist() == [float(row) for row in _oracle_held_rows(cu, cp_size, rank)]
    back = _emulate(held, plans, inverse=True)
    for rank in range(cp_size):
        torch.testing.assert_close(back[rank], slabs[rank])


def _emulate(tensors: list[torch.Tensor], plans: list[ExchangeMetadata], *, inverse: bool = False):
    """Run the pair of collectives over every rank, in process."""
    size = len(tensors)
    presents = [t if inverse else t[plans[r].send_index] for r, t in enumerate(tensors)]
    splits = [
        (plans[r].recv_split, plans[r].send_split) if inverse else (plans[r].send_split, plans[r].recv_split)
        for r in range(size)
    ]
    offsets = []
    for send, _ in splits:
        running, rows = [0], 0
        for value in send:
            rows += value
            running.append(rows)
        offsets.append(running)
    outs = []
    for rank in range(size):
        received = torch.cat(
            [presents[source][offsets[source][rank] : offsets[source][rank + 1]] for source in range(size)]
        )
        if inverse:
            slab = torch.empty((sum(splits[rank][0]), *received.shape[1:]), dtype=received.dtype)
            slab[plans[rank].send_index] = received
            outs.append(slab)
        else:
            outs.append(received)
    return outs


def test_the_frames_say_the_queries_are_the_chunks_and_the_kv_streams_are_the_row():
    """The three frames under CP: chunk queries, row KV streams, and the reach in ``seqused``.

    ``seqused`` is the whole point of this layout on the kernel side: the KV tensors are the
    gathered row -- so their ``cu_seqlens`` must address that row -- while the queries only
    reach ``K_d = (cp_rank + 1) · Q_d`` of each document, which is what the kernels' end-aligned
    window and causal limit are anchored at.
    """
    for cu, cp_size in _CASES:
        chunk = (torch.tensor(cu[1:]) - torch.tensor(cu[:-1])) // cp_size
        for rank in range(cp_size):
            masks = _masks(_positions(*[e - s for s, e in pairwise(cu)]), cp_size, rank)
            assert masks.exchange is not None
            assert masks.q.cu_seqlens.tolist() == (torch.tensor(cu) // cp_size).tolist()
            assert masks.q.seqused.tolist() == chunk.tolist()
            assert masks.swa_k.cu_seqlens.tolist() == list(cu)
            assert masks.swa_k.seqused.tolist() == ((rank + 1) * chunk).tolist()
            frame = masks.cmp_k
            assert frame is not None and masks.ratio == _RATIO
            assert frame.cu_seqlens.tolist() == (torch.tensor(cu) // _RATIO).tolist()
            assert frame.seqused.tolist() == (((rank + 1) * chunk) // _RATIO).tolist()
            assert frame.residual.tolist() == [0] * (len(cu) - 1)
            # ``positions_q`` rides shaped like the input, so the oracle is compared flat.
            assert masks.positions_q.reshape(-1).tolist() == _oracle_positions(list(cu), cp_size, rank)


def test_the_unsharded_forward_is_unchanged():
    """CP=1 keeps every frame the row, for every stack.

    The grid's arithmetic degenerates on its own at ``cp_size == 1`` -- one chunk per document
    *is* the document -- so the unsharded path is the same derivation rather than a branch, and
    the metadata it builds is the one geometry both the torch-native and the fused paths read.
    """
    masks = _masks(_positions(8, 16))
    assert masks.exchange is None
    torch.testing.assert_close(masks.positions_q, _positions(8, 16))
    assert masks.q.cu_seqlens.tolist() == [0, 8, 24]
    assert masks.q.seqused.tolist() == [8, 16]
    assert masks.swa_k.cu_seqlens.tolist() == [0, 8, 24]
    assert masks.swa_k.seqused.tolist() == [8, 16]
    assert masks.cmp_k is not None and masks.ratio == _RATIO
    assert masks.cmp_k.seqused.tolist() == [4, 8]
    assert masks.cmp_k.cu_seqlens.tolist() == [0, 4, 12]


def test_the_cp_transform_carries_the_plan_and_recuts_the_frames():
    """The transform is handed the **global** row and returns this rank's chunks.

    The plan cannot be derived from the slab and the frames cannot either, so this is the one
    place both the whole row and the rank meet: the split vectors say where this rank's rows
    go, and the frames address the chunks those rows become.  The input slice itself is the
    framework's -- one even ``Shard(1)`` slab per rank when no load balancer is configured --
    which is why the transform is handed the shard *width* rather than the tensor.
    """
    row = _positions(8, 16)
    for cp_size in (2, 4):
        for cp_rank in range(cp_size):
            slab = row.numel() // cp_size
            masks = _masks(row, cp_size, cp_rank)
            plan = masks.exchange
            assert plan is not None
            assert list(plan.send_split) == _oracle_splits([0, 8, 24], cp_size, slab)[cp_rank].tolist()
            assert sum(plan.recv_split) == slab
            assert sorted(plan.send_index.tolist()) == list(range(slab))
            # The carried positions ride shaped like the input, so no consumer reshapes.
            torch.testing.assert_close(masks.positions_q, plan.positions_q.reshape(1, slab))
            assert masks.q.seqused.tolist() == [8 // cp_size, 16 // cp_size]
            assert masks.swa_k.seqused.tolist() == [(cp_rank + 1) * 8 // cp_size, (cp_rank + 1) * 16 // cp_size]


def test_the_transform_rejects_a_slab_the_row_does_not_grid():
    """A shard width the documents do not add up to means the frames describe another row.

    The launcher's cut is even, so the chunks sum to the slab by construction; anything else
    means the metadata came from a different row than the inputs were sliced from, and the
    kernels would grid it silently.
    """
    with pytest.raises(ValueError, match="not the chunks' total"):
        shard_asc_v41_metadata(
            build_fused_metadata(_row(_positions(8, 16)), ratio=_RATIO, geometry=_GEOMETRY),
            cp_size=2,
            cp_rank=0,
            batch_size=2,
            seq_len=12,
        )


def test_the_model_takes_a_context_parallel_degree():
    """The text stack admits CP; the layout above is what it was waiting for."""
    assert DeepSeekV41Model.Config.accepts_context_parallel is True
