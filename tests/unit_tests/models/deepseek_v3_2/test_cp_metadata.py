# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU oracle for DeepSeek-V3.2 context-parallel varlen metadata.

The DSV3.2 CP2 integration case proves that the NPU training entry can be
triggered.  These tests cover the metadata hand-off before that entry:
document boundaries, causal K ranges, and gather indices for each rank.  The
expected values below are derived from the document layout and the head/tail
permutation, not from ``CPVarlenMetadata`` itself.  The query-index union check
only guards rank partition coverage; production input sharding and the
Attention consumer remain the responsibility of the integration case.

No NPU or collective is required here.  Cross-rank communication and kernel
execution remain the responsibility of ``dsv3_2_dsa_cp2``.
"""

from bisect import bisect_right
from itertools import pairwise

import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata


class _FakeCPMesh:
    """Small CPU stand-in for the 1-D DeviceMesh metadata contract."""

    ndim = 1

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def size(self) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _global_metadata() -> VarlenMetadata:
    # Two batches with deliberately different document boundaries.  The
    # flattened offsets are [0, 3, 8, 13, 16], which exposes both a document
    # split inside a rank and a split across CP ranks.
    cu = torch.tensor([0, 3, 8, 13, 16], dtype=torch.int32)
    return VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=5, max_k=5)


def _headtail_indices(seq_length: int, cp_world_size: int) -> list[int]:
    """Independent reference for the head/tail rank permutation."""
    chunk = seq_length // (2 * cp_world_size)
    return [
        pos
        for rank in range(cp_world_size)
        for chunk_id in (rank, 2 * cp_world_size - 1 - rank)
        for pos in range(chunk_id * chunk, (chunk_id + 1) * chunk)
    ]


def _reference_layout(
    cu_seq: list[int],
    *,
    batch_size: int,
    seq_length: int,
    cp_world_size: int,
    rank: int,
    permutation: list[int],
) -> tuple[list[int], list[int], list[int], list[int], list[list[int]]]:
    """Derive rank-local metadata with a small, non-tensor reference.

    ``permutation`` maps a position in the head/tail stream to its original
    position.  The reference deliberately uses Python list operations and
    document boundaries instead of reproducing ``CPVarlenMetadata``'s tensor
    implementation.
    """
    shard_len = seq_length // cp_world_size
    rank_q = [
        batch * seq_length + position
        for batch in range(batch_size)
        for position in permutation[rank * shard_len : (rank + 1) * shard_len]
    ]
    inverse = [permutation.index(position) for position in range(seq_length)]

    q_lengths: list[int] = []
    k_lengths: list[int] = []
    gather: list[int] = []
    causal_ranges: list[list[int]] = []
    segment_start = 0
    for index in range(1, len(rank_q) + 1):
        is_end = index == len(rank_q)
        if not is_end:
            previous, current = rank_q[index - 1], rank_q[index]
            previous_doc = bisect_right(cu_seq, previous) - 1
            current_doc = bisect_right(cu_seq, current) - 1
            is_end = previous_doc != current_doc or current != previous + 1
        if not is_end:
            continue

        segment = rank_q[segment_start:index]
        document = bisect_right(cu_seq, segment[0]) - 1
        document_start = cu_seq[document]
        causal_end = segment[-1]
        causal = list(range(document_start, causal_end + 1))
        q_lengths.append(len(segment))
        k_lengths.append(len(causal))
        causal_ranges.append(causal)
        for position in causal:
            batch, local_position = divmod(position, seq_length)
            gather.append(batch * seq_length + inverse[local_position])
        segment_start = index

    def cumulative(lengths: list[int]) -> list[int]:
        result = [0]
        for length in lengths:
            result.append(result[-1] + length)
        return result

    return cumulative(q_lengths), cumulative(k_lengths), gather, rank_q, causal_ranges


def _assert_tensor(actual: torch.Tensor, expected: list[int]) -> None:
    torch.testing.assert_close(actual.cpu(), torch.tensor(expected, dtype=actual.dtype))


def _assert_metadata_dtypes(metadata: CPVarlenMetadata) -> None:
    assert metadata.cu_seq_q.dtype == torch.int32
    assert metadata.cu_seq_k.dtype == torch.int32
    assert metadata.k_global_gather_indices.dtype == torch.int64


def test_cp_varlen_metadata_cp1_is_the_global_layout():
    metadata = _global_metadata()
    expected_q, expected_k, expected_gather, _, _ = _reference_layout(
        metadata.cu_seq_q.tolist(),
        batch_size=2,
        seq_length=8,
        cp_world_size=1,
        rank=0,
        permutation=list(range(8)),
    )
    result = CPVarlenMetadata.from_global(
        metadata,
        _FakeCPMesh(rank=0, world_size=1),
        batch_size=2,
        seq_length=8,
    )

    _assert_metadata_dtypes(result)
    _assert_tensor(result.cu_seq_q, expected_q)
    _assert_tensor(result.cu_seq_k, expected_k)
    _assert_tensor(result.k_global_gather_indices, expected_gather)
    expected_max_q = max(q - p for p, q in pairwise(expected_q))
    expected_max_k = max(k - p for p, k in pairwise(expected_k))
    assert (result.max_q, result.max_k) == (expected_max_q, expected_max_k)


def test_cp_varlen_metadata_cp2_headtail_preserves_cp1_document_semantics():
    metadata = _global_metadata()
    batch_size = 2
    sequence_length = 8
    permutation = _headtail_indices(sequence_length, cp_world_size=2)
    assert permutation == [0, 1, 6, 7, 2, 3, 4, 5]

    load_balancer = _HeadTailLoadBalancer(sequence_length, 2, "cpu")
    torch.testing.assert_close(load_balancer._generate_indices()[0], torch.tensor(permutation, dtype=torch.int32))

    all_rank_q: list[int] = []
    for rank in range(2):
        cu_q, cu_k, gather, rank_q, causal_ranges = _reference_layout(
            metadata.cu_seq_q.tolist(),
            batch_size=batch_size,
            seq_length=sequence_length,
            cp_world_size=2,
            rank=rank,
            permutation=permutation,
        )
        all_rank_q.extend(rank_q)
        result = CPVarlenMetadata.from_global(
            metadata,
            _FakeCPMesh(rank=rank, world_size=2),
            batch_size=batch_size,
            seq_length=sequence_length,
            load_balancer=load_balancer,
        )
        _assert_metadata_dtypes(result)
        _assert_tensor(result.cu_seq_q, cu_q)
        _assert_tensor(result.cu_seq_k, cu_k)
        _assert_tensor(result.k_global_gather_indices, gather)
        assert result.max_q == max(q - p for p, q in pairwise(cu_q))
        assert result.max_k == max(k - p for p, k in pairwise(cu_k))

        # The production gather indices address the head/tail-reordered K
        # stream.  Mapping them back through the independent permutation must
        # recover each CP segment's ordinary causal prefix from the CP=1
        # layout.  This is the actual semantic equivalence being protected.
        restored = [
            permutation[index % sequence_length] + (index // sequence_length) * sequence_length for index in gather
        ]
        token_ids = torch.arange(batch_size * sequence_length)
        restored_tokens = token_ids[restored].tolist()
        for segment, expected_causal in enumerate(causal_ranges):
            k_start, k_end = cu_k[segment], cu_k[segment + 1]
            assert restored_tokens[k_start:k_end] == expected_causal

    assert sorted(all_rank_q) == list(range(2 * sequence_length))
