# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Context-Parallel awareness of DeepSeek-V4 SMLA metadata.

Pins ``_cp_smla_seq_lengths`` (the per-rank query/key lengths used to size the
SMLA attention masks / op metadata) to the shapes that
``CompressorAttentionCP._post_hook`` actually produces, so the metadata cannot
drift away from the tensors the kernel receives under CP.

Also pins the CP mask-handler registry wiring: importing ``npu_smla`` registers
``_smla_cp_mask_handler`` so the generic ``adjust_cp_mask`` dispatch owns SMLA
varlen metadata (resizes it per rank, skips sequence-sharding) and defers every
other mask type to upstream.

Run with::

    pytest tests/unit_tests/converters/test_npu_smla_cp_metadata.py -v
"""

import pytest
import torch

from torchtitan_npu.converters.kernels.npu_smla import (
    _cp_smla_seq_lengths,
    _smla_cp_mask_handler,
    DeepSeekV4SMLAAttentionMasks,
)
from torchtitan_npu.distributed.context_parallel import adjust_cp_mask


def _post_hook_lengths(seq_len, cp_degree, cp_rank, ratio):
    """Reproduce CompressorAttentionCP._post_hook's per-rank kernel shapes."""
    local_s = seq_len // cp_degree
    slice_blocks = (cp_rank + 1) * local_s // ratio
    target_ori_len = slice_blocks * ratio
    return local_s, target_ori_len, slice_blocks


def test_no_cp_is_identity():
    # cp_degree == 1 must reproduce the original non-CP metadata sizing.
    assert _cp_smla_seq_lengths(4096, 1, 0) == (4096, 4096)


@pytest.mark.parametrize(
    "seq_len, cp_degree, cp_rank, expected",
    [
        (4096, 2, 0, (2048, 2048)),  # rank0 attends to its own shard only
        (4096, 2, 1, (2048, 4096)),  # rank1 causally attends to the full seq
        (8192, 4, 0, (2048, 2048)),
        (8192, 4, 3, (2048, 8192)),
    ],
)
def test_per_rank_lengths(seq_len, cp_degree, cp_rank, expected):
    assert _cp_smla_seq_lengths(seq_len, cp_degree, cp_rank) == expected


@pytest.mark.parametrize("cp_degree, cp_rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_matches_post_hook_shapes(cp_degree, cp_rank, ratio):
    seq_len = 4096
    q_seq_len, kv_seq_len = _cp_smla_seq_lengths(seq_len, cp_degree, cp_rank)
    local_s, target_ori_len, slice_blocks = _post_hook_lengths(
        seq_len, cp_degree, cp_rank, ratio
    )
    assert q_seq_len == local_s
    assert kv_seq_len == target_ori_len
    assert kv_seq_len // ratio == slice_blocks


def test_seq_len_not_divisible_by_cp_raises():
    with pytest.raises(ValueError):
        _cp_smla_seq_lengths(4097, 2, 0)


# --- CP mask-handler registry wiring -----------------------------------------
# ``npu_smla`` calls ``register_cp_mask_handler(_smla_cp_mask_handler)`` at
# import time, so ``adjust_cp_mask`` should own SMLA masks and defer the rest.


class _FakeCpMesh:
    """Minimal cp_mesh stub exposing the two methods the handler calls."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _make_smla_masks(seq_len, batch_size=2, ratios=(1, 4, 128)):
    full = torch.full((batch_size,), seq_len, dtype=torch.int32)
    return DeepSeekV4SMLAAttentionMasks(
        actual_seq_qlen=full.clone(),
        actual_seq_klen=full.clone(),
        cmp_residual_kv={
            ratio: torch.full((batch_size,), seq_len % ratio, dtype=torch.int32)
            for ratio in ratios
        },
        batch_size=batch_size,
        seq_len=seq_len,
    )


@pytest.mark.parametrize(
    "seq_len, cp_degree, cp_rank",
    [
        (384, 2, 0),  # kv_len=192 -> 192 % 128 == 64, a non-trivial residual
        (384, 2, 1),
        (8192, 4, 0),
        (8192, 4, 3),
    ],
)
def test_registry_dispatches_smla_masks_per_rank(seq_len, cp_degree, cp_rank):
    # adjust_cp_mask must route SMLA masks to the registered handler, mark them
    # handled (so they are NOT sequence-sharded) and resize every field to this
    # rank's per-rank q/kv lengths.
    masks = _make_smla_masks(seq_len)
    handled, adjusted = adjust_cp_mask(masks, _FakeCpMesh(cp_degree, cp_rank))
    q_len, kv_len = _cp_smla_seq_lengths(seq_len, cp_degree, cp_rank)
    assert handled is True
    assert adjusted.q_seq_len == q_len
    assert adjusted.kv_seq_len == kv_len
    assert int(adjusted.actual_seq_qlen[0]) == q_len
    assert int(adjusted.actual_seq_klen[0]) == kv_len
    # Residuals must be recomputed from the per-rank kv_len, not the full seq_len.
    for ratio, residual in adjusted.cmp_residual_kv.items():
        assert int(residual[0]) == kv_len % ratio


def test_registry_defers_unknown_mask_type():
    # A mask type no handler owns is returned unchanged with handled=False, so
    # upstream cp_shard performs its normal sequence-sharding.
    foreign = torch.ones(4, 4)
    handled, returned = adjust_cp_mask(foreign, _FakeCpMesh(2, 0))
    assert handled is False
    assert returned is foreign


def test_handler_returns_none_for_unknown_mask_type():
    # The handler itself defers (returns None) on non-SMLA inputs.
    assert _smla_cp_mask_handler(object(), _FakeCpMesh(2, 0)) is None


def test_registry_handles_smla_without_cp_as_identity():
    # cp_degree == 1 is still owned by SMLA (handled=True) but left unchanged.
    masks = _make_smla_masks(4096)
    handled, adjusted = adjust_cp_mask(masks, _FakeCpMesh(1, 0))
    assert handled is True
    assert adjusted is masks
