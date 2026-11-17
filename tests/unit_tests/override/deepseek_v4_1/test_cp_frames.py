# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the AscendC attention reads off a context-parallel forward's frames.

A sharded forward expresses each rank's per-document reach as the frame's per-sequence length, so
these tests pin the operands the operators actually receive: the query axis is the rank's chunk
grid, the KV axes are the gathered row's grids, and ``seqused_ori_kv - seqused_q`` is exactly this
rank's chunk offset inside the document -- the identity the kernels anchor their window and their
compressed causal limit at.  At one degree the same options carry the boundary differences, which
is what makes a ``cp = 1`` versus ``cp > 1`` comparison a comparison of the scheme.
"""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as selector_ascendc
from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as core_ascendc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
    build_fused_metadata,
    shard_asc_v41_metadata,
)

# Only the frames are under test, so any window does.
_WINDOW = 128
# One geometry per entry: a ratio is what the *entry* is built for, so a test states the one
# it exercises instead of a stack-wide table.
_RATIO = 2
# The port's kernel geometry, at the small widths these tests run.  Only the frames are read
# here, but a build always produces the metadata, so the numbers are stated rather than omitted.
_GEOMETRY = dict(num_heads=2, head_dim=4, index_topk=8, index_num_heads=4, index_head_dim=4)


def _row(positions: torch.Tensor, ratio: int):
    """The row a fused entry starts from, built by the reference core's own config.

    ``cu_seq_k`` is that config's ratio applied to the row, so the entry's ratio has to be the
    row's here -- the fused entry never folds a grid its own backend did not produce.
    """
    config = CompressedSparseInnerAttention2.Config(window_size=_WINDOW, softmax_scale=0.5, compress_ratio=ratio)
    return config.build_attention_metadata(positions)


def _masks(bounds: list[int], cp_size: int = 1, cp_rank: int = 0, ratio: int = _RATIO):
    """One geometry's frames, or this rank's after the CP re-cut.

    This is what the AscendC attention's ``AscV41Metadata.prepare_cp_metadata`` runs -- the seam's hook
    delegates to the same function -- and the frames it returns are the ones the operators grid,
    so the tests below read them through the same option builder the kernels are called with.
    """
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)
    entry = build_fused_metadata(
        _row(positions, ratio),
        ratio=ratio,
        geometry=AicpuGeometry(window_size=_WINDOW, **_GEOMETRY),
    )
    if cp_size == 1:
        return entry
    return shard_asc_v41_metadata(
        entry,
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch_size=1,
        seq_len=positions.shape[-1] // cp_size,
    )


def _core_options(metadata, ratio: int = _RATIO):
    """The options the SMLA kernels are called with, read off the entry's own frames."""
    return core_ascendc.kernel_options(
        ratio=ratio,
        q=metadata.q,
        swa_k=metadata.swa_k,
        cmp_k=metadata.cmp_k,
        window_size=_WINDOW,
    )


def _selector_options(metadata, ratio: int = _RATIO):
    """The same, for the selector's two operators: its queries and the pool grid they address."""
    return selector_ascendc.kernel_options(ratio=ratio, q=metadata.q, cmp_k=metadata.cmp_k)


def _lengths(bounds: list[int], dtype=torch.int32) -> torch.Tensor:
    return torch.tensor([end - begin for begin, end in pairwise(bounds)], dtype=dtype)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("bounds", [[0, 8, 24], [0, 8, 32, 40]], ids=["two-docs", "three-docs"])
def test_one_degree_passes_the_boundary_differences(bounds, ratio):
    """At one degree every per-sequence length is the boundary difference, so they are no-ops."""
    options = _core_options(_masks(bounds, ratio=ratio), ratio)
    lengths = _lengths(bounds)
    assert torch.equal(options["cu_seqlens_q"], torch.tensor(bounds, dtype=torch.int32))
    assert torch.equal(options["seqused_q"], lengths)
    assert torch.equal(options["seqused_ori_kv"], lengths)
    if ratio == 0:
        # A ratio-0 layer has no compressed stream, and the operators spell that ``None``.
        assert options["seqused_cmp_kv"] is None
        assert options["cu_seqlens_cmp_kv"] is None
        assert options["cmp_residual_kv"] is None
    else:
        assert torch.equal(options["seqused_cmp_kv"], lengths // ratio)
        assert torch.equal(options["cu_seqlens_cmp_kv"], torch.tensor(bounds, dtype=torch.int32) // ratio)


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 1), (4, 3), (8, 7)], ids=["R2r1", "R4r3", "R8r7"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_the_anchor_is_this_ranks_chunk_offset(cp_size, cp_rank, ratio):
    """The core gets chunk queries over the row, and the two lengths differ by the chunk offset.

    ``seqused_ori_kv - seqused_q = r * Q_d`` is the whole kernel contract of this mode: it is how
    the window and the compressed causal limit end up anchored at this rank's chunk instead of at
    the row's end.
    """
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    reach = (cp_rank + 1) * chunk
    options = _core_options(_masks(bounds, cp_size, cp_rank, ratio=ratio), ratio)

    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["cu_seqlens_ori_kv"], row)
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["seqused_ori_kv"], reach)
    assert torch.equal(options["seqused_ori_kv"] - options["seqused_q"], cp_rank * chunk)
    assert torch.equal(options["seqused_cmp_kv"], reach // ratio)
    # The pool grid is the row's own, folded by the backend that built the entry: it is the same
    # boundaries at every rank because the KV streams stay the whole row.
    assert torch.equal(options["cu_seqlens_cmp_kv"], (row // ratio).to(torch.int32))


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 1), (4, 3)], ids=["R2r1", "R4r3"])
def test_the_selector_gets_the_same_chunk_and_reach(cp_size, cp_rank):
    """The selection is scored in the query frame against the row's compressed grid.

    Without these two lengths a rank would score its chunks against the whole row and spend its
    top-k on blocks it may not read.  They sit in the same option dict as the boundaries, because
    every operator this selector can pick takes the same pair.
    """
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    masks = _masks(bounds, cp_size, cp_rank)
    options = _selector_options(masks)

    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["cu_seqlens_k"], (row // 2).to(torch.int32))
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["seqused_k"], (cp_rank + 1) * chunk // 2)
