# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The local query and key frames passed to the AscendC kernels."""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.context_parallel import CPAttention
from torchtitan_npu.override.deepseek_v4.cp_metadata import DocumentCPLoadBalancer, DocumentCPPlan
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as selector_ascendc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as core_ascendc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

# Only the frames are under test, so any window does.
_WINDOW = 128
# One geometry per entry: a ratio is what the *entry* is built for, so a test states the one
# it exercises instead of a stack-wide table.
_RATIO = 2
# The port's kernel geometry, at the small widths these tests run.  Only the frames are read
# here, but a build always produces the metadata, so the numbers are stated rather than omitted.
_GEOMETRY = dict(num_heads=2, head_dim=4, index_topk=8, index_num_heads=4, index_head_dim=4)


@pytest.fixture(autouse=True)
def stub_aicpu_record_generation(monkeypatch):
    monkeypatch.setattr(core_ascendc, "build_aicpu_metadata", lambda **kwargs: None)


def _masks(bounds: list[int], cp_size: int = 1, cp_rank: int = 0, ratio: int = _RATIO):
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)])
    kernel = AscV41SparseAttention.Config(window_size=_WINDOW, softmax_scale=0.5, compress_ratio=ratio, **_GEOMETRY)
    if cp_size == 1:
        return kernel.build_attention_metadata(positions)

    packed = CPAttention.Config(kernel=kernel).build_attention_metadata(positions)
    layout = DocumentCPLoadBalancer.Config().build(
        seq_len=positions.numel(), attention_metadata=packed, cp_size=cp_size
    )
    permutation = layout.generate_permutation()
    plan = DocumentCPPlan.build(permutation, packed.row.q.cu_seqlens, cp_size=cp_size, cp_rank=cp_rank, head_tail=False)
    return CPAttention.prepare_cp_metadata(packed, permutation=permutation, plan=plan).entries[0]


def _core_options(metadata, ratio: int = _RATIO):
    """The options the SMLA kernels are called with, read off the entry's own frames.

    ``entry_options`` is what the forward and the backward pass to the kernels, so this reads
    exactly what they are handed -- the entry's ``a3_workaround`` frames on arch22.
    """
    return core_ascendc.entry_options(metadata, ratio=ratio, window_size=_WINDOW)


def _selector_options(metadata, ratio: int = _RATIO):
    """The same, for the selector's two operators: its queries and the pool it addresses."""
    return selector_ascendc.entry_options(ratio, metadata)


def _prefix(lengths: torch.Tensor) -> torch.Tensor:
    """The frame of a per-document prefix: the running sum of the reach, from zero."""
    return torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0).to(torch.int32)])


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


@pytest.mark.parametrize("is_a5", [False, True], ids=["a3", "a5"])
def test_a_ratio_one_frame_keeps_the_targets_residual_spelling(monkeypatch, is_a5):
    """The selector omits the ratio-1 residual; the sparse core uses its target's convention."""
    monkeypatch.setattr(core_ascendc, "_IS_A5", is_a5)
    monkeypatch.setattr(selector_ascendc, "_IS_A5", is_a5)
    metadata = _masks([0, 8, 24], ratio=1)

    assert _selector_options(metadata, 1)["cmp_residual_k"] is None
    core_residual = _core_options(metadata, 1)["cmp_residual_kv"]
    if is_a5:
        assert core_residual is None
    else:
        torch.testing.assert_close(core_residual, torch.zeros(2, dtype=torch.int32))

    pooled = _masks([0, 8, 24], ratio=2)
    assert _core_options(pooled, 2)["cmp_residual_kv"] is pooled.cmp_k.residual
    assert _selector_options(pooled, 2)["cmp_residual_k"] is pooled.cmp_k.residual


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 1), (4, 3), (8, 7)], ids=["R2r1", "R4r3", "R8r7"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_the_anchor_is_this_ranks_chunk_offset(cp_size, cp_rank, ratio):
    """The sliding-window buffer ends at this rank's last query."""
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    reach = (cp_rank + 1) * chunk
    metadata = _masks(bounds, cp_size, cp_rank, ratio=ratio)
    options = _core_options(metadata, ratio)

    assert torch.equal(metadata.q.cu_seqlens, (row // cp_size).to(torch.int32))
    assert torch.equal(metadata.q.seqused, chunk)
    assert torch.equal(metadata.swa_k.cu_seqlens, _prefix(reach))
    assert torch.equal(metadata.swa_k.seqused, reach)

    # The kernel's frames: the reach laid out as the frame, so ``seqused`` is its boundary
    # difference and the window is anchored at the chunk's last row.
    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["cu_seqlens_ori_kv"], _prefix(reach))
    assert torch.equal(options["seqused_ori_kv"], reach)
    if not _IS_A5:
        assert torch.equal(options["seqused_ori_kv"], torch.diff(options["cu_seqlens_ori_kv"]))
    cmp_row = (row // ratio).to(torch.int32)
    assert torch.equal(options["cu_seqlens_cmp_kv"], cmp_row if _IS_A5 else _prefix(reach // ratio))
    assert torch.equal(options["seqused_cmp_kv"], reach // ratio)
    if not _IS_A5:
        assert torch.equal(options["seqused_cmp_kv"], torch.diff(options["cu_seqlens_cmp_kv"]))


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 0), (2, 1), (4, 3)], ids=["R2r0", "R2r1", "R4r3"])
def test_the_selector_gets_the_same_chunk_and_reach(cp_size, cp_rank):
    """The selection is scored in the query frame against this rank's reach.

    The selector right-aligns its selection at the frame's end too, which is why it reads its keys
    with the entry's ``a3`` pool: the picks stay document-local, so the same numbers still address
    the whole-row pool SMLA/SMLAG are handed.  Without either the rank would spend its top-k on
    blocks it may not read.
    """
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    reach = (cp_rank + 1) * chunk
    metadata = _masks(bounds, cp_size, cp_rank)
    options = _selector_options(metadata)

    # The row's pool grid stays the entry's ``cu_seq_k``; ``cmp_k`` is the frame the selector reads:
    # the row itself where the kernel honours ``seqused``, this rank's reach where it does not.
    assert torch.equal(metadata.cu_seq_k, (row // 2).to(torch.int32))
    expected = (row // 2).to(torch.int32) if _IS_A5 else _prefix(reach // 2)
    assert torch.equal(metadata.cmp_k.cu_seqlens, expected)
    assert torch.equal(metadata.cmp_k.seqused, reach // 2)
    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["cu_seqlens_k"], metadata.cmp_k.cu_seqlens)
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["seqused_k"], reach // 2)
    if not _IS_A5:
        # arch22's frame *is* the reach, so its boundary difference is the reach itself.
        assert torch.equal(options["seqused_k"], torch.diff(options["cu_seqlens_k"]))
