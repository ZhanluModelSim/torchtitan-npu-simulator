# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The aligned text dataloader's padding and row-cut contracts."""

from pathlib import Path

import pytest
import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import preprocess, tiny_v41_model
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.model import compression_alignment
from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import (
    AlignedHuggingfaceDataloader,
    AlignedTextDataset,
    pad_segments_to_multiple,
)

ASSETS = Path(__file__).resolve().parents[3] / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=str(ASSETS / "deepseek_v3"))


def _loader(tokenizer, *, rank=0, world=1, alignment=2, seq_len=512, infinite=False):
    return AlignedHuggingfaceDataloader.Config(
        dataset="c4_test",
        dataset_path=str(ASSETS / "c4_test"),
        infinite=infinite,
        per_doc_alignment=alignment,
    ).build(
        tokenizer=tokenizer,
        dp_rank=rank,
        dp_world_size=world,
        seq_len=seq_len,
        local_batch_size=1,
    )


def _stream(loader):
    """Every supervisable (token, label) pair and token count the loader emitted."""
    pairs: list[tuple[int, int]] = []
    tokens_total = 0
    for inputs, labels in loader:
        tokens = inputs["input"][0]
        row_labels = labels[0]
        tokens_total += int(tokens.numel())
        supervised = row_labels != IGNORE_INDEX
        pairs.extend(zip(tokens[:-1][supervised[:-1]].tolist(), row_labels[:-1][supervised[:-1]].tolist(), strict=True))
    return tokens_total, pairs


def test_compression_alignment_is_the_lcm_of_the_pooling_ratios():
    assert compression_alignment(()) == 1
    assert compression_alignment((0, 0)) == 1
    assert compression_alignment((0, 1, 1)) == 1
    assert compression_alignment((0, 2, 1)) == 2
    assert compression_alignment((0, 2, 4)) == 4
    # V4.1's own ratios: the 16 ratio-2 layers pool, everything else does not.
    assert compression_alignment((0, 0) + (2,) * 18 + (1,) * 20) == 2


def test_pad_segments_to_multiple_appends_only_pads():
    ids = [2002, 11, 12]
    labels = [IGNORE_INDEX, 12, 2003]
    padded_ids, padded_labels = pad_segments_to_multiple((ids, labels), multiple=2)
    assert padded_ids == [*ids, 0]
    assert padded_labels == [*labels, IGNORE_INDEX]
    # The original ids and labels are a prefix of the padded ones, in order.
    assert padded_ids[: len(ids)] == ids and padded_labels[: len(labels)] == labels
    # An already-aligned segment is returned untouched.
    assert pad_segments_to_multiple((ids, labels), multiple=3) == (ids, labels)
    assert pad_segments_to_multiple((ids, labels), multiple=1) == (ids, labels)
    # A single-token segment is padded to the alignment, not dropped.
    assert pad_segments_to_multiple(([7], [IGNORE_INDEX]), multiple=4) == ([7, 0, 0, 0], [IGNORE_INDEX] * 4)
    # Tensors go through the same way: the pads are appended, never prepended.
    tensor_ids, tensor_labels = pad_segments_to_multiple((torch.tensor(ids), torch.tensor(labels)), multiple=2)
    assert tensor_ids.tolist() == padded_ids
    assert tensor_labels.tolist() == padded_labels
    # A length mismatch is rejected instead of silently truncating one side.
    with pytest.raises(ValueError, match="same length"):
        pad_segments_to_multiple((ids, labels[:-1]), multiple=2)


def test_alignment_adds_a_bounded_number_of_untrained_tokens(tokenizer):
    """Padding adds at most one untrained position per document.

    The pads themselves are never targets.  What padding does cost is one target per
    odd-length document: labels are shifted within the document, so the EOS token's
    label pointed at the pad that now follows it and is no longer supervised.
    """
    aligned_tokens, aligned = _stream(_loader(tokenizer, alignment=2))
    unaligned_tokens, unaligned = _stream(_loader(tokenizer, alignment=1))
    # At most one pad per emitted document, and each costs at most one EOS target.
    assert 0 <= aligned_tokens - unaligned_tokens <= len(aligned)
    assert abs(len(aligned) - len(unaligned)) <= len(aligned)
    assert aligned and unaligned


def test_row_length_is_a_multiple_of_the_alignment(tokenizer):
    for inputs, _ in _loader(tokenizer, alignment=2):
        assert int(inputs["input"].numel()) % 2 == 0


def test_row_cut_rebases_positions(tokenizer):
    """A row is cut to exactly ``seq_len``; the row's own positions start at 0 and
    count up, so every row is an independent pooling problem."""
    loader = _loader(tokenizer, alignment=2, seq_len=128)
    inputs, _ = next(iter(loader))
    positions = inputs["positions"][0]
    assert int(positions.numel()) == 128
    assert int(positions[0]) == 0
    assert torch.equal(positions, torch.arange(128))


def test_alignment_must_divide_seq_len(tokenizer):
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=3, seq_len=128)
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=0)
    # The dataset validates before it touches the tokenizer or the dataset path.
    with pytest.raises(ValueError, match="multiple of per_doc_alignment"):
        AlignedTextDataset.__init__(
            object.__new__(AlignedTextDataset),
            dataset_name="c4_test",
            dataset_path=str(ASSETS / "c4_test"),
            tokenizer=None,
            seq_len=10,
            per_doc_alignment=4,
        )


def test_dp_shards_partition_the_aligned_stream(tokenizer):
    """Sharding happens on the source documents, before padding, so both ranks
    still emit an aligned row length and a non-empty supervised stream."""
    ranks = [_stream(_loader(tokenizer, rank=rank, world=2)) for rank in range(2)]
    assert all(tokens > 0 and pairs for tokens, pairs in ranks)
    # Neither rank is a prefix of the other: the shard really partitioned the source.
    assert ranks[0][1] != ranks[1][1]
    for inputs, _ in _loader(tokenizer, rank=0, world=2, alignment=2):
        assert int(inputs["input"].numel()) % 2 == 0


def test_the_seam_builds_the_metadata_and_forwards_nothing_extra(monkeypatch):
    """The attention backend turns packed positions into the packed frames.

    The builder consumes ``positions`` and returns the metadata the layers read; it must not
    leave behind a batch key the forward does not accept, which is how the metadata stays the
    backends' business rather than the model's signature.
    """
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]], dtype=torch.long)
    model, _ = tiny_v41_model(monkeypatch)
    _, _, extra = preprocess(model, positions, positions, positions)

    # The seam's own keys and nothing else: no ``valid_tokens``, no second mask object.
    assert set(extra) == {"positions", "attention_metadata"}
    metadata = extra["attention_metadata"]
    # One entry per ``(backend config, compression ratio)``: the entry *is* one geometry, and
    # ``cu_seq_k`` is that geometry's own grid, so a stack that pools at three ratios carries
    # three entries rather than one ratio table every layer divides.
    assert set(metadata) == {(CompressedSparseInnerAttention2.Config, ratio) for ratio in (0, 1, 2)}
    # The ragged cumulative form of the two document starts plus the row total.
    expected_cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    for (backend, ratio), frames in metadata.items():
        assert backend is CompressedSparseInnerAttention2.Config
        torch.testing.assert_close(frames.cu_seq_q, expected_cu, rtol=0, atol=0)
        # The pool grid is the same backend's ratio applied once, where the ratio is known: a
        # layer that pools reads this boundary set instead of dividing one it was handed.  A
        # layer that does not pool addresses the row itself.
        expected_pool = expected_cu if ratio <= 1 else (expected_cu // ratio).to(torch.int32)
        torch.testing.assert_close(frames.cu_seq_k, expected_pool, rtol=0, atol=0)


def test_the_metadata_carries_no_validity_marks(monkeypatch):
    """Row validity is the label mask, so the metadata carries no mark tensor.

    The packed loader marks structural padding with ``IGNORE_INDEX``, which the loss consumes
    directly; the model must not expect a mark field the loader no longer emits.
    """
    positions = torch.arange(6).unsqueeze(0)
    tokens = torch.zeros_like(positions)
    labels = torch.full_like(positions, IGNORE_INDEX)
    labels[0, 1] = 7
    model, _ = tiny_v41_model(monkeypatch)
    _, _, extra = preprocess(model, tokens, labels, positions)

    assert "valid_tokens" not in extra
    for frames in extra["attention_metadata"].values():
        assert not hasattr(frames, "valid_tokens")
        assert not hasattr(frames, "valid_tokens_BL")
