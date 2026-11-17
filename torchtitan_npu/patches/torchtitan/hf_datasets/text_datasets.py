# Backports the per-document pooling alignment API onto the pinned upstream text
# dataset: ``pad_segments_to_multiple`` as a data-pipeline field and its
# implementation.  Mirrors upstream ``torchtitan/components/data/packing.py``
# (``_pad_segments_to_multiple``) and ``torchtitan/components/data/types.py``
# (``DatasetBuildContext.pad_segments_to_multiple``), which the 0.3.0 release
# predates.  Remove this module once the pinned TorchTitan dependency carries the
# equivalent field and the compatibility tests pass.
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-document pooling alignment for the pinned text data path.

Models with a compressed KV pool each group of ``R`` consecutive tokens into one
main-KV entry, and the indexer addresses that entry axis as the row's boundaries over
``R``.  A document whose token count is not a multiple of ``R`` therefore gets a partial
trailing group that the *next* document's tokens complete: the pooled value mixes two
documents, and the boundaries no longer say which document the group belongs to.

Padding every packed document segment up to a multiple of the model's alignment keeps
each group inside one document.  The pad stays inside its segment -- positions keep
running through it, so the next document still starts at position 0 -- and its labels
are ``IGNORE_INDEX``, so it never becomes a target.

The alignment value itself is the model's business (``V41Model``'s
``compression_alignment``); this module only applies it.

What the backport does not carry from upstream
----------------------------------------------
Upstream also maintains a ``padding_mask`` on its sequence type and marks the appended
pads there.  The 0.3.0 pipeline has no such field, so nothing here can consume one;
a consumer that needs to tell a pad from a real token must derive it (the pads are
exactly the unsupervised positions after a segment's last supervised token).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer  # noqa: TC002
from torchtitan.hf_datasets.text_datasets import (
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)
from torchtitan.tools.logging import logger

# Id appended by the alignment padding.  The pads are never targets (their labels are
# IGNORE_INDEX), so any in-vocabulary id works.
PAD_ID = 0


def pad_segments_to_multiple(segment: tuple, *, multiple: int) -> tuple:
    """Append pad tokens so a packed segment is a multiple of ``multiple``.

    A segment is a run of tokens between position resets, i.e. one document (or the
    row-tail remainder of one).  ``segment`` is its ``(input_ids, labels)`` pair;
    positions are not carried here because the caller owns them and keeps them running
    through the pads.  Both members must be the same length and share a container
    type (a list, or a tensor/array whose dtype survives concatenation).
    ``multiple <= 1`` and an already-aligned segment both fall out with nothing to pad.

    The pads are appended, so the original ids and labels are a prefix of the result,
    and every pad's label is ``IGNORE_INDEX``.
    """
    input_ids, labels = segment
    if len(input_ids) != len(labels):
        raise ValueError(f"input_ids and labels must have the same length, got {len(input_ids)} and {len(labels)}.")
    pad_len = -len(input_ids) % multiple
    if pad_len == 0:
        return input_ids, labels
    if isinstance(input_ids, list):
        return (
            [*input_ids, *([PAD_ID] * pad_len)],
            [*labels, *([IGNORE_INDEX] * pad_len)],
        )
    pads = input_ids.new_full((pad_len,), PAD_ID)
    ignore = labels.new_full((pad_len,), IGNORE_INDEX)
    return torch.cat((input_ids, pads)), torch.cat((labels, ignore))


class AlignedTextDataset(HuggingFaceTextDataset):
    """``HuggingFaceTextDataset`` with per-document pooling alignment.

    The alignment makes a document that fits in one row pool exactly.  It does not
    extend to the row cut: a document longer than ``seq_len`` is still chunked by the
    generic row cut, and the continuation row starts its own group grid, so a group can
    straddle the cut.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        *,
        seq_len: int = 2048,
        per_doc_alignment: int = 1,
        **kwargs,
    ) -> None:
        if per_doc_alignment < 1:
            raise ValueError("per_doc_alignment must be positive")
        if seq_len % per_doc_alignment:
            raise ValueError(f"seq_len ({seq_len}) must be a multiple of per_doc_alignment ({per_doc_alignment}).")
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            **kwargs,
        )
        self.per_doc_alignment = per_doc_alignment

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)
                # Pad this document's segment before it enters the buffers, so the
                # row cut never splits a pooling group.
                input_ids, label_ids = pad_segments_to_multiple(
                    (sample_tokens[:-1], sample_tokens[1:]),
                    multiple=self.per_doc_alignment,
                )
                self._inputs_buffer.extend(input_ids)
                self._labels_buffer.extend(label_ids)
                self._positions_buffer.extend(range(len(input_ids)))
                self._sample_idx += 1

                while len(self._inputs_buffer) >= self.seq_len:
                    row_inputs = torch.LongTensor(self._inputs_buffer[: self.seq_len])
                    row_labels = torch.LongTensor(self._labels_buffer[: self.seq_len])
                    positions = torch.LongTensor(self._normalize_positions(self._positions_buffer[: self.seq_len]))
                    self._inputs_buffer = self._inputs_buffer[self.seq_len :]
                    self._labels_buffer = self._labels_buffer[self.seq_len :]
                    self._positions_buffer = self._positions_buffer[self.seq_len :]
                    yield {"input": row_inputs, "positions": positions}, row_labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            self.reloop()


class AlignedHuggingfaceDataloader(HuggingFaceTextDataLoader):
    """``HuggingFaceTextDataLoader`` building :class:`AlignedTextDataset`."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        per_doc_alignment: int = 1
        """Per-document pooling granularity (the model's compression alignment)."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        hf_ds = AlignedTextDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            per_doc_alignment=config.per_doc_alignment,
        )
        # Bypass ``HuggingFaceTextDataLoader.__init__``: it would build the unaligned
        # dataset.  The assembly below is that method's body verbatim.
        ParallelAwareDataloader.__init__(
            self,
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )


__all__ = [
    "PAD_ID",
    "AlignedHuggingfaceDataloader",
    "AlignedTextDataset",
    "pad_segments_to_multiple",
]
