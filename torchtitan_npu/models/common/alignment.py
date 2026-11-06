# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The document alignment a packed row's compression groups need.

V4 and V4.1 both pool every group of ``R`` consecutive tokens into one main-KV entry and
address that entry axis by the row's boundaries over ``R``.  A document whose token count
is not a multiple of ``R`` therefore gets a partial trailing group that the *next*
document's tokens complete: the pooled value mixes two documents, and the boundaries no
longer say which document the group belongs to.  The dataloaders pad each document to the
least common multiple of every ratio the stack pools with, which keeps a whole group of
each ratio inside one document.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from torchtitan.components.data import TextSequence
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.text_datasets import TextProcessor

if TYPE_CHECKING:
    from torchtitan.components.data import DatasetBuildContext

import torch


def compression_alignment(compress_ratios: Sequence[int]) -> int:
    """The ``A`` a packed document length must be a multiple of, ``1`` when none.

    Taking the least common multiple of every positive ratio gives the alignment that
    keeps a whole group of each ratio inside one document; that is what the dataloaders
    pad each document to.  V4's ``(1, 4, 128)`` gives 128, V4.1's ``(0, 0, 2, ...)``
    gives 2.
    """
    alignment = 1
    for ratio in compress_ratios:
        if ratio > 0:
            alignment = math.lcm(alignment, ratio)
    return alignment


def require_document_alignment(
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int = 1,
    ratio: int = 1,
) -> int:
    """Validate one packed row's document lengths against the packing grid.

    The required multiple is ``cp_size * max(ratio, 1)``: ``ratio`` keeps every pooling group
    inside one document, and ``cp_size`` keeps every rank's chunk of that document
    ratio-aligned -- each rank owns ``length / cp_size`` tokens, so a length that is only a
    multiple of ``ratio`` can leave a chunk with a partial compressed block.  That strict form
    is the contract (intentional, do not relax it): the fused compressors grid whole groups
    only.

    The loader satisfies it by padding through the processor's ``per_doc_alignment``, which must
    be a multiple of the returned value.  The check runs on the boundaries tensor (no host
    round-trip).  Returns the required multiple.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    required = max(cp_size, 1) * max(ratio, 1)
    # One device read, not two: the two failures share a message, and this runs once per
    # geometry per microbatch, where a second sync per call is measurable.
    if bool(((lengths <= 0) | (lengths % required != 0)).any()):
        factors = []
        if cp_size > 1:
            factors.append(f"cp_size={cp_size}")
        if ratio > 1:
            factors.append(f"compress_ratio={ratio}")
        what = " x ".join(factors) if factors else "the packing grid"
        raise ValueError(
            f"document lengths must be positive multiples of {required} ({what}), got "
            f"{lengths.tolist()}. Pad every document through the processor's `per_doc_alignment` "
            f"(a multiple of {required}): under context parallel each rank's chunk must stay "
            "ratio-aligned, and partial compression blocks are not supported."
        )
    return required


def require_per_doc_alignment(
    per_doc_alignment: int,
    compress_ratios: Sequence[int],
    cp_degree: int,
) -> int:
    """Validate the loader's per-document padding against the model's packing grid.

    ``required = compression_alignment(ratios) * cp_degree``: every document is padded to the
    stack-wide lcm (each ratio's groups stay inside one document) and, under context parallel,
    to a multiple of the degree as well -- each document is split into ``cp_degree`` equal
    chunks and every chunk must keep a whole number of groups.  That is intentional: partial
    compressed blocks are not part of the validated kernel contract.

    Raises a CLI-pointed error when the loader would produce a row the compressor cannot grid;
    returns ``required`` so a caller can log it or derive a default.
    """
    required = compression_alignment(compress_ratios) * max(cp_degree, 1)
    if per_doc_alignment < 1 or per_doc_alignment % required:
        raise ValueError(
            f"per-document alignment ({per_doc_alignment}) is not a multiple of "
            f"{required} (= compression_alignment({tuple(compress_ratios)}) * "
            f"cp_degree={cp_degree}). A context-parallel run splits every document into "
            f"{cp_degree} chunks and each chunk must stay ratio-aligned; set "
            f"the dataloader or its dataset processor alignment to a multiple of {required}."
        )
    return required


def require_dataloader_alignment(dataloader, compress_ratios: Sequence[int], cp_degree: int) -> None:
    """Validate document padding for the model's configured data pipeline."""
    alignment = getattr(dataloader, "per_doc_alignment", None)
    if alignment is None:
        packing = getattr(dataloader, "dataset", None)
        dataset = getattr(packing, "dataset", packing)
        processor = getattr(dataset, "processor", None)
        alignment = getattr(processor, "per_doc_alignment", 1)
    require_per_doc_alignment(alignment, compress_ratios, cp_degree)


def validate_alignment(multiple: int, *, context: "DatasetBuildContext") -> None:
    if multiple < 1:
        raise ValueError("per_doc_alignment must be positive")
    if context.max_context_length % multiple or context.num_tokens_per_microbatch % multiple:
        raise ValueError("context length and token batch must be multiples of per_doc_alignment")


def align_text_sequence(sequence: TextSequence, *, multiple: int) -> TextSequence:
    pad_len = -len(sequence.input_ids) % multiple
    if not pad_len:
        return sequence
    positions = sequence.positions
    if positions is not None:
        positions = np.concatenate((positions, positions[-1] + np.arange(1, pad_len + 1, dtype=positions.dtype)))
    # ``IGNORE_INDEX`` marks loss masking, while the target data contract uses
    # a separate structural mask for routing, attention and auxiliary-loss
    # accounting.  Alignment padding is real document-tail padding and must be
    # visible to the model even though its label is also ignored.
    padding_mask = sequence.padding_mask
    if padding_mask is None:
        padding_mask = np.zeros(len(sequence.input_ids), dtype=np.bool_)
    else:
        padding_mask = np.asarray(padding_mask, dtype=np.bool_)
    padding_mask = np.concatenate((padding_mask, np.ones(pad_len, dtype=np.bool_)))
    return replace(
        sequence,
        input_ids=np.pad(sequence.input_ids, (0, pad_len), constant_values=0),
        labels=np.pad(sequence.labels, (0, pad_len), constant_values=IGNORE_INDEX),
        positions=positions,
        padding_mask=padding_mask,
    )


class AlignedTextProcessor(TextProcessor):
    """Pad tokenized documents before native Grain packing forms compression groups."""

    @dataclass(kw_only=True, slots=True)
    class Config(TextProcessor.Config):
        per_doc_alignment: int = 1

    def __init__(self, config: Config, *, context: "DatasetBuildContext") -> None:
        super().__init__(config, context=context)
        validate_alignment(config.per_doc_alignment, context=context)
        self._per_doc_alignment = config.per_doc_alignment

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> TextSequence | None:
        sequence = super().__call__(sample, rng)
        if sequence is None:
            return None
        return align_text_sequence(sequence, multiple=self._per_doc_alignment)
