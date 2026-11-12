# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4.1 multimodal data adapter.

The image token protocol and packed batching over TorchTitan's multimodal stack;
the sample parser reads the CC12M WebDataset ``jpg``/``txt`` fields (``cc12m`` /
``cc12m-test``).  Each document is padded to the model's per-document pooling
alignment, so a compressor group never straddles a document edge."""

import copy
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

import grain.python as grain
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchtitan.components.data import GrainDataLoader, HuggingFaceStreamingSource
from torchtitan.components.data.collators import Collator
from torchtitan.components.data.dataset import SampleProcessor
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS, MMSamplePackingConfig
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

from torchtitan_npu.models.common.alignment import validate_alignment

from .data import TEXT, ImagePatchProcessor, build_image_token_layout


def _process_mm_sample(sample, tokenizer, *, per_doc_alignment: int = 1, **kwargs):
    image = sample["jpg"]
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    patches, grid = ImagePatchProcessor().from_image(image)
    caption = tokenizer.encode(sample["txt"], add_bos=False, add_eos=False)
    if not caption:
        return None
    image_ids, _, _ = build_image_token_layout(
        [(int(grid[0]), int(grid[1]))], span_start=0, vocab_size=tokenizer.get_vocab_size()
    )
    ids = torch.tensor([tokenizer.bos_id, *image_ids.tolist(), *caption, tokenizer.eos_id], dtype=torch.long)
    labels = ids.clone()
    labels[: 1 + image_ids.numel()] = -100
    # One document is one segment here, so the whole sample is padded at once.  The
    # pad stays inside its own document (positions continue; the next BOS still
    # resets) and is unsupervised, so it only rounds the document up to a whole
    # number of pooling groups.
    padding = -ids.numel() % per_doc_alignment
    padding_mask = torch.zeros(ids.numel() + padding, dtype=torch.bool)
    if padding:
        padding_mask[-padding:] = True
    ids = torch.nn.functional.pad(ids, (0, padding))
    labels = torch.nn.functional.pad(labels, (0, padding), value=-100)
    return {
        "input_ids": ids,
        "labels": labels,
        "positions": torch.arange(ids.numel()),
        "padding_mask": padding_mask,
        # The upstream packer preserves this ordered list for every document.
        "pixel_values": [(patches, grid)],
    }


@dataclass
class _V41Collator:
    seq_len: int
    vocab_size: int

    def __call__(self, batch) -> tuple[dict[str, Any], torch.Tensor]:
        # The current model uses one document-packed stream per DP rank.
        (sample,) = batch
        ids, labels, positions = (sample[k] for k in ("input_ids", "labels", "positions"))
        sample_padding_mask = sample.get("padding_mask", torch.zeros_like(ids, dtype=torch.bool))
        images = sample["pixel_values"]
        real_length = sample.get("packed_length", ids.numel())
        sample_padding_mask = sample_padding_mask.clone()
        if real_length < ids.numel():
            sample_padding_mask[real_length:] = True
        # Grain zero-fills the packed row tail. Reconstruct one padding segment
        # from its segment IDs rather than treating every zero as a document.
        positions = positions.clone()
        positions[real_length:] = torch.arange(ids.numel() - real_length)
        starts = (positions[:real_length] == 0).nonzero().flatten().tolist()
        types = torch.full_like(ids, TEXT)
        indices = torch.full_like(ids, -1)
        feature_base = 0
        for start, (_, grid) in zip(starts, images, strict=True):
            _, image_types, feature_ids = build_image_token_layout(
                [(int(grid[0]), int(grid[1]))], span_start=0, vocab_size=self.vocab_size
            )
            span = slice(start + 1, start + 1 + image_types.numel())
            types[span] = image_types
            indices[span] = torch.where(feature_ids >= 0, feature_ids + feature_base, -1)
            feature_base += int((feature_ids >= 0).sum())
        pad = self.seq_len - ids.numel()
        tokens = torch.nn.functional.pad(ids, (0, pad))
        token_types = torch.nn.functional.pad(types, (0, pad), value=TEXT)
        feature_indices = torch.nn.functional.pad(indices, (0, pad), value=-1)
        padding_mask = torch.nn.functional.pad(sample_padding_mask, (0, pad), value=True)
        # BOS labels are already masked, so shifting a pack never supervises
        # EOS -> the next document's BOS. Padding is a separate document.
        targets = torch.nn.functional.pad(labels[1:], (0, pad + 1), value=-100)
        positions = torch.cat((positions, torch.arange(ids.numel() - real_length, self.seq_len - real_length)))
        return {
            "input": tokens,
            "positions": positions,
            "padding_mask": padding_mask,
            "token_types": token_types,
            "image_feature_indices": feature_indices,
            "pixel_values": pad_sequence([p for p, _ in images], batch_first=True),
            "image_grid": torch.stack([g for _, g in images]),
        }, targets


class _GrainV41Collator(Collator):
    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context):
        del config
        self.collate = _V41Collator(context.num_tokens_per_microbatch, context.tokenizer.get_vocab_size())

    def __call__(self, rows):
        inputs, labels = self.collate(rows)
        padding_mask = inputs.pop("padding_mask")
        return TokenizedTrainingMicrobatch(
            input=inputs.pop("input"),
            labels=labels,
            positions=inputs.pop("positions"),
            padding_mask=padding_mask,
            loss_token_counts=(labels != -100).sum(),
            routing_token_counts=(~padding_mask).sum(),
            model_kwargs=inputs,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _V41PackingConfig(MMSamplePackingConfig):
    def build(self, *, context, dataset_iteration_policy):
        dataset = self.dataset.build(context=context, dataset_iteration_policy=dataset_iteration_policy)
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= context.max_context_length)
        dataset = dataset.map(_packing_input)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct=dict.fromkeys(
                ("input_ids", "labels", "positions", "padding_mask"),
                context.num_tokens_per_microbatch,
            ),
            padding_struct={
                "input_ids": 0,
                "labels": -100,
                "positions": 0,
                "padding_mask": True,
            },
            num_packing_bins=self.num_packing_bins,
            meta_features=("labels", "positions", "padding_mask", "pixel_values"),
            seed=dataset_iteration_policy.seed,
            shuffle_bins=dataset_iteration_policy.shuffle,
        )
        return dataset.map(_packed_sample)


def _packing_input(sample):
    return {
        **{key: np.asarray(sample[key]) for key in ("input_ids", "labels", "positions", "padding_mask")},
        "pixel_values": sample["pixel_values"],
    }


def _packed_sample(sample):
    return {
        **{key: torch.from_numpy(sample[key]) for key in ("input_ids", "labels", "positions", "padding_mask")},
        "pixel_values": [image for images in sample["pixel_values"] for image in images],
        "packed_length": int(np.count_nonzero(sample["input_ids_segment_ids"])),
    }


class CaptionProcessor(SampleProcessor):
    """Decode aligned image-caption documents before Grain packing."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        per_doc_alignment: int = 1

    def __init__(self, config: Config, *, context):
        validate_alignment(config.per_doc_alignment, context=context)
        self.tokenizer = context.tokenizer
        self.max_context_length = context.max_context_length
        self.per_doc_alignment = config.per_doc_alignment

    def __call__(self, sample, rng):
        del rng
        processed = _process_mm_sample(sample, self.tokenizer, per_doc_alignment=self.per_doc_alignment)
        if processed is not None and len(processed["input_ids"]) > self.max_context_length:
            return None
        return processed


def caption_dataloader_config(
    *,
    dataset: str = "cc12m-test",
    dataset_path: str | None = None,
    num_packing_bins: int = 0,
    per_doc_alignment: int = 1,
    seed: int = 42,
    shuffle: bool = False,
    repeat: bool = True,
) -> GrainDataLoader.Config:
    """Compose the caption source, processor and collator for the native loader."""
    if dataset not in ("cc12m", "cc12m-test"):
        raise ValueError("V4.1 caption data requires cc12m or cc12m-test jpg/txt samples")
    if num_packing_bins < 0:
        raise ValueError("num_packing_bins must be nonnegative (zero disables packing)")
    samples = copy.deepcopy(MM_DATASETS[dataset])
    if dataset_path is not None:
        source = samples.source
        assert isinstance(source, HuggingFaceStreamingSource.Config)
        samples = replace(samples, source=replace(source, path=dataset_path))
    samples = replace(samples, processor=CaptionProcessor.Config(per_doc_alignment=per_doc_alignment))
    return GrainDataLoader.Config(
        dataset=(
            _V41PackingConfig(dataset=samples, num_packing_bins=num_packing_bins) if num_packing_bins else samples
        ),
        collator=_GrainV41Collator.Config(),
        seed=seed,
        shuffle=shuffle,
        repeat=repeat,
    )
