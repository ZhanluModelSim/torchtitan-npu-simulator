# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4.1 multimodal data adapter.

The image token protocol and packed batching over TorchTitan's multimodal stack;
the sample parser reads the CC12M WebDataset ``jpg``/``txt`` fields (``cc12m`` /
``cc12m-test``).  Each document is padded to the model's per-document pooling
alignment, so a compressor group never straddles a document edge (see the
``text_datasets`` patch)."""

import functools
from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets.multimodal.mm_datasets import HuggingFaceMultiModalDataset
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_pixel_budget
from torchtitan.hf_datasets.multimodal.utils.packing import MMSamplePacker

from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import pad_segments_to_multiple

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
    ids, labels = pad_segments_to_multiple((ids, labels), multiple=per_doc_alignment)
    return {
        "input_ids": ids,
        "labels": labels,
        "positions": torch.arange(ids.numel()),
        # The upstream packer preserves this ordered list for every document.
        "pixel_values": [(patches, grid)],
    }


class _BufferedSamplePacker(MMSamplePacker):
    """Use upstream packing with bounded admission even without an exact fit."""

    def add_sample(self, sample):
        super().add_sample(sample)
        if len(self._sample_buffer) >= self.buffer_size:
            self.flush()


class _MultiModalDataset(HuggingFaceMultiModalDataset):
    """Advance HF streaming epochs as TorchTitan's text dataset does.

    TorchTitan 0.3.0's multimodal iterator omits this step, so a restored
    HF starting state would otherwise be replayed on every subsequent epoch.
    """

    def _get_data_iter(self):
        yield from super()._get_data_iter()
        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._data.set_epoch(state_dict["hf_dataset_state"]["epoch"])
        super().load_state_dict(state_dict)


@dataclass
class _V41Collator:
    seq_len: int
    vocab_size: int

    def __call__(self, batch):
        # The current model uses one document-packed stream per DP rank.
        (sample,) = batch
        ids, labels, positions = (sample[k] for k in ("input_ids", "labels", "positions"))
        images = sample["pixel_values"]
        starts = (positions == 0).nonzero().flatten().tolist()
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
        # BOS labels are already masked, so shifting a pack never supervises
        # EOS -> the next document's BOS. Padding is a separate document.
        targets = torch.nn.functional.pad(labels[1:], (0, pad + 1), value=-100)
        positions = torch.cat((positions, torch.arange(pad)))
        return {
            "input": tokens.unsqueeze(0),
            "positions": positions.unsqueeze(0),
            "token_types": token_types.unsqueeze(0),
            "image_feature_indices": feature_indices.unsqueeze(0),
            "pixel_values": pad_sequence([p for p, _ in images], batch_first=True),
            "image_grid": torch.stack([g for _, g in images]),
        }, targets.unsqueeze(0)


class DeepSeekV41DataLoader(ParallelAwareDataloader):
    """Reuse upstream HF sharding, packing, cycling and checkpoint state."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "cc12m-test"
        packing_buffer_size: int = 0
        infinite: bool = True
        per_doc_alignment: int = 1
        """Per-document pooling granularity (the model's compression alignment);
        the recipe derives it from the model spec."""

    def __init__(
        self,
        config: Config,
        *,
        tokenizer,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if local_batch_size != 1:
            raise ValueError("DSV4.1 expects one packed sequence per rank (local_batch_size=1)")
        if config.per_doc_alignment < 1:
            raise ValueError("per_doc_alignment must be positive")
        if seq_len % config.per_doc_alignment:
            raise ValueError(
                f"seq_len ({seq_len}) must be a multiple of per_doc_alignment ({config.per_doc_alignment})"
            )
        processor = ImagePatchProcessor()
        dataset = _MultiModalDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=processor.patch_size,
            temporal_patch_size=1,
            spatial_merge_size=processor.downsample_ratio,
            min_pixels=processor.min_pixels,
            max_pixels=0,
            image_mean=processor.mean,
            image_std=processor.std,
            packing_buffer_size=config.packing_buffer_size,
            resize_fn=resize_to_pixel_budget,
            max_patches=0,
            max_patches_per_side=0,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        # Instance-local adaptation: no global dataset registry mutation, and
        # iteration/packing/state remain entirely owned by upstream TorchTitan.
        dataset.sample_processor = functools.partial(_process_mm_sample, per_doc_alignment=config.per_doc_alignment)
        if dataset.enable_packing:
            dataset.packer = _BufferedSamplePacker(
                max_seq_length=seq_len, buffer_size=config.packing_buffer_size, batch_size=local_batch_size
            )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            collate_fn=_V41Collator(seq_len, tokenizer.get_vocab_size()),
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )
