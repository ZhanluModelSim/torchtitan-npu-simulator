# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataloader for glm5_next with a uniform square grid.

The glm5_next vision tower v1 requires every image in a batch to share one
square patch grid (MODEL_CONTRACT.md section 7): the 2D spatial-merge conv
needs a rectangular ``[N, C, H, W]`` layout whose H/W must be config-known to
stay meta-clean. The upstream cc12m sample processor resizes with a preserved
aspect ratio, so this loader swaps in a square-resize sample processor over
the same ``cc12m-test`` asset and reuses the upstream NLD collator verbatim.

The image token id comes from the tokenizer's ``<|image|>`` special token
(``SpecialTokens.img_id``); the model config's ``image_token_id`` must match
(enforced here at build time is the flavor's responsibility).
"""

import io
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from PIL import Image

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.experiments.vlm.datasets.mm_collator_nld import MultiModalCollatorNLD
from torchtitan.experiments.vlm.datasets.mm_datasets import HuggingFaceMultiModalDataset
from torchtitan.experiments.vlm.datasets.utils.image import calculate_image_tokens
from torchtitan.experiments.vlm.datasets.utils.text import process_text_with_images
from torchtitan.experiments.vlm.datasets.mm_datasets import HuggingFaceMultiModalDataLoader
from torchtitan.experiments.vlm.model.args import SpecialTokens

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])

_UNSET = object()


def _process_image_square(image: bytes | Image.Image, image_size: int) -> torch.Tensor | None:
    """Resize to a fixed square and apply CLIP normalization.

    Returns ``[1, image_size, image_size, 3]`` (dummy temporal dim, HWC) to
    match the upstream ``process_image`` contract.
    """
    try:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        array = np.array(image).astype(np.float32) / 255.0
        array = (array - _CLIP_MEAN) / _CLIP_STD
        return torch.from_numpy(array).float().unsqueeze(0)
    except Exception:
        return None


def _process_glm5_mm_sample(
    *,
    sample: dict,
    tokenizer: BaseTokenizer,
    patch_size: int,
    spatial_merge_size: int,
    max_patch_per_image: int,
    special_tokens: SpecialTokens,
    image_size: int,
) -> dict | None:
    """Upstream ``_process_mm_sample`` with a forced square resize.

    Keeps the same interleaved sample contract and output keys
    (``input_ids``/``labels``/``pixel_values``) as the cc12m processor.
    Samples follow the cc12m-wd format (``txt`` + ``jpg`` keys).
    """
    try:
        text = sample.get("txt", "")
        image = sample.get("jpg")
        if image is None:
            return None
        texts = [None, text]
        images = [image, None]

        processed_images = []
        image_dimensions = []
        texts_list = list(texts)
        for idx, img in enumerate(images):
            if img is None:
                continue
            processed_img = _process_image_square(img, image_size)
            if processed_img is None:
                texts_list[idx] = ""
                continue
            num_tokens, width, height = calculate_image_tokens(
                processed_img,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            if num_tokens > max_patch_per_image // (spatial_merge_size * spatial_merge_size):
                texts_list[idx] = ""
                continue
            processed_images.append(processed_img)
            image_dimensions.append((num_tokens, width, height))
            texts_list[idx] = special_tokens.img_token

        if not processed_images:
            return None

        processed_text = process_text_with_images(
            texts_list, image_dimensions, tokenizer, special_tokens, add_eos=True
        )
        tokens = tokenizer.encode(processed_text)
        input_ids = torch.tensor(tokens)
        labels = torch.tensor(tokens)
        special_token_ids = torch.tensor(
            [special_tokens.boi_id, special_tokens.eoi_id, special_tokens.img_id]
        )
        labels = torch.where(
            torch.isin(labels, special_token_ids), special_tokens.ignore_id, labels
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": processed_images,
        }
    except Exception:
        return None


class Glm5NextMultiModalDataLoader(HuggingFaceMultiModalDataLoader):
    """Upstream NLD multimodal loader with a uniform square image grid."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceMultiModalDataLoader.Config):
        image_size: int = 56
        """Forced square image size; patches per side = image_size // patch_size."""

        def __post_init__(self) -> None:
            factor = self.patch_size * self.spatial_merge_size
            if self.image_size % factor != 0:
                raise ValueError(
                    f"image_size={self.image_size} must be divisible by "
                    f"patch_size*spatial_merge_size={factor}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        special_tokens = SpecialTokens.from_tokenizer(tokenizer)
        mm_ds = HuggingFaceMultiModalDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            max_patches_per_image=config.max_patches_per_image,
            max_images_per_batch=config.max_images_per_batch,
            packing_buffer_size=config.packing_buffer_size,
            special_tokens=special_tokens,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        mm_ds.sample_processor = partial(
            _process_glm5_mm_sample,
            image_size=config.image_size,
        )

        collate_fn = MultiModalCollatorNLD(
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=config.patch_size,
            max_images_per_batch=config.max_images_per_batch,
            max_patches_per_image=config.max_patches_per_image,
            special_tokens=special_tokens,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
            "collate_fn": collate_fn,
        }
        from torchtitan.components.dataloader import ParallelAwareDataloader

        ParallelAwareDataloader.__init__(
            self,
            mm_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
