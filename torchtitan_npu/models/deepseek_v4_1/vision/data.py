# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

TEXT = -1
IMAGE_START = 0
IMAGE = 1
IMAGE_NEW_LINE = 2
IMAGE_END = 3


@dataclass(frozen=True, slots=True)
class VisionBatch:
    """Device-ready multimodal metadata shared by the vision and decoder paths."""

    patches: torch.Tensor
    grid_hw: torch.Tensor
    patch_offsets: torch.Tensor
    image_feature_indices: torch.Tensor
    token_types: torch.Tensor

    def validate(self, *, batch_size: int, seq_len: int) -> None:
        if self.grid_hw.ndim != 2 or self.grid_hw.shape[-1] != 2:
            raise ValueError("grid_hw must have shape [num_images, 2]")
        if self.patch_offsets.ndim != 1 or self.patch_offsets.numel() != self.grid_hw.shape[0] + 1:
            raise ValueError("patch_offsets must have one entry per image plus a sentinel")
        if self.image_feature_indices.shape != (batch_size, seq_len):
            raise ValueError("image_feature_indices must have shape [batch, seq]")
        if self.token_types.shape != (batch_size, seq_len):
            raise ValueError("token_types must have shape [batch, seq]")


def _downsampled_grid(height: int, width: int, ratio: int) -> tuple[int, int]:
    return (height + ratio - 1) // ratio, (width + ratio - 1) // ratio


def build_image_token_layout(
    grid_hw: Iterable[tuple[int, int]],
    *,
    span_start: int,
    vocab_size: int,
    downsample_ratio: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ids, token types, and feature indices for one text sample.

    Each image row is encoded as IMAGE tokens followed by one NEW_LINE token,
    wrapped by START and END markers. All image protocol ids are legal
    vocabulary ids; token_types carries their semantic role.
    """

    image_token_id = vocab_size - 16
    if image_token_id < 0:
        raise ValueError("vocab_size is too small for the reserved image token")
    ids: list[int] = []
    types: list[int] = []
    feature_indices: list[int] = []
    feature_base = 0
    for height, width in grid_hw:
        out_h, out_w = _downsampled_grid(int(height), int(width), downsample_ratio)
        ids.append(image_token_id)
        types.append(IMAGE_START)
        feature_indices.append(-1)
        feature = 0
        for _ in range(out_h):
            for _ in range(out_w):
                ids.append(image_token_id)
                types.append(IMAGE)
                feature_indices.append(feature_base + feature)
                feature += 1
            ids.append(image_token_id)
            types.append(IMAGE_NEW_LINE)
            feature_indices.append(-1)
        ids.append(image_token_id)
        types.append(IMAGE_END)
        feature_indices.append(-1)
        feature_base += out_h * out_w
    if span_start < 0:
        raise ValueError("span_start must be non-negative")
    prefix = [0] * span_start
    ids = prefix + ids
    types = [TEXT] * span_start + types
    feature_indices = [-1] * span_start + feature_indices
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(types, dtype=torch.long),
        torch.tensor(feature_indices, dtype=torch.long),
    )


def build_shifted_labels(input_ids: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
    """Create next-token labels, masking targets that belong to image protocol."""
    if input_ids.shape != token_types.shape:
        raise ValueError("input_ids and token_types must have the same shape")
    labels = torch.full_like(input_ids, -100)
    if input_ids.shape[-1] > 1:
        labels[..., :-1] = input_ids[..., 1:]
        labels[..., :-1] = labels[..., :-1].masked_fill(token_types[..., 1:] >= 0, -100)
    return labels


def scatter_image_features(
    hidden: torch.Tensor,
    visual_features: torch.Tensor,
    image_feature_indices: torch.Tensor,
) -> torch.Tensor:
    """Differentiably write concatenated visual features into token embeddings."""
    if hidden.ndim != 3 or image_feature_indices.shape != hidden.shape[:2]:
        raise ValueError("hidden and image_feature_indices must describe [batch, seq]")
    if visual_features.ndim != 2 or visual_features.shape[1] != hidden.shape[-1]:
        raise ValueError("visual_features must have shape [features, hidden]")
    positions = torch.nonzero(image_feature_indices >= 0, as_tuple=False)
    if positions.numel() == 0:
        return hidden + visual_features.sum() * 0
    feature_ids = image_feature_indices[positions[:, 0], positions[:, 1]]
    if int(feature_ids.max()) >= visual_features.shape[0]:
        raise ValueError("image feature index exceeds visual feature count")
    flat_positions = positions[:, 0] * hidden.shape[1] + positions[:, 1]
    values = visual_features.index_select(0, feature_ids).to(hidden.dtype)
    output = hidden.clone().reshape(-1, hidden.shape[-1])
    return output.index_copy(0, flat_positions, values).view_as(hidden)


@dataclass(frozen=True, slots=True)
class ImagePatchProcessor:
    """Convert a real image into the reference DeepSeek ViT patch contract."""

    patch_size: int = 14
    downsample_ratio: int = 3
    min_pixels: int = 544 * 544
    max_n_token: int = 1024
    max_wh_ratio: float | None = None
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def _llm_grid(self, height: int, width: int) -> tuple[int, int]:
        patch_grid_h = height // self.patch_size
        patch_grid_w = width // self.patch_size
        return (
            math.ceil(patch_grid_h / self.downsample_ratio),
            math.ceil(patch_grid_w / self.downsample_ratio),
        )

    def _solve_resize_ratio(self, height: float, width: float) -> tuple[int, int]:
        ratio = height / width
        max_w_float = math.sqrt((self.max_n_token - 2) / ratio + 0.25) - 0.5
        max_h_float = max_w_float * ratio
        cell = self.patch_size * self.downsample_ratio
        if max_w_float < 1.0:
            return (self.max_n_token - 2) // 2 * cell, cell
        if max_h_float < 1.0:
            return cell, (self.max_n_token - 3) * cell
        beta = min(
            math.floor(max_w_float) * cell / width,
            math.floor(max_h_float) * cell / height,
        )
        return (
            math.floor(height * beta / self.patch_size) * self.patch_size,
            math.floor(width * beta / self.patch_size) * self.patch_size,
        )

    def _safe_resize(self, height: float, width: float) -> tuple[int, int]:
        best_height = math.ceil(height / self.patch_size) * self.patch_size
        best_width = math.ceil(width / self.patch_size) * self.patch_size
        llm_h, llm_w = self._llm_grid(best_height, best_width)
        if llm_h * (llm_w + 1) + 2 > self.max_n_token:
            best_height, best_width = self._solve_resize_ratio(height, width)
        return best_height, best_width

    def target_grid(self, height: float, width: float) -> tuple[int, int]:
        """The one raw-size -> patch-grid decision.

        Shared by ``from_path`` (full pixel decode) and offline
        shape/budget calculations, so preparation and runtime can never
        drift apart.  The aspect clamp, min-pixels upscale and
        ``_safe_resize`` chain below are exact, in the original order.
        """
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = height * self.max_wh_ratio
        if 0 < width * height < self.min_pixels:
            scale = (self.min_pixels / (width * height)) ** 0.5
            # Match the model asset's reference image planner: round down the
            # scaled dimensions before snapping them to the patch grid.
            width = int(width * scale)
            height = int(height * scale)
        target_h, target_w = self._safe_resize(height, width)
        return target_h // self.patch_size, target_w // self.patch_size

    def from_path(self, path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        with Image.open(path) as image:
            return self.from_image(image)

    def from_image(self, image) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the decoded image returned by the upstream HF dataset."""
        from PIL import Image, ImageOps

        image = image.convert("RGB")
        width, height = image.size
        grid_h, grid_w = self.target_grid(height, width)
        target_h, target_w = grid_h * self.patch_size, grid_w * self.patch_size
        if self.max_wh_ratio is not None and image.width >= self.max_wh_ratio * image.height:
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        else:
            image = ImageOps.pad(
                image,
                (target_w, target_h),
                method=Image.Resampling.BICUBIC,
                color=(127, 127, 127),
            )
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone()
        pixels = pixels.view(target_h, target_w, 3).permute(2, 0, 1).float() / 255.0
        pixels = ((pixels - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]).to(
            torch.bfloat16
        )
        patches = (
            pixels.view(3, grid_h, self.patch_size, grid_w, self.patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, 3 * self.patch_size * self.patch_size)
        )
        return patches, torch.tensor([grid_h, grid_w], dtype=torch.long)
