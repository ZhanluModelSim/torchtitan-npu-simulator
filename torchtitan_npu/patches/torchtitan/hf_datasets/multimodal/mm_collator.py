# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport contiguous-video MRoPE positions to the pinned upstream.

Upstream PR: pending publication of the Qwen3.5 contiguous video-MRoPE fix.
Upstream base: ``b175497ea3502388bb9513391810a86ca46dbfa8``.

The collator builds positions before the model Config exists, so a late Config
override cannot repair this behavior. Remove when the pinned upstream revision
contains the contiguous-video MRoPE fix.
"""

from __future__ import annotations

import inspect

import torch
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator

_ORIGINAL = MultiModalCollator._build_mrope_positions
_EXPECTED_PARAMETERS = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("grid_thw", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("grid_thw_videos", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("positions", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("image_token_id", inspect.Parameter.KEYWORD_ONLY),
    ("video_token_id", inspect.Parameter.KEYWORD_ONLY),
)


def _parameter_contract(fn):
    return tuple((parameter.name, parameter.kind) for parameter in inspect.signature(fn).parameters.values())


def patched_build_mrope_positions(
    self,
    tokens: torch.Tensor,
    grid_thw: torch.Tensor | None,
    grid_thw_videos: torch.Tensor | None,
    positions: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """Build 3D MRoPE positions without flattening a video into frame grids."""
    if self.patch_order != "block":
        raise ValueError(f"MRoPE requires patch_order='block', got {self.patch_order!r}.")

    spatial_merge_size = self.spatial_merge_size
    batch_size, seq_len = tokens.shape
    mrope_positions = torch.zeros(batch_size, seq_len, 3, dtype=tokens.dtype, device=tokens.device)

    resets = torch.zeros(
        batch_size,
        max(seq_len - 1, 0),
        dtype=torch.bool,
        device=tokens.device,
    )
    if positions is not None:
        resets = positions[:, 1:] < positions[:, :-1]
    grid_thw_images = grid_thw if grid_thw is not None else tokens.new_empty((0, 3))
    grid_thw_video = grid_thw_videos if grid_thw_videos is not None else tokens.new_empty((0, 3))
    vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
    prev_vision = torch.cat([torch.zeros_like(vision_mask[:, :1]), vision_mask[:, :-1]], dim=1)
    batch_vision_starts = vision_mask & ~prev_vision
    grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    image_index, video_index = 0, 0
    for sample_i in range(batch_size):
        llm_pos_ids_list: list[torch.Tensor] = []

        if positions is not None:
            reset_indices = torch.where(resets[sample_i])[0] + 1
            doc_starts = [0, *reset_indices.tolist()]
            doc_ranges = [
                (
                    doc_starts[d],
                    doc_starts[d + 1] if d + 1 < len(doc_starts) else seq_len,
                )
                for d in range(len(doc_starts))
            ]
        else:
            doc_ranges = [(0, seq_len)]

        sample_tokens = tokens[sample_i]
        sample_vision_starts = torch.where(batch_vision_starts[sample_i])[0].tolist()
        vision_start_index = 0

        for doc_start, doc_end in doc_ranges:
            doc_pos_ids_list: list[torch.Tensor] = []
            doc_vision_starts: list[int] = []
            while vision_start_index < len(sample_vision_starts) and sample_vision_starts[vision_start_index] < doc_end:
                doc_vision_starts.append(sample_vision_starts[vision_start_index])
                vision_start_index += 1

            pair_cursor = doc_start
            for vision_start in doc_vision_starts:
                if sample_tokens[vision_start] == image_token_id:
                    t, h, w = grid_thw_images[image_index]
                    image_index += 1
                else:
                    t, h, w = grid_thw_video[video_index]
                    video_index += 1

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    int(t.item()),
                    int(h.item()) // spatial_merge_size,
                    int(w.item()) // spatial_merge_size,
                )
                text_len = vision_start - pair_cursor
                pos_id_offset = doc_pos_ids_list[-1].max() + 1 if len(doc_pos_ids_list) > 0 else 0
                doc_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset)

                grid_key = (llm_grid_t, llm_grid_h, llm_grid_w)
                if grid_key not in grid_cache:
                    hw = llm_grid_h * llm_grid_w
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, hw).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    grid_cache[grid_key] = torch.stack([t_index, h_index, w_index])
                doc_pos_ids_list.append(grid_cache[grid_key] + text_len + pos_id_offset)
                pair_cursor = vision_start + llm_grid_t * llm_grid_h * llm_grid_w

            if pair_cursor < doc_end:
                pos_id_offset = doc_pos_ids_list[-1].max() + 1 if len(doc_pos_ids_list) > 0 else 0
                text_len = doc_end - pair_cursor
                doc_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset)

            llm_pos_ids_list.extend(doc_pos_ids_list)

        mrope_positions[sample_i] = torch.cat(llm_pos_ids_list, dim=1).T

    return mrope_positions


def apply() -> None:
    """Install the backport once, rejecting an unknown upstream contract."""
    current = MultiModalCollator._build_mrope_positions
    if current is patched_build_mrope_positions:
        return
    if current is not _ORIGINAL or _parameter_contract(current) != _EXPECTED_PARAMETERS:
        raise RuntimeError("Unsupported upstream MultiModalCollator signature")
    MultiModalCollator._build_mrope_positions = patched_build_mrope_positions


__all__ = ["apply"]
