# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic dataset and dataloader for MAGI-2-preview flow-matching training.

Fork reason: real video datasets and VAE/text-encoder preprocessing are
deferred; training is exercised on synthetic latents that mirror the official
MAGI-2-preview packing layout (video/audio/text segments in original order,
channel-padded tokens, RoPE coords mapping, sinusoidal diffusion-time
embedding) so every ``Magi2PreviewModel`` forward kwarg is populated.
Reference: SandAI-org/MAGI-2 inference/pipeline/preview_data_proxy.py

The flow-matching sample construction (``_flow_matching_sample``), the RoPE
coordinate blocks (``_build_*_coords``) and multi-sample packing
(``_pack_packed_samples``) are shared with ``latent_dataset.py`` so the
offline-latent loader emits exactly the same input_dict contract.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import BaseDataLoader

from .embeddings import sinusoidal_embedding_1d

logger = logging.getLogger(__name__)

# Modality ids matching the official MAGI-2 ``Modality`` enum. TIME tokens
# never appear in synthetic samples; the model remaps them to TEXT anyway.
MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2
MODALITY_TIME = 3

# Channel layout of the official model (fixed, independent of model flavor):
# packed input tokens are channel-padded to the text width, predictions and
# labels live in the 64-dim video+audio channel space.
MAX_IN_CHANNELS = 5120
VIDEO_CHANNELS = 48
AUDIO_CHANNELS = 64
LABEL_CHANNELS = 64
TIME_CHANNEL_DIM = 64
AUDIO_TIME_COMPRESSION = 8


def _grid_coords(
    shape: tuple[int, int, int],
    ref: tuple[int, int, int],
    offset: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Build (t, h, w, T, H, W, refT, refH, refW) rows for a (t, h, w) grid.

    Mirrors the official ``get_coords``: token coordinates are t-major
    meshgrid indices (plus ``offset``), broadcast with the original grid
    sizes and the reference sizes consumed by ``ElementWiseFourierEmbed``.
    """
    t, h, w = shape
    ot, oh, ow = offset
    time_rng = torch.arange(t, dtype=torch.float32) + ot
    height_rng = torch.arange(h, dtype=torch.float32) + oh
    width_rng = torch.arange(w, dtype=torch.float32) + ow
    t_grid, h_grid, w_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")
    coords = torch.stack([t_grid, h_grid, w_grid], dim=-1).reshape(-1, 3)
    sizes = torch.tensor(shape, dtype=torch.float32).expand(coords.shape[0], 3)
    refs = torch.tensor(ref, dtype=torch.float32).expand(coords.shape[0], 3)
    return torch.cat([coords, sizes, refs], dim=1)


def _build_video_coords(
    video_frames: int, video_height: int, video_width: int
) -> torch.Tensor:
    """RoPE coords block for one video latent grid (reference == grid shape)."""
    return _grid_coords(
        (video_frames, video_height, video_width),
        (video_frames, video_height, video_width),
    )


def _build_audio_coords(audio_len: int) -> torch.Tensor:
    """RoPE coords block for audio tokens.

    The audio time axis is compressed 8x through its reference length,
    mirroring the official ``magic_audio_ref_t`` convention.
    """
    audio_ref_t = (audio_len - 1) // AUDIO_TIME_COMPRESSION + 1
    return _grid_coords((audio_len, 1, 1), (audio_ref_t, 1, 1))


def _build_text_coords(text_len: int) -> torch.Tensor:
    """RoPE coords block for text tokens at negative times, ref (1, 1, 1)."""
    return _grid_coords((text_len, 1, 1), (1, 1, 1), offset=(-text_len, 0, 0))


def _flow_matching_sample(
    video_x0: torch.Tensor,
    audio_x0: torch.Tensor,
    text_x0: torch.Tensor,
    video_eps: torch.Tensor,
    audio_eps: torch.Tensor,
    text_eps: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Pack one clean-latent sample into a noisy flow-matching training pair.

    Pure construction (no RNG): given clean latents ``x0``, noise ``eps`` and
    the sample's diffusion time ``sigma``, builds the noisy input
    ``x_t = (1 - sigma) * x0 + sigma * eps`` and the velocity labels
    ``v = eps - x0`` packed exactly as ``Magi2PreviewModel.forward`` consumes
    them. Shared by the synthetic and offline-latent datasets.

    Args:
        video_x0: clean video latents ``(T, H, W, 48)``.
        audio_x0: clean audio latents ``(L_a, 64)``.
        text_x0: clean text embeddings ``(L_t, 5120)``.
        video_eps: noise matching ``video_x0``.
        audio_eps: noise matching ``audio_x0``.
        text_eps: noise matching ``text_x0``.
        sigma: scalar flow-matching time shared by the sample's noisy tokens.

    Returns:
        ``(input_dict, labels)`` where ``input_dict`` carries the exact
        forward kwargs ``input``/``coords_mapping``/``modality_mapping``/
        ``time_embedding``/``cu_seqlens`` (single segment) and ``labels`` is
        ``(T_total, 64)`` with zero text rows.
    """
    video_frames, video_height, video_width, _ = video_x0.shape
    video_x0 = video_x0.to(torch.float32)
    audio_x0 = audio_x0.to(torch.float32)
    text_x0 = text_x0.to(torch.float32)
    video_eps = video_eps.to(torch.float32)
    audio_eps = audio_eps.to(torch.float32)
    text_eps = text_eps.to(torch.float32)

    video_xt = (1 - sigma) * video_x0 + sigma * video_eps
    audio_xt = (1 - sigma) * audio_x0 + sigma * audio_eps
    text_xt = (1 - sigma) * text_x0 + sigma * text_eps
    video_velocity = video_eps - video_x0
    audio_velocity = audio_eps - audio_x0

    n_video = video_frames * video_height * video_width
    n_audio = audio_x0.shape[0]
    n_text = text_x0.shape[0]
    total = n_video + n_audio + n_text

    input_tokens = torch.zeros(total, MAX_IN_CHANNELS)
    input_tokens[:n_video, :VIDEO_CHANNELS] = video_xt.reshape(n_video, VIDEO_CHANNELS)
    input_tokens[n_video : n_video + n_audio, :AUDIO_CHANNELS] = audio_xt
    input_tokens[n_video + n_audio :] = text_xt

    modality_mapping = torch.cat(
        [
            torch.full((n_video,), MODALITY_VIDEO, dtype=torch.int32),
            torch.full((n_audio,), MODALITY_AUDIO, dtype=torch.int32),
            torch.full((n_text,), MODALITY_TEXT, dtype=torch.int32),
        ]
    )

    # Text sits at negative times with ref (1, 1, 1), so every RoPE axis
    # either scales to zero (time) or sits at its center (h, w).
    coords_mapping = torch.cat(
        [
            _build_video_coords(video_frames, video_height, video_width),
            _build_audio_coords(n_audio),
            _build_text_coords(n_text),
        ],
        dim=0,
    )

    per_token_sigma = torch.cat([sigma.expand(n_video + n_audio), torch.zeros(n_text)])
    time_embedding = sinusoidal_embedding_1d(TIME_CHANNEL_DIM, per_token_sigma)

    cu_seqlens = torch.tensor([0, total], dtype=torch.int32)

    labels = torch.zeros(total, LABEL_CHANNELS)
    labels[:n_video, :VIDEO_CHANNELS] = video_velocity.reshape(n_video, VIDEO_CHANNELS)
    labels[n_video : n_video + n_audio, :AUDIO_CHANNELS] = audio_velocity

    input_dict = {
        "input": input_tokens,
        "coords_mapping": coords_mapping,
        "modality_mapping": modality_mapping,
        "time_embedding": time_embedding,
        "cu_seqlens": cu_seqlens,
    }
    return input_dict, labels


def _pack_packed_samples(
    samples: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Concatenate single-sample packed pairs into one multi-sample pack.

    Each entry is an ``(input_dict, labels)`` pair from
    ``_flow_matching_sample``; per-token fields are concatenated in sample
    order and ``cu_seqlens`` is rebuilt as the cumulative segment ends,
    mirroring the official ``SimplePackedData`` layout.
    """
    input_dict = {
        key: torch.cat([sample[0][key] for sample in samples], dim=0)
        for key in ("input", "coords_mapping", "modality_mapping", "time_embedding")
    }
    seqlens = torch.tensor(
        [int(sample[0]["cu_seqlens"][-1]) for sample in samples], dtype=torch.int32
    )
    # torch.cumsum promotes int32 to int64 on CPU; the model contract is int32.
    input_dict["cu_seqlens"] = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(seqlens, dim=0).to(torch.int32)]
    )
    labels = torch.cat([sample[1] for sample in samples], dim=0)
    return input_dict, labels


class Magi2SyntheticDataset(IterableDataset):
    """Infinite synthetic flow-matching dataset for MAGI-2-preview.

    Each sample is one fully packed sequence:
    - clean latents x0 ~ N(0, 1): video (F, H, W, 48), audio (L_a, 64),
      text (L_t, 5120);
    - sigma ~ U(0, 1) scalar shared by every noisy token of the sample;
    - noisy latents x_t = (1 - sigma) * x0 + sigma * eps with eps ~ N(0, 1)
      and flow-matching velocity labels v = eps - x0;
    - segments packed video-first in original order, matching the official
      modality_mapping / coords_mapping conventions (audio time axis
      compressed 8x via its reference length, text placed at negative times
      with ref (1, 1, 1) so its RoPE scale collapses to zero).

    Samples are deterministically seeded from ``seed + iteration`` so the
    stream is reproducible and checkpoint-resumable.
    """

    def __init__(
        self,
        video_frames: int = 2,
        video_height: int = 4,
        video_width: int = 4,
        audio_len: int = 16,
        text_len: int = 16,
        seed: int = 0,
    ):
        self.video_frames = video_frames
        self.video_height = video_height
        self.video_width = video_width
        self.audio_len = audio_len
        self.text_len = text_len
        self.seed = seed
        self.iteration = 0

    def _build_sample(self, iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build one packed flow-matching training sample."""
        gen = torch.Generator()
        gen.manual_seed(self.seed + iteration)

        sigma = torch.rand((), generator=gen)

        video_x0 = torch.randn((self.video_frames, self.video_height, self.video_width, VIDEO_CHANNELS), generator=gen)
        audio_x0 = torch.randn((self.audio_len, AUDIO_CHANNELS), generator=gen)
        text_x0 = torch.randn((self.text_len, MAX_IN_CHANNELS), generator=gen)
        video_eps = torch.randn_like(video_x0, generator=gen)
        audio_eps = torch.randn_like(audio_x0, generator=gen)
        text_eps = torch.randn_like(text_x0, generator=gen)

        return _flow_matching_sample(
            video_x0, audio_x0, text_x0, video_eps, audio_eps, text_eps, sigma
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        iteration = self.iteration
        while True:
            sample = self._build_sample(iteration)
            iteration += 1
            self.iteration = iteration
            yield sample


class Magi2SyntheticDataLoader(BaseDataLoader):
    """Infinite synthetic dataloader for MAGI-2-preview training.

    Each iteration yields one fully packed sample as ``(input_dict, labels)``
    where ``input_dict`` carries exactly the ``Magi2PreviewModel`` forward
    kwargs: ``input``, ``coords_mapping``, ``modality_mapping``,
    ``time_embedding``, ``cu_seqlens``. Text label rows are zero and the
    model's PostAdapter emits zero text rows, so sum-MSE is naturally masked.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        video_frames: int = 2
        """Number of synthetic latent video frames (t extent of the grid)."""

        video_height: int = 4
        """Latent video grid height."""

        video_width: int = 4
        """Latent video grid width."""

        audio_len: int = 16
        """Number of audio latent tokens."""

        text_len: int = 16
        """Number of text tokens."""

        seed: int = 0
        """Base RNG seed; the data-parallel rank is added on top so each rank streams different data."""

    def __init__(
        self,
        config: "Magi2SyntheticDataLoader.Config",
        dp_world_size: int | None = None,
        dp_rank: int | None = None,
        tokenizer: Any = None,
        seq_len: int | None = None,
        local_batch_size: int | None = None,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        # ``tokenizer`` / ``seq_len`` / ``local_batch_size`` are part of the
        # shared dataloader build contract but unused here: synthetic samples
        # are already packed to their final token count and every iteration
        # yields one sample.
        self.dataset = Magi2SyntheticDataset(
            video_frames=config.video_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            audio_len=config.audio_len,
            text_len=config.text_len,
            seed=config.seed + (dp_rank if dp_rank is not None else 0),
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        return iter(self.dataset)

    def state_dict(self) -> dict[str, Any]:
        return {"iteration": self.dataset.iteration}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # An empty state (fresh checkpoint) is valid: keep the stream start.
        if state_dict:
            self.dataset.iteration = state_dict["iteration"]
