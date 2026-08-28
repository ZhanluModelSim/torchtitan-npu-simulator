# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline pre-encoded latent dataset and dataloader for MAGI-2-preview.

Trains on real VAE/text-encoder latents written to a shard directory by
``scripts/magi2_preprocess_latents.py`` (see
``docs/user-guides/magi2_preview_data_pipeline.md`` for the full spec):

- One ``.safetensors`` file per shard (``.pt`` shards are accepted as a
  fallback container), tensors named ``{sample_id}.video_latent``
  ``(48, T, H, W)`` fp16/bf16, ``{sample_id}.audio_latent`` ``(L_a, 64)``
  and ``{sample_id}.text_emb`` ``(L_t, 5120)``.
- The shard file carries a JSON sample listing (``format`` / ``samples``
  metadata for safetensors, ``format`` / ``samples`` payload entries for
  ``.pt``) recording per-sample ``video_shape`` / ``audio_len`` /
  ``text_len`` plus arbitrary attrs (``fps``, ``num_frames``, ...). The
  listing is derived from tensor names/shapes when absent.
- An optional ``index.json`` manifest summarizes the directory for tooling;
  the loader only needs the shard files.

Behavior: samples are bucketed by video latent shape ``(T, H, W)`` — every
pack contains one bucket only — and packed greedily up to
``max_tokens_per_pack`` tokens. Each yielded ``(input_dict, labels)`` pair
honors the exact ``Magi2PreviewModel`` forward contract built by the
synthetic loader (input / coords_mapping / modality_mapping /
time_embedding / cu_seqlens, labels ``(T_total, 64)`` with zero text rows);
multi-sample packs carry one ``cu_seqlens`` segment per sample, mirroring
the official ``SimplePackedData`` layout. Shard files are split across
data-parallel ranks (``files[dp_rank::dp_world_size]``) and the stream is
infinite and checkpoint-resumable via ``state_dict`` (epoch + sample
pointer; flow-matching noise is re-derived from ``seed``).
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import BaseDataLoader

from .dataset import (
    AUDIO_CHANNELS,
    MAX_IN_CHANNELS,
    VIDEO_CHANNELS,
    _flow_matching_sample,
    _pack_packed_samples,
)

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as _safetensors_save_file

    _SAFETENSORS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - safetensors is a repo dep
    safe_open = None
    _safetensors_save_file = None
    _SAFETENSORS_IMPORT_ERROR = exc

# Shard format tag stored in every shard's metadata.
SHARD_FORMAT = "magi2-latent-v1"
VIDEO_KEY = "video_latent"
AUDIO_KEY = "audio_latent"
TEXT_KEY = "text_emb"
_SAMPLE_TENSOR_KEYS = (VIDEO_KEY, AUDIO_KEY, TEXT_KEY)
_SHARD_SUFFIXES = (".safetensors", ".pt")
# Stride between per-(epoch, position) noise seeds; bounds samples per epoch.
_EPOCH_SEED_STRIDE = 1_000_000_007


def sample_noise_rng(seed: int, epoch: int, position: int) -> torch.Generator:
    """Deterministic per-sample noise generator, resumable without stream state."""
    return torch.Generator().manual_seed(seed + epoch * _EPOCH_SEED_STRIDE + position)


@dataclass(slots=True)
class _ShardSample:
    """Address of one sample inside a shard file plus its listing metadata."""

    shard: str
    sample_id: str
    video_shape: tuple[int, int, int]
    audio_len: int
    text_len: int
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        frames, height, width = self.video_shape
        return frames * height * width + self.audio_len + self.text_len


def _list_shard_files(data_path: str) -> list[str]:
    """Sorted shard file paths under ``data_path``; raises when none exist."""
    path = Path(data_path)
    if not path.is_dir():
        raise FileNotFoundError(f"MAGI-2 latent data_path is not a directory: {data_path}")
    files = sorted(
        str(entry)
        for entry in path.iterdir()
        if entry.is_file() and entry.suffix in _SHARD_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No shard files (*.safetensors / *.pt) found under {data_path}; "
            "generate them with scripts/magi2_preprocess_latents.py"
        )
    return files


def _entry_from_shapes(sample_id: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """Build one canonical listing entry from per-tensor shapes."""
    video_shape = shapes[VIDEO_KEY]
    if len(video_shape) != 4 or video_shape[0] != VIDEO_CHANNELS:
        raise ValueError(
            f"Sample {sample_id!r}: {VIDEO_KEY} must be ({VIDEO_CHANNELS}, T, H, W), "
            f"got {tuple(video_shape)}"
        )
    audio_shape = shapes[AUDIO_KEY]
    if len(audio_shape) != 2 or audio_shape[1] != AUDIO_CHANNELS:
        raise ValueError(
            f"Sample {sample_id!r}: {AUDIO_KEY} must be (L_a, {AUDIO_CHANNELS}), "
            f"got {tuple(audio_shape)}"
        )
    text_shape = shapes[TEXT_KEY]
    if len(text_shape) != 2 or text_shape[1] != MAX_IN_CHANNELS:
        raise ValueError(
            f"Sample {sample_id!r}: {TEXT_KEY} must be (L_t, {MAX_IN_CHANNELS}), "
            f"got {tuple(text_shape)}"
        )
    return {
        "id": sample_id,
        "video_shape": list(video_shape[1:]),
        "audio_len": int(audio_shape[0]),
        "text_len": int(text_shape[0]),
        "attrs": {},
    }


def _derive_listing_from_keys(shapes: dict[str, tuple[int, ...]]) -> list[dict[str, Any]]:
    """Derive the sample listing from tensor names when metadata is absent."""
    grouped: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, shape in shapes.items():
        sample_id, _, tensor_name = key.partition(".")
        if not sample_id or tensor_name not in _SAMPLE_TENSOR_KEYS:
            raise ValueError(
                f"Unexpected tensor key {key!r}; expected '<sample_id>.<{'|'.join(_SAMPLE_TENSOR_KEYS)}]>'"
            )
        grouped.setdefault(sample_id, {})[tensor_name] = shape
    listing = []
    for sample_id, shapes_by_key in grouped.items():
        missing = set(_SAMPLE_TENSOR_KEYS) - shapes_by_key.keys()
        if missing:
            raise ValueError(f"Sample {sample_id!r} is missing tensors: {sorted(missing)}")
        listing.append(_entry_from_shapes(sample_id, shapes_by_key))
    return listing


def _listing_entry_from_metadata(entry: dict[str, Any], available_keys: set[str]) -> dict[str, Any]:
    """Validate one metadata listing entry and normalize it."""
    for required in ("id", "video_shape", "audio_len", "text_len"):
        if required not in entry:
            raise ValueError(f"Sample listing entry is missing {required!r}: {entry}")
    sample_id = str(entry["id"])
    for tensor_name in _SAMPLE_TENSOR_KEYS:
        key = f"{sample_id}.{tensor_name}"
        if key not in available_keys:
            raise ValueError(f"Shard listing references missing tensor {key!r}")
    return {
        "id": sample_id,
        "video_shape": [int(dim) for dim in entry["video_shape"]],
        "audio_len": int(entry["audio_len"]),
        "text_len": int(entry["text_len"]),
        "attrs": dict(entry.get("attrs", {})),
    }


def _read_safetensors_listing(shard_path: str) -> list[dict[str, Any]]:
    """Read the sample listing of one ``.safetensors`` shard (header only)."""
    if safe_open is None:
        raise RuntimeError(
            "safetensors is required to read .safetensors shards but is not "
            f"importable ({_SAFETENSORS_IMPORT_ERROR}); install safetensors or "
            "convert the shards to the .pt container"
        )
    with safe_open(shard_path, framework="pt") as shard:
        metadata = shard.metadata() or {}
        keys = set(shard.keys())
        raw_listing = metadata.get("samples")
        if raw_listing:
            return [
                _listing_entry_from_metadata(entry, keys) for entry in json.loads(raw_listing)
            ]
        shapes = {key: tuple(shard.get_slice(key).get_shape()) for key in shard.keys()}
    return _derive_listing_from_keys(shapes)


def _read_pt_shard(shard_path: str) -> dict[str, Any]:
    data = torch.load(shard_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "tensors" not in data:
        raise ValueError(
            f"{shard_path} is not a MAGI-2 latent .pt shard (missing 'tensors' payload)"
        )
    return data


def _read_shard_listing(shard_path: str) -> list[dict[str, Any]]:
    """Canonical per-sample listing of one shard file, whichever container."""
    if shard_path.endswith(".safetensors"):
        return _read_safetensors_listing(shard_path)
    data = _read_pt_shard(shard_path)
    tensors = data["tensors"]
    raw_listing = data.get("samples")
    if raw_listing:
        return [_listing_entry_from_metadata(entry, set(tensors.keys())) for entry in raw_listing]
    return _derive_listing_from_keys({key: tuple(tensor.shape) for key, tensor in tensors.items()})


def write_latent_shard(path: str | Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write one MAGI-2 latent shard; returns the listing stored alongside it.

    The file extension selects the container: ``.safetensors`` (preferred,
    metadata in the safetensors header) or ``.pt`` (torch.save payload with
    ``format`` / ``samples`` / ``tensors`` entries).

    Args:
        path: destination shard file.
        samples: one mapping per sample carrying ``id`` (str without ``.``),
            ``video_latent`` ``(48, T, H, W)`` fp16/bf16, ``audio_latent``
            ``(L_a, 64)`` and ``text_emb`` ``(L_t, 5120)`` float tensors,
            plus arbitrary JSON-serializable attrs (``fps``, ``num_frames``,
            ...) recorded in the listing.
    """
    path = Path(path)
    if path.suffix not in _SHARD_SUFFIXES:
        raise ValueError(f"Shard path must end in one of {_SHARD_SUFFIXES}, got {path.suffix!r}")

    tensors: dict[str, torch.Tensor] = {}
    listing: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for sample in samples:
        sample_id = str(sample["id"])
        if not sample_id or "." in sample_id:
            raise ValueError(f"Sample ids must be non-empty strings without '.': {sample_id!r}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id {sample_id!r} in shard {path}")
        seen_ids.add(sample_id)

        video = sample[VIDEO_KEY]
        audio = sample[AUDIO_KEY]
        text = sample[TEXT_KEY]
        if video.ndim != 4 or video.shape[0] != VIDEO_CHANNELS:
            raise ValueError(
                f"Sample {sample_id!r}: {VIDEO_KEY} must be ({VIDEO_CHANNELS}, T, H, W), "
                f"got {tuple(video.shape)}"
            )
        if video.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"Sample {sample_id!r}: {VIDEO_KEY} must be float16/bfloat16, got {video.dtype}"
            )
        if audio.ndim != 2 or audio.shape[1] != AUDIO_CHANNELS:
            raise ValueError(
                f"Sample {sample_id!r}: {AUDIO_KEY} must be (L_a, {AUDIO_CHANNELS}), "
                f"got {tuple(audio.shape)}"
            )
        if text.ndim != 2 or text.shape[1] != MAX_IN_CHANNELS:
            raise ValueError(
                f"Sample {sample_id!r}: {TEXT_KEY} must be (L_t, {MAX_IN_CHANNELS}), "
                f"got {tuple(text.shape)}"
            )
        if not audio.is_floating_point() or not text.is_floating_point():
            raise ValueError(f"Sample {sample_id!r}: audio/text tensors must be floating point")

        tensors[f"{sample_id}.{VIDEO_KEY}"] = video.contiguous()
        tensors[f"{sample_id}.{AUDIO_KEY}"] = audio.contiguous()
        tensors[f"{sample_id}.{TEXT_KEY}"] = text.contiguous()
        attrs = {
            key: value
            for key, value in sample.items()
            if key not in ("id", *_SAMPLE_TENSOR_KEYS)
        }
        listing.append(
            {
                "id": sample_id,
                "video_shape": list(video.shape[1:]),
                "audio_len": int(audio.shape[0]),
                "text_len": int(text.shape[0]),
                "attrs": attrs,
            }
        )
    samples_json = json.dumps(listing)  # raises early on non-serializable attrs

    if path.suffix == ".safetensors":
        if _safetensors_save_file is None:
            raise RuntimeError(
                "safetensors is required to write .safetensors shards but is not "
                f"importable ({_SAFETENSORS_IMPORT_ERROR}); install safetensors or "
                "write a .pt shard instead"
            )
        _safetensors_save_file(
            tensors, str(path), metadata={"format": SHARD_FORMAT, "samples": samples_json}
        )
    else:
        torch.save({"format": SHARD_FORMAT, "samples": listing, "tensors": tensors}, path)
    return listing


def build_latent_index(data_path: str) -> dict[str, Any]:
    """Summarize a shard directory as the optional ``index.json`` manifest."""
    return {
        "format": SHARD_FORMAT,
        "shards": [
            {"file": Path(shard).name, "samples": _read_shard_listing(shard)}
            for shard in _list_shard_files(data_path)
        ],
    }


class Magi2LatentDataset(IterableDataset):
    """Infinite, bucketed, packed stream over a directory of latent shards.

    Iteration order is deterministic for a given ``(data_path, seed,
    dp_world_size, dp_rank)``: shard files are sharded across data-parallel
    ranks, samples are grouped into buckets by video latent shape ``(T, H,
    W)`` (buckets iterated in sorted order, samples within a bucket shuffled
    per epoch with ``seed + epoch``), and samples are packed greedily within
    a bucket up to ``max_tokens_per_pack`` tokens — a sample larger than the
    budget is packed alone. Flow-matching noise is drawn per sample from
    ``sample_noise_rng(seed, epoch, position)`` so the stream is exactly
    reproducible from the ``state_dict`` position.
    """

    def __init__(
        self,
        data_path: str,
        max_tokens_per_pack: int = 4096,
        seed: int = 0,
        dp_world_size: int = 1,
        dp_rank: int = 0,
    ):
        if max_tokens_per_pack < 1:
            raise ValueError(f"max_tokens_per_pack must be >= 1, got {max_tokens_per_pack}")
        if dp_world_size < 1 or not 0 <= dp_rank < dp_world_size:
            raise ValueError(
                f"Invalid dp sharding: dp_rank={dp_rank}, dp_world_size={dp_world_size}"
            )
        self.data_path = data_path
        self.max_tokens_per_pack = max_tokens_per_pack
        self.seed = seed
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank

        all_files = _list_shard_files(data_path)
        self.shard_files = all_files[dp_rank::dp_world_size]
        if not self.shard_files:
            raise ValueError(
                f"dp_rank {dp_rank} received no shard files: the directory holds "
                f"{len(all_files)} shard(s) but dp_world_size={dp_world_size}; add "
                "more shards or reduce the data-parallel degree"
            )

        self._samples: list[_ShardSample] = []
        for shard in self.shard_files:
            for entry in _read_shard_listing(shard):
                self._samples.append(
                    _ShardSample(
                        shard=shard,
                        sample_id=entry["id"],
                        video_shape=tuple(entry["video_shape"]),
                        audio_len=entry["audio_len"],
                        text_len=entry["text_len"],
                        attrs=entry["attrs"],
                    )
                )
        if not self._samples:
            raise ValueError(f"Shard files under {data_path} contain no samples")

        self.epoch = 0
        self.sample_pointer = 0
        self._safetensors_handles: dict[str, Any] = {}
        self._pt_cache: tuple[str, dict[str, Any]] | None = None

    def _epoch_order(self, epoch: int) -> list[_ShardSample]:
        """Bucket-major sample order for one epoch (deterministic in seed)."""
        buckets: dict[tuple[int, int, int], list[_ShardSample]] = {}
        for sample in self._samples:
            buckets.setdefault(sample.video_shape, []).append(sample)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order: list[_ShardSample] = []
        for bucket_key in sorted(buckets):
            members = buckets[bucket_key]
            permutation = torch.randperm(len(members), generator=generator).tolist()
            order.extend(members[index] for index in permutation)
        return order

    def _open_safetensors(self, shard_path: str) -> Any:
        handle = self._safetensors_handles.get(shard_path)
        if handle is None:
            if safe_open is None:
                raise RuntimeError(
                    "safetensors is required to read .safetensors shards but is "
                    f"not importable ({_SAFETENSORS_IMPORT_ERROR})"
                )
            handle = safe_open(shard_path, framework="pt")
            self._safetensors_handles[shard_path] = handle
        return handle

    def _load_tensors(self, sample: _ShardSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = f"{sample.sample_id}."
        if sample.shard.endswith(".safetensors"):
            handle = self._open_safetensors(sample.shard)
            video = handle.get_tensor(prefix + VIDEO_KEY)
            audio = handle.get_tensor(prefix + AUDIO_KEY)
            text = handle.get_tensor(prefix + TEXT_KEY)
        else:
            tensors = self._load_pt_shard(sample.shard)["tensors"]
            video = tensors[prefix + VIDEO_KEY]
            audio = tensors[prefix + AUDIO_KEY]
            text = tensors[prefix + TEXT_KEY]

        expected_video = (VIDEO_CHANNELS, *sample.video_shape)
        if tuple(video.shape) != expected_video or tuple(audio.shape) != (
            sample.audio_len,
            AUDIO_CHANNELS,
        ) or tuple(text.shape) != (sample.text_len, MAX_IN_CHANNELS):
            raise ValueError(
                f"Sample {sample.sample_id!r} in {sample.shard} does not match its "
                f"listing: got video {tuple(video.shape)} / audio {tuple(audio.shape)} "
                f"/ text {tuple(text.shape)}, expected video {expected_video} / "
                f"audio ({sample.audio_len}, {AUDIO_CHANNELS}) / "
                f"text ({sample.text_len}, {MAX_IN_CHANNELS})"
            )
        return video, audio, text

    def _load_pt_shard(self, shard_path: str) -> dict[str, Any]:
        if self._pt_cache is not None and self._pt_cache[0] == shard_path:
            return self._pt_cache[1]
        data = _read_pt_shard(shard_path)
        self._pt_cache = (shard_path, data)
        return data

    def _build_sample_pair(
        self, sample: _ShardSample, epoch: int, position: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Load one sample and build its noisy flow-matching packed pair."""
        video_latent, audio_latent, text_emb = self._load_tensors(sample)
        generator = sample_noise_rng(self.seed, epoch, position)
        sigma = torch.rand((), generator=generator)
        video_x0 = video_latent.permute(1, 2, 3, 0).to(torch.float32)
        audio_x0 = audio_latent.to(torch.float32)
        text_x0 = text_emb.to(torch.float32)
        video_eps = torch.randn_like(video_x0, generator=generator)
        audio_eps = torch.randn_like(audio_x0, generator=generator)
        text_eps = torch.randn_like(text_x0, generator=generator)
        return _flow_matching_sample(
            video_x0, audio_x0, text_x0, video_eps, audio_eps, text_eps, sigma
        )

    def _build_pack(
        self, order: list[_ShardSample], pointer: int, epoch: int
    ) -> tuple[tuple[dict[str, torch.Tensor], torch.Tensor], int]:
        """Greedily pack same-bucket samples starting at ``pointer``."""
        bucket = order[pointer].video_shape
        members: list[tuple[_ShardSample, int]] = []
        pack_tokens = 0
        while pointer < len(order):
            sample = order[pointer]
            if sample.video_shape != bucket:
                break
            if members and pack_tokens + sample.num_tokens > self.max_tokens_per_pack:
                break
            members.append((sample, pointer))
            pack_tokens += sample.num_tokens
            pointer += 1
        pairs = [
            self._build_sample_pair(sample, epoch, position) for sample, position in members
        ]
        packed = pairs[0] if len(pairs) == 1 else _pack_packed_samples(pairs)
        return packed, pointer

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        epoch, pointer = self.epoch, self.sample_pointer
        while True:
            order = self._epoch_order(epoch)
            total = len(order)
            if pointer >= total:
                # Resumed at the end of an epoch (or the dataset shrank):
                # roll forward to the start of the next epoch.
                epoch += 1
                pointer = 0
                order = self._epoch_order(epoch)
                total = len(order)
            while pointer < total:
                packed, pointer = self._build_pack(order, pointer, epoch)
                if pointer >= total:
                    epoch += 1
                    pointer = 0
                self.epoch = epoch
                self.sample_pointer = pointer
                yield packed

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "sample_pointer": self.sample_pointer}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # An empty state (fresh checkpoint) is valid: keep the stream start.
        if state_dict:
            self.epoch = int(state_dict["epoch"])
            self.sample_pointer = int(state_dict["sample_pointer"])


class Magi2LatentDataLoader(BaseDataLoader):
    """Infinite dataloader over pre-encoded MAGI-2-preview latent shards.

    Each iteration yields one packed sequence ``(input_dict, labels)`` with
    exactly the ``Magi2PreviewModel`` forward kwargs; ``cu_seqlens`` carries
    one segment per packed sample. Shard files are split across dp ranks, so
    every rank streams a disjoint subset of the directory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        data_path: str = ""
        """Directory of latent shards; see docs/user-guides/magi2_preview_data_pipeline.md."""

        max_tokens_per_pack: int = 4096
        """Token budget per packed sequence; same-bucket samples are concatenated up to it."""

        seed: int = 0
        """Base seed for per-epoch sample shuffling and flow-matching noise."""

    def __init__(
        self,
        config: "Magi2LatentDataLoader.Config",
        dp_world_size: int | None = None,
        dp_rank: int | None = None,
        tokenizer: Any = None,
        seq_len: int | None = None,
        local_batch_size: int | None = None,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        # ``tokenizer`` / ``seq_len`` / ``local_batch_size`` are part of the
        # shared dataloader build contract but unused here: latent samples are
        # already encoded and each iteration yields one packed sequence.
        self.dataset = Magi2LatentDataset(
            data_path=config.data_path,
            max_tokens_per_pack=config.max_tokens_per_pack,
            seed=config.seed,
            dp_world_size=dp_world_size if dp_world_size else 1,
            dp_rank=dp_rank if dp_rank else 0,
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        return iter(self.dataset)

    def state_dict(self) -> dict[str, Any]:
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.dataset.load_state_dict(state_dict)
