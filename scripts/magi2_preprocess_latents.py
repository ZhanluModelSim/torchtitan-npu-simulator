# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preprocess raw videos/captions into MAGI-2-preview latent training shards.

Writes the shard directory consumed by
``torchtitan_npu.models.magi2_preview.latent_dataset.Magi2LatentDataLoader``
(see docs/user-guides/magi2_preview_data_pipeline.md for the format spec).

Three modes:

Dry run (CPU, no weights): emits a tiny random-latent shard set usable by
the loader end to end:

    python3 scripts/magi2_preprocess_latents.py --dry-run \
        --output-dir ./magi2_latent_shards

Self test (CPU, no weights, no video files): stub encoders exercise the full
manifest -> decode -> encode -> shard -> load plumbing on synthetic tensors:

    python3 scripts/magi2_preprocess_latents.py --self-test \
        --output-dir /tmp/magi2_self_test

Real encoding against a local clone of SandAI-org/MAGI-2-preview (weights
downloaded with ``hf download sand-ai/MAGI-2-preview``):

    python3 scripts/magi2_preprocess_latents.py \
        --magi2-repo /path/to/MAGI-2-preview \
        --input-manifest ./manifest.jsonl \
        --output-dir ./magi2_latent_shards \
        --vae-ckpt /weights/ckpt/vae \
        --text-encoder-path /weights/ckpt/text_encoder \
        [--audio-vae-ckpt /weights/ckpt/stable-audio-open-1.0] [--device cuda]

The manifest is JSON Lines, one object per sample: ``{"video": <path>,
"caption": <str>, "audio": <optional path>, "id": <optional str>}``; relative
paths resolve against the manifest directory. Every encoder import is lazy
and fails with an actionable message when the repo, a dependency, or a weight
is missing.
"""

import argparse
import json
import logging
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
LATENT_DTYPE = "float16"

# Official MAGI-2-preview sampling conventions (DataProxyConfig /
# EvaluationConfig in SandAI-org/MAGI-2-preview inference/common/magi2_config.py):
# pixel video at 25 fps, VAE stride (8, 16, 16) with causal time handling
# (T_latent = (T - 1) // 8 + 1), audio latents at 25 rows per second.
VIDEO_TARGET_FPS = 25.0
VAE_TEMPORAL_STRIDE = 8
VAE_SPATIAL_STRIDE = 16
AUDIO_LATENT_FPS = 25.0
AUDIO_LATENT_CHANNELS = 64
TEXT_EMBED_DIM = 5120
# Keep at least two latent frames per sample (one full temporal window).
MIN_SAMPLED_FRAMES = 1 + VAE_TEMPORAL_STRIDE

# Dry-run latent shapes (T, H, W) cycled across samples so the output
# exercises bucketing and multi-shape packing.
DRY_RUN_SHAPES = ((2, 4, 4), (2, 4, 8), (4, 4, 4))
DRY_RUN_AUDIO_LEN = 16
DRY_RUN_TEXT_LEN = 16

# Self-test pixel shapes (T, H, W): T is 1 mod the temporal stride and the
# spatial sides are multiples of the spatial stride, cycled across samples so
# the output exercises multi-shape bucketing.
SELF_TEST_SHAPES = ((9, 32, 32), (17, 48, 48), (9, 32, 64), (17, 64, 32))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory of <name>.mp4 videos with paired <name>.txt captions.",
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=None,
        help=(
            "JSON Lines manifest: one object per sample with 'video' (path), "
            "'caption' (str), optional 'audio' (path) and optional 'id'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the latent shards and index.json into.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=64)
    parser.add_argument(
        "--magi2-repo",
        type=str,
        default=None,
        help=(
            "Root of a local SandAI-org/MAGI-2-preview clone; its inference "
            "package is imported lazily. Omit only when it is already on PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="Official Wan2.2 video VAE: the ckpt/vae directory or its Wan2.2_VAE.pth.",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default=None,
        help="HuggingFace directory of the Qwen3.5 text encoder (ckpt/text_encoder).",
    )
    parser.add_argument(
        "--audio-vae-ckpt",
        type=str,
        default=None,
        help=(
            "Audio VAE directory with model_config.json + model.safetensors "
            "(ckpt/stable-audio-open-1.0); omit for video+text only."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a tiny random-latent shard set without any encoders.",
    )
    parser.add_argument("--num-dry-run-samples", type=int, default=8)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run the full manifest -> decode -> encode -> shard -> load path with "
            "weight-less stub encoders on synthetic tensors (CPU-safe, deterministic)."
        ),
    )
    parser.add_argument("--num-self-test-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dry_run and args.self_test:
        parser.error("--dry-run and --self-test are mutually exclusive")
    if args.output_dir is None and not args.self_test:
        parser.error("--output-dir is required")
    if not args.dry_run and not args.self_test:
        if bool(args.input) == bool(args.input_manifest):
            parser.error("real encoding needs exactly one of --input / --input-manifest")
        missing = [
            name
            for name, value in (("--vae-ckpt", args.vae_ckpt),
                                ("--text-encoder-path", args.text_encoder_path))
            if not value
        ]
        if missing:
            parser.error(f"real encoding requires {', '.join(missing)} (or use --dry-run / --self-test)")
    return args


# ---------------------------------------------------------------------------
# Official encoder wiring (lazy imports with actionable errors)
# ---------------------------------------------------------------------------


def _prepare_magi2_repo(magi2_repo: str | None) -> None:
    """Validate the MAGI-2-preview clone root and put it on ``sys.path``."""
    if not magi2_repo:
        return
    repo = Path(magi2_repo)
    marker = repo / "inference" / "model" / "vae2_2.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"--magi2-repo does not look like a SandAI-org/MAGI-2-preview clone "
            f"(missing {marker}); clone it with: "
            "git clone https://github.com/SandAI-org/MAGI-2-preview"
        )
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_magi2_module(module_name: str):
    """Import one module from the official repo with an actionable error."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import {module_name} from the official MAGI-2-preview repo; "
            "pass --magi2-repo <clone root> (or add it to PYTHONPATH) and install "
            "its requirements (SandAI-org/MAGI-2-preview requirements.txt; the "
            "text encoder also needs a transformers with Qwen3_5TextModel support)"
        ) from exc


def _resolve_vae_ckpt(vae_ckpt: str) -> Path:
    """Accept the ckpt/vae directory or the Wan2.2_VAE.pth file itself."""
    path = Path(vae_ckpt)
    if path.is_dir():
        candidate = path / "Wan2.2_VAE.pth"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"--vae-ckpt directory {path} holds no Wan2.2_VAE.pth; download the "
                "'vae' directory of sand-ai/MAGI-2-preview "
                "(hf download sand-ai/MAGI-2-preview --include 'vae/*')"
            )
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"--vae-ckpt not found: {vae_ckpt}")
    return path


def _load_video_vae(vae_ckpt: str, device: str):
    """Lazy-load the official Wan2.2 video VAE (get_vae2_2 builder)."""
    import torch

    vae2_2 = _import_magi2_module("inference.model.vae2_2")
    return vae2_2.get_vae2_2(
        str(_resolve_vae_ckpt(vae_ckpt)), device=device, weight_dtype=torch.float32
    )


def _load_text_encoder(model_path: str, device: str):
    """Lazy-load the official Qwen3.5 text encoder."""
    import torch

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"--text-encoder-path not found: {model_path}")
    qwen35 = _import_magi2_module("inference.model.qwen35")
    # Mirror the official inference pipeline (inference/pipeline/inference_engine.py):
    # bfloat16 weights, hidden states two layers above the last one.
    return qwen35.Qwen35TextEncoder(
        model_path=model_path, device=device, precision=torch.bfloat16, skip_layer=2
    )


def _load_audio_vae(audio_vae_ckpt: str):
    """Lazy-load the official Stable Audio Open VAE feature extractor."""
    path = Path(audio_vae_ckpt)
    if not path.is_dir():
        raise FileNotFoundError(
            f"--audio-vae-ckpt must be the audio VAE directory (got {audio_vae_ckpt}); "
            "download 'stable-audio-open-1.0' from sand-ai/MAGI-2-preview"
        )
    for required in ("model_config.json", "model.safetensors"):
        if not (path / required).is_file():
            raise FileNotFoundError(
                f"--audio-vae-ckpt is missing {required}: {path / required}; download "
                "'stable-audio-open-1.0' from sand-ai/MAGI-2-preview"
            )
    audio_decoder = _import_magi2_module("inference.pipeline.audio_decoder")
    return audio_decoder.SAAudioFeatureExtractor(str(path))


def _encoder_device(encoder) -> str:
    """Device an encoder lives on (works for the official wrappers and stubs)."""
    module = getattr(encoder, "vae", encoder)
    try:
        return str(next(module.parameters()).device)
    except (AttributeError, StopIteration):
        return str(getattr(encoder, "device", "cpu"))


# ---------------------------------------------------------------------------
# Input handling: manifest / directory entries, frame sampling
# ---------------------------------------------------------------------------


def _id_from_video_path(video_path: Path) -> str:
    """Deterministic shard-safe sample id derived from the video file name."""
    return video_path.stem.replace(".", "_")


def read_manifest(manifest_path: str) -> list[dict]:
    """Parse a JSON Lines manifest into normalized sample entries."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"--input-manifest not found: {manifest_path}")
    entries = []
    seen_ids = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}:{line_no}: each line must be a JSON object, got {type(record).__name__}"
            )
        for key in ("video", "caption"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{path}:{line_no}: missing required string field {key!r}")
        if record.get("audio") is not None and (
            not isinstance(record["audio"], str) or not record["audio"].strip()
        ):
            raise ValueError(f"{path}:{line_no}: 'audio' must be a non-empty path string when present")
        video = Path(record["video"])
        if not video.is_absolute():
            video = path.parent / video
        audio = None
        if record.get("audio"):
            audio = Path(record["audio"])
            if not audio.is_absolute():
                audio = path.parent / audio
        sample_id = record.get("id") or _id_from_video_path(video)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_no}: 'id' must be a non-empty string when present")
        if "." in sample_id:
            raise ValueError(
                f"{path}:{line_no}: sample id must not contain '.': {sample_id!r} "
                "(shard tensor keys are '<id>.<tensor>')"
            )
        if sample_id in seen_ids:
            raise ValueError(f"{path}:{line_no}: duplicate sample id {sample_id!r}")
        seen_ids.add(sample_id)
        entries.append({"id": sample_id, "video": video, "caption": record["caption"], "audio": audio})
    if not entries:
        raise ValueError(f"--input-manifest contains no samples: {manifest_path}")
    return entries


def _directory_entries(input_dir: str, audio_expected: bool) -> list[dict]:
    """Legacy directory mode: <name>.mp4 + <name>.txt (+ optional <name>.wav)."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--input is not a directory: {input_dir}")
    video_paths = sorted(
        path for path in root.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos ({', '.join(VIDEO_SUFFIXES)}) under {root}")
    entries = []
    for video_path in video_paths:
        caption_path = video_path.with_suffix(".txt")
        if not caption_path.is_file():
            raise FileNotFoundError(f"Caption file missing for {video_path}: expected {caption_path}")
        audio_path = video_path.with_suffix(".wav")
        if audio_expected and not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio waveform missing for {video_path}: expected {audio_path} "
                "(or drop --audio-vae-ckpt to encode video+text only)"
            )
        entries.append(
            {
                "id": _id_from_video_path(video_path),
                "video": video_path,
                "caption": caption_path.read_text(encoding="utf-8"),
                "audio": audio_path if audio_expected else None,
            }
        )
    return entries


def sample_frame_indices(num_frames: int, native_fps: float) -> list[int]:
    """Indices of the frames to keep, subsampled and VAE-stride aligned.

    Keeps roughly one frame per ``1 / VIDEO_TARGET_FPS`` seconds and trims the
    kept count to ``1 mod VAE_TEMPORAL_STRIDE`` so the causal VAE yields
    ``T_latent = (T - 1) // VAE_TEMPORAL_STRIDE + 1`` latent frames.
    """
    if num_frames < 1:
        raise ValueError(f"Video has no frames to sample from (got {num_frames})")
    stride = max(1, round(native_fps / VIDEO_TARGET_FPS)) if native_fps > 0 else 1
    indices = list(range(0, num_frames, stride))
    usable = 1 + ((len(indices) - 1) // VAE_TEMPORAL_STRIDE) * VAE_TEMPORAL_STRIDE
    if usable < MIN_SAMPLED_FRAMES:
        raise ValueError(
            f"Video too short: {len(indices)} frame(s) left after sampling to "
            f"{VIDEO_TARGET_FPS} fps, need at least {MIN_SAMPLED_FRAMES} to fill one "
            "VAE temporal window"
        )
    return indices[:usable]


def _crop_to_vae_grid(video):
    """Center-crop the spatial sides to multiples of ``VAE_SPATIAL_STRIDE``."""
    height, width = video.shape[-2:]
    if height < VAE_SPATIAL_STRIDE or width < VAE_SPATIAL_STRIDE:
        raise ValueError(
            f"Video too small for the VAE spatial stride {VAE_SPATIAL_STRIDE}: "
            f"{height}x{width}"
        )
    new_height = height - height % VAE_SPATIAL_STRIDE
    new_width = width - width % VAE_SPATIAL_STRIDE
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    return video[..., top : top + new_height, left : left + new_width]


# ---------------------------------------------------------------------------
# Video / audio decoding (lazy, optional dependencies)
# ---------------------------------------------------------------------------


def _decode_video(video_path: Path):
    """Decode one video to ((1, 3, T, H, W) float32 in [-1, 1], native fps)."""
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    errors = []
    try:
        return _decode_video_torchvision(video_path)
    except ImportError as exc:
        errors.append(f"torchvision unavailable: {exc}")
    try:
        return _decode_video_imageio(video_path)
    except ImportError as exc:
        errors.append(f"imageio unavailable: {exc}")
    except RuntimeError as exc:
        errors.append(f"imageio could not decode (missing imageio-ffmpeg?): {exc}")
    raise RuntimeError(
        "Video decoding needs torchvision (torchvision.io.read_video) or imageio "
        "with the imageio-ffmpeg plugin; install one of them (the official "
        "MAGI-2-preview requirements ship torchvision) or pre-encode the latents "
        f"elsewhere. Import errors: {'; '.join(errors)}"
    )


def _decode_video_torchvision(video_path: Path):
    from torchvision.io import read_video

    import torch

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"No video frames decoded from {video_path}")
    native_fps = float(info.get("video_fps") or 0.0)
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32)
    return video / 127.5 - 1.0, native_fps


def _decode_video_imageio(video_path: Path):
    import imageio

    import torch

    with imageio.get_reader(str(video_path)) as reader:
        metadata = reader.get_meta_data()
        frames = [torch.as_tensor(frame) for frame in reader]
    if not frames:
        raise ValueError(f"No video frames decoded from {video_path}")
    native_fps = float(metadata.get("fps") or 0.0)
    video = torch.stack(frames).permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32)
    return video / 127.5 - 1.0, native_fps


def _load_waveform_wav(audio_path: Path):
    """Load one waveform as ((C, L) float32 in [-1, 1], sample rate)."""
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio waveform not found: {audio_path}")
    try:
        from scipy.io import wavfile
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to load audio waveforms (scipy.io.wavfile); install scipy"
        ) from exc
    import torch

    rate, data = wavfile.read(str(audio_path))
    waveform = torch.as_tensor(data)
    if waveform.dtype == torch.int16:
        waveform = waveform.to(torch.float32) / 32768.0
    elif waveform.dtype == torch.int32:
        waveform = waveform.to(torch.float32) / 2147483648.0
    else:
        waveform = waveform.to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # mono: (1, L)
    else:
        waveform = waveform.transpose(0, 1)  # (L, C) -> (C, L)
    return waveform, int(rate)


def _scipy_resample(tensor, num: int, axis: int):
    """Resample a tensor along ``axis`` with scipy (official audio_decoder idiom)."""
    from scipy.signal import resample

    import torch

    resampled = resample(tensor.detach().cpu().numpy(), num, axis=axis)
    return torch.from_numpy(resampled).float()


def _encode_audio_latent(audio_vae, audio_path: Path, target_len: int, write_dtype, load_waveform):
    """Encode one waveform to a ``(target_len, 64)`` audio latent."""
    import torch

    waveform, rate = load_waveform(audio_path)
    in_channels = getattr(getattr(audio_vae, "vae_model", None), "in_channels", 2)
    if waveform.shape[0] == 1 and in_channels > 1:
        waveform = waveform.repeat(in_channels, 1)  # mono -> model channel count
    waveform = waveform[:in_channels]
    sample_rate = int(getattr(audio_vae, "sample_rate", rate))
    if rate != sample_rate:
        waveform = _scipy_resample(
            waveform, int(waveform.shape[1] * sample_rate / rate), axis=1
        )
    waveform = waveform.unsqueeze(0).to(_encoder_device(audio_vae))
    latent = audio_vae.encode(waveform)[0].transpose(0, 1).detach().to("cpu", torch.float32)
    if latent.shape[0] != target_len:
        # Align with the official audio_latent_fps contract (rows per second).
        latent = _scipy_resample(latent, target_len, axis=0)
    return latent.to(write_dtype)


# ---------------------------------------------------------------------------
# Encoding and shard writing
# ---------------------------------------------------------------------------


def _encode_entry(
    entry: dict,
    video_vae,
    text_encoder,
    audio_vae,
    write_dtype,
    decode_video=_decode_video,
    load_waveform=_load_waveform_wav,
) -> dict:
    """Encode one manifest entry into a shard sample mapping."""
    import torch

    video, native_fps = decode_video(entry["video"])
    indices = sample_frame_indices(video.shape[2], native_fps)
    video = _crop_to_vae_grid(video.index_select(2, torch.tensor(indices, dtype=torch.long)))
    video_latent = video_vae.encode(video.to(_encoder_device(video_vae), torch.float32))
    video_latent = video_latent.squeeze(0).detach().to("cpu", write_dtype)

    text_emb = text_encoder.encode(entry["caption"])
    text_emb = text_emb.squeeze(0).detach().to("cpu", write_dtype)

    num_frames = video.shape[2]
    audio_target_len = int(round(num_frames / VIDEO_TARGET_FPS * AUDIO_LATENT_FPS))
    if audio_vae is not None and entry["audio"] is not None:
        audio_latent = _encode_audio_latent(
            audio_vae, entry["audio"], audio_target_len, write_dtype, load_waveform
        )
    else:
        audio_latent = torch.zeros((0, AUDIO_LATENT_CHANNELS), dtype=write_dtype)

    return {
        "id": entry["id"],
        "video_latent": video_latent,
        "audio_latent": audio_latent,
        "text_emb": text_emb,
        "fps": VIDEO_TARGET_FPS,
        "num_frames": int(num_frames),
        "source": Path(entry["video"]).name,
    }


def _write_shards(samples: list[dict], output_dir: Path, samples_per_shard: int) -> None:
    from torchtitan_npu.models.magi2_preview.latent_dataset import (
        build_latent_index,
        write_latent_shard,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, start in enumerate(range(0, len(samples), samples_per_shard)):
        shard_path = output_dir / f"shard_{shard_index:04d}.safetensors"
        write_latent_shard(shard_path, samples[start : start + samples_per_shard])
        logger.info("wrote %s (%d samples)", shard_path, min(samples_per_shard, len(samples) - start))
    index = build_latent_index(str(output_dir))
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d shards)", index_path, len(index["shards"]))


# ---------------------------------------------------------------------------
# Modes: dry run, self test, real encoding
# ---------------------------------------------------------------------------


def _run_dry_run(args: argparse.Namespace) -> None:
    import torch

    generator = torch.Generator().manual_seed(args.seed)
    dtype = getattr(torch, LATENT_DTYPE)
    samples = []
    for index in range(args.num_dry_run_samples):
        frames, height, width = DRY_RUN_SHAPES[index % len(DRY_RUN_SHAPES)]
        samples.append(
            {
                "id": f"dryrun_{index:04d}",
                "video_latent": torch.randn((48, frames, height, width), generator=generator, dtype=torch.float32).to(dtype),
                "audio_latent": torch.randn((DRY_RUN_AUDIO_LEN, 64), generator=generator, dtype=torch.float32).to(dtype),
                "text_emb": torch.randn((DRY_RUN_TEXT_LEN, 5120), generator=generator, dtype=torch.float32).to(dtype),
                "fps": 25.0,
                "num_frames": (frames - 1) * 4 + 1,
                "source": "dry-run",
            }
        )
    _write_shards(samples, Path(args.output_dir), args.samples_per_shard)


class StubVideoVAE:
    """Weight-less deterministic stand-in for the Wan2.2 video VAE encoder."""

    device = "cpu"

    def encode(self, video):
        import torch
        import torch.nn.functional as F

        # Causal-VAE time layout: the first frame maps to its own latent frame.
        first = F.avg_pool3d(
            video[:, :, :1], kernel_size=(1, VAE_SPATIAL_STRIDE, VAE_SPATIAL_STRIDE)
        )
        rest = F.avg_pool3d(
            video[:, :, 1:],
            kernel_size=(VAE_TEMPORAL_STRIDE, VAE_SPATIAL_STRIDE, VAE_SPATIAL_STRIDE),
            stride=(VAE_TEMPORAL_STRIDE, VAE_SPATIAL_STRIDE, VAE_SPATIAL_STRIDE),
        )
        return torch.cat([first, rest], dim=2).repeat(1, 16, 1, 1, 1)  # 3 -> 48 channels


class StubTextEncoder:
    """Weight-less deterministic stand-in for the Qwen3.5 text encoder."""

    max_tokens = 16

    def encode(self, caption: str):
        import torch

        generator = torch.Generator().manual_seed(zlib.crc32(caption.encode("utf-8")))
        length = min(max(1, len(caption.split())), self.max_tokens)
        return torch.randn((1, length, TEXT_EMBED_DIM), generator=generator)


class StubAudioVAE:
    """Weight-less deterministic stand-in for the Stable Audio Open VAE."""

    sample_rate = 44100
    downsampling_ratio = 2048
    device = "cpu"
    vae_model = SimpleNamespace(in_channels=2)

    @property
    def latent_fps(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    def encode(self, waveform):
        import torch.nn.functional as F

        latent = F.avg_pool1d(
            waveform, kernel_size=self.downsampling_ratio, stride=self.downsampling_ratio
        )
        return latent.repeat(1, 32, 1)  # 2 -> 64 channels


def _self_test_entries(num_samples: int) -> list[dict]:
    """Synthetic manifest entries; the paths are never opened by the stubs."""
    entries = []
    for index in range(num_samples):
        entries.append(
            {
                "id": f"selftest_{index:04d}",
                "video": Path(f"synthetic_{index:04d}.mp4"),
                "caption": f"synthetic clip {index} exercising the preprocessing plumbing",
                "audio": Path(f"synthetic_{index:04d}.wav") if index % 2 == 0 else None,
            }
        )
    return entries


def _self_test_shape(video_path: Path) -> tuple[int, int, int]:
    index = int(video_path.stem.rsplit("_", 1)[1])
    return SELF_TEST_SHAPES[index % len(SELF_TEST_SHAPES)]


def _self_test_decode_video(video_path: Path):
    """Deterministic synthetic frames for the self test (no video files)."""
    import torch

    frames, height, width = _self_test_shape(video_path)
    generator = torch.Generator().manual_seed(zlib.crc32(video_path.stem.encode("utf-8")))
    video = torch.randn((1, 3, frames, height, width), generator=generator)
    return video, VIDEO_TARGET_FPS


def _self_test_load_waveform(audio_path: Path):
    """Deterministic synthetic waveform at half the stub sample rate."""
    import torch

    frames, _, _ = _self_test_shape(audio_path)
    generator = torch.Generator().manual_seed(zlib.crc32(audio_path.stem.encode("utf-8")) ^ 0xA5A5)
    length = frames * StubAudioVAE.downsampling_ratio
    waveform = torch.randn((StubAudioVAE.vae_model.in_channels, length), generator=generator)
    return waveform, StubAudioVAE.sample_rate // 2


def _run_self_test(args: argparse.Namespace) -> None:
    import tempfile

    import torch

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="magi2_preprocess_self_test_"))
    )
    entries = _self_test_entries(args.num_self_test_samples)
    video_vae, text_encoder, audio_vae = StubVideoVAE(), StubTextEncoder(), StubAudioVAE()
    write_dtype = getattr(torch, LATENT_DTYPE)
    samples = [
        _encode_entry(
            entry,
            video_vae,
            text_encoder,
            audio_vae,
            write_dtype,
            decode_video=_self_test_decode_video,
            load_waveform=_self_test_load_waveform,
        )
        for entry in entries
    ]
    _write_shards(samples, output_dir, args.samples_per_shard)

    # Load side: the full Magi2LatentDataset pack contract over the shards.
    from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

    dataset = Magi2LatentDataset(str(output_dir), max_tokens_per_pack=4096, seed=args.seed)
    input_dict, labels = next(iter(dataset))
    if labels.shape[1] != AUDIO_LATENT_CHANNELS or "cu_seqlens" not in input_dict:
        raise RuntimeError("self-test failed: loaded pack does not match the model contract")
    logger.info(
        "self-test ok: %d samples encoded and re-loaded from %s", len(samples), output_dir
    )


def _run_encode(args: argparse.Namespace) -> None:
    import torch

    _prepare_magi2_repo(args.magi2_repo)
    if args.input_manifest:
        entries = read_manifest(args.input_manifest)
    else:
        entries = _directory_entries(args.input, audio_expected=args.audio_vae_ckpt is not None)
    audio_entries = sum(1 for entry in entries if entry["audio"] is not None)
    if audio_entries and args.audio_vae_ckpt is None:
        raise ValueError(
            f"{audio_entries} sample(s) carry audio tracks but --audio-vae-ckpt is not set"
        )

    video_vae = _load_video_vae(args.vae_ckpt, args.device)
    text_encoder = _load_text_encoder(args.text_encoder_path, args.device)
    audio_vae = _load_audio_vae(args.audio_vae_ckpt) if args.audio_vae_ckpt else None
    write_dtype = getattr(torch, LATENT_DTYPE)

    samples = []
    for entry in entries:
        logger.info("encoding %s", entry["id"])
        samples.append(_encode_entry(entry, video_vae, text_encoder, audio_vae, write_dtype))
    _write_shards(samples, Path(args.output_dir), args.samples_per_shard)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.dry_run:
        _run_dry_run(args)
    elif args.self_test:
        _run_self_test(args)
    else:
        _run_encode(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
