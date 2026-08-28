# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preprocess raw videos/captions into MAGI-2-preview latent training shards.

Writes the shard directory consumed by
``torchtitan_npu.models.magi2_preview.latent_dataset.Magi2LatentDataLoader``
(see docs/user-guides/magi2_preview_data_pipeline.md for the format spec).

Dry run (CPU, no weights): emits a tiny random-latent shard set usable by
the loader end to end:

    python3 scripts/magi2_preprocess_latents.py --dry-run \
        --output-dir ./magi2_latent_shards

Real encoding requires the official MAGI-2 inference repo on PYTHONPATH and
the VAE / text-encoder weights; every encoder import is lazy and fails with
an actionable message when a dependency or weight is missing:

    PYTHONPATH=/path/to/MAGI-2 python3 scripts/magi2_preprocess_latents.py \
        --input ./videos --output-dir ./magi2_latent_shards \
        --vae-ckpt /weights/Wan2.2_VAE.pth \
        --text-encoder-path /weights/qwen3.5 \
        [--audio-vae-ckpt /weights/sa_audio_vae] [--device cuda]

``--input`` holds ``<name>.mp4`` videos with paired ``<name>.txt`` captions
(and optional ``<name>.wav`` audio when ``--audio-vae-ckpt`` is given).
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
LATENT_DTYPE = "float16"

# Dry-run latent shapes (T, H, W) cycled across samples so the output
# exercises bucketing and multi-shape packing.
DRY_RUN_SHAPES = ((2, 4, 4), (2, 4, 8), (4, 4, 4))
DRY_RUN_AUDIO_LEN = 16
DRY_RUN_TEXT_LEN = 16


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory of <name>.mp4 videos with paired <name>.txt captions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the latent shards and index.json into.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=64)
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="Official Wan2.2 video VAE checkpoint (Wan2.2_VAE.pth).",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default=None,
        help="HuggingFace directory of the Qwen3.5 text encoder.",
    )
    parser.add_argument(
        "--audio-vae-ckpt",
        type=str,
        default=None,
        help="Optional official audio VAE checkpoint; omit for video+text only.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a tiny random-latent shard set without any encoders.",
    )
    parser.add_argument("--num-dry-run-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.dry_run:
        missing = [
            name
            for name, value in (("--input", args.input), ("--vae-ckpt", args.vae_ckpt),
                                ("--text-encoder-path", args.text_encoder_path))
            if not value
        ]
        if missing:
            parser.error(f"real encoding requires {', '.join(missing)} (or use --dry-run)")
    return args


def _load_video_vae(ckpt_path: str, device: str):
    """Lazy-load the official video VAE with an actionable error message."""
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"--vae-ckpt not found: {ckpt_path}")
    try:
        from inference.model.vae2_2 import get_vae2_2
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import inference.model.vae2_2 from the official MAGI-2 repo; "
            "clone https://github.com/SandAI-org/MAGI-2 and add its root to PYTHONPATH"
        ) from exc
    import torch

    return get_vae2_2(ckpt_path, device=device, weight_dtype=torch.float32)


def _load_text_encoder(model_path: str, device: str):
    """Lazy-load the official Qwen3.5 text encoder."""
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"--text-encoder-path not found: {model_path}")
    try:
        from inference.model.qwen35 import Qwen35TextEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import inference.model.qwen35 from the official MAGI-2 repo; "
            "clone https://github.com/SandAI-org/MAGI-2 and add its root to PYTHONPATH "
            "(the encoder also needs transformers with Qwen3_5TextModel support)"
        ) from exc
    return Qwen35TextEncoder(model_path, device=device)


def _load_audio_vae(ckpt_path: str):
    """Lazy-load the official audio VAE feature extractor."""
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"--audio-vae-ckpt not found: {ckpt_path}")
    try:
        from inference.pipeline.audio_decoder import SAAudioFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import inference.pipeline.audio_decoder from the official MAGI-2 "
            "repo; clone https://github.com/SandAI-org/MAGI-2 and add its root to PYTHONPATH"
        ) from exc
    return SAAudioFeatureExtractor(ckpt_path)


def _decode_video(video_path: Path):
    """Decode one video to a (1, 3, T, H, W) float tensor in [-1, 1]."""
    try:
        from torchvision.io import read_video
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to decode videos; install it or pre-encode "
            "the latents elsewhere"
        ) from exc
    import torch

    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"No video frames decoded from {video_path}")
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32)
    return video / 127.5 - 1.0


def _encode_sample(
    video_path: Path,
    vae,
    text_encoder,
    audio_vae,
    write_dtype,
) -> dict:
    """Encode one video/caption pair into a shard sample mapping."""
    import torch

    caption_path = video_path.with_suffix(".txt")
    if not caption_path.is_file():
        raise FileNotFoundError(f"Caption file missing for {video_path}: expected {caption_path}")

    video = _decode_video(video_path)
    video_latent = vae.encode(video).squeeze(0).to(write_dtype)

    caption = caption_path.read_text(encoding="utf-8")
    text_emb = text_encoder.encode(caption).squeeze(0).to(write_dtype)

    audio_latent = torch.zeros(0, 64, dtype=write_dtype)
    waveform_path = video_path.with_suffix(".wav")
    if audio_vae is not None:
        if not waveform_path.is_file():
            raise FileNotFoundError(
                f"Audio waveform missing for {video_path}: expected {waveform_path} "
                "(or drop --audio-vae-ckpt to encode video+text only)"
            )
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required to load audio waveforms") from exc
        waveform, _ = torchaudio.load(str(waveform_path))
        audio_latent = audio_vae.encode(waveform).squeeze(0).to(write_dtype)

    return {
        "id": video_path.stem,
        "video_latent": video_latent,
        "audio_latent": audio_latent,
        "text_emb": text_emb,
        "fps": 0.0,
        "num_frames": int(video.shape[2]),
        "source": video_path.name,
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


def _run_encode(args: argparse.Namespace) -> None:
    import torch

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input is not a directory: {input_dir}")
    video_paths = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos ({', '.join(VIDEO_SUFFIXES)}) under {input_dir}")

    vae = _load_video_vae(args.vae_ckpt, args.device)
    text_encoder = _load_text_encoder(args.text_encoder_path, args.device)
    audio_vae = _load_audio_vae(args.audio_vae_ckpt) if args.audio_vae_ckpt else None
    write_dtype = getattr(torch, LATENT_DTYPE)

    samples = []
    for video_path in video_paths:
        logger.info("encoding %s", video_path.name)
        samples.append(_encode_sample(video_path, vae, text_encoder, audio_vae, write_dtype))
    _write_shards(samples, Path(args.output_dir), args.samples_per_shard)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.dry_run:
        _run_dry_run(args)
    else:
        _run_encode(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
