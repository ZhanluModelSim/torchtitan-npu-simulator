# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scripts/magi2_preprocess_latents.py preprocessing wiring.

Covers the JSON Lines manifest round trip, frame-sampling conventions, the
weight-less stub-encoder plumbing (decode -> encode -> shard -> load), the
``--self-test`` / ``--dry-run`` entry points, and the actionable error paths
for missing repos, weights, and video decoders.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

INPUT_DICT_KEYS = {
    "input",
    "coords_mapping",
    "modality_mapping",
    "time_embedding",
    "cu_seqlens",
}


def _load_script_module():
    candidates = []
    # Repo layout from this test file: tests/unit_tests/tools/<file>.
    test_root = Path(__file__).resolve()
    if len(test_root.parents) > 3:
        candidates.append(test_root.parents[3] / "scripts" / "magi2_preprocess_latents.py")
    # Resolve through any harness symlink: magi2_preview/__init__.py sits three
    # levels below the repo root (models/ -> torchtitan_npu/ -> repo root).
    try:
        from torchtitan_npu.models import magi2_preview

        pkg_root = Path(magi2_preview.__file__).resolve()
        if len(pkg_root.parents) > 3:
            candidates.append(pkg_root.parents[3] / "scripts" / "magi2_preprocess_latents.py")
    except ImportError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("magi2_preprocess_latents", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    pytest.skip("magi2_preprocess_latents.py not found relative to the test file or package")


@pytest.fixture(scope="module")
def script():
    return _load_script_module()


def _write_manifest(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def _expected_video_shape(script, pixel_shape):
    frames, height, width = pixel_shape
    latent_frames = (frames - 1) // script.VAE_TEMPORAL_STRIDE + 1
    return (latent_frames, height // script.VAE_SPATIAL_STRIDE, width // script.VAE_SPATIAL_STRIDE)


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_round_trip(self, script, tmp_path):
        manifest = _write_manifest(
            tmp_path / "manifest.jsonl",
            [
                {"video": "videos/a.mp4", "caption": "first clip", "audio": "audio/a.wav", "id": "clip_a"},
                {"video": "/data/videos/b.mov", "caption": "second clip"},
                {"video": "videos/c.part1.mkv", "caption": "third clip", "audio": "audio/c.wav"},
            ],
        )
        entries = script.read_manifest(manifest)
        assert [entry["id"] for entry in entries] == ["clip_a", "b", "c_part1"]
        assert entries[0]["video"] == tmp_path / "videos" / "a.mp4"
        assert entries[0]["audio"] == tmp_path / "audio" / "a.wav"
        assert entries[0]["caption"] == "first clip"
        # Absolute paths pass through untouched; missing audio stays None.
        assert entries[1]["video"] == Path("/data/videos/b.mov")
        assert entries[1]["audio"] is None
        # Default id derives deterministically from the video stem ('.' sanitized).
        assert entries[2]["id"] == "c_part1"

    def test_manifest_skips_blank_lines(self, script, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '\n{"video": "a.mp4", "caption": "c"}\n   \n', encoding="utf-8"
        )
        entries = script.read_manifest(manifest)
        assert len(entries) == 1

    def test_manifest_rejects_missing_fields(self, script, tmp_path):
        manifest = _write_manifest(tmp_path / "bad.jsonl", [{"video": "a.mp4"}])
        with pytest.raises(ValueError, match="bad.jsonl:1.*caption"):
            script.read_manifest(manifest)

    def test_manifest_rejects_invalid_json(self, script, tmp_path):
        manifest = tmp_path / "bad.jsonl"
        manifest.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:1: invalid JSON"):
            script.read_manifest(manifest)

    def test_manifest_rejects_duplicate_and_dotted_ids(self, script, tmp_path):
        manifest = _write_manifest(
            tmp_path / "dup.jsonl",
            [{"video": "a.mp4", "caption": "x", "id": "same"},
             {"video": "b.mp4", "caption": "y", "id": "same"}],
        )
        with pytest.raises(ValueError, match="duplicate sample id 'same'"):
            script.read_manifest(manifest)
        manifest = _write_manifest(
            tmp_path / "dot.jsonl", [{"video": "a.mp4", "caption": "x", "id": "has.dot"}]
        )
        with pytest.raises(ValueError, match=r"must not contain '\.'"):
            script.read_manifest(manifest)

    def test_manifest_missing_file_and_empty_raise(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="--input-manifest not found"):
            script.read_manifest(tmp_path / "nope.jsonl")
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no samples"):
            script.read_manifest(manifest)


# ---------------------------------------------------------------------------
# Frame sampling conventions
# ---------------------------------------------------------------------------


class TestFrameSampling:
    def test_subsample_and_trim_to_vae_stride(self, script):
        # 50 fps source: keep every 2nd frame (100), trim to 97 (1 mod 8).
        indices = script.sample_frame_indices(200, native_fps=50.0)
        assert len(indices) == 97
        assert indices[0] == 0 and indices[1] == 2 and indices[-1] == 192

    def test_native_fps_below_target_keeps_all_then_trims(self, script):
        indices = script.sample_frame_indices(100, native_fps=12.0)
        assert indices == list(range(97))

    def test_exact_convention_counts(self, script):
        # 250 frames at 25 fps (the official 10 s clip): 32 latent frames.
        indices = script.sample_frame_indices(250, native_fps=25.0)
        assert len(indices) == 249
        latent_frames = (len(indices) - 1) // script.VAE_TEMPORAL_STRIDE + 1
        assert latent_frames == 32

    def test_unknown_fps_keeps_every_frame(self, script):
        assert script.sample_frame_indices(20, native_fps=0.0) == list(range(17))

    def test_too_short_raises(self, script):
        with pytest.raises(ValueError, match="too short"):
            script.sample_frame_indices(8, native_fps=25.0)

    def test_crop_to_vae_grid_center_crops(self, script):
        video = torch.arange(3 * 40 * 36, dtype=torch.float32).reshape(1, 3, 1, 40, 36)
        cropped = script._crop_to_vae_grid(video)
        assert cropped.shape[-2:] == (32, 32)
        assert torch.equal(cropped, video[..., 4:36, 2:34])

    def test_crop_rejects_tiny_frames(self, script):
        video = torch.zeros(1, 3, 1, 12, 20)
        with pytest.raises(ValueError, match="too small"):
            script._crop_to_vae_grid(video)


# ---------------------------------------------------------------------------
# Stub-encoder plumbing
# ---------------------------------------------------------------------------


class TestStubEncoderPlumbing:
    def _encode_all(self, script, num_samples=4):
        entries = script._self_test_entries(num_samples)
        write_dtype = torch.float16
        return [
            script._encode_entry(
                entry,
                script.StubVideoVAE(),
                script.StubTextEncoder(),
                script.StubAudioVAE(),
                write_dtype,
                decode_video=script._self_test_decode_video,
                load_waveform=script._self_test_load_waveform,
            )
            for entry in entries
        ]

    def test_encode_entry_shapes_dtype_and_attrs(self, script):
        samples = self._encode_all(script)
        for index, sample in enumerate(samples):
            pixel_shape = script.SELF_TEST_SHAPES[index % len(script.SELF_TEST_SHAPES)]
            expected = _expected_video_shape(script, pixel_shape)
            assert sample["id"] == f"selftest_{index:04d}"
            assert sample["video_latent"].shape == (48, *expected)
            assert sample["video_latent"].dtype == torch.float16
            assert sample["text_emb"].shape[1] == script.TEXT_EMBED_DIM
            assert sample["text_emb"].dtype == torch.float16
            assert sample["fps"] == script.VIDEO_TARGET_FPS
            assert sample["num_frames"] == pixel_shape[0]
            # Audio on even entries only; rows align with the pixel frame count.
            if index % 2 == 0:
                assert sample["audio_latent"].shape == (pixel_shape[0], script.AUDIO_LATENT_CHANNELS)
            else:
                assert sample["audio_latent"].shape == (0, script.AUDIO_LATENT_CHANNELS)

    def test_stub_encoders_are_deterministic(self, script):
        first, second = self._encode_all(script), self._encode_all(script)
        for sample_a, sample_b in zip(first, second, strict=True):
            for key in ("video_latent", "audio_latent", "text_emb"):
                assert torch.equal(sample_a[key], sample_b[key])

    def test_self_test_waveform_exercises_both_resamples(self, script):
        # The synthetic waveform arrives at half the stub sample rate, so the
        # plumbing resamples the waveform up and the latent back down.
        waveform, rate = script._self_test_load_waveform(Path("synthetic_0000.wav"))
        assert rate == script.StubAudioVAE.sample_rate // 2
        latent = script._encode_audio_latent(
            script.StubAudioVAE(),
            Path("synthetic_0000.wav"),
            target_len=9,
            write_dtype=torch.float16,
            load_waveform=script._self_test_load_waveform,
        )
        assert latent.shape == (9, script.AUDIO_LATENT_CHANNELS)


# ---------------------------------------------------------------------------
# End-to-end --self-test and --dry-run invocations
# ---------------------------------------------------------------------------


class TestSelfTestEndToEnd:
    def test_self_test_full_encode_shard_load_path(self, script, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataLoader,
        )

        output_dir = tmp_path / "self_test_shards"
        # 8 samples cycle the 4 stub shapes twice, so every bucket holds two
        # samples and the loader yields genuine multi-sample packs.
        exit_code = script.main(
            ["--self-test", "--output-dir", str(output_dir), "--num-self-test-samples", "8",
             "--samples-per-shard", "2", "--seed", "0"]
        )
        assert exit_code == 0

        shard_files = sorted(output_dir.glob("shard_*.safetensors"))
        assert len(shard_files) == 4
        index = json.loads((output_dir / "index.json").read_text())
        assert sum(len(shard["samples"]) for shard in index["shards"]) == 8

        config = Magi2LatentDataLoader.Config(
            data_path=str(output_dir), max_tokens_per_pack=4096, seed=0
        )
        loader = Magi2LatentDataLoader(config, dp_world_size=1, dp_rank=0)
        input_dict, labels = next(iter(loader))
        assert set(input_dict) == INPUT_DICT_KEYS
        assert labels.shape == (input_dict["input"].shape[0], 64)
        # Multi-sample packs carry one cu_seqlens segment per sample.
        cu_seqlens = input_dict["cu_seqlens"].tolist()
        assert len(cu_seqlens) >= 3 and cu_seqlens[0] == 0
        # Stream checkpointing survives the shard set built by the script.
        state = loader.state_dict()
        rebuilt = Magi2LatentDataLoader(config, dp_world_size=1, dp_rank=0)
        rebuilt.load_state_dict(state)
        resumed_dict, resumed_labels = next(iter(rebuilt))
        next_dict, next_labels = next(iter(loader))
        assert torch.equal(resumed_dict["input"], next_dict["input"])
        assert torch.equal(resumed_labels, next_labels)

    def test_self_test_is_deterministic_across_runs(self, script, tmp_path):
        for run in ("first", "second"):
            assert script.main(
                ["--self-test", "--output-dir", str(tmp_path / run),
                 "--samples-per-shard", "4", "--seed", "0"]
            ) == 0
        first = load_file(str(tmp_path / "first" / "shard_0000.safetensors"))
        second = load_file(str(tmp_path / "second" / "shard_0000.safetensors"))
        assert first.keys() == second.keys()
        for key in first:
            assert torch.equal(first[key], second[key])

    def test_self_test_without_output_dir_uses_tempdir(self, script, tmp_path, monkeypatch):
        scratch = tmp_path / "auto"
        monkeypatch.setattr(
            "tempfile.mkdtemp", lambda prefix=None: str(scratch), raising=True
        )
        assert script.main(["--self-test", "--num-self-test-samples", "2"]) == 0
        assert sorted(path.name for path in scratch.glob("shard_*.safetensors"))

    def test_dry_run_cli_still_supported(self, script, tmp_path):
        output_dir = tmp_path / "dryrun"
        assert script.main(
            ["--dry-run", "--output-dir", str(output_dir),
             "--num-dry-run-samples", "4", "--samples-per-shard", "2", "--seed", "1"]
        ) == 0
        assert len(list(output_dir.glob("shard_*.safetensors"))) == 2
        assert (output_dir / "index.json").is_file()


# ---------------------------------------------------------------------------
# Actionable error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_prepare_repo_rejects_non_clone(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="SandAI-org/MAGI-2-preview"):
            script._prepare_magi2_repo(str(tmp_path / "missing"))
        empty = tmp_path / "empty_repo"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="vae2_2.py"):
            script._prepare_magi2_repo(str(empty))

    def test_encoder_import_error_is_actionable(self, script, monkeypatch, tmp_path):
        repo = tmp_path / "MAGI-2-preview"
        (repo / "inference" / "model").mkdir(parents=True)
        (repo / "inference" / "model" / "vae2_2.py").write_text("", encoding="utf-8")
        script._prepare_magi2_repo(str(repo))  # path validation passes
        try:
            monkeypatch.setitem(sys.modules, "inference.model.vae2_2", None)
            with pytest.raises(RuntimeError, match="--magi2-repo"):
                script._load_video_vae(str(tmp_path / "no.pth"), "cpu")
        finally:
            sys.path.remove(str(repo.resolve()))

    def test_vae_ckpt_dir_without_weights(self, script, tmp_path):
        empty = tmp_path / "vae"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
            script._resolve_vae_ckpt(str(empty))
        with pytest.raises(FileNotFoundError, match="--vae-ckpt not found"):
            script._resolve_vae_ckpt(str(tmp_path / "nope.pth"))

    def test_audio_vae_ckpt_must_be_a_complete_dir(self, script, tmp_path):
        stray = tmp_path / "audio.pth"
        stray.write_text("", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="audio VAE directory"):
            script._load_audio_vae(str(stray))
        partial = tmp_path / "stable-audio-open-1.0"
        partial.mkdir()
        (partial / "model_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            script._load_audio_vae(str(partial))

    def test_text_encoder_path_missing(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="--text-encoder-path not found"):
            script._load_text_encoder(str(tmp_path / "nope"), "cpu")

    def test_decoder_missing_error_lists_both_options(self, script, monkeypatch, tmp_path):
        video = tmp_path / "whatever.mp4"
        video.write_bytes(b"")
        for name in ("torchvision", "torchvision.io", "imageio"):
            monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(RuntimeError, match="torchvision") as excinfo:
            script._decode_video(video)
        assert "imageio-ffmpeg" in str(excinfo.value)

    def test_real_encode_requires_sources_and_weights(self, script, tmp_path):
        with pytest.raises(SystemExit):
            script.main(["--output-dir", str(tmp_path)])
        with pytest.raises(SystemExit):
            script.main(
                ["--output-dir", str(tmp_path), "--input", "x", "--input-manifest", "y",
                 "--vae-ckpt", "v", "--text-encoder-path", "t"]
            )
        with pytest.raises(SystemExit):
            script.main(["--dry-run", "--self-test", "--output-dir", str(tmp_path)])

    def test_audio_entries_require_audio_vae(self, script, tmp_path):
        manifest = _write_manifest(
            tmp_path / "manifest.jsonl",
            [{"video": "a.mp4", "caption": "x", "audio": "a.wav"}],
        )
        args = script._parse_args(
            ["--input-manifest", str(manifest), "--output-dir", str(tmp_path / "out"),
             "--vae-ckpt", str(tmp_path / "v.pth"), "--text-encoder-path", str(tmp_path)]
        )
        # The guard fires before any encoder weight is touched.
        with pytest.raises(ValueError, match="--audio-vae-ckpt is not set"):
            script._run_encode(args)

    def test_missing_media_files_raise_before_decoders(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            script._decode_video(tmp_path / "gone.mp4")
        with pytest.raises(FileNotFoundError, match="Audio waveform not found"):
            script._load_waveform_wav(tmp_path / "gone.wav")

    def test_directory_mode_requires_captions(self, script, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="Caption file missing"):
            script._directory_entries(str(tmp_path), audio_expected=False)
