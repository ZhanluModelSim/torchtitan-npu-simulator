# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the phase-3 preprocessing pipeline additions.

Covers the pluggable encoder registry (registration, lookup, build),
CLI aliases and environment-variable checkpoint resolution, the
``--dry-run`` end-to-end shard writing path, and the
``Magi2LatentShardReader`` round-trip against the shard format produced
by the script.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

INPUT_DICT_KEYS = {
    "input",
    "coords_mapping",
    "modality_mapping",
    "time_embedding",
    "cu_seqlens",
}


def _load_script_module():
    """Locate and import scripts/magi2_preprocess_latents.py as a module."""
    candidates = []
    test_root = Path(__file__).resolve()
    # tests/unit_tests/scripts/<this> -> 3 levels up = repo root.
    if len(test_root.parents) > 3:
        candidates.append(test_root.parents[3] / "scripts" / "magi2_preprocess_latents.py")
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


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------


class TestEncoderRegistry:
    def test_built_in_encoders_registered(self, script):
        names = script.EncoderRegistry.names()
        for expected in ("video_vae", "text", "audio_vae"):
            assert expected in names, f"{expected} not in registry: {names}"

    def test_stub_encoders_registered(self, script):
        names = script.EncoderRegistry.names()
        for expected in ("stub_video_vae", "stub_text", "stub_audio_vae"):
            assert expected in names, f"{expected} not in registry: {names}"

    def test_get_unknown_encoder_raises(self, script):
        with pytest.raises(KeyError, match="Unknown encoder"):
            script.EncoderRegistry.get("nonexistent_encoder")

    def test_register_custom_encoder(self, script):
        original_names = set(script.EncoderRegistry.names())

        class _MyEncoder(script.BaseEncoder):
            name = "test_custom"

            @classmethod
            def from_config(cls, *, ckpt="fake", **kw):
                return cls()

            def encode(self, x):
                return x * 2

        script.EncoderRegistry.register(_MyEncoder)
        try:
            assert "test_custom" in script.EncoderRegistry.names()
            assert script.EncoderRegistry.get("test_custom") is _MyEncoder
            instance = script.EncoderRegistry.build("test_custom")
            assert instance.encode(3) == 6
        finally:
            # Clean up: remove the test registration.
            script.EncoderRegistry._encoders.pop("test_custom", None)
            assert script.EncoderRegistry.names() == sorted(original_names)

    def test_register_rejects_nameless_encoder(self, script):
        class _NamelessEncoder(script.BaseEncoder):
            name = ""

        with pytest.raises(ValueError, match="non-empty 'name'"):
            script.EncoderRegistry.register(_NamelessEncoder)

    def test_stub_video_encoder_from_config(self, script):
        encoder = script.EncoderRegistry.build("stub_video_vae")
        assert hasattr(encoder, "encode")
        assert encoder.device == "cpu"

    def test_stub_text_encoder_from_config(self, script):
        encoder = script.EncoderRegistry.build("stub_text")
        result = encoder.encode("hello world")
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3  # (1, L, D)

    def test_stub_audio_encoder_from_config(self, script):
        encoder = script.EncoderRegistry.build("stub_audio_vae")
        assert encoder.device == "cpu"
        assert encoder.sample_rate == script.StubAudioVAE.sample_rate


# ---------------------------------------------------------------------------
# CLI aliases and env-var resolution
# ---------------------------------------------------------------------------


class TestCLIAliases:
    def test_video_ckpt_new_name(self, script):
        args = script._parse_args(["--dry-run", "--output-dir", "x", "--video-ckpt", "/v"])
        assert args.video_ckpt == "/v"

    def test_video_ckpt_legacy_alias(self, script):
        args = script._parse_args(["--dry-run", "--output-dir", "x", "--vae-ckpt", "/legacy"])
        assert args.video_ckpt == "/legacy"

    def test_text_ckpt_new_name(self, script):
        args = script._parse_args(["--dry-run", "--output-dir", "x", "--text-ckpt", "/t"])
        assert args.text_ckpt == "/t"

    def test_text_ckpt_legacy_alias(self, script):
        args = script._parse_args(
            ["--dry-run", "--output-dir", "x", "--text-encoder-path", "/legacy_t"]
        )
        assert args.text_ckpt == "/legacy_t"

    def test_audio_ckpt_new_name(self, script):
        args = script._parse_args(["--dry-run", "--output-dir", "x", "--audio-ckpt", "/a"])
        assert args.audio_ckpt == "/a"

    def test_audio_ckpt_legacy_alias(self, script):
        args = script._parse_args(
            ["--dry-run", "--output-dir", "x", "--audio-vae-ckpt", "/legacy_a"]
        )
        assert args.audio_ckpt == "/legacy_a"


class TestEnvVarOverride:
    def test_video_ckpt_from_env(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_VIDEO_CKPT", "/env/video")
        args = script._parse_args(["--dry-run", "--output-dir", "x"])
        assert args.video_ckpt == "/env/video"

    def test_text_ckpt_from_env(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_TEXT_CKPT", "/env/text")
        args = script._parse_args(["--dry-run", "--output-dir", "x"])
        assert args.text_ckpt == "/env/text"

    def test_audio_ckpt_from_env(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_AUDIO_CKPT", "/env/audio")
        args = script._parse_args(["--dry-run", "--output-dir", "x"])
        assert args.audio_ckpt == "/env/audio"

    def test_cli_overrides_env(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_VIDEO_CKPT", "/env/video")
        args = script._parse_args(
            ["--dry-run", "--output-dir", "x", "--video-ckpt", "/cli/video"]
        )
        assert args.video_ckpt == "/cli/video"

    def test_legacy_cli_overrides_env(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_TEXT_CKPT", "/env/text")
        args = script._parse_args(
            ["--dry-run", "--output-dir", "x", "--text-encoder-path", "/legacy/text"]
        )
        assert args.text_ckpt == "/legacy/text"

    def test_no_env_no_cli_yields_none(self, script, monkeypatch):
        monkeypatch.delenv("MAGI2_VIDEO_CKPT", raising=False)
        monkeypatch.delenv("MAGI2_TEXT_CKPT", raising=False)
        monkeypatch.delenv("MAGI2_AUDIO_CKPT", raising=False)
        args = script._parse_args(["--dry-run", "--output-dir", "x"])
        assert args.video_ckpt is None
        assert args.text_ckpt is None
        assert args.audio_ckpt is None


# ---------------------------------------------------------------------------
# --dry-run end-to-end writes shards + index
# ---------------------------------------------------------------------------


class TestDryRunEndToEnd:
    def test_dry_run_writes_shards_and_index(self, script, tmp_path):
        output_dir = tmp_path / "dryrun_shards"
        exit_code = script.main([
            "--dry-run", "--output-dir", str(output_dir),
            "--num-dry-run-samples", "6", "--samples-per-shard", "3", "--seed", "42",
        ])
        assert exit_code == 0

        shard_files = sorted(output_dir.glob("shard_*.safetensors"))
        assert len(shard_files) == 2

        index_path = output_dir / "index.json"
        assert index_path.is_file()
        index = json.loads(index_path.read_text())
        assert index["format"] == script.LATENT_DTYPE or "format" in index
        assert len(index["shards"]) == 2
        total_samples = sum(len(s["samples"]) for s in index["shards"])
        assert total_samples == 6

    def test_dry_run_deterministic_across_runs(self, script, tmp_path):
        for run_name in ("run_a", "run_b"):
            script.main([
                "--dry-run", "--output-dir", str(tmp_path / run_name),
                "--num-dry-run-samples", "4", "--samples-per-shard", "4", "--seed", "7",
            ])
        from safetensors.torch import load_file

        a = load_file(str(tmp_path / "run_a" / "shard_0000.safetensors"))
        b = load_file(str(tmp_path / "run_b" / "shard_0000.safetensors"))
        assert set(a.keys()) == set(b.keys())
        for key in a:
            assert torch.equal(a[key], b[key])


# ---------------------------------------------------------------------------
# Shard reader round-trip
# ---------------------------------------------------------------------------


class TestShardReaderRoundTrip:
    def test_reader_metadata_after_dry_run(self, script, tmp_path):
        output_dir = tmp_path / "reader_test"
        script.main([
            "--dry-run", "--output-dir", str(output_dir),
            "--num-dry-run-samples", "4", "--samples-per-shard", "2", "--seed", "0",
        ])

        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentShardReader,
        )

        reader = Magi2LatentShardReader(str(output_dir))
        assert reader.num_shards == 2
        assert reader.num_samples == 4
        assert len(reader) == 4
        ids = reader.sample_ids()
        assert len(ids) == 4
        assert ids[0] == "dryrun_0000"

    def test_reader_iterates_all_samples(self, script, tmp_path):
        output_dir = tmp_path / "iter_test"
        script.main([
            "--dry-run", "--output-dir", str(output_dir),
            "--num-dry-run-samples", "6", "--samples-per-shard", "3", "--seed", "1",
        ])

        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentShardReader,
        )

        reader = Magi2LatentShardReader(str(output_dir))
        samples = list(reader)
        assert len(samples) == 6
        for sample in samples:
            assert "id" in sample
            assert "video_latent" in sample
            assert "audio_latent" in sample
            assert "text_emb" in sample
            # Video latent shape: (48, T, H, W).
            assert sample["video_latent"].shape[0] == 48
            assert sample["video_latent"].ndim == 4
            # Audio latent: (L, 64).
            assert sample["audio_latent"].ndim == 2
            assert sample["audio_latent"].shape[1] == 64
            # Text emb: (L, 5120).
            assert sample["text_emb"].ndim == 2
            assert sample["text_emb"].shape[1] == script.TEXT_EMBED_DIM

    def test_reader_round_trip_via_self_test(self, script, tmp_path):
        """Write via --self-test, read back with the shard reader."""
        output_dir = tmp_path / "selftest_shards"
        script.main([
            "--self-test", "--output-dir", str(output_dir),
            "--num-self-test-samples", "4", "--samples-per-shard", "2",
        ])

        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentShardReader,
        )

        reader = Magi2LatentShardReader(str(output_dir))
        assert reader.num_samples == 4
        samples = list(reader)
        ids = [s["id"] for s in samples]
        assert "selftest_0000" in ids
        assert "selftest_0003" in ids
        # Every loaded sample can be re-written (round-trip through write_latent_shard).
        from torchtitan_npu.models.magi2_preview.latent_dataset import write_latent_shard

        re_written = write_latent_shard(tmp_path / "round_trip.safetensors", samples)
        assert len(re_written) == 4

    def test_reader_load_sample_preserves_attrs(self, script, tmp_path):
        output_dir = tmp_path / "attrs_test"
        script.main([
            "--dry-run", "--output-dir", str(output_dir),
            "--num-dry-run-samples", "2", "--samples-per-shard", "2", "--seed", "0",
        ])

        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentShardReader,
        )

        reader = Magi2LatentShardReader(str(output_dir))
        sample = next(iter(reader))
        # The dry-run samples carry fps, num_frames, source as attrs,
        # which load_sample flattens into the top-level dict.
        assert "fps" in sample
        assert "num_frames" in sample
        assert sample["fps"] == 25.0


# ---------------------------------------------------------------------------
# Lazy-import error messages
# ---------------------------------------------------------------------------


class TestLazyImportErrors:
    def test_video_vae_import_error_is_actionable(self, script, monkeypatch, tmp_path):
        repo = tmp_path / "MAGI-2-preview"
        (repo / "inference" / "model").mkdir(parents=True)
        (repo / "inference" / "model" / "vae2_2.py").write_text("", encoding="utf-8")
        script._prepare_magi2_repo(str(repo))
        try:
            monkeypatch.setitem(sys.modules, "inference.model.vae2_2", None)
            with pytest.raises(RuntimeError, match="--magi2-repo"):
                script.EncoderRegistry.build("video_vae", ckpt=str(tmp_path / "no.pth"))
        finally:
            sys.path.remove(str(repo.resolve()))

    def test_text_encoder_missing_dir(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="--text-encoder-path not found"):
            script.EncoderRegistry.build("text", ckpt=str(tmp_path / "nonexistent"))

    def test_audio_vae_missing_dir(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="audio VAE directory"):
            script.EncoderRegistry.build("audio_vae", ckpt=str(tmp_path / "nonexistent"))

    def test_audio_vae_incomplete_dir(self, script, tmp_path):
        partial = tmp_path / "audio_partial"
        partial.mkdir()
        (partial / "model_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            script.EncoderRegistry.build("audio_vae", ckpt=str(partial))


# ---------------------------------------------------------------------------
# Resolve helper unit tests
# ---------------------------------------------------------------------------


class TestResolveEncoderCkpt:
    def test_cli_value_wins(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_VIDEO_CKPT", "/env/path")
        result = script._resolve_encoder_ckpt("/cli/path", None, "MAGI2_VIDEO_CKPT")
        assert result == "/cli/path"

    def test_legacy_value_when_cli_absent(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_TEXT_CKPT", "/env/path")
        result = script._resolve_encoder_ckpt(None, "/legacy/path", "MAGI2_TEXT_CKPT")
        assert result == "/legacy/path"

    def test_env_when_both_cli_absent(self, script, monkeypatch):
        monkeypatch.setenv("MAGI2_AUDIO_CKPT", "/env/audio")
        result = script._resolve_encoder_ckpt(None, None, "MAGI2_AUDIO_CKPT")
        assert result == "/env/audio"

    def test_none_when_nothing_set(self, script, monkeypatch):
        monkeypatch.delenv("MAGI2_VIDEO_CKPT", raising=False)
        result = script._resolve_encoder_ckpt(None, None, "MAGI2_VIDEO_CKPT")
        assert result is None
