# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MAGI-2-preview offline latent dataset pipeline.

Covers the shard container round trip, bucketed multi-sample packing against
the phase-1 input_dict contract, flow-matching noise reconstruction, dp-rank
file sharding, checkpoint resume, and the ``--dry-run`` preprocessing script.
"""

import importlib.util
import json
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


def _fake_sample(sample_id, video_shape, audio_len=16, text_len=16, generator=None, **attrs):
    """One fake shard sample with N(0, 1) latents in the official layout."""
    frames, height, width = video_shape
    return {
        "id": sample_id,
        "video_latent": torch.randn((48, frames, height, width), generator=generator).half(),
        "audio_latent": torch.randn((audio_len, 64), generator=generator).half(),
        "text_emb": torch.randn((text_len, 5120), generator=generator).half(),
        "fps": 25.0,
        "num_frames": (frames - 1) * 4 + 1,
        **attrs,
    }


def _write_two_bucket_dir(tmp_path, samples_per_bucket=3):
    """Shard dir with two video shapes (two buckets) across two shard files."""
    from torchtitan_npu.models.magi2_preview.latent_dataset import write_latent_shard

    generator = torch.Generator().manual_seed(0)
    small = [
        _fake_sample(f"small_{i}", (2, 4, 4), generator=generator)
        for i in range(samples_per_bucket)
    ]
    large = [
        _fake_sample(f"large_{i}", (2, 4, 8), generator=generator)
        for i in range(samples_per_bucket)
    ]
    write_latent_shard(tmp_path / "shard_0000.safetensors", small)
    write_latent_shard(tmp_path / "shard_0001.safetensors", large)
    return tmp_path


def _segments(input_dict):
    cu_seqlens = input_dict["cu_seqlens"].tolist()
    return list(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True))


def _video_grid_shape(input_dict, start):
    """(T, H, W) latent shape carried by the coords rows of a sample's video block."""
    sizes = input_dict["coords_mapping"][start, 3:6]
    return tuple(int(value) for value in sizes.tolist())


# ---------------------------------------------------------------------------
# Shard container
# ---------------------------------------------------------------------------


class TestShardFormat:
    def test_safetensors_round_trip(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            SHARD_FORMAT,
            _read_shard_listing,
            build_latent_index,
            write_latent_shard,
        )

        samples = [
            _fake_sample("a", (2, 4, 4), audio_len=8, text_len=4),
            _fake_sample("b", (4, 2, 2), audio_len=0, text_len=12, caption="hello"),
        ]
        shard_path = tmp_path / "shard_0000.safetensors"
        listing = write_latent_shard(shard_path, samples)

        assert [entry["id"] for entry in listing] == ["a", "b"]
        assert listing[0]["video_shape"] == [2, 4, 4]
        assert listing[1]["audio_len"] == 0
        assert listing[1]["attrs"]["caption"] == "hello"
        assert listing[0]["attrs"]["fps"] == 25.0

        read_back = _read_shard_listing(str(shard_path))
        assert read_back == listing

        index = build_latent_index(str(tmp_path))
        assert index["format"] == SHARD_FORMAT
        assert index["shards"][0]["file"] == "shard_0000.safetensors"
        assert index["shards"][0]["samples"] == listing

    def test_pt_fallback_round_trip(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataset,
            _read_shard_listing,
            write_latent_shard,
        )

        samples = [_fake_sample("a", (2, 4, 4)), _fake_sample("b", (2, 4, 4))]
        shard_path = tmp_path / "shard_0000.pt"
        listing = write_latent_shard(shard_path, samples)
        assert _read_shard_listing(str(shard_path)) == listing

        dataset = Magi2LatentDataset(str(tmp_path))
        input_dict, labels = next(iter(dataset))
        assert set(input_dict) == INPUT_DICT_KEYS
        assert labels.shape[1] == 64

    def test_listing_derived_without_metadata(self, tmp_path):
        """Hand-built metadata-less safetensors shards are discovered by keys."""
        from safetensors.torch import save_file

        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataset,
            _read_shard_listing,
        )

        tensors = {
            "x.video_latent": torch.randn(48, 2, 4, 4).half(),
            "x.audio_latent": torch.randn(16, 64).half(),
            "x.text_emb": torch.randn(16, 5120).half(),
        }
        save_file(tensors, str(tmp_path / "shard_0000.safetensors"))

        listing = _read_shard_listing(str(tmp_path / "shard_0000.safetensors"))
        assert listing[0]["id"] == "x"
        assert listing[0]["video_shape"] == [2, 4, 4]

        dataset = Magi2LatentDataset(str(tmp_path))
        input_dict, _ = next(iter(dataset))
        assert input_dict["cu_seqlens"].tolist() == [0, 2 * 4 * 4 + 16 + 16]

    def test_writer_rejects_bad_samples(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import write_latent_shard

        bad_channels = _fake_sample("a", (2, 4, 4))
        bad_channels["video_latent"] = torch.randn(16, 2, 4, 4).half()
        with pytest.raises(ValueError, match="video_latent"):
            write_latent_shard(tmp_path / "bad.safetensors", [bad_channels])

        bad_dtype = _fake_sample("a", (2, 4, 4))
        bad_dtype["video_latent"] = torch.randn(48, 2, 4, 4)
        with pytest.raises(ValueError, match="float16/bfloat16"):
            write_latent_shard(tmp_path / "bad.safetensors", [bad_dtype])

        duplicate = [_fake_sample("a", (2, 4, 4)), _fake_sample("a", (2, 4, 4))]
        with pytest.raises(ValueError, match="Duplicate sample id"):
            write_latent_shard(tmp_path / "bad.safetensors", duplicate)

        with pytest.raises(ValueError, match="must end in"):
            write_latent_shard(tmp_path / "bad.bin", [_fake_sample("a", (2, 4, 4))])

    def test_missing_and_empty_dirs_raise(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        with pytest.raises(FileNotFoundError):
            Magi2LatentDataset(str(tmp_path / "does-not-exist"))
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="No shard files"):
            Magi2LatentDataset(str(tmp_path / "empty"))


# ---------------------------------------------------------------------------
# Packing contract
# ---------------------------------------------------------------------------


class TestPackingContract:
    def test_multi_sample_pack_honors_input_dict_contract(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.dataset import MODALITY_VIDEO
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        _write_two_bucket_dir(tmp_path, samples_per_bucket=3)
        dataset = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=4096)
        input_dict, labels = next(iter(dataset))

        assert set(input_dict) == INPUT_DICT_KEYS
        segments = _segments(input_dict)
        # One bucket (2, 4, 4) holds 3 samples * (32 video + 16 audio + 16 text)
        # tokens; the 4096 budget packs all of them into the first sequence.
        assert len(segments) == 3
        for start, end in segments:
            assert end - start == 2 * 4 * 4 + 16 + 16
            modalities = input_dict["modality_mapping"][start:end]
            counts = [(modalities == modality).sum().item() for modality in (0, 1, 2)]
            assert counts == [32, 16, 16]
            # Per-sample layout is video-first in original order.
            assert (modalities[:32] == MODALITY_VIDEO).all()

        total = segments[-1][1]
        assert input_dict["input"].shape == (total, 5120)
        assert input_dict["input"].dtype == torch.float32
        assert input_dict["coords_mapping"].shape == (total, 9)
        assert input_dict["modality_mapping"].shape == (total,)
        assert input_dict["modality_mapping"].dtype == torch.int32
        assert input_dict["time_embedding"].shape == (total, 64)
        assert input_dict["cu_seqlens"].dtype == torch.int32
        assert labels.shape == (total, 64)
        assert torch.isfinite(input_dict["input"]).all()
        assert torch.isfinite(labels).all()
        # Text label rows are zero so the sum-MSE loss stays masked.
        text_rows = input_dict["modality_mapping"] == 2
        assert text_rows.any()
        assert torch.equal(labels[text_rows], torch.zeros_like(labels[text_rows]))

    def test_bucket_homogeneity_and_token_budget(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        _write_two_bucket_dir(tmp_path, samples_per_bucket=4)
        max_tokens = 2 * 4 * 8 + 16 + 16 + 1  # fits one large sample plus slack
        dataset = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=max_tokens)

        shapes_by_pack = []
        iterator = iter(dataset)
        for _ in range(6):
            input_dict, _ = next(iterator)
            pack_shapes = set()
            for start, _ in _segments(input_dict):
                pack_shapes.add(_video_grid_shape(input_dict, start))
            assert len(pack_shapes) == 1, "a pack mixed video shapes"
            shapes_by_pack.append(pack_shapes.pop())
            assert input_dict["input"].shape[0] <= max_tokens
        assert set(shapes_by_pack) == {(2, 4, 4), (2, 4, 8)}

    def test_oversized_sample_is_packed_alone(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataset,
            write_latent_shard,
        )

        sample = _fake_sample("big", (4, 8, 8))  # 256 video + 16 + 16 = 288 tokens
        write_latent_shard(tmp_path / "shard_0000.safetensors", [sample])
        dataset = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=10)
        input_dict, _ = next(iter(dataset))
        assert len(_segments(input_dict)) == 1
        assert input_dict["input"].shape[0] == 288


# ---------------------------------------------------------------------------
# Flow-matching noise
# ---------------------------------------------------------------------------


class TestFlowMatchingNoise:
    def test_noise_reconstruction_identity(self, tmp_path):
        """x0 = x_t - sigma * v recovers the stored latents for every sample."""
        from torchtitan_npu.models.magi2_preview.embeddings import sinusoidal_embedding_1d
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataset,
            sample_noise_rng,
            write_latent_shard,
        )

        generator = torch.Generator().manual_seed(0)
        samples = [_fake_sample(f"s{i}", (2, 4, 4), generator=generator) for i in range(3)]
        write_latent_shard(tmp_path / "shard_0000.safetensors", samples)
        stored = {
            sample["id"]: sample["video_latent"].permute(1, 2, 3, 0).float()
            for sample in samples
        }

        dataset = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=4096, seed=7)
        input_dict, labels = next(iter(dataset))

        candidate_sigmas = [
            torch.rand((), generator=sample_noise_rng(7, 0, position)) for position in range(3)
        ]
        seen_sigmas = set()
        for start, end in _segments(input_dict):
            n_video = 2 * 4 * 4
            x_t = input_dict["input"][start : start + n_video, :48].reshape(2, 4, 4, 48)
            velocity = labels[start : start + n_video, :48].reshape(2, 4, 4, 48)

            # Exactly one candidate sigma reproduces the stored latents; the
            # identity is exact up to fp32 rounding, so match on the argmin
            # residual rather than a loose tolerance.
            best_residual, sigma = None, None
            for candidate in candidate_sigmas:
                for x0 in stored.values():
                    residual = (x_t - candidate * velocity - x0).abs().max().item()
                    if best_residual is None or residual < best_residual:
                        best_residual, sigma = residual, candidate
            # The true (sigma, x0) pair reconstructs to fp32 rounding (~1e-6);
            # any wrong pair is O(1) away, so 1e-4 separates them cleanly.
            assert best_residual < 1e-4
            seen_sigmas.add(round(sigma.item(), 6))

            # Every sample draws its own sigma; the per-token time embedding is
            # constant over the noisy (video + audio) tokens and uses sigma 0
            # for the trailing text tokens.
            modalities = input_dict["modality_mapping"][start:end]
            n_noisy = int((modalities != 2).sum())
            expected = sinusoidal_embedding_1d(64, sigma.expand(1)).expand(end - start, 64).clone()
            expected[n_noisy:] = sinusoidal_embedding_1d(
                64, torch.zeros(end - start - n_noisy)
            )
            assert torch.allclose(input_dict["time_embedding"][start:end], expected, atol=1e-6)

            # Audio rows reconstruct with the same sigma as the video rows.
            audio_x_t = input_dict["input"][start + n_video : start + n_noisy, :64]
            audio_v = labels[start + n_video : start + n_noisy, :64]
            assert any(
                torch.allclose(audio_x_t - sigma * audio_v, sample["audio_latent"].float(), atol=1e-4)
                for sample in samples
            )
        assert len(seen_sigmas) == 3, "each packed sample must draw its own sigma"

    def test_deterministic_stream_for_fixed_seed(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        _write_two_bucket_dir(tmp_path)
        first = Magi2LatentDataset(str(tmp_path), seed=3)
        second = Magi2LatentDataset(str(tmp_path), seed=3)
        for _ in range(4):
            input_a, labels_a = next(iter(first))
            input_b, labels_b = next(iter(second))
            for key in INPUT_DICT_KEYS:
                assert torch.equal(input_a[key], input_b[key])
            assert torch.equal(labels_a, labels_b)


# ---------------------------------------------------------------------------
# dp sharding and resume
# ---------------------------------------------------------------------------


class TestDpShardingAndResume:
    def _four_shard_dir(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import write_latent_shard

        generator = torch.Generator().manual_seed(0)
        for index in range(4):
            sample = _fake_sample(f"sample_{index}", (2, 4, 4), generator=generator)
            write_latent_shard(tmp_path / f"shard_{index:04d}.safetensors", [sample])
        return tmp_path

    def test_dp_rank_file_sharding(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        self._four_shard_dir(tmp_path)
        rank0 = Magi2LatentDataset(str(tmp_path), dp_world_size=2, dp_rank=0)
        rank1 = Magi2LatentDataset(str(tmp_path), dp_world_size=2, dp_rank=1)

        assert len(rank0.shard_files) == 2 and len(rank1.shard_files) == 2
        assert not set(rank0.shard_files) & set(rank1.shard_files)
        assert {Path(path).name for path in rank0.shard_files} == {
            "shard_0000.safetensors",
            "shard_0002.safetensors",
        }
        assert {Path(path).name for path in rank1.shard_files} == {
            "shard_0001.safetensors",
            "shard_0003.safetensors",
        }
        assert {sample.sample_id for sample in rank0._samples} == {"sample_0", "sample_2"}

        with pytest.raises(ValueError, match="no shard files"):
            Magi2LatentDataset(str(tmp_path), dp_world_size=5, dp_rank=4)
        with pytest.raises(ValueError, match="Invalid dp sharding"):
            Magi2LatentDataset(str(tmp_path), dp_world_size=2, dp_rank=2)

    def test_state_dict_resume_reproduces_stream(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        _write_two_bucket_dir(tmp_path, samples_per_bucket=4)
        uninterrupted = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=80, seed=1)

        warm = iter(uninterrupted)
        for _ in range(3):
            next(warm)
        state = uninterrupted.state_dict()
        assert set(state) == {"epoch", "sample_pointer"}

        resumed = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=80, seed=1)
        resumed.load_state_dict(state)
        for _ in range(5):
            input_a, labels_a = next(warm)
            input_b, labels_b = next(iter(resumed))
            for key in INPUT_DICT_KEYS:
                assert torch.equal(input_a[key], input_b[key])
            assert torch.equal(labels_a, labels_b)

    def test_load_state_dict_empty_is_valid(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        _write_two_bucket_dir(tmp_path)
        dataset = Magi2LatentDataset(str(tmp_path))
        dataset.load_state_dict({})
        assert dataset.state_dict() == {"epoch": 0, "sample_pointer": 0}

    def test_infinite_iteration_wraps_epochs(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataset,
            write_latent_shard,
        )

        write_latent_shard(
            tmp_path / "shard_0000.safetensors",
            [_fake_sample("only", (2, 4, 4))],
        )
        dataset = Magi2LatentDataset(str(tmp_path), max_tokens_per_pack=4096)
        iterator = iter(dataset)
        for _ in range(3):
            input_dict, _ = next(iterator)
            assert len(_segments(input_dict)) == 1
        assert dataset.epoch >= 2
        assert dataset.sample_pointer == 0


# ---------------------------------------------------------------------------
# Dataloader build contract and config registry
# ---------------------------------------------------------------------------


class TestDataLoaderContract:
    def test_config_build_with_trainer_kwargs(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataLoader,
            write_latent_shard,
        )

        write_latent_shard(
            tmp_path / "shard_0000.safetensors",
            [_fake_sample("a", (2, 4, 4)), _fake_sample("b", (2, 4, 4))],
        )
        config = Magi2LatentDataLoader.Config(
            data_path=str(tmp_path), max_tokens_per_pack=4096, seed=5
        )
        loader = config.build(
            dp_world_size=1, dp_rank=0, tokenizer=None, seq_len=64, local_batch_size=1
        )

        input_dict, labels = next(iter(loader))
        assert set(input_dict) == INPUT_DICT_KEYS
        assert len(_segments(input_dict)) == 2
        assert labels.shape[1] == 64
        # Both 64-token samples fit one pack, so consuming it wraps to the
        # start of the next epoch.
        assert loader.state_dict() == {"epoch": 1, "sample_pointer": 0}
        loader.load_state_dict({"epoch": 0, "sample_pointer": 0})
        assert loader.dataset.sample_pointer == 0

    def test_latent_smoketest_factory(self):
        from torchtitan_npu.models.magi2_preview.config_registry import (
            magi2_preview_latent_smoketest,
            magi2_preview_smoketest,
        )
        from torchtitan_npu.models.magi2_preview.latent_dataset import (
            Magi2LatentDataLoader,
        )

        config = magi2_preview_latent_smoketest()
        synthetic_config = magi2_preview_smoketest()
        assert isinstance(config.dataloader, Magi2LatentDataLoader.Config)
        assert config.dataloader.data_path
        assert config.training.steps == synthetic_config.training.steps == 2
        assert config.parallelism == synthetic_config.parallelism
        assert config.model_spec.flavor == "debug"

    def test_latent_smoketest_kept_out_of_simulator_registry(self):
        from torchtitan_npu.simulator import config_registry as simulator_registry

        assert not hasattr(simulator_registry, "magi2_preview_latent_smoketest")


# ---------------------------------------------------------------------------
# Synthetic stream determinism (refactor guard)
# ---------------------------------------------------------------------------


class TestSyntheticRefactorGuard:
    @staticmethod
    def _reference_sample(dataset, iteration):
        """Phase-1 sample construction, inlined as the determinism oracle."""
        from torchtitan_npu.models.magi2_preview.dataset import (
            AUDIO_CHANNELS,
            AUDIO_TIME_COMPRESSION,
            LABEL_CHANNELS,
            MAX_IN_CHANNELS,
            MODALITY_AUDIO,
            MODALITY_TEXT,
            MODALITY_VIDEO,
            TIME_CHANNEL_DIM,
            VIDEO_CHANNELS,
            _grid_coords,
        )
        from torchtitan_npu.models.magi2_preview.embeddings import (
            sinusoidal_embedding_1d,
        )

        gen = torch.Generator()
        gen.manual_seed(dataset.seed + iteration)
        sigma = torch.rand((), generator=gen)
        video_x0 = torch.randn(
            (dataset.video_frames, dataset.video_height, dataset.video_width, VIDEO_CHANNELS),
            generator=gen,
        )
        audio_x0 = torch.randn((dataset.audio_len, AUDIO_CHANNELS), generator=gen)
        text_x0 = torch.randn((dataset.text_len, MAX_IN_CHANNELS), generator=gen)
        video_eps = torch.randn_like(video_x0, generator=gen)
        audio_eps = torch.randn_like(audio_x0, generator=gen)
        text_eps = torch.randn_like(text_x0, generator=gen)

        video_xt = (1 - sigma) * video_x0 + sigma * video_eps
        audio_xt = (1 - sigma) * audio_x0 + sigma * audio_eps
        text_xt = (1 - sigma) * text_x0 + sigma * text_eps
        video_velocity = video_eps - video_x0
        audio_velocity = audio_eps - audio_x0

        n_video = dataset.video_frames * dataset.video_height * dataset.video_width
        n_audio = dataset.audio_len
        n_text = dataset.text_len
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
        audio_ref_t = (dataset.audio_len - 1) // AUDIO_TIME_COMPRESSION + 1
        coords_mapping = torch.cat(
            [
                _grid_coords(
                    (dataset.video_frames, dataset.video_height, dataset.video_width),
                    (dataset.video_frames, dataset.video_height, dataset.video_width),
                ),
                _grid_coords((dataset.audio_len, 1, 1), (audio_ref_t, 1, 1)),
                _grid_coords((dataset.text_len, 1, 1), (1, 1, 1), offset=(-dataset.text_len, 0, 0)),
            ],
            dim=0,
        )
        per_token_sigma = torch.cat([sigma.expand(n_video + n_audio), torch.zeros(n_text)])
        time_embedding = sinusoidal_embedding_1d(TIME_CHANNEL_DIM, per_token_sigma)
        labels = torch.zeros(total, LABEL_CHANNELS)
        labels[:n_video, :VIDEO_CHANNELS] = video_velocity.reshape(n_video, VIDEO_CHANNELS)
        labels[n_video : n_video + n_audio, :AUDIO_CHANNELS] = audio_velocity
        return {
            "input": input_tokens,
            "coords_mapping": coords_mapping,
            "modality_mapping": modality_mapping,
            "time_embedding": time_embedding,
            "cu_seqlens": torch.tensor([0, total], dtype=torch.int32),
        }, labels

    def test_synthetic_stream_bitwise_unchanged(self):
        from torchtitan_npu.models.magi2_preview.dataset import Magi2SyntheticDataset

        for seed in (0, 7):
            dataset = Magi2SyntheticDataset(seed=seed)
            reference = Magi2SyntheticDataset(seed=seed)
            for iteration in range(4):
                expected_dict, expected_labels = self._reference_sample(reference, iteration)
                actual_dict, actual_labels = next(iter(dataset))
                for key in INPUT_DICT_KEYS:
                    assert torch.equal(actual_dict[key], expected_dict[key])
                assert torch.equal(actual_labels, expected_labels)


# ---------------------------------------------------------------------------
# Preprocessing script dry run
# ---------------------------------------------------------------------------


def _script_path() -> Path:
    from torchtitan_npu.models import magi2_preview

    candidates = []
    # Resolve through any harness symlink: magi2_preview/__init__.py sits three
    # levels below the repo root (models/ -> torchtitan_npu/ -> repo root).
    pkg_root = Path(magi2_preview.__file__).resolve()
    if len(pkg_root.parents) > 3:
        candidates.append(pkg_root.parents[3] / "scripts" / "magi2_preprocess_latents.py")
    # Repo layout from this test file: tests/unit_tests/models/<file>.
    test_root = Path(__file__).resolve()
    if len(test_root.parents) > 3:
        candidates.append(test_root.parents[3] / "scripts" / "magi2_preprocess_latents.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("magi2_preprocess_latents.py not found relative to the package or test file")


class TestPreprocessDryRun:
    def test_dry_run_output_is_loadable(self, tmp_path):
        from torchtitan_npu.models.magi2_preview.latent_dataset import Magi2LatentDataset

        spec = importlib.util.spec_from_file_location(
            "magi2_preprocess_latents", _script_path()
        )
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        output_dir = tmp_path / "dryrun_shards"
        exit_code = script.main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--num-dry-run-samples",
                "6",
                "--samples-per-shard",
                "2",
                "--seed",
                "3",
            ]
        )
        assert exit_code == 0

        shard_files = sorted(output_dir.glob("shard_*.safetensors"))
        assert len(shard_files) == 3
        index = json.loads((output_dir / "index.json").read_text())
        assert len(index["shards"]) == 3
        assert sum(len(shard["samples"]) for shard in index["shards"]) == 6

        dataset = Magi2LatentDataset(str(output_dir), max_tokens_per_pack=4096)
        input_dict, labels = next(iter(dataset))
        assert set(input_dict) == INPUT_DICT_KEYS
        assert len(_segments(input_dict)) >= 2
        assert labels.shape == (input_dict["input"].shape[0], 64)
        # The dry-run cycles at least two shapes, so two epochs of packs must
        # cover more than one bucket.
        shapes = set()
        iterator = iter(dataset)
        for _ in range(4):
            pack, _ = next(iterator)
            for start, _ in _segments(pack):
                shapes.add(_video_grid_shape(pack, start))
        assert len(shapes) >= 2
