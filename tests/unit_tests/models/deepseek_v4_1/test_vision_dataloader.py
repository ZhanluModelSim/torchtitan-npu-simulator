# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The image-conditioned caption loader: protocol, alignment, packing and resume.

The text-pipeline loader contracts live in ``test_text_dataloader``; what is here is what
only the vision path has -- the image protocol, the per-document alignment of an image
document, and the packing that carries the visual items along.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.parallel_dims import ParallelDims

from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_debugmodel_multimodal
from torchtitan_npu.models.deepseek_v4_1.vision.data import IMAGE_END, IMAGE_START, ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import _process_mm_sample, caption_dataloader_config

ASSETS = Path(__file__).resolve().parents[3] / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=str(ASSETS / "deepseek_v3"))


def _loader(tokenizer, *, rank=0, world=1, packing=0, infinite=False, alignment=2):
    return caption_dataloader_config(
        dataset="cc12m-test",
        dataset_path=str(ASSETS / "cc12m_test"),
        num_packing_bins=packing,
        repeat=infinite,
        per_doc_alignment=alignment,
    ).build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=world, max_context_length=512, num_tokens_per_batch=512)


def _captions(loader):
    try:
        return [tuple(labels[labels != -100].tolist()) for _, labels in loader]
    finally:
        loader.close()


def test_packed_images_supervision_and_positions(tokenizer):
    unpacked = _captions(_loader(tokenizer))
    loader = _loader(tokenizer, packing=4)
    try:
        iterator = iter(loader)
        first = next(iterator)
        packed = [first, *iterator]
    finally:
        loader.close()
    assert len(packed) < len(unpacked)
    actual_captions = []
    for inputs, labels in packed:
        tokens, positions, types, indices = [
            inputs[k] for k in ("input", "positions", "token_types", "image_feature_indices")
        ]
        assert tokens.ndim == labels.ndim == positions.ndim == 1
        assert type(inputs["num_valid_tokens"]) is int
        assert inputs["num_valid_tokens"] == (labels != -100).sum()
        starts = (tokens == tokenizer.bos_id).nonzero().flatten().tolist()
        ends = (tokens == tokenizer.eos_id).nonzero().flatten().tolist()
        assert len(starts) == len(ends) == inputs["pixel_values"].shape[0]
        assert (types == IMAGE_START).sum() == (types == IMAGE_END).sum() == len(starts)
        assert torch.equal(indices[indices >= 0], torch.arange((indices >= 0).sum()))
        for start, end in zip(starts, ends, strict=True):
            assert torch.equal(positions[start : end + 1], torch.arange(end - start + 1))
            actual_captions.append(tuple(labels[start:end][labels[start:end] != -100].tolist()))
            assert labels[end] == -100  # no cross-document BOS prediction
        assert (labels[:-1][types[1:] >= 0] == -100).all()
        mask = labels[:-1] != -100
        assert torch.equal(labels[:-1][mask], tokens[1:][mask])
        assert (labels[int(ends[-1]) :] == -100).all()
        # All document and padding boundaries must preserve compressor groups.
        boundaries = (positions == 0).nonzero().flatten()
        assert (boundaries % 2 == 0).all()
        tail_start = ((ends[-1] + 2) // 2) * 2
        assert torch.equal(positions[tail_start:], torch.arange(positions.numel() - tail_start))
    assert sorted(actual_captions) == sorted(unpacked)


@pytest.mark.parametrize("consumed", [1, 33])
def test_packed_loader_resume_pending_samples(tokenizer, consumed):
    original = _loader(tokenizer, packing=8, infinite=True)
    restored = _loader(tokenizer, packing=8, infinite=True)
    try:
        iterator = iter(original)
        for _ in range(consumed):
            next(iterator)
        state = deepcopy(original.state_dict())
        restored.load_state_dict(state)
        resumed = iter(restored)
        # Restore buffered packs both in the first epoch and after all 32 source samples.
        for _ in range(5):
            expected, expected_labels = next(iterator)
            actual, actual_labels = next(resumed)
            assert torch.equal(actual_labels, expected_labels)
            assert actual.keys() == expected.keys()
            for key in expected:
                # ``num_valid_tokens`` rides the batch as the int the trainer pops.
                if isinstance(expected[key], torch.Tensor):
                    assert torch.equal(actual[key], expected[key]), key
                else:
                    assert actual[key] == expected[key], key
    finally:
        original.close()
        restored.close()


def test_target_grid_fixed_oracle():
    processor = ImagePatchProcessor()
    expected = {
        (100, 100): (39, 39),
        (60, 1600): (201, 8),
        (1600, 60): (8, 201),
        (701, 1024): (74, 51),
        (555, 416): (34, 45),
    }
    for (width, height), grid in expected.items():
        assert processor.target_grid(height, width) == grid


def test_mm_sample_is_padded_to_the_alignment(tokenizer):
    """The multimodal sample parser rounds each document up to the alignment."""
    import tarfile

    with tarfile.open(ASSETS / "cc12m_test" / "cc12m-train-0000.tar") as tar:
        members = {name: tar.extractfile(name).read() for name in tar.getnames()}
    stems = sorted(name[:-4] for name in members if name.endswith(".txt"))
    # A sample whose token stream is odd, so the pad actually fires.
    for stem in stems:
        sample = {
            "jpg": members[f"{stem}.jpg"],
            "txt": members[f"{stem}.txt"].decode("utf-8", "replace").strip(),
        }
        unaligned = _process_mm_sample(dict(sample), tokenizer, per_doc_alignment=1)
        if int(unaligned["input_ids"].numel()) % 2:
            break
    else:
        pytest.skip("no odd-length document in the test tar")

    aligned = _process_mm_sample(dict(sample), tokenizer, per_doc_alignment=2)
    # One pad, appended; the real ids are a prefix and the pad is unsupervised.
    assert int(aligned["input_ids"].numel()) == int(unaligned["input_ids"].numel()) + 1
    assert aligned["input_ids"][: unaligned["input_ids"].numel()].tolist() == unaligned["input_ids"].tolist()
    assert (aligned["labels"][unaligned["labels"].numel() :] == -100).all()
    # Positions keep running through the pad, so it stays inside its document.
    assert torch.equal(aligned["positions"], torch.arange(aligned["input_ids"].numel()))


def test_mm_loader_rejects_an_alignment_that_does_not_divide_context_length(tokenizer):
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=3)
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=0)


def test_recipe_passes_the_model_alignment_to_the_loader():
    """The recipes derive the loader's per-document alignment from the model spec
    (the LCM of its compression ratios), so a packed document never straddles a
    pooled group."""
    from torchtitan_npu.models.deepseek_v4_1 import config_registry as cr

    trainer = cr.deepseek_v4_1_debugmodel_multimodal()
    assert isinstance(trainer.dataloader, GrainDataLoader.Config)
    assert trainer.dataloader.dataset.processor.per_doc_alignment == cr._per_doc_alignment(trainer.model_spec) == 2


def test_empty_caption_is_filtered_before_packing(tokenizer):
    assert _process_mm_sample({"jpg": Image.new("RGB", (20, 40)), "txt": ""}, tokenizer) is None


def test_packed_vision_batch_reaches_folded_attention_metadata(tokenizer):
    recipe = deepseek_v4_1_debugmodel_multimodal()
    recipe.parallelism.spmd_backend = "partial_dtensor"
    with torch.device("meta"):
        model = recipe.model_spec.model.build()
    loader = _loader(tokenizer, packing=4)
    try:
        batch, labels = next(iter(loader))
    finally:
        loader.close()
    valid = batch.pop("num_valid_tokens")
    dimensions = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1)
    tokens, targets, kwargs = model.preprocess_inputs(
        {**batch, "labels": labels}, parallel_dims=dimensions, parallelism=recipe.parallelism
    )
    assert tokens.ndim == targets.ndim == 1
    assert valid == (targets != -100).sum()
    for key in ("positions", "token_types", "image_feature_indices", "pixel_values", "image_grid"):
        torch.testing.assert_close(kwargs[key], batch[key], rtol=0, atol=0)
    boundaries = torch.cat(((batch["positions"] == 0).nonzero().flatten(), torch.tensor([512]))).int()
    for layer in recipe.model_spec.model.layers:
        attention = layer.attention.inner_attention
        metadata = kwargs["attention_metadata"][attention.attention_metadata_key]
        torch.testing.assert_close(metadata.cu_seq_q, boundaries, rtol=0, atol=0)
        torch.testing.assert_close(metadata.cu_seq_k, boundaries // max(attention.compress_ratio, 1), rtol=0, atol=0)
