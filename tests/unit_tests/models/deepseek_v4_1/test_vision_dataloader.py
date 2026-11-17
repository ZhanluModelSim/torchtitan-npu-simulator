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
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4_1.vision.data import IMAGE_END, IMAGE_START, ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import DeepSeekV41DataLoader

ASSETS = Path(__file__).resolve().parents[3] / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=str(ASSETS / "deepseek_v3"))


def _loader(tokenizer, *, rank=0, world=1, packing=0, infinite=False, alignment=2):
    return DeepSeekV41DataLoader.Config(
        dataset="cc12m-test",
        dataset_path=str(ASSETS / "cc12m_test"),
        packing_buffer_size=packing,
        infinite=infinite,
        per_doc_alignment=alignment,
    ).build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=world, seq_len=512, local_batch_size=1)


def _captions(loader):
    return [tuple(labels[labels != -100].tolist()) for _, labels in loader]


def test_packed_images_supervision_and_positions(tokenizer):
    unpacked = _captions(_loader(tokenizer))
    loader = _loader(tokenizer, packing=4)
    iterator = iter(loader)
    first = next(iterator)
    assert loader.dataset._sample_idx == 4  # produce before EOF without an exact-length fit
    packed = [first, *iterator]
    assert len(packed) < len(unpacked)
    actual_captions = []
    for inputs, labels in packed:
        tokens, positions, types, indices = [
            inputs[k][0] for k in ("input", "positions", "token_types", "image_feature_indices")
        ]
        starts = (tokens == tokenizer.bos_id).nonzero().flatten().tolist()
        ends = (tokens == tokenizer.eos_id).nonzero().flatten().tolist()
        assert len(starts) == len(ends) == inputs["pixel_values"].shape[0]
        assert (types == IMAGE_START).sum() == (types == IMAGE_END).sum() == len(starts)
        assert torch.equal(indices[indices >= 0], torch.arange((indices >= 0).sum()))
        for start, end in zip(starts, ends, strict=True):
            assert torch.equal(positions[start : end + 1], torch.arange(end - start + 1))
            actual_captions.append(tuple(labels[0, start:end][labels[0, start:end] != -100].tolist()))
            assert labels[0, end] == -100  # no cross-document BOS prediction
        assert (labels[0, :-1][types[1:] >= 0] == -100).all()
        mask = labels[0, :-1] != -100
        assert torch.equal(labels[0, :-1][mask], tokens[1:][mask])
        assert (labels[0, int(ends[-1]) :] == -100).all()
    assert sorted(actual_captions) == sorted(unpacked)


@pytest.mark.parametrize("consumed", [1, 33])
def test_packed_loader_resume_pending_samples(tokenizer, consumed):
    original = _loader(tokenizer, packing=8, infinite=True)
    iterator = iter(original)
    for _ in range(consumed):
        next(iterator)
    state = deepcopy(original.state_dict())
    restored = _loader(tokenizer, packing=8, infinite=True)
    restored.load_state_dict(state)
    resumed = iter(restored)
    # Restore buffered packs both in the first epoch and after all 32 source samples.
    for _ in range(5):
        expected, expected_labels = next(iterator)
        actual, actual_labels = next(resumed)
        assert torch.equal(actual_labels, expected_labels)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key


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

    from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import _process_mm_sample

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


def test_mm_loader_rejects_an_alignment_that_does_not_divide_seq_len(tokenizer):
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
    assert trainer.dataloader.per_doc_alignment == cr._per_doc_alignment(trainer.model_spec) == 2
