# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core CPU contracts for DeepSeek V4.1 vision-language SFT data."""

import json
from copy import deepcopy
from itertools import islice
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu_recipes.models.deepseek_v4_1 import (
    deepseek_v4_1_debugmodel_multimodal_sft,
    deepseek_v4_1_vision_sft,
)
from torchtitan_npu.models.deepseek_v4_1.vision import language_dataset as dataset_module
from torchtitan_npu.models.deepseek_v4_1.vision.data import ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import (
    DeepSeekV41VisionLanguageProcessor,
    _GrainSFTProcessor,
    _SFTCollator,
    sft_dataloader_config,
)

pytestmark = pytest.mark.cpu
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _unpack(microbatch):
    inputs = microbatch.as_input_dict()
    labels = inputs.pop("labels")
    inputs["num_valid_tokens"] = microbatch.num_valid_tokens
    return inputs, labels


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer.Config().build(tokenizer_path=str(ASSETS / "deepseek_v3"))


class _StubVisionLanguageEncoder:
    """Encode plain text/image blocks only; exclude the official chat template."""

    def encode_messages_with_assistant_mask(self, messages, tokenizer, **_kwargs):
        token_ids = []
        assistant_mask = []
        image_positions = []
        images = []

        def add_text(text, supervised=False):
            ids = tokenizer.tokenizer.encode(text).ids
            token_ids.extend(ids)
            assistant_mask.extend([supervised] * len(ids))

        for message in messages:
            supervised = message["role"] == "assistant"
            for block in message["content"]:
                if block["type"] == "text":
                    add_text(block["text"], supervised)
                elif block["type"] in ("image", "image_url"):
                    image_positions.append(len(token_ids))
                    images.append(block)
            if supervised and not message.get("wo_eos"):
                token_ids.append(tokenizer.eos_id)
                assistant_mask.append(True)
        return "", token_ids, assistant_mask, image_positions, images


@pytest.fixture(autouse=True)
def stub_official_encoder(monkeypatch):
    monkeypatch.setattr(
        dataset_module,
        "DeepSeekV41VisionLanguageEncoder",
        lambda *_args, **_kwargs: _StubVisionLanguageEncoder(),
    )


def _processor(dataset_path, tokenizer):
    return DeepSeekV41VisionLanguageProcessor(
        dataset_path=str(dataset_path),
        image_root="",
        tokenizer=tokenizer,
        vision_language_encoder=_StubVisionLanguageEncoder(),
        seq_len=512,
    )


def _loader(
    dataset_path,
    tokenizer,
    *,
    rank=0,
    world=1,
    infinite=False,
    batch_size=1,
):
    config = sft_dataloader_config(
        dataset_path=str(dataset_path),
        repeat=infinite,
    )
    return config.build(
        tokenizer=tokenizer,
        dp_rank=rank,
        dp_world_size=world,
        max_context_length=512,
        num_tokens_per_microbatch=512 * batch_size,
    )


def test_supported_chat_schemas_normalize_to_official_roles():
    llava = DeepSeekV41VisionLanguageProcessor._message_parts(
        {
            "image": "train2014/example.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "A red bus."},
            ],
        }
    )
    assert [message["role"] for message in llava] == ["user", "assistant"]
    assert llava[0]["content"][0] == {"type": "image", "source": "train2014/example.jpg"}

    messages = DeepSeekV41VisionLanguageProcessor._message_parts(
        {
            "messages": [
                {"role": "developer", "content": "Follow the image."},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "last_reminder", "content": "Be concise."},
            ]
        }
    )
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "latest_reminder"]


def test_target_grid_matches_the_rounding_sensitive_reference_case():
    assert ImagePatchProcessor().target_grid(100, 101) == (39, 39)


def test_vqa_labels_only_assistant_text_and_eos(tmp_path, tokenizer):
    Image.new("RGB", (101, 100), color=(255, 0, 0)).save(tmp_path / "sample.jpg")
    row = {
        "image": "sample.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat color is the image?"},
            {"from": "gpt", "value": "It is red."},
        ],
    }
    dataset_path = tmp_path / "vqa.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    inputs, labels = _processor(dataset_path, tokenizer)._sample(row)

    # Labels are shifted targets, so the reply's EOS is supervised at the reply's
    # last token and the row tail (also filled with EOS) is unsupervised.
    last_supervised = int((labels != -100).nonzero().flatten().max())
    assert labels[last_supervised] == tokenizer.eos_id
    assert inputs["input"][last_supervised + 1] == tokenizer.eos_id
    assert (labels[last_supervised + 1 :] == -100).all()
    assert (labels[:-1][inputs["token_types"][1:] >= 0] == -100).all()
    supervised = labels[:-1] != -100
    assert torch.equal(labels[:-1][supervised], inputs["input"][1:][supervised])


def test_streaming_dp_shards_and_resume_round_trip(tmp_path, tokenizer):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ]
        }
        for index in range(6)
    ]
    dataset_path = tmp_path / "text_vqa.jsonl"
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    shards = []
    for rank in range(2):
        loader = _loader(dataset_path, tokenizer, rank=rank, world=2, infinite=True)
        try:
            samples = list(islice(loader, 3))
        finally:
            loader.close()
        shards.append({tuple(labels[labels != -100].tolist()) for _, labels in map(_unpack, samples)})
    assert shards[0] and shards[1] and shards[0].isdisjoint(shards[1])

    original = _loader(dataset_path, tokenizer, infinite=True)
    restored = _loader(dataset_path, tokenizer, infinite=True)
    try:
        iterator = iter(original)
        next(iterator)
        state = deepcopy(original.state_dict())
        expected = _unpack(next(iterator))
        restored.load_state_dict(state)
        actual = _unpack(next(iter(restored)))
        assert torch.equal(actual[1], expected[1])
        for key in expected[0]:
            # ``num_valid_tokens`` rides the batch as the int the trainer pops.
            if isinstance(expected[0][key], torch.Tensor):
                assert torch.equal(actual[0][key], expected[0][key]), key
            else:
                assert actual[0][key] == expected[0][key], key
    finally:
        original.close()
        restored.close()


def test_single_context_budget_and_shifted_batch_contract(tmp_path, tokenizer):
    dataset_path = tmp_path / "text_vqa.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_tokens_per_microbatch == max_context_length"):
        _loader(dataset_path, tokenizer, batch_size=2)

    recipe = deepseek_v4_1_vision_sft(dataset_path=str(dataset_path), flavor="deepseek_v4_1_debugmodel")
    assert isinstance(recipe.dataloader, GrainDataLoader.Config)
    assert isinstance(recipe.dataloader.collator, _SFTCollator.Config)
    assert isinstance(recipe.dataloader.dataset.processor, _GrainSFTProcessor.Config)
    # One chat context per batch: the SFT graph has no packing stage.
    assert not hasattr(recipe.dataloader.dataset, "num_packing_bins")
    loader = recipe.dataloader.build(
        tokenizer=tokenizer, dp_rank=0, dp_world_size=1, max_context_length=512, num_tokens_per_microbatch=512
    )
    try:
        inputs, labels = _unpack(next(iter(loader)))
    finally:
        loader.close()
    raw = json.loads(dataset_path.read_text())
    expected_inputs, expected_labels = _processor(dataset_path, tokenizer)._sample(raw)
    assert inputs["input"].shape == labels.shape == (512,)
    assert type(inputs["num_valid_tokens"]) is int
    assert inputs["num_valid_tokens"] == (expected_labels != -100).sum()
    torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
    for key, value in expected_inputs.items():
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)


def test_registry_entry_reads_the_sft_environment(tmp_path, monkeypatch):
    dataset_path = tmp_path / "registry.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSV41_SFT_DATASET", str(dataset_path))
    monkeypatch.setenv("DSV41_SFT_THINKING_MODE", "thinking")
    recipe = deepseek_v4_1_debugmodel_multimodal_sft()
    assert isinstance(recipe.dataloader, GrainDataLoader.Config)
    assert isinstance(recipe.dataloader.collator, _SFTCollator.Config)
    processor = recipe.dataloader.dataset.processor
    assert processor.dataset_path == str(dataset_path.resolve())
    assert processor.thinking_mode == "thinking"
    assert not hasattr(recipe.dataloader.dataset, "num_packing_bins")


def test_sft_recipe_rejects_text_only_flavors(tmp_path):
    with pytest.raises(ValueError, match="multimodal flavor"):
        deepseek_v4_1_vision_sft(dataset_path=str(tmp_path / "unused.jsonl"), flavor="deepseek_v4_1_debugmodel_text")


def test_sft_recipe_requires_a_dataset(monkeypatch):
    monkeypatch.delenv("DSV41_SFT_DATASET", raising=False)
    monkeypatch.delenv("DATASET_PATH", raising=False)
    with pytest.raises(ValueError, match="DSV41_SFT_DATASET"):
        deepseek_v4_1_vision_sft()


def test_config_name_installs_the_sft_loader_with_cli_policy(tmp_path, monkeypatch):
    """The recipe, not an override, installs the loader; tyro still owns the policy."""
    from torchtitan.config import ConfigManager

    dataset_path = tmp_path / "cli.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSV41_SFT_DATASET", str(dataset_path))
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu_recipes.models.deepseek_v4_1",
            "--config",
            "deepseek_v4_1_debugmodel_multimodal_sft",
            "--dataloader.seed",
            "137",
        ]
    )
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.collator, _SFTCollator.Config)
    assert isinstance(config.dataloader.dataset.processor, _GrainSFTProcessor.Config)
    # The old override had to carry the policy fields over; the recipe is parsed by
    # tyro like any other config, so a CLI value lands on the freshly built loader.
    assert config.dataloader.seed == 137
    assert config.dataloader.dataset.processor.dataset_path == str(dataset_path.resolve())


@pytest.mark.parametrize("num_images", (0, 2), ids=("text-dummy", "interleaved-images"))
def test_grain_sft_preserves_processed_multimodal_fixture(tmp_path, tokenizer, num_images):
    for index, size in enumerate(((101, 100), (140, 70))):
        Image.new("RGB", size, color=(index * 127, 37, 255)).save(tmp_path / f"image{index}.png")
    blocks = []
    for index in range(num_images):
        blocks.extend(
            [
                {"type": "text", "text": f"Look at image {index}:"},
                {"type": "image", "source": f"image{index}.png"},
            ]
        )
    blocks.append({"type": "text", "text": "Describe the colors."})
    row = {"messages": [{"role": "user", "content": blocks}, {"role": "assistant", "content": "Blue and red."}]}
    dataset_path = tmp_path / "multimodal.jsonl"
    # Overlong rows are filtered before collation; they must not affect labels.
    overlong = {"messages": [{"role": "user", "content": "word " * 700}, {"role": "assistant", "content": "Skipped."}]}
    dataset_path.write_text(json.dumps(overlong) + "\n" + json.dumps(row) + "\n")
    expected_inputs, expected_labels = _processor(dataset_path, tokenizer)._sample(row)
    loader = _loader(dataset_path, tokenizer)
    try:
        inputs, labels = _unpack(next(iter(loader)))
    finally:
        loader.close()
    torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
    for key, value in expected_inputs.items():
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)
    expected_reply = [*tokenizer.tokenizer.encode("Blue and red.").ids, tokenizer.eos_id]
    assert labels[labels != -100].tolist() == expected_reply
    assert inputs["num_valid_tokens"] == len(expected_reply)
    assert inputs["pixel_values"].shape[0] == max(num_images, 1)
    if not num_images:
        assert (inputs["image_feature_indices"] == -1).all()
        assert torch.count_nonzero(inputs["pixel_values"]) == 0
