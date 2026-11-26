# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core CPU contracts for DeepSeek V4.1 vision-language SFT data."""

import json
from copy import deepcopy
from itertools import islice
from pathlib import Path
from textwrap import dedent

import pytest
import torch
from datasets import Dataset
from PIL import Image
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4_1 import build_model_config
from torchtitan_npu.models.deepseek_v4_1.vision import language_dataset as dataset_module
from torchtitan_npu.models.deepseek_v4_1.vision.data import ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import (
    DeepSeekV41VisionLanguageProcessor,
    _GrainSFTProcessor,
    _SFTCollator,
    sft_dataloader_config,
)
from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import DeepSeekV41VisionLanguageEncoder
from torchtitan_npu.override.deepseek_v4_1.vision_language_dataloader import sft as sft_override
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import (
    _vision_sft_dataloader_config,
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_debugmodel_multimodal_sft,
)

pytestmark = pytest.mark.cpu
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _unpack(microbatch):
    inputs = microbatch.as_input_dict()
    labels = inputs.pop("labels")
    inputs["loss_token_count"] = int(microbatch.loss_token_counts)
    # Same contract as the caption loader: routing counts stay a one-entry vector.
    assert microbatch.routing_token_counts.shape == (1,)
    return inputs, labels


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer.Config().build(tokenizer_path=str(ASSETS / "deepseek_v3"))


@pytest.mark.parametrize("with_tool_result", (True, False))
def test_tool_call_is_supervised_when_a_shorter_prefix_renders_differently(tmp_path, tokenizer, with_tool_result):
    asset = tmp_path / "encoding.py"
    asset.write_text(
        "IMAGE_PLACEHOLDER = '<image>'\n"
        "def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,\n"
        "                    add_default_bos_token=True, reasoning_effort=None, return_multi_modal_data=False):\n"
        "    if not messages and context:\n"
        "        raise ValueError('This encoder cannot render an empty tool-call suffix')\n"
        "    prompt = ''.join(message['content'] for message in messages)\n"
        f"    if not context and len(messages) == {3 if with_tool_result else 2}:\n"
        "        prompt += ' Earlier thinking.'\n"
        "    return (prompt, {'images': []}) if return_multi_modal_data else prompt\n"
    )
    messages = [
        {"role": "user", "content": "Question!\n"},
        {"role": "assistant", "content": "Look up the rec", "wo_eos": True},
        {"role": "assistant", "content": "ord.", "tool_calls": [{"id": "lookup"}]},
    ]
    if with_tool_result:
        messages.append({"role": "tool", "content": "\nTool result."})
    encoder = DeepSeekV41VisionLanguageEncoder(str(asset), thinking_mode="thinking")
    prompt, tokens, keep, _, _ = encoder.encode_messages_with_assistant_mask(messages, tokenizer)
    assert prompt == "Question!\nLook up the record." + ("\nTool result." if with_tool_result else "")
    assert tokens == tokenizer.tokenizer.encode(prompt, add_special_tokens=False).ids
    assert [token for token, supervise in zip(tokens, keep, strict=True) if supervise] == (
        tokenizer.tokenizer.encode("Look up the record.").ids
    )


@pytest.mark.parametrize("has_merge_helper", (True, False), ids=("ordered-images", "missing-helper"))
def test_tool_result_images_follow_call_order(tmp_path, tokenizer, has_merge_helper):
    asset = tmp_path / "encoding.py"
    source = dedent(
        """\
        IMAGE_PLACEHOLDER = "<image>"

        def merge_tool_messages(messages):
            return messages

        def sort_tool_results_by_call_order(messages):
            ordinary = [message for message in messages if "tool_call_id" not in message]
            order = [call["id"] for message in ordinary for call in message.get("tool_calls", [])]
            results = {message["tool_call_id"]: message for message in messages if "tool_call_id" in message}
            return ordinary + [results[key] for key in order if key in results]

        def process_image_messages(messages):
            return messages, [message["image"] for message in messages if "image" in message]

        def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                            add_default_bos_token=True, reasoning_effort=None, return_multi_modal_data=False):
            ordered = sort_tool_results_by_call_order(messages)
            prompt = "".join(message["content"] for message in ordered)
            _, images = process_image_messages(messages)
            return (prompt, {"images": images}) if return_multi_modal_data else prompt
        """
    )
    asset.write_text(source if has_merge_helper else source + "\ndel merge_tool_messages\n")
    messages = [
        {"role": "user", "content": "Compare two images.\n"},
        {"role": "assistant", "content": "Look them up.\n", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "content": "B <image>\n", "tool_call_id": "b", "image": "b.png"},
        {"role": "tool", "content": "A <image>\n", "tool_call_id": "a", "image": "a.png"},
    ]
    encoder = DeepSeekV41VisionLanguageEncoder(str(asset))
    if has_merge_helper:
        prompt, _, _, positions, images = encoder.encode_messages_with_assistant_mask(messages, tokenizer)
        assert prompt.endswith("A <image>\nB <image>\n")
        assert images == ["a.png", "b.png"]
        assert len(positions) == 2
    else:
        with pytest.raises(RuntimeError, match="missing tool image helpers: merge_tool_messages"):
            encoder.encode_messages_with_assistant_mask(messages, tokenizer)


def test_empty_historical_assistant_span(tmp_path, tokenizer):
    asset = tmp_path / "encoding.py"
    asset.write_text(
        "IMAGE_PLACEHOLDER = '<image>'\n"
        "def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,\n"
        "                    add_default_bos_token=True, reasoning_effort=None, return_multi_modal_data=False):\n"
        "    prompt = ''.join(message['content'] for message in messages)\n"
        "    if len(messages) == 2:\n"
        "        prompt += ' Earlier thinking.'\n"
        "    return (prompt, {'images': []}) if return_multi_modal_data else prompt\n"
    )
    messages = [
        {"role": "user", "content": "Old question.\n"},
        {"role": "assistant", "content": "", "wo_eos": True},
        {"role": "user", "content": "New question.\n"},
        {"role": "assistant", "content": "Answer."},
    ]
    encoder = DeepSeekV41VisionLanguageEncoder(str(asset), thinking_mode="thinking")
    _, tokens, keep, _, _ = encoder.encode_messages_with_assistant_mask(messages, tokenizer)
    assert [token for token, supervise in zip(tokens, keep, strict=True) if supervise] == (
        tokenizer.tokenizer.encode("Answer.").ids
    )


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
    first_unsupervised = last_supervised + 1
    assert labels[last_supervised] == tokenizer.eos_id
    assert inputs["input"][first_unsupervised] == tokenizer.eos_id
    assert (labels[first_unsupervised:] == -100).all()
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
            # The batch carries the count of supervised labels independently of padding.
            if isinstance(expected[0][key], torch.Tensor):
                assert torch.equal(actual[0][key], expected[0][key]), key
            else:
                assert actual[0][key] == expected[0][key], key
    finally:
        original.close()
        restored.close()


@pytest.mark.parametrize("has_valid", (True, False), ids=("valid-after-1000", "all-invalid"))
def test_parquet_stream_checks_all_rows_before_declaring_a_shard_empty(tmp_path, tokenizer, has_valid):
    data = tmp_path / "questions_[train].parquet"
    rows = [{"instruction": "word " * 700, "output": "Too long"}] * 1_000
    if has_valid:
        rows.append({"instruction": "Question", "output": "Answer"})
    Dataset.from_list(rows).to_parquet(str(data))

    if not has_valid:
        with pytest.raises(ValueError, match="SFT data shard 0/1 has no valid samples"):
            _loader(data, tokenizer, infinite=True)
        return

    loader = _loader(data, tokenizer, infinite=True)
    try:
        assert next(iter(loader)).loss_token_counts.item() > 0
    finally:
        loader.close()


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

    recipe = deepseek_v4_1_debugmodel_multimodal_sft(dataset_path=str(dataset_path))
    assert isinstance(recipe.dataloader, GrainDataLoader.Config)
    assert isinstance(recipe.dataloader.collator, _SFTCollator.Config)
    assert isinstance(recipe.dataloader.dataset.processor, _GrainSFTProcessor.Config)
    # One chat context per batch: the SFT graph has no packing stage.
    assert not hasattr(recipe.dataloader.dataset, "num_packing_bins")
    loader = recipe.dataloader.build(
        tokenizer=tokenizer, dp_rank=0, dp_world_size=1, max_context_length=512, num_tokens_per_microbatch=512
    )
    try:
        batch = next(iter(loader))
        inputs, labels = _unpack(batch)
    finally:
        loader.close()
    raw = json.loads(dataset_path.read_text())
    expected_inputs, expected_labels = _processor(dataset_path, tokenizer)._sample(raw)
    assert batch.routing_token_counts.shape == (1,)
    assert batch.loss_token_counts.ndim == 0
    torch.testing.assert_close(batch.loss_token_counts, (batch.labels != -100).sum())
    torch.testing.assert_close(batch.routing_token_counts[0], (~batch.padding_mask).sum())
    assert inputs["input"].shape == labels.shape == (512,)
    assert type(inputs["loss_token_count"]) is int
    assert inputs["loss_token_count"] == (expected_labels != -100).sum()
    torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
    for key, value in expected_inputs.items():
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)


def test_registry_entry_uses_explicit_sft_options(tmp_path):
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
    recipe = deepseek_v4_1_debugmodel_multimodal_sft(
        dataset_path=str(dataset_path), thinking_mode="thinking"
    )
    assert isinstance(recipe.dataloader, GrainDataLoader.Config)
    assert isinstance(recipe.dataloader.collator, _SFTCollator.Config)
    processor = recipe.dataloader.dataset.processor
    assert processor.dataset_path == str(dataset_path.resolve())
    assert processor.thinking_mode == "thinking"
    assert not hasattr(recipe.dataloader.dataset, "num_packing_bins")


def test_sft_recipe_rejects_text_only_flavors():
    model = build_model_config("deepseek_v4_1_debugmodel_text", seq_len=512)
    with pytest.raises(ValueError, match="multimodal flavor"):
        _vision_sft_dataloader_config(model)


def test_sft_recipe_uses_the_python_default_dataset():
    recipe = deepseek_v4_1_debugmodel_multimodal_sft()
    assert recipe.dataloader.dataset.processor.dataset_path == "/path/to/train.jsonl"


@pytest.mark.parametrize("num_images", (0, 2), ids=("text-dummy", "interleaved-images"))
@pytest.mark.parametrize("suffix", (".jsonl", ".json"))
def test_grain_sft_preserves_processed_multimodal_fixture(tmp_path, tokenizer, num_images, suffix):
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
    row = {
        "messages": [
            {"role": "user", "content": blocks},
            {"role": "assistant", "content_blocks": [{"type": "text", "text": "Blue and red."}]},
        ]
    }
    dataset_path = tmp_path / f"multimodal{suffix}"
    # Overlong rows are filtered before collation; they must not affect labels.
    overlong = {"messages": [{"role": "user", "content": "word " * 700}, {"role": "assistant", "content": "Skipped."}]}
    rows = [overlong, row]
    dataset_path.write_text(
        json.dumps(rows) if suffix == ".json" else "".join(json.dumps(sample) + "\n" for sample in rows)
    )
    expected_inputs, expected_labels = _processor(dataset_path, tokenizer)._sample(row)
    loader_config = sft_override(deepseek_v4_1_debugmodel_multimodal().dataloader, dataset_path=str(dataset_path))
    loader_config.repeat = False
    loader = loader_config.build(
        tokenizer=tokenizer, dp_rank=0, dp_world_size=1, max_context_length=512, num_tokens_per_microbatch=512
    )
    try:
        inputs, labels = _unpack(next(iter(loader)))
    finally:
        loader.close()
    torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
    for key, value in expected_inputs.items():
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)
    expected_reply = [*tokenizer.tokenizer.encode("Blue and red.").ids, tokenizer.eos_id]
    assert labels[labels != -100].tolist() == expected_reply
    assert inputs["loss_token_count"] == len(expected_reply)
    assert inputs["pixel_values"].shape[0] == max(num_images, 1)
    if not num_images:
        assert (inputs["image_feature_indices"] == -1).all()
        assert torch.count_nonzero(inputs["pixel_values"]) == 0


def test_repeating_sft_rejects_a_dp_shard_with_no_valid_samples(tmp_path, tokenizer):
    dataset_path = tmp_path / "shards.jsonl"
    rows = [
        {"instruction": "Question", "output": "Answer"},
        {"instruction": "word " * 700, "output": "Too long"},
    ]
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = sft_override(deepseek_v4_1_debugmodel_multimodal().dataloader, dataset_path=str(dataset_path))
    config.shuffle = False
    loader = config.build(
        tokenizer=tokenizer, dp_rank=1, dp_world_size=2, max_context_length=512, num_tokens_per_microbatch=512
    )
    try:
        with pytest.raises(ValueError, match="SFT data shard 1/2 has no valid samples"):
            next(iter(loader))
    finally:
        loader.close()
