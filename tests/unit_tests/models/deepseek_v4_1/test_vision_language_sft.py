# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core CPU contracts for DeepSeek V4.1 vision-language SFT data."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4_1.vision import language_dataset as dataset_module
from torchtitan_npu.models.deepseek_v4_1.vision.data import ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import (
    DeepSeekV41VisionLanguageDataLoader,
    DeepSeekV41VisionLanguageProcessor,
)

pytestmark = pytest.mark.cpu
ASSETS = Path(__file__).resolve().parents[3] / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=str(ASSETS / "deepseek_v3"))


class _StubVisionLanguageEncoder:
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
    num_workers=0,
):
    config = DeepSeekV41VisionLanguageDataLoader.Config(
        dataset_path=str(dataset_path),
        infinite=infinite,
        num_workers=num_workers,
    )
    return DeepSeekV41VisionLanguageDataLoader(
        config,
        tokenizer=tokenizer,
        dp_rank=rank,
        dp_world_size=world,
        seq_len=512,
        local_batch_size=batch_size,
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
        samples = list(_loader(dataset_path, tokenizer, rank=rank, world=2))
        shards.append({tuple(labels[labels != -100].tolist()) for _, labels in samples})
    assert shards[0] and shards[1] and shards[0].isdisjoint(shards[1])

    original = _loader(dataset_path, tokenizer, infinite=True)
    iterator = iter(original)
    next(iterator)
    state = deepcopy(original.state_dict())
    expected = next(iterator)
    restored = _loader(dataset_path, tokenizer, infinite=True)
    restored.load_state_dict(state)
    actual = next(iter(restored))
    assert torch.equal(actual[1], expected[1])
    for key in expected[0]:
        assert torch.equal(actual[0][key], expected[0][key]), key


def test_batch_contract_and_worker_forwarding(tmp_path, tokenizer, monkeypatch):
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
    with pytest.raises(ValueError, match="local_batch_size=1"):
        _loader(dataset_path, tokenizer, batch_size=2)

    captured = {}

    def capture_init(_self, _dataset, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ParallelAwareDataloader, "__init__", capture_init)
    _loader(dataset_path, tokenizer, num_workers=2)
    assert captured["batch_size"] == 1
    assert captured["num_workers"] == 2
