# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conversation semantics independent of the external DSV4 encoding asset."""

import json
from copy import deepcopy

import grain
import numpy as np
import pytest
from torchtitan.components.data import DatasetBuildContext
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4EncoderConfig
from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor


class _StubEncoder:
    """Fixed token/mask oracle; records messages at the encoding boundary."""

    def __init__(self):
        self.tokens = [10, 11, 12, 13, 14, 15]
        self.mask = [False, False, True, True, True, True]
        self.messages = None

    def encode_messages_with_assistant_mask(self, messages, tokenizer, eos_id):
        self.messages = messages
        return "", self.tokens, self.mask


@pytest.fixture
def processor_and_encoder(monkeypatch):
    encoder = _StubEncoder()
    monkeypatch.setattr(DSV4EncoderConfig, "build", lambda self, path: encoder)
    context = DatasetBuildContext(
        tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/deepseek_v3"),
        max_context_length=8,
        num_tokens_per_microbatch=16,
        read_options=grain.ReadOptions(num_threads=0),
    )
    processor = DeepSeekV4ChatProcessor.Config(per_doc_alignment=8).build(context=context)
    return processor, encoder


def test_chat_processor_shifts_supervision_then_aligns(processor_and_encoder):
    processor, _ = processor_and_encoder
    sample = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}

    sequence = processor(sample, np.random.default_rng(0))

    assert sequence.input_ids.tolist() == [10, 11, 12, 13, 14, 0, 0, 0]
    assert sequence.labels.tolist() == [-100, 12, 13, 14, 15, -100, -100, -100]
    assert sequence.positions is None


def test_chat_processor_preserves_tools_and_normalizes_reminders_without_mutation(processor_and_encoder):
    processor, encoder = processor_and_encoder
    messages = [
        {"role": "developer", "content": "use tools"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "ok"},
        {"role": "last_reminder", "content": "remember"},
    ]
    sample = {"messages": json.dumps(messages), "tools": '[{"name":"search"}]'}
    original = deepcopy(sample)

    sequence = processor(sample, np.random.default_rng(0))

    assert sequence is not None
    assert sample == original
    assert encoder.messages[0] == {"role": "developer", "content": "use tools", "tools": [{"name": "search"}]}
    assert encoder.messages[-1] == {"role": "latest_reminder", "content": "remember"}
    assert encoder.messages[1:5] == messages[1:5]


@pytest.mark.parametrize("case", ("oversized", "all-masked"), ids=("oversized", "all-masked"))
def test_chat_processor_drops_unusable_conversations(processor_and_encoder, case):
    processor, encoder = processor_and_encoder
    if case == "oversized":
        encoder.tokens = list(range(10))
        encoder.mask = [True] * 10
    else:
        encoder.mask = [False] * len(encoder.tokens)

    sequence = processor({"messages": [{"role": "assistant", "content": "ok"}]}, np.random.default_rng(0))

    assert sequence is None


def test_chat_processor_accepts_exact_context_boundary(processor_and_encoder):
    processor, encoder = processor_and_encoder
    encoder.tokens = list(range(9))
    encoder.mask = [False, False] + [True] * 7

    sequence = processor({"messages": [{"role": "assistant", "content": "ok"}]}, np.random.default_rng(0))

    assert sequence.input_ids.tolist() == list(range(8))
    assert sequence.labels.tolist() == [-100, 2, 3, 4, 5, 6, 7, 8]


def test_chat_processor_rejects_unknown_role(processor_and_encoder):
    processor, _ = processor_and_encoder
    with pytest.raises(ValueError, match="Unsupported DeepSeek-V4 role"):
        processor({"messages": [{"role": "unknown", "content": "hi"}]}, np.random.default_rng(0))
