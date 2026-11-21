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
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torchtitan.components.data import DatasetBuildContext
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4ChatEncoder, DSV4EncoderConfig
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


@pytest.mark.parametrize("add_bos", (False, True))
@pytest.mark.parametrize("wo_eos", (False, True))
def test_chat_processor_supervises_tokens_across_assistant_fragments(tmp_path, add_bos, wo_eos):
    backend = Tokenizer(BPE(vocab={"?": 0, "a": 1, "b": 2, "ab": 3, "^": 4, "!": 5}, merges=[("a", "b")]))
    backend.add_special_tokens(["<s>", "</s>"])
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>"}))
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "bos_token = '<s>'\n"
        "def merge_tool_messages(messages):\n"
        "    return messages\n"
        "def sort_tool_results_by_call_order(messages):\n"
        "    return messages\n"
        "def render_message(index, messages, *, reasoning_effort, **kwargs):\n"
        "    message = messages[index]\n"
        "    prefix = ''\n"
        "    if message['role'] == 'user':\n"
        "        prefix += '^' if index and messages[index - 1]['role'] == 'system' else ''\n"
        "        prefix += '!' if reasoning_effort == 'high' else ''\n"
        "    eos = '</s>' if message['role'] == 'assistant' and not message.get('wo_eos') else ''\n"
        "    return prefix + message['content'] + eos\n"
        "def encode_messages(messages, *, context=None, add_default_bos_token, **kwargs):\n"
        "    history = context or []\n"
        "    conversation = history + messages\n"
        "    prefix = bos_token if add_default_bos_token and not history else ''\n"
        "    return prefix + ''.join(render_message(i, conversation, **kwargs)\n"
        "                            for i in range(len(history), len(conversation)))\n"
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    context = DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=8,
        num_tokens_per_microbatch=8,
        read_options=grain.ReadOptions(num_threads=0),
    )
    config = DSV4EncoderConfig(thinking_mode="chat", add_default_bos_token=add_bos)
    processor = DeepSeekV4ChatProcessor.Config(chat_encoder=config).build(context=context)
    messages = [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content": "a", "wo_eos": True},
        {"role": "assistant", "content": "b", "wo_eos": wo_eos},
    ]

    sequence = processor({"messages": messages}, np.random.default_rng(0))

    assert sequence is not None
    expected_prompt = ("<s>" if add_bos else "") + "?ab" + ("" if wo_eos else "</s>")
    assert sequence.input_ids.tolist() == backend.encode(expected_prompt, add_special_tokens=False).ids[:-1]
    assert sequence.labels.tolist() == ([-100] if add_bos else []) + [3] + ([] if wo_eos else [tokenizer.eos_id])

    encoder = DSV4ChatEncoder(
        str(encoding / "encoding_dsv4.py"),
        context=[{"role": "system", "content": "history"}],
        add_default_bos_token=add_bos,
        reasoning_effort="high",
    )
    prompt, tokens, mask = encoder.encode_messages_with_assistant_mask(messages, tokenizer, tokenizer.eos_id)
    assert prompt == "^!?ab" + ("" if wo_eos else "</s>")
    assert tokens == backend.encode(prompt, add_special_tokens=False).ids
    assert mask == [False, False, False, True] + ([] if wo_eos else [True])


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
