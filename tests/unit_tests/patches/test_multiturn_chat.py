# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the unified chat_dataset multi-turn patch."""

from unittest.mock import MagicMock

import pytest
import torch

from torchtitan.components.loss import IGNORE_INDEX
import torchtitan_npu.patches.torchtitan.chat_dataset as chat_dataset_patch
from torchtitan_npu.patches.torchtitan.chat_dataset import (
    _patched_loader_init,
    _patched_tokenize_sample,
    _patched_validate_messages,
)


# ---------------------------------------------------------------------------
# _patched_loader_init
# ---------------------------------------------------------------------------


def test_loader_init_resolves_chat_processor_import_path_before_upstream_init(monkeypatch):
    from torchtitan_npu.config.configs import ChatDataLoaderConfig
    from torchtitan_npu.hf_datasets.chat_processors import process_gsm8k_sample

    captured = {}

    def fake_orig_loader_init(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        seq_len,
        local_batch_size,
        **kwargs,
    ):
        del dp_world_size, dp_rank, tokenizer, seq_len, local_batch_size, kwargs
        captured["sample_processor"] = config.sample_processor
        self.dataset = MagicMock()

    config = ChatDataLoaderConfig(chat_processor="torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample")
    loader = MagicMock()

    monkeypatch.setattr(chat_dataset_patch, "_orig_loader_init", fake_orig_loader_init)
    monkeypatch.setattr(chat_dataset_patch, "_mtp_seq_len_delta", lambda: 0)

    _patched_loader_init(
        loader,
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=MagicMock(),
        seq_len=128,
        local_batch_size=1,
    )

    assert captured["sample_processor"] is process_gsm8k_sample
    assert config.sample_processor is None


def test_loader_init_requires_sample_processor_or_chat_processor(monkeypatch):
    from torchtitan_npu.config.configs import ChatDataLoaderConfig

    monkeypatch.setattr(chat_dataset_patch, "_mtp_seq_len_delta", lambda: 0)

    with pytest.raises(ValueError, match="requires either sample_processor or chat_processor"):
        _patched_loader_init(
            MagicMock(),
            ChatDataLoaderConfig(),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=MagicMock(),
            seq_len=128,
            local_batch_size=1,
        )


def test_loader_init_merges_typed_dataset_fields_without_mutating_config(monkeypatch):
    from torchtitan_npu.config.configs import ChatDataLoaderConfig

    captured = {}
    copy_calls = []

    def fake_orig_loader_init(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        seq_len,
        local_batch_size,
        **kwargs,
    ):
        del dp_world_size, dp_rank, tokenizer, seq_len, local_batch_size, kwargs
        captured["load_dataset_kwargs"] = config.load_dataset_kwargs
        self.dataset = MagicMock()

    monkeypatch.setattr(chat_dataset_patch, "_orig_loader_init", fake_orig_loader_init)
    monkeypatch.setattr(chat_dataset_patch, "_mtp_seq_len_delta", lambda: 0)
    original_copy = chat_dataset_patch.copy.copy

    def counting_copy(value):
        copy_calls.append(value)
        return original_copy(value)

    monkeypatch.setattr(chat_dataset_patch.copy, "copy", counting_copy)

    original_kwargs = {"split": "validation", "name": "main", "extra": 3}
    config = ChatDataLoaderConfig(
        load_dataset_kwargs=original_kwargs,
        dataset_split="train",
        data_files="demo.parquet",
        dataset_config_name="default",
        chat_processor="torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample",
    )

    _patched_loader_init(
        MagicMock(),
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=MagicMock(),
        seq_len=128,
        local_batch_size=1,
    )

    assert captured["load_dataset_kwargs"] == {
        "split": "train",
        "name": "default",
        "extra": 3,
        "data_files": "demo.parquet",
    }
    assert config.load_dataset_kwargs == original_kwargs
    assert len(copy_calls) == 1


# ---------------------------------------------------------------------------
# _patched_validate_messages
# ---------------------------------------------------------------------------


class TestPatchedValidate:
    def test_valid_single_turn(self):
        _patched_validate_messages(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )

    def test_valid_single_turn_bound_method_call(self):
        class DatasetLike:
            _validate_messages = staticmethod(_patched_validate_messages)

        DatasetLike()._validate_messages(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )

    def test_valid_multi_turn(self):
        _patched_validate_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "ok"},
        ])

    def test_valid_with_tool_role(self):
        _patched_validate_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": []},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "got it"},
        ])

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown role"):
            _patched_validate_messages(
                [{"role": "user", "content": "hi"}, {"role": "bot", "content": "hello"}]
            )

    def test_rejects_single_message(self):
        with pytest.raises(ValueError, match="at least 2"):
            _patched_validate_messages([{"role": "user", "content": "hi"}])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            _patched_validate_messages([])

    # --- 2-message strict mode ---

    def test_2msg_rejects_system_first(self):
        with pytest.raises(ValueError, match="First message must be 'user'"):
            _patched_validate_messages(
                [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "ok"}]
            )

    def test_2msg_rejects_user_user(self):
        with pytest.raises(ValueError, match="Second message must be 'assistant'"):
            _patched_validate_messages(
                [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]
            )

    # --- multi-turn relaxed mode ---

    def test_multiturn_allows_system_first(self):
        _patched_validate_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_multiturn_rejects_assistant_first(self):
        with pytest.raises(ValueError, match="must be 'user' or 'system'"):
            _patched_validate_messages([
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ])


# ---------------------------------------------------------------------------
# Hardcoded exact-match tests for Qwen3 multiturn chat with real tokenizer.
#
# In each test:
#  - input_ids = full_tokens[:-1]: the BOS-prefixed token sequence with the
#    last token stripped (<|im_end|> may be absent if it was the final token).
#  - label_ids = full_tokens[1:] with only assistant content tokens and their
#    closing <|im_end|> kept (≠ IGNORE_INDEX).  All format markers
#    (<|im_start|>, \n, role words) and non-assistant content are masked.
#
# Notice these important tokens:
# - 151644 = <|im_start|>
# - 151645 = <|im_end|> (= eos_id in Qwen3)
# - 151667 = <think>
# - 151668 = </think>
#
# <think> / </think> appear in the last assistant turn because Qwen3's chat
# template unconditionally wraps the final assistant message with
# <think>\n\n</think>\n\n even when reasoning_content is not provided.
# These format tokens are treated as part of the assistant span and kept
# in labels — the model must learn to emit them to follow the template.
# Non-last assistant turns use plain format (no think wrapper).
# -    271 = \n\n  (double newline, used in think wrapper)
# -    198 = \n    (single newline)
# - -100 = IGNORE_INDEX (masked label)
# ---------------------------------------------------------------------------


class TestHardcodedExactMatch:
    """Each test compares _patched_tokenize_sample output against a hardcoded reference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from pathlib import Path
        from transformers import AutoTokenizer

        tokenizer_path = Path("tests/assets/tokenizer/qwen3-tokenizer")
        if not (tokenizer_path / "tokenizer.json").exists():
            pytest.skip("Real tokenizer assets not available")
        self._tok = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )

    def _run(self, messages, seq_len=256):
        mock = MagicMock()
        mock.seq_len = seq_len
        mock._eos_id = self._tok.eos_token_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = self._tok
        mock._sample_idx = 0
        mock._chat_encoder = None
        return _patched_tokenize_sample(mock, {"sample": 1})

    def test_system_user_assistant(self):
        input_ids, label_ids = self._run([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ])
        assert input_ids == [
            151644, 8948, 198, 3430, 10950, 13, 151645, 198,
            151644, 872, 198, 13048, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 9707, 13,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, 151667, 271, 151668, 271, 9707, 13, 151645,
        ]

    def test_two_assistant_turns(self):
        input_ids, label_ids = self._run([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ])
        assert input_ids == [
            151644, 872, 198, 80, 16, 151645, 198,
            151644, 77091, 198, 64, 16, 151645, 198,
            151644, 872, 198, 80, 17, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 64, 17,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100,
            -100, -100,   64,   16, 151645, -100, -100,
            -100, -100, -100, -100, -100, -100, -100,
            -100, -100, 151667,  271, 151668,  271,   64,   17, 151645,
        ]

    def test_single_token_responses(self):
        input_ids, label_ids = self._run([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "Y"},
            {"role": "user", "content": "z"},
            {"role": "assistant", "content": "W"},
        ])
        assert input_ids == [
            151644, 872, 198, 87, 151645, 198,
            151644, 77091, 198, 56, 151645, 198,
            151644, 872, 198, 89, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 54,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100,
            -100, -100,   56, 151645, -100, -100,
            -100, -100, -100, -100, -100, -100, -100,
            -100, 151667,  271, 151668,  271,   54, 151645,
        ]

    def test_drop_sample_exceeds_seq_len(self):
        result = self._run(
            [{"role": "user", "content": "a" * 200},
             {"role": "assistant", "content": "b" * 200},
             {"role": "user", "content": "c" * 200}],
            seq_len=5,
        )
        assert result is None

    def test_first_label_always_masked(self):
        _, label_ids = self._run([
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},
        ])
        assert label_ids[0] == IGNORE_INDEX
        kept = [lid for lid in label_ids if lid != IGNORE_INDEX]
        assert 151645 in kept, "<|im_end|> from assistant turn should be kept"

    def test_three_assistant_turns_long_dialogue(self):
        """7 messages, 3 assistant turns: [system, user, asst, user, asst, user, asst]."""
        input_ids, label_ids = self._run([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome!"},
        ])
        # system: You are a helpful assistant.
        # user1:  Hello
        # asst1:  Hi there!
        # user2:  What is Python?
        # asst2:  A programming language.
        # user3:  Thanks
        # asst3:  You are welcome!  (last → with <think> wrapper)
        assert input_ids == [
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
            151644, 872, 198, 9707, 151645, 198,
            151644, 77091, 198, 13048, 1052, 0, 151645, 198,
            151644, 872, 198, 3838, 374, 13027, 30, 151645, 198,
            151644, 77091, 198, 32, 15473, 4128, 13, 151645, 198,
            151644, 872, 198, 12658, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 2610, 525, 10565, 0,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, 13048, 1052, 0, 151645, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, 32, 15473, 4128, 13, 151645, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, -100, 151667, 271, 151668, 271, 2610, 525, 10565, 0, 151645,
        ]


# ---------------------------------------------------------------------------
# 2-message delegation test (uses mock — upstream code calls tokenizer
# without tokenize=False, which crashes with the real Qwen3 tokenizer)
# ---------------------------------------------------------------------------


class TestTwoMessageDelegation:
    """2-message samples delegate to upstream _ORIG_TOKENIZE."""

    def test_delegates_and_returns_valid_tuple(self):
        mock = MagicMock()
        mock.seq_len = 256
        mock._eos_id = 151645
        mock._sample_processor = MagicMock(return_value=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        # Mock tokenizer: apply_chat_template returns a string (upstream expects this)
        mock._tokenizer = MagicMock()
        mock._tokenizer.apply_chat_template = MagicMock(
            side_effect=[
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>",
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            ]
        )
        mock._tokenizer.encode = MagicMock(
            side_effect=[
                [100, 1, 2, 3, 151645],
                [100, 1],
            ]
        )
        mock._validate_messages = MagicMock()
        mock._logged_first_sample = True
        mock._sample_idx = 0
        mock._chat_encoder = None

        result = _patched_tokenize_sample(mock, {"sample": 1})
        assert result is not None
        input_ids, label_ids = result
        assert len(input_ids) == len(label_ids)
        assert len(input_ids) > 0


class _RewriteThinkingTokenizer:
    eos_id = 3

    @staticmethod
    def _strip_think(content):
        start = content.find("<think>")
        end = content.find("</think>")
        if start == -1 or end == -1:
            return content
        return content[:start] + content[end + len("</think>"):]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for idx, msg in enumerate(messages):
            content = msg["content"]
            if msg["role"] == "assistant" and idx != len(messages) - 1:
                content = self._strip_think(content)
            parts.append(f"<{msg['role']}>{content}</{msg['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [ord(ch) + 10 for ch in text]
        if add_bos:
            ids.insert(0, 2)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            if token_id in (2, self.eos_id):
                continue
            chars.append(chr(token_id - 10))
        return "".join(chars)


class TestAssistantRangeMasking:
    @staticmethod
    def test_prefix_match_does_not_train_assistant_header():
        class HeaderTokenizer:
            eos_id = 3

            @staticmethod
            def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
                text = "".join(
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    for msg in messages
                ).rstrip("\n")
                if add_generation_prompt:
                    text += "<|im_start|>assistant\n"
                return text

            @staticmethod
            def encode(text, add_bos=True, add_eos=False):
                ids = [ord(ch) + 10 for ch in text]
                if add_bos:
                    ids.insert(0, 2)
                if add_eos:
                    ids.append(HeaderTokenizer.eos_id)
                return ids

            @staticmethod
            def decode(ids):
                return "".join(chr(token_id - 10) for token_id in ids if token_id not in (2, HeaderTokenizer.eos_id))

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        tokenizer = HeaderTokenizer()
        mock = MagicMock()
        mock.seq_len = 4096
        mock._eos_id = tokenizer.eos_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = tokenizer
        mock._sample_idx = 0
        mock._logged_first_sample = True
        mock._chat_encoder = None

        _input_ids, label_ids = _patched_tokenize_sample(mock, {"sample": 1})
        kept = [token_id for token_id in label_ids if token_id != IGNORE_INDEX]
        kept_text = tokenizer.decode(kept)

        assert "answer" in kept_text
        assert "<|im_start|>assistant" not in kept_text
        assert "question" not in kept_text

    @staticmethod
    def test_unknown_non_chatml_template_raises():
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": "<think>hidden crane reasoning</think><guess>[crane]</guess>",
            },
            {"role": "user", "content": "feedback should be masked"},
            {
                "role": "assistant",
                "content": "<think>hidden plane reasoning</think><guess>[plane]</guess>",
            },
            {"role": "user", "content": "game end should be masked"},
        ]
        tokenizer = _RewriteThinkingTokenizer()
        mock = MagicMock()
        mock.seq_len = 4096
        mock._eos_id = tokenizer.eos_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = tokenizer
        mock._sample_idx = 0
        mock._logged_first_sample = True
        mock._chat_encoder = None

        with pytest.raises(ValueError, match="Unable to locate ChatML assistant ranges"):
            _patched_tokenize_sample(mock, {"sample": 1})

    @staticmethod
    def test_real_dsv4_encoder_masks_prompt_marker_but_keeps_assistant_and_eos(tmp_path):
        from torchtitan_npu.patches.encoders.dsv4 import DSV4ChatEncoder

        class StubTokenizer:
            eos_id = 3

            _special_tokens = [
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<think>",
                "</think>",
                "<tool_result>",
                "</tool_result>",
                "<tool_call>",
                "</tool_call>",
            ]

            _special_ids = {token: 10_000 + idx for idx, token in enumerate(_special_tokens)}

            @classmethod
            def encode(cls, text, add_bos=True, add_eos=False):
                ids = []
                pos = 0
                while pos < len(text):
                    match = None
                    for token in sorted(cls._special_tokens, key=len, reverse=True):
                        if text.startswith(token, pos):
                            match = token
                            break
                    if match is not None:
                        ids.append(cls._special_ids[match])
                        pos += len(match)
                    else:
                        ids.append(ord(text[pos]) + 100)
                        pos += 1
                if add_bos:
                    ids.insert(0, 2)
                if add_eos:
                    ids.append(cls.eos_id)
                return ids

            @classmethod
            def decode(cls, ids):
                reverse = {v: k for k, v in cls._special_ids.items()}
                chars = []
                for token_id in ids:
                    if token_id in (2, cls.eos_id):
                        continue
                    if token_id in reverse:
                        chars.append(reverse[token_id])
                    else:
                        chars.append(chr(token_id - 100))
                return "".join(chars)

        encoding_module = tmp_path / "encoding_dsv4.py"
        encoding_module.write_text(
            """
def merge_tool_messages(messages):
    return messages


def sort_tool_results_by_call_order(messages):
    return messages


def _drop_thinking_messages(messages):
    raise AssertionError("_drop_thinking_messages should not run when tools are present")


def render_message(index, messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
    msg = messages[index]
    role = msg["role"]
    content = msg.get("content", "")
    if role == "system":
        return "<｜begin▁of▁sentence｜>" + content
    if role == "user":
        prompt = "<｜User｜>" + content + "<｜Assistant｜>"
        if thinking_mode == "thinking":
            prompt += "<think>"
        return prompt
    if role == "tool":
        return "<tool_result>" + content + "</tool_result>"
    if role == "assistant":
        reasoning = msg.get("reasoning_content", "")
        if reasoning and drop_thinking:
            reasoning = ""
        if reasoning:
            return "<think>" + reasoning + "</think>" + content + "<｜end▁of▁sentence｜>"
        return content + "<｜end▁of▁sentence｜>"
    raise NotImplementedError(role)
""",
            encoding="utf-8",
        )

        messages = [
            {
                "role": "system",
                "content": "system",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            },
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "plan tool use",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "reasoning_content": "final reasoning", "content": "final answer"},
        ]

        tokenizer = StubTokenizer()
        encoder = DSV4ChatEncoder(
            encoding_module_path=str(encoding_module),
            thinking_mode="thinking",
            drop_thinking=True,
            reasoning_effort=None,
        )

        mock = MagicMock()
        mock.seq_len = 4096
        mock._eos_id = tokenizer.eos_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = tokenizer
        mock._sample_idx = 9
        mock._logged_first_sample = True
        mock._chat_encoder = encoder

        _input_ids, label_ids = _patched_tokenize_sample(mock, {"sample": 1})
        kept = [token_id for token_id in label_ids if token_id != IGNORE_INDEX]
        kept_text = tokenizer.decode(kept)
        full_text = tokenizer.decode(_input_ids + [tokenizer.eos_id])

        assert "<｜Assistant｜>" in full_text
        assert "<｜Assistant｜>" not in kept_text
        assert "plan tool use" in kept_text
        assert "final answer" in kept_text
        assert "<｜end▁of▁sentence｜>" in kept_text
        assert "question" not in kept_text

    @staticmethod
    def test_dsv4_encoder_rejects_boundary_retokenization():
        from torchtitan_npu.patches.encoders.dsv4 import DSV4ChatEncoder

        class ContextSensitiveTokenizer:
            eos_id = 3

            @staticmethod
            def encode(text, add_bos=True, add_eos=False):
                if text == "<A>":
                    ids = [11, 12]
                elif text == "<A>B":
                    ids = [99]
                else:
                    ids = [ord(ch) + 10 for ch in text]
                if add_bos:
                    ids.insert(0, 2)
                if add_eos:
                    ids.append(ContextSensitiveTokenizer.eos_id)
                return ids

        class BrokenEncImpl:
            @staticmethod
            def merge_tool_messages(messages):
                return messages

            @staticmethod
            def sort_tool_results_by_call_order(messages):
                return messages

            @staticmethod
            def _drop_thinking_messages(messages):
                return messages

            @staticmethod
            def render_message(index, messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
                del messages, thinking_mode, drop_thinking, reasoning_effort
                return "<A>" if index == 0 else "B"

        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        encoder = DSV4ChatEncoder.__new__(DSV4ChatEncoder)
        encoder._enc = BrokenEncImpl()
        encoder.thinking_mode = "thinking"
        encoder.drop_thinking = True
        encoder.reasoning_effort = None
        tokenizer = ContextSensitiveTokenizer()

        with pytest.raises(ValueError, match="assistant span prefix alignment failed"):
            DSV4ChatEncoder.encode_messages_with_assistant_ranges(encoder, messages, tokenizer, tokenizer.eos_id)

    @staticmethod
    def test_all_masked_sample_is_dropped(caplog):
        class CharTokenizer:
            eos_id = 3

            @staticmethod
            def encode(text, add_bos=True, add_eos=False):
                ids = [ord(ch) + 10 for ch in text]
                if add_bos:
                    ids.insert(0, 2)
                if add_eos:
                    ids.append(CharTokenizer.eos_id)
                return ids

            @staticmethod
            def decode(ids):
                return "".join(chr(token_id - 10) for token_id in ids if token_id not in (2, CharTokenizer.eos_id))

        class EmptySpanEncoder:
            @staticmethod
            def encode_messages_with_assistant_ranges(messages, tokenizer, eos_id):
                del messages, eos_id
                full_text = "<|im_start|>user\nquestion<|im_end|>\n"
                full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
                if full_tokens[-1] != tokenizer.eos_id:
                    full_tokens.append(tokenizer.eos_id)
                return full_text, full_tokens, []

        messages = [
            {"role": "user", "content": "question"},
        ]
        tokenizer = CharTokenizer()
        mock = MagicMock()
        mock.seq_len = 4096
        mock._eos_id = tokenizer.eos_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = tokenizer
        mock._sample_idx = 7
        mock._logged_first_sample = True
        mock._chat_encoder = EmptySpanEncoder()

        with caplog.at_level("WARNING"):
            result = _patched_tokenize_sample(mock, {"sample": 1})

        assert result is None
        assert "Dropping sample 7: all labels are masked" in caplog.text
