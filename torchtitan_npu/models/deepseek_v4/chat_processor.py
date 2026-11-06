# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 conversation processing for TorchTitan's Grain dataset graph."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
import tyro
from torchtitan.components.data import DatasetBuildContext, SampleProcessor, TextSequence
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.common.alignment import align_text_sequence, validate_alignment

from .chat_encoder import DSV4EncoderConfig

logger = logging.getLogger(__name__)


def _sft_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = sample["messages"]
    return json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages


class DeepSeekV4ChatProcessor(SampleProcessor):
    """Encode one conversation; Grain owns iteration and packing."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        per_doc_alignment: int = 1
        messages_fn: Annotated[Callable[[dict[str, Any]], list[dict[str, Any]]], tyro.conf.Suppress] = _sft_messages
        chat_encoder: DSV4EncoderConfig = field(default_factory=DSV4EncoderConfig)

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        validate_alignment(config.per_doc_alignment, context=context)
        if not isinstance(context.tokenizer, HuggingFaceTokenizer):
            raise TypeError("DeepSeek-V4 chat SFT requires HuggingFaceTokenizer")
        if context.tokenizer.eos_id is None:
            raise ValueError("DeepSeek-V4 chat SFT requires a tokenizer eos_id")
        self._tokenizer = context.tokenizer
        self._eos_id = context.tokenizer.eos_id
        self._chat_encoder = config.chat_encoder.build(context.tokenizer.tokenizer_path)
        self._max_context_length = context.max_context_length
        self._per_doc_alignment = config.per_doc_alignment
        self._messages_fn = config.messages_fn

    @staticmethod
    def _validate_v4_messages(messages: list[dict[str, Any]]) -> None:
        roles = {"system", "developer", "user", "assistant", "tool", "latest_reminder"}
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in roles:
                raise ValueError(f"Unsupported DeepSeek-V4 role at messages[{index}]: {role!r}")
            if role == "developer" and not (isinstance(message.get("content"), str) and message["content"]):
                raise ValueError(f"messages[{index}] developer requires non-empty string content")
            if role == "latest_reminder" and not isinstance(message.get("content"), str):
                raise ValueError(f"messages[{index}] latest_reminder requires string content")

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> TextSequence | None:
        del rng
        messages = [dict(message) for message in self._messages_fn(sample)]
        for message in messages:
            if message.get("role") == "last_reminder":
                message["role"] = "latest_reminder"  # Official encoding name.
        raw_tools = sample.get("tools", [])
        tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        if tools:
            if messages and messages[0].get("role") in {"system", "developer"}:
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})
        self._validate_v4_messages(messages)
        # Official V4 permits a reminder after the last supervised assistant turn.
        if not messages or messages[-1].get("role") not in {"assistant", "latest_reminder"}:
            raise ValueError("SFT sample must end with an assistant or latest_reminder turn")
        if not any(message.get("role") == "assistant" for message in messages):
            raise ValueError("SFT sample requires an assistant turn for labels")
        # V4 encoding merges tool results into user turns; assistant tool calls remain supervised.
        _, full_tokens, assistant_mask = self._chat_encoder.encode_messages_with_assistant_mask(
            messages, self._tokenizer, self._eos_id
        )

        if len(full_tokens) - 1 > self._max_context_length:
            logger.debug("Dropping SFT sample exceeding max_context_length")
            return None

        if len(full_tokens) != len(assistant_mask):
            raise ValueError("Chat encoder must return one assistant mask entry per token")
        input_ids = full_tokens[:-1]
        label_ids = [
            token if keep else IGNORE_INDEX for token, keep in zip(full_tokens[1:], assistant_mask[1:], strict=True)
        ]
        if all(label == IGNORE_INDEX for label in label_ids):
            logger.debug("Dropping SFT sample with no supervised labels")
            return None
        return align_text_sequence(
            TextSequence(
                input_ids=np.asarray(input_ids, dtype=np.int64),
                labels=np.asarray(label_ids, dtype=np.int64),
            ),
            multiple=self._per_doc_alignment,
        )
