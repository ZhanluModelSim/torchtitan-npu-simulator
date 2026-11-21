# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's model-owned structured chat encoder."""

import importlib.util
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torchtitan.components.tokenizer import HuggingFaceTokenizer

ThinkingMode = Literal["chat", "thinking"]
ReasoningEffort = Literal["low", "high", "max"]


class DSV4ChatEncoder:
    """Render messages with the model asset's official ``encoding_dsv4.py``."""

    def __init__(
        self,
        encoding_module_path: str,
        thinking_mode: ThinkingMode = "thinking",
        context: list[dict[str, Any]] | None = None,
        drop_thinking: bool = True,
        add_default_bos_token: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
        if reasoning_effort not in (None, "low", "high", "max"):
            raise ValueError(f"Unsupported reasoning_effort: {reasoning_effort!r}")

        spec = importlib.util.spec_from_file_location("encoding_dsv4", encoding_module_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load encoding module: {encoding_module_path}")
        self._encoding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._encoding)
        self.thinking_mode = thinking_mode
        self.context = context
        self.drop_thinking = drop_thinking
        self.add_default_bos_token = add_default_bos_token
        # The original V4 file accepts None, while later files also call it "low".
        self.reasoning_effort = None if reasoning_effort == "low" else reasoning_effort

    def encode_messages_with_assistant_mask(
        self,
        messages: list[dict[str, Any]],
        tokenizer: HuggingFaceTokenizer,
        eos_id: int,
    ) -> tuple[str, list[int], list[bool]]:
        del eos_id  # The official template controls EOS, including wo_eos.
        full_text = self._encoding.encode_messages(
            messages,
            thinking_mode=self.thinking_mode,
            context=deepcopy(self.context),
            drop_thinking=self.drop_thinking,
            add_default_bos_token=self.add_default_bos_token,
            reasoning_effort=self.reasoning_effort,
        )
        # Per-message rendering only identifies the assistant boundaries.
        context = deepcopy(self.context) or []
        bos = self._encoding.bos_token if self.add_default_bos_token and not context else ""
        messages = self._encoding.merge_tool_messages(messages)
        context_len = len(context)
        messages = self._encoding.sort_tool_results_by_call_order(context + messages)[context_len:]
        if context:
            context = self._encoding.sort_tool_results_by_call_order(self._encoding.merge_tool_messages(context))
        messages = context + messages
        context_len = len(context)
        drop_thinking = self.drop_thinking and not any(message.get("tools") for message in messages)
        if self.thinking_mode == "thinking" and drop_thinking:
            # render_message drops earlier reasoning; encode_messages also removes earlier developer turns.
            last_user = max(
                (index for index, message in enumerate(messages) if message.get("role") in {"user", "developer"}),
                default=-1,
            )
            last_context_user = max(
                (index for index, message in enumerate(context) if message.get("role") in {"user", "developer"}),
                default=-1,
            )
            context_len -= sum(
                message.get("role") == "developer" and index < last_context_user
                for index, message in enumerate(context)
            )
            messages = [
                message
                for index, message in enumerate(messages)
                if message.get("role") != "developer" or index >= last_user
            ]

        rendered = [
            self._encoding.render_message(
                index,
                messages,
                thinking_mode=self.thinking_mode,
                drop_thinking=drop_thinking,
                reasoning_effort=self.reasoning_effort,
            )
            for index in range(context_len, len(messages))
        ]
        if bos + "".join(rendered) != full_text:
            raise ValueError("Per-message rendering diverges from the official DeepSeek V4 prompt")
        encoded = tokenizer.tokenizer.encode(full_text, add_special_tokens=False)
        assistant_spans = []
        cursor = len(bos)
        for message, message_text in zip(messages[context_len:], rendered, strict=True):
            end = cursor + len(message_text)
            if message.get("role") == "assistant" and cursor < end:
                assistant_spans.append((cursor, end))
            cursor = end
        assistant_mask = [
            any(begin < span_end and finish > span_start for span_start, span_end in assistant_spans)
            for begin, finish in encoded.offsets
        ]
        return full_text, encoded.ids, assistant_mask


@dataclass(kw_only=True, slots=True)
class DSV4EncoderConfig:
    thinking_mode: ThinkingMode = "thinking"
    drop_thinking: bool = True
    add_default_bos_token: bool = True
    reasoning_effort: ReasoningEffort | None = None

    def build(self, tokenizer_path: str) -> DSV4ChatEncoder:
        return DSV4ChatEncoder(
            encoding_module_path=str(Path(tokenizer_path) / "encoding" / "encoding_dsv4.py"),
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            add_default_bos_token=self.add_default_bos_token,
            reasoning_effort=self.reasoning_effort,
        )
