# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin training adapter for the model asset's official V4.1 encoding."""

import copy
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Literal

from torchtitan.components.tokenizer import HuggingFaceTokenizer

ThinkingMode = Literal["chat", "thinking"]
ReasoningEffort = str | int


class DeepSeekV41VisionLanguageEncoder:
    """Apply the model asset's official template and assistant supervision."""

    def __init__(
        self,
        encoding_module_path: str,
        thinking_mode: ThinkingMode = "chat",
        drop_thinking: bool = True,
        add_default_bos_token: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
        if not Path(encoding_module_path).is_file():
            raise FileNotFoundError(
                f"DeepSeek V4.1 SFT could not find {encoding_module_path}. "
                "Set --hf-assets-path to the model directory containing encoding/encoding.py."
            )
        spec = importlib.util.spec_from_file_location("encoding_dsv41", encoding_module_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load encoding module: {encoding_module_path}")
        encoding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoding)
        encode_messages = getattr(encoding, "encode_messages", None)
        image_placeholder = getattr(encoding, "IMAGE_PLACEHOLDER", None)
        if not callable(encode_messages) or not isinstance(image_placeholder, str):
            raise RuntimeError("The V4.1 encoding must provide encode_messages and IMAGE_PLACEHOLDER")
        required = {
            "thinking_mode",
            "context",
            "drop_thinking",
            "add_default_bos_token",
            "reasoning_effort",
            "return_multi_modal_data",
        }
        missing = required - inspect.signature(encode_messages).parameters.keys()
        if missing:
            raise RuntimeError(f"The V4.1 encoding API is missing parameters: {sorted(missing)}")
        self._encoding = encoding
        self._encode_messages: Any = encode_messages
        self._image_placeholder = image_placeholder
        self.thinking_mode = thinking_mode
        self.drop_thinking = drop_thinking
        self.add_default_bos_token = add_default_bos_token
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _mark_content(message: dict[str, Any], marker: str, *, prepend: bool) -> None:
        key = "content_blocks" if "content_blocks" in message else "content"
        content = message.get(key)
        if isinstance(content, list):
            block = {"type": "text", "text": marker}
            message[key] = [block, *content] if prepend else [*content, block]
        elif content is None or isinstance(content, str):
            content = content or ""
            message[key] = marker + content if prepend else content + marker
        else:
            raise RuntimeError("assistant content cannot be probed by the official V4.1 encoding")

    def _assistant_span_by_probe(
        self,
        messages: list[dict[str, Any]],
        index: int,
        prompt: str,
        encoding_options: dict[str, Any],
    ) -> tuple[int, int]:
        """Derive one span with the full conversation available to the encoder."""
        if messages[index].get("tool_calls"):
            # Keep the later user turns visible to drop_thinking and include the
            # serialized tool calls that follow the assistant's text.
            context = encoding_options["context"] or []
            options = encoding_options | {"add_default_bos_token": False}
            with_call = self._encode_messages(messages[index:], **(options | {"context": context + messages[:index]}))
            next_index = index + 1
            after_call = ""
            if next_index < len(messages):
                after_call = self._encode_messages(
                    messages[next_index:], **(options | {"context": context + messages[:next_index]})
                )
            start, end = len(prompt) - len(with_call), len(prompt) - len(after_call)
            if start >= end or not prompt.endswith(with_call) or not with_call.endswith(after_call):
                raise RuntimeError("the official V4.1 encoding did not preserve the tool-call SFT span")
            return start, end
        marker = f"\ue000torchtitan_sft_{index}\ue001"
        if marker in prompt:
            raise RuntimeError("SFT span probe marker occurs in the encoded prompt")

        start_messages = copy.deepcopy(messages)
        start_message = start_messages[index]
        reasoning = start_message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise RuntimeError("assistant reasoning_content must be text")
        start_message["reasoning_content"] = marker + reasoning
        self._mark_content(start_message, marker, prepend=True)
        start_prompt = self._encode_messages(start_messages, **encoding_options)
        start = start_prompt.find(marker)
        if start < 0 or start_prompt[:start] != prompt[:start]:
            raise RuntimeError("the official V4.1 encoding did not preserve the SFT start probe")

        end_messages = copy.deepcopy(messages)
        end_message = end_messages[index]
        self._mark_content(end_message, marker, prepend=False)
        end_message["wo_eos"] = True
        end_prompt = self._encode_messages(end_messages, **encoding_options)
        marker_end = end_prompt.find(marker) + len(marker)
        suffix = end_prompt[marker_end:]
        if marker_end < len(marker) or not prompt.endswith(suffix):
            raise RuntimeError("the official V4.1 encoding did not preserve the SFT end probe")
        end = len(prompt) - len(suffix)
        if start > end:
            raise RuntimeError("the official V4.1 encoding produced an inverted assistant span")
        return start, end

    def _merge_tool_messages(self, messages):
        normalized = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "user" and message.get("content_blocks") is None and isinstance(content, list):
                # The first release expects user blocks to be set before merging tools.
                message = message | {"content_blocks": content}
            normalized.append(message)
        return self._encoding.merge_tool_messages(normalized)

    def encode_messages_with_assistant_mask(
        self,
        messages: list[dict[str, Any]],
        tokenizer: HuggingFaceTokenizer,
        *,
        context: list[dict[str, Any]] | None = None,
        thinking_mode: ThinkingMode | None = None,
        drop_thinking: bool | None = None,
        add_default_bos_token: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> tuple[str, list[int], list[bool], list[int], list[dict[str, Any]]]:
        """Return prompt, text tokens, assistant mask, image offsets, and images."""
        thinking_mode = thinking_mode or self.thinking_mode
        drop_thinking = self.drop_thinking if drop_thinking is None else drop_thinking
        add_bos = self.add_default_bos_token if add_default_bos_token is None else add_default_bos_token
        reasoning_effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort

        encoding_options = {
            "thinking_mode": thinking_mode,
            "context": context or None,
            "drop_thinking": drop_thinking,
            "add_default_bos_token": add_bos,
            "reasoning_effort": reasoning_effort,
        }
        prompt, media = self._encode_messages(
            messages,
            **encoding_options,
            return_multi_modal_data=True,
        )
        if not isinstance(prompt, str) or not isinstance(media, dict):
            raise RuntimeError("The V4.1 encoding returned an unsupported multimodal result")
        images = media.get("images")
        if not isinstance(images, list):
            raise RuntimeError("The V4.1 encoding did not return an images list")
        if images and any(message.get("tool_calls") for message in (context or []) + messages):
            # The first V4.1 release collected images before sorting tool results.
            helpers = ("merge_tool_messages", "sort_tool_results_by_call_order", "process_image_messages")
            missing = [name for name in helpers if not callable(getattr(self._encoding, name, None))]
            if missing:
                raise RuntimeError(f"The V4.1 encoding is missing tool image helpers: {', '.join(missing)}")
            history = self._merge_tool_messages(context or [])
            history_length = len(history)
            ordered = self._encoding.sort_tool_results_by_call_order(history + self._merge_tool_messages(messages))
            _, images = self._encoding.process_image_messages(ordered[history_length:])

        # Use official prefixes when stable and recover the span against the
        # full conversation when later messages change the prefix rendering.
        assistant_spans = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            before = self._encode_messages(messages[:index], **encoding_options)
            through = self._encode_messages(messages[: index + 1], **encoding_options)
            if through.startswith(before) and prompt.startswith(through):
                start, end = len(before), len(through)
            else:
                start, end = self._assistant_span_by_probe(messages, index, prompt, encoding_options)
            if start < end:
                assistant_spans.append((start, end))

        placeholder = self._image_placeholder
        image_spans = []
        cursor = 0
        while (start := prompt.find(placeholder, cursor)) >= 0:
            image_spans.append((start, start + len(placeholder)))
            cursor = start + len(placeholder)
        if len(image_spans) != len(images):
            raise ValueError("Encoded image count does not match the prompt")

        token_ids: list[int] = []
        assistant_mask: list[bool] = []
        image_positions: list[int] = []
        cursor = 0
        for start, end in [*image_spans, (len(prompt), len(prompt))]:
            encoded = tokenizer.tokenizer.encode(prompt[cursor:start], add_special_tokens=False)
            token_ids.extend(encoded.ids)
            assistant_mask.extend(
                any(
                    cursor + begin < span_end and cursor + finish > span_start
                    for span_start, span_end in assistant_spans
                )
                for begin, finish in encoded.offsets
            )
            if end > start:
                image_positions.append(len(token_ids))
            cursor = end
        return prompt, token_ids, assistant_mask, image_positions, images
