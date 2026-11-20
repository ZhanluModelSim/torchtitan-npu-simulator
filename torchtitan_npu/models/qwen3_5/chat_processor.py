# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Raw Qwen text and tool conversations with supervision on the rendered tokens."""

import json
import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from jinja2 import nodes
from torchtitan.components.data import SampleProcessor, TextSequence
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_LOSS_MASK = "default+ignore_empty_think"
_MASK_SCOPES = ("default", "last_round", "template", "last-assistant")
_THINK_MASKS = ("ignore_empty_think", "ignore_think_prefix", "ignore_think")
_HEADER, _STOP = "<|im_start|>assistant\n", "<|im_end|>"
_THINK, _THINK_END = "<think>", "</think>"
_TEMPLATE_LOCK = Lock()  # Transformers caches one stateful assistant tracker per template.
_BEGIN = "\x00qwen-assistant-begin\x00"
_END = "\x00qwen-assistant-end\x00"


GENERATION_BLOCK = re.compile(r"\{%[-+]?\s*generation\s*[-+]?%\}")


class QwenSFTTokenizer(HuggingFaceTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def set_chat_template(self, template: str) -> None:
        if GENERATION_BLOCK.search(template):
            from transformers.utils.chat_template_utils import _compile_jinja_template

            self._chat_template = _compile_jinja_template(template)
        else:
            super().set_chat_template(template)


def assistant_turn_supervision(messages: list[dict[str, Any]], *, last_round: bool) -> list[bool]:
    last_user = (
        max((index for index, turn in enumerate(messages) if turn.get("role") == "user"), default=-1)
        if last_round
        else -1
    )
    keep = []
    for index, turn in enumerate(messages):
        if turn.get("role") == "assistant":
            loss = turn.get("loss")
            keep.append(index >= last_user if loss is None else loss)
    return keep


def _order_tool_results(messages):
    ordered = []
    for is_result, group in groupby(messages, key=lambda message: message["role"] == "tool"):
        if not is_result:
            ordered.extend(group)
            continue
        results = {message["tool_call_id"]: message for message in group}
        preceding = ordered[-1] if ordered else {}
        calls = (preceding.get("tool_calls") or []) if preceding.get("role") == "assistant" else []
        count = len(results)
        call_ids = [call["id"] for call in calls[:count]]
        if results.keys() != set(call_ids):
            raise ValueError("Tool results must follow their assistant call batch and identify a prefix of that batch")
        ordered.extend(results[call_id] for call_id in call_ids)
    return ordered


def _messages(row):
    raw = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("Qwen SFT requires a JSONL row with a nonempty messages list")
    messages = []
    call_ids = set()
    pending_ids = set()
    for turn in raw:
        if not isinstance(turn, dict) or turn.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Qwen SFT expects system, user, assistant or tool messages")
        message = dict(turn)
        content = message.get("content")
        if isinstance(content, list):
            if not all(isinstance(part, dict) and isinstance(part.get("text"), str) for part in content):
                raise ValueError("The Qwen text SFT recipe requires text content")
            message["content"] = "".join(part["text"] for part in content)
        if message["role"] != "assistant" and not isinstance(message.get("content"), str):
            raise ValueError("System, user and tool messages require text content")
        if message.get("role") == "assistant":
            if "loss" in message and message["loss"] is not None and type(message["loss"]) is not bool:
                raise ValueError("Assistant loss must be true, false or null")
            for field in ("content", "reasoning_content"):
                value = message.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"Expected text in assistant {field}")
                if value and (_THINK in value or _THINK_END in value):
                    raise ValueError("Provide reasoning_content without thinking tags")
            if message.get("tool_calls") is not None:
                if not isinstance(message["tool_calls"], list):
                    raise ValueError("Assistant tool_calls must be a list")
                calls = []
                for call in message["tool_calls"]:
                    if not isinstance(call, dict) or call.get("type", "function") != "function":
                        raise ValueError("Each tool call must describe a function")
                    function = call.get("function")
                    if not isinstance(function, dict):
                        raise ValueError("Each tool call must describe a function")
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        raise ValueError("Each tool call must name a function")
                    function = dict(function)
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must decode to a JSON object")
                    function["arguments"] = args
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                        raise ValueError("Tool call IDs must be unique nonempty strings")
                    call_ids.add(call_id)
                    pending_ids.add(call_id)
                    calls.append({**call, "function": function})
                message["tool_calls"] = calls
        elif message["role"] == "tool":
            result_id = message.get("tool_call_id")
            if not isinstance(result_id, str) or result_id not in pending_ids:
                raise ValueError("Tool result must identify a matching preceding call")
            pending_ids.remove(result_id)
        messages.append(message)
    if messages[-1]["role"] not in ("assistant", "tool"):
        raise ValueError("Qwen SFT conversations must end with an assistant or tool message")
    return _order_tool_results(messages) if call_ids != pending_ids else messages


@dataclass(frozen=True, slots=True)
class _AssistantSpan:
    body_start: int
    target_start: int
    end: int
    masked_think_end: int | None


def _assistant_range(turn, offset, think_mask):
    if not turn.startswith(_HEADER) or not turn.rstrip().endswith(_STOP):
        raise ValueError("Expected Qwen assistant framing; use --loss-mask template for other framing")
    body = start = len(_HEADER)
    end = len(turn.rstrip())
    masked_think_end = None
    if turn.startswith(_THINK, body):
        start += len(_THINK)
        closing = turn.find(_THINK_END, start, end)
        if closing < 0:
            raise ValueError("Assistant thinking has no closing tag")
        reasoning = turn[start:closing]
        start += len(reasoning) - len(reasoning.lstrip("\n"))
        if think_mask == "ignore_think_prefix":
            masked_think_end = offset + start
        elif think_mask == "ignore_think" or (think_mask == "ignore_empty_think" and not reasoning.strip()):
            after_think = closing + len(_THINK_END)
            masked_think_end = offset + end - len(turn[after_think:end].lstrip("\n"))
    return _AssistantSpan(offset + body, offset + start, offset + end, masked_think_end)


def _instrument(template):
    # The Blueown loader locates assistant turns at the official Jinja message
    # loop so that literal ChatML examples in code or tool results stay context.
    from transformers.utils.chat_template_utils import _compile_jinja_template

    env = _compile_jinja_template(template).environment
    tree = env.parse(template)
    loops = [
        node
        for node in tree.find_all(nodes.For)
        if isinstance(node.iter, nodes.Name)
        and node.iter.name == "messages"
        and isinstance(node.target, nodes.Name)
        and node.target.name == "message"
    ]
    if len(loops) != 1:
        raise ValueError("Expected the Qwen message loop; use --loss-mask template for custom framing")

    def mark(value):
        assistant = nodes.Compare(
            nodes.Getattr(nodes.Name("message", "load"), "role", "load"),
            [nodes.Operand("eq", nodes.Const("assistant"))],
        )
        return nodes.If(assistant, [nodes.Output([nodes.Const(value)])], [], [])

    loop = loops[0]
    loop.body = [mark(_BEGIN), *loop.body, mark(_END)]
    return env.template_class.from_code(env, env.compile(tree), env.globals, None)


def _strip_markers(marked, count, think_mask):
    parts = marked.split(_BEGIN)
    if len(parts) != count + 1 or _END in parts[0]:
        raise ValueError("Assistant boundaries differ from the input conversation")
    clean = [parts[0]]
    ranges = []
    offset = len(parts[0])
    for part in parts[1:]:
        if part.count(_END) != 1:
            raise ValueError("Invalid assistant template boundary")
        turn, rest = part.split(_END)
        ranges.append(_assistant_range(turn, offset, think_mask))
        clean.extend((turn, rest))
        offset += len(turn) + len(rest)
    return "".join(clean).rstrip("\n"), ranges


def _tokens_in_spans(offsets, spans, *, overlap=False):
    keep = np.zeros(len(offsets), dtype=np.bool_)
    span = 0
    for index, (start, end) in enumerate(offsets):
        while span < len(spans) and start >= spans[span][1]:
            span += 1
        if span < len(spans) and end > start:
            low, high = spans[span]
            keep[index] = end > low if overlap else start >= low and end <= high
    return keep


class QwenSFTChatProcessor(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        chat_template: str | None = None
        loss_mask: str = DEFAULT_LOSS_MASK

        def __post_init__(self):
            if self.loss_mask == "thinking-safe":
                self.loss_mask = DEFAULT_LOSS_MASK
            elif self.loss_mask == "last-assistant":
                self.loss_mask = "last-assistant+ignore_empty_think"
            elif self.loss_mask in _THINK_MASKS:
                self.loss_mask = f"default+{self.loss_mask}"
            scope, separator, modifier = self.loss_mask.partition("+")
            if scope not in _MASK_SCOPES or (separator and modifier not in _THINK_MASKS):
                raise ValueError(
                    "Use --loss-mask default, last_round, template or a thinking modifier "
                    "(ignore_empty_think, ignore_think_prefix, ignore_think). Join scope and modifier with '+'."
                )

    def __init__(self, config, *, context):
        from transformers import AutoTokenizer

        self.tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(context.tokenizer.tokenizer_path, local_files_only=True),
        )
        self.backend = context.tokenizer.tokenizer
        self.eos_id = context.tokenizer.eos_id
        if not self.tokenizer.is_fast or self.eos_id != self.tokenizer.eos_token_id:
            raise ValueError("Qwen SFT requires the matching fast HF tokenizer")
        self.max_length = context.max_context_length
        self.loss_mask = config.loss_mask
        self.policy, _, self.think_mask = config.loss_mask.partition("+")
        self.template = (
            self.tokenizer.get_chat_template()
            if config.chat_template is None
            else Path(config.chat_template).read_text(encoding="utf-8")
        )
        self.use_generation_mask = self.policy == "template" and bool(GENERATION_BLOCK.search(self.template))
        if self.policy == "template" and config.chat_template is not None and not self.use_generation_mask:
            raise ValueError("--loss-mask template requires Hugging Face generation blocks in a custom template")
        self.instrumented = _instrument(self.template) if not self.use_generation_mask or self.think_mask else None

    def __call__(self, row, rng):
        del rng
        messages = _messages(row)
        tools = row.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or any(
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or not isinstance(tool.get("function"), dict)
                for tool in tools
            )
        ):
            raise ValueError("Qwen tools must be a list of function definitions")
        options = dict(tools=tools, add_generation_prompt=False, preserve_thinking=True, enable_thinking=True)
        if self.use_generation_mask:
            ids, keep = self._template_mask(messages, options)
        else:
            ids, keep = self._span_mask(messages, options)
        ids = np.asarray(ids, dtype=np.int64)
        keep = np.asarray(keep, dtype=np.bool_)
        if len(ids) - 1 > self.max_length or not keep[1:].any():
            return None
        # As in MindSpeed, context and targets share the input axis. TorchTitan
        # expects prediction alignment at the processor boundary, before packing.
        return TextSequence(input_ids=ids[:-1], labels=np.where(keep[1:], ids[1:], IGNORE_INDEX))

    def _assistant_encoding(self, messages, options, count):
        if self.instrumented is None:
            raise RuntimeError("Assistant boundaries require an instrumented template")
        with _TEMPLATE_LOCK:
            marked = self.instrumented.render(
                messages=messages,
                documents=None,
                **self.tokenizer.special_tokens_map,
                **options,
            )
        rendered, ranges = _strip_markers(marked, count, self.think_mask)
        return self.backend.encode(rendered, add_special_tokens=False), ranges

    def _span_mask(self, messages, options):
        if self.policy == "template":
            # Older official templates emit the same assistant framing without
            # HF generation blocks. Supervise the rendered body and EOS.
            keep = [True for turn in messages if turn["role"] == "assistant"]
        else:
            keep = assistant_turn_supervision(messages, last_round=self.policy == "last_round")
        encoded, ranges = self._assistant_encoding(messages, options, len(keep))
        spans = []
        for index, (selected, span) in enumerate(zip(keep, ranges, strict=True)):
            if not selected or (self.policy == "last-assistant" and index != len(keep) - 1):
                continue
            start = span.body_start if self.policy == "template" else span.target_start
            spans.append((span.masked_think_end if span.masked_think_end is not None else start, span.end))
        return encoded.ids, _tokens_in_spans(encoded.offsets, spans)

    def _template_mask(self, messages, options):
        with _TEMPLATE_LOCK:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                **options,
                chat_template=self.template,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        # return_dict=True selects the mapping form of the Hugging Face API.
        batch = cast("dict[str, list[int]]", encoded)
        ids, keep = batch["input_ids"], batch["assistant_masks"]
        if len(ids) != len(keep):
            raise ValueError("The chat template must return one mask entry per token")
        while ids and ids[-1] != self.eos_id and not cast("str", self.tokenizer.decode([ids[-1]])).strip():
            ids.pop()
            keep.pop()
        if not ids or ids[-1] != self.eos_id:
            raise ValueError("The custom chat template must end with Qwen EOS")
        if self.think_mask:
            count = sum(turn["role"] == "assistant" for turn in messages)
            qwen, ranges = self._assistant_encoding(messages, options, count)
            if qwen.ids != ids:
                raise ValueError("Thinking masks require matching Qwen and HF template tokenization")
            thinking = []
            for span in ranges:
                if span.masked_think_end is not None:
                    thinking.append((span.body_start, span.masked_think_end))
            keep = np.asarray(keep, dtype=np.bool_) & ~_tokens_in_spans(
                qwen.offsets, thinking, overlap=self.think_mask != "ignore_empty_think"
            )
        return ids, keep
