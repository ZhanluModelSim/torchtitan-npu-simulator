# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Callable
from importlib import import_module


def process_tau_sample(sample):
    raw_messages = sample["messages"]
    messages = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
    messages = [dict(message) for message in messages]
    raw_tools = sample.get("tools", [])
    tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools

    if tools:
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "tools": tools}
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return messages


def process_gsm8k_sample(sample):
    reasoning, final_answer = sample["answer"].rsplit("####", 1)
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "reasoning_content": reasoning.strip(), "content": final_answer.strip()},
    ]


def process_wordle_sample(sample: dict) -> list[dict]:
    if "messages" in sample:
        messages = list(sample["messages"])
    elif "prompt" in sample and "completion" in sample:
        messages = list(sample["prompt"]) + list(sample["completion"])
    else:
        raise KeyError(f"Wordle sample must have 'messages' or 'prompt'+'completion'. Got keys: {list(sample.keys())}")

    normalized = []
    for message in messages:
        normalized_message = dict(message)
        content = normalized_message.get("content")
        if isinstance(content, list):
            normalized_message["content"] = "\n".join(str(item) for item in content)
        normalized.append(normalized_message)
    return normalized


def import_chat_processor(path: str) -> Callable:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(
            f"chat_processor must be an import path like "
            f"'torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample', got {path!r}"
        )
    module = import_module(module_name)
    try:
        processor = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown chat_processor import path {path!r}") from exc
    if not callable(processor):
        raise TypeError(f"chat_processor import path {path!r} resolved to non-callable {processor!r}")
    return processor
