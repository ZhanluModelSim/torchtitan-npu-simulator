# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

from torchtitan_npu.patches.encoders.dsv4 import DSV4EncoderConfig

_DSV4_ENCODER_PATH = "./assets/hf/DeepSeek-V4-Flash-Base/encoding/encoding_dsv4.py"


def dsv4_chat_encoder() -> DSV4EncoderConfig:
    return DSV4EncoderConfig(encoding_module_path=_DSV4_ENCODER_PATH)


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
