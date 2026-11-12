# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from renderers import Message


def _messages(sample: dict[str, Any]) -> list["Message"]:
    return cast("list[Message]", sample["messages"])


def _long_text_messages(sample: dict[str, Any]) -> list["Message"]:
    if "messages" in sample:
        return cast("list[Message]", sample["messages"])
    if "prompt" in sample:
        return cast(
            "list[Message]",
            [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": sample["response"]}],
        )
    user = sample["instruction"]
    if sample.get("input"):
        user = f"{user}\n\n{sample['input']}"
    return cast(
        "list[Message]",
        [{"role": "user", "content": user}, {"role": "assistant", "content": sample["output"]}],
    )
