# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 chat dataset, adapting TorchTitan's ChatDataset/ChatDataLoader.

The upstream implementation is in torchtitan/hf_datasets/text_datasets.py.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import tyro
from datasets import load_dataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, ChatDataset
from torchtitan.tools.logging import logger

from .chat_encoder import DSV4ChatEncoder, DSV4EncoderConfig


def _sft_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the JSON record's messages column; V4 semantics belong to the dataset."""
    raw_messages = sample["messages"]
    return json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages


class DeepSeekV4ChatDataset(ChatDataset):
    """V4 developer/reminder/tool turns with assistant-only supervision."""

    def __init__(self, *args, chat_encoder: DSV4ChatEncoder, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chat_encoder = chat_encoder

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

    def _tokenize_sample(self, sample: dict[str, Any]) -> tuple[list[int], list[int]] | None:
        messages = [dict(message) for message in self._sample_processor(sample)]
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

        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(f"Dropping sample {self._sample_idx}: tokens exceeds seq_len {self.seq_len}")
            return None

        input_ids = full_tokens[:-1]
        label_ids = [
            token if keep else IGNORE_INDEX for token, keep in zip(full_tokens[1:], assistant_mask[1:], strict=True)
        ]
        if all(label == IGNORE_INDEX for label in label_ids):
            logger.warning(f"Dropping sample {self._sample_idx}: all labels are masked")
            return None
        return input_ids, label_ids


class DeepSeekV4ChatDataLoader(ChatDataLoader):
    """Construct the DSV4 dataset while retaining upstream greedy packing."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChatDataLoader.Config):
        dataset_path: str | None = "json"
        dataset: str = "json"  # accepted from the shared CPT launcher
        data_files: str = ""
        sample_processor: Annotated[Callable[[dict[str, Any]], list[dict[str, Any]]], tyro.conf.Suppress] = (
            _sft_messages
        )
        chat_encoder: DSV4EncoderConfig = field(default_factory=DSV4EncoderConfig)

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
    ) -> None:
        if not config.data_files:
            raise ValueError("SFT requires data_files")
        if not isinstance(tokenizer, HuggingFaceTokenizer):
            raise TypeError("DeepSeek-V4 chat SFT requires HuggingFaceTokenizer")
        chat_encoder = config.chat_encoder.build(tokenizer.tokenizer_path)
        extra_kwargs = {
            key: value for key, value in config.load_dataset_kwargs.items() if key not in {"data_files", "split"}
        }
        chat_dataset = DeepSeekV4ChatDataset(
            dataset=load_dataset(
                config.dataset_path or "json",
                data_files=config.data_files,
                split="train",
                **extra_kwargs,
            ),
            tokenizer=tokenizer,
            sample_processor=config.sample_processor,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            chat_encoder=chat_encoder,
        )
        ParallelAwareDataloader.__init__(
            self,
            chat_dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )
