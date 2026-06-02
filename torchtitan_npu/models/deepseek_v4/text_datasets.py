# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def process_sample(sample):
    if "question" in sample and "answer" in sample:
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    elif "instruction" in sample and "input" in sample and "output" in sample:
        return [
            {"role": "user", "content": sample["instruction"] + "\n" + sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    else:
        raise ValueError(
            "Unknown Sample Format. Only 'query + answer' or 'instruction + input + output'"
            "data formats are supported at this time."
            "Please make sure the required keys are included in the data."
        )


class DSV4ChatDataset(IterableDataset, Stateful):
    """Dataset for single-turn chat/instruction-tuning.

    Tokenizes [user, assistant] message pairs, masks prompt tokens with
    IGNORE_INDEX in labels, and uses greedy sequence packing with
    per-document positions. Implements Stateful for checkpointing.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: HuggingFaceTokenizer,
        sample_processor: Callable,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatDataset requires a tokenizer with a valid EOS token."
            )

        # Shuffle the initial data to promote an even distribution across nodes. For map-style
        # datasets, split_dataset_by_node assigns contiguous data chunks to consecutive nodes, which
        # can lead to token imbalances, causing some nodes' epoch_idx to run ahead of others.
        self._original_data = split_dataset_by_node(
            cast(Dataset, dataset.shuffle(seed=42)), dp_rank, dp_world_size
        )
        self._data = self._original_data
        self._tokenizer = tokenizer

        import os
        import sys
        from functools import partial

        sys.path.append(os.path.join(tokenizer.tokenizer_path, "encoding"))

        try:
            from encoding_dsv4 import encode_messages  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please ensure that the 'encoding_dsv4' package is included in the model directory."
            ) from e
        self._apply_chat_template_func = partial(encode_messages, thinking_mode="chat")
        self._eos_id = tokenizer.eos_id
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_processor = sample_processor

        self._dataset_id = f"{dataset.info.dataset_name}/{dataset.split}"

        # Variables for checkpointing
        self._sample_idx = 0
        self._epoch: int = 0

        self._logged_first_sample = False

    def __iter__(self):
        yield from self._iter_simple()

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def state_dict(self):
        _state_dict: dict[str, Any] = {
            "epoch": self._epoch,
        }

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()

        return _state_dict

    def load_state_dict(self, state_dict):
        self._epoch = state_dict["epoch"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
            # Replay shuffles so _data matches the order at checkpoint time
            if self._epoch > 0:
                self._data = cast(
                    Dataset, self._original_data.shuffle(seed=42 + self._epoch)
                )
        else:
            data_state = state_dict["data"]
            # HuggingFace IterableDataset sync epoch
            saved_epoch = data_state.get("epoch", 0)
            self._data.set_epoch(saved_epoch)
            self._data.load_state_dict(data_state)

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def _tokenize_sample(
        self, sample: dict[str, Any]
    ) -> tuple[list[int], list[int]] | None:
        """Tokenize a single-turn sample and create input/label pairs.

        Returns (input_ids, label_ids) where input_ids = tokens[:-1] and
        label_ids = tokens[1:] with prompt tokens masked as IGNORE_INDEX.
        Returns None if the sample exceeds seq_len (dropped to avoid
        training on truncated responses).

        Uses incremental prefix re-tokenization to find the prompt/response
        token boundary, avoiding BPE merge errors.
        """
        messages = self._sample_processor(sample)
        self._validate_messages(messages)

        full_text = self._apply_chat_template_func(messages)
        # Strip extra newline and ensure the sequence ends with EOS without duplicates
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        # Drop examples exceeding seq_len rather than truncating.
        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(
                f"Dropping sample {self._sample_idx}: "
                f"tokens exceeds seq_len {self.seq_len}"
            )
            return None

        input_ids = full_tokens[:-1]
        label_ids = full_tokens[1:]

        # Find prompt/response boundary by tokenizing just the user message
        prompt_text = self._apply_chat_template_func(messages[:1])
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)

        # Labels are shifted by one token, so the first assistant token is
        # predicted at index prompt_len - 1 and must remain unmasked.
        mask_end = min(max(prompt_len - 1, 0), len(label_ids))
        label_ids[:mask_end] = [IGNORE_INDEX] * mask_end

        if not self._logged_first_sample:
            logger.info(
                f"[ChatDataset] First sample full:\n{full_text}\n{input_ids}\n{label_ids}"
            )
            self._logged_first_sample = True

        return input_ids, label_ids

    def _iter_simple(self):
        """Simple iteration: each example occupies exactly one seq_len row.
        Truncates if long, pads with EOS/IGNORE_INDEX if short.
        Positions buffer is omitted.
        """
        while True:
            for sample in self._get_data_iter():
                if not isinstance(sample, dict):
                    raise TypeError(f"Expected dict, got {type(sample)}")
                result = self._tokenize_sample(sample)
                if result is None:
                    self._sample_idx += 1
                    continue

                input_ids, label_ids = result

                # 1. 裁剪或填充到固定长度 seq_len
                if len(input_ids) >= self.seq_len:
                    # 超长截断
                    final_inputs = input_ids[: self.seq_len]
                    final_labels = label_ids[: self.seq_len]
                else:
                    # 不足填充
                    pad_len = self.seq_len - len(input_ids)
                    final_inputs = input_ids + [self._eos_id] * pad_len
                    final_labels = label_ids + [IGNORE_INDEX] * pad_len

                self._sample_idx += 1

                # 2. 直接组装成 Tensor 返回（不通过类 buffer 中转，防止污染打包状态）
                input_tensor = torch.tensor(final_inputs, dtype=torch.long)
                label_tensor = torch.tensor(final_labels, dtype=torch.long)

                yield {"input": input_tensor}, label_tensor

            # 数据集末尾的无限循环/洗牌逻辑
            if not self.infinite:
                logger.warning(f"Chat dataset '{self._dataset_id}' has run out of data")
                break
            else:
                self._sample_idx = 0
                self._epoch += 1
                if isinstance(self._data, Dataset):
                    self._data = cast(
                        Dataset,
                        self._original_data.shuffle(seed=42 + self._epoch),
                    )
                elif hasattr(self._data, "set_epoch"):
                    self._data.set_epoch(self._epoch)
                logger.warning(
                    f"Chat dataset '{self._dataset_id}' is being re-looped "
                    f"(epoch {self._epoch})"
                )


def build_chat_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: HuggingFaceTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for Chat datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    if dataset_path is None:
        raise ValueError("dataset_path must be a valid path")
    dataset = load_dataset(dataset_path, split="train")

    chat_ds = DSV4ChatDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sample_processor=process_sample,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        chat_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
