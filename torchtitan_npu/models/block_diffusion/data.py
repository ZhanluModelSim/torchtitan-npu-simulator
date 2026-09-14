# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sliding-window data and corruption for Block Diffusion training."""

import math
from dataclasses import dataclass

import torch
from datasets import Dataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)
from torchtitan.tools.logging import logger


def corrupt_last_canvas(
    tokens: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    generator: torch.Generator,
    min_mask_ratio: float = 0.01,
    max_mask_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt the final block and create same-position masked-only labels."""

    if tokens.ndim != 1:
        raise ValueError(f"tokens must be one-dimensional, got shape={tokens.shape}")
    if tokens.numel() < block_size or tokens.numel() % block_size != 0:
        raise ValueError(
            "token count must be a multiple of block_size and contain one canvas"
        )
    if not 0.0 < min_mask_ratio <= max_mask_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 < min <= max <= 1")

    sample = torch.rand((), generator=generator).item()
    mask_ratio = min_mask_ratio + sample * (max_mask_ratio - min_mask_ratio)
    num_masked = min(block_size, max(1, math.ceil(mask_ratio * block_size)))
    canvas_start = tokens.numel() - block_size
    canvas_mask = torch.zeros(block_size, dtype=torch.bool)
    selected = torch.randperm(block_size, generator=generator)[:num_masked]
    canvas_mask[selected] = True

    masked_positions = torch.zeros_like(tokens, dtype=torch.bool)
    masked_positions[canvas_start:] = canvas_mask
    corrupted = tokens.clone()
    corrupted[masked_positions] = mask_token_id
    labels = torch.full_like(tokens, IGNORE_INDEX)
    labels[masked_positions] = tokens[masked_positions]
    return corrupted, labels, masked_positions


class BlockDiffusionDataset(HuggingFaceTextDataset):
    """Text dataset yielding overlapping windows one canvas stride apart."""

    def __init__(
        self,
        *,
        block_size: int,
        mask_token_id: int,
        corruption_seed: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.seq_len < block_size or self.seq_len % block_size != 0:
            raise ValueError(
                "seq_len must be a multiple of block_size and contain one canvas"
            )
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        if not 0.0 < min_mask_ratio <= max_mask_ratio <= 1.0:
            raise ValueError("mask ratios must satisfy 0 < min <= max <= 1")

        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self._corruption_generator = torch.Generator().manual_seed(corruption_seed)

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text,
                    add_bos=True,
                    add_eos=True,
                )
                self._inputs_buffer.extend(sample_tokens)
                self._positions_buffer.extend(
                    index % self.seq_len for index in range(len(sample_tokens))
                )
                self._sample_idx += 1

                while len(self._inputs_buffer) >= self.seq_len:
                    clean_tokens = torch.tensor(
                        self._inputs_buffer[: self.seq_len],
                        dtype=torch.long,
                    )
                    positions = torch.tensor(
                        self._positions_buffer[: self.seq_len],
                        dtype=torch.long,
                    )
                    corrupted, labels, _ = corrupt_last_canvas(
                        clean_tokens,
                        block_size=self.block_size,
                        mask_token_id=self.mask_token_id,
                        generator=self._corruption_generator,
                        min_mask_ratio=self.min_mask_ratio,
                        max_mask_ratio=self.max_mask_ratio,
                    )

                    # A stride of one canvas makes every source block a target
                    # while preserving a fixed seq_len for compile and CP.
                    del self._inputs_buffer[: self.block_size]
                    del self._positions_buffer[: self.block_size]
                    yield {"input": corrupted, "positions": positions}, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break

            self._sample_idx = 0
            self._epoch += 1
            logger.warning(f"Dataset {self.dataset_name} is being re-looped")
            if not isinstance(self._data, Dataset):
                if hasattr(self._data, "set_epoch") and hasattr(self._data, "epoch"):
                    self._data.set_epoch(self._data.epoch + 1)

    def state_dict(self):
        state = super().state_dict()
        state["corruption_generator_state"] = self._corruption_generator.get_state()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        generator_state = state_dict.get("corruption_generator_state")
        if generator_state is not None:
            self._corruption_generator.set_state(generator_state)


class BlockDiffusionDataLoader(ParallelAwareDataloader):
    """TorchTitan dataloader for prefix-plus-canvas diffusion batches."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        block_size: int = 256
        mask_token_id: int = 100
        corruption_seed: int = 42
        min_mask_ratio: float = 0.01
        max_mask_ratio: float = 1.0

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ) -> None:
        del kwargs
        dataset = BlockDiffusionDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            block_size=config.block_size,
            mask_token_id=config.mask_token_id,
            corruption_seed=config.corruption_seed + dp_rank,
            min_mask_ratio=config.min_mask_ratio,
            max_mask_ratio=config.max_mask_ratio,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            batch_size=local_batch_size,
        )
