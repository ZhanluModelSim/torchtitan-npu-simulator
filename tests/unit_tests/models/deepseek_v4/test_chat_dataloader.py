# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real recipe selection and Grain packing with a fixed chat-encoding oracle."""

import json
from contextlib import ExitStack
from copy import deepcopy

import grain
import pytest
import torch
from torchtitan.components.data import FirstFitPackingConfig, GrainDataLoader, SingleDatasetConfig
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4EncoderConfig
from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash_sft


class _StubEncoder:
    """Each integer prompt identifies one six-token conversation."""

    def encode_messages_with_assistant_mask(self, messages, tokenizer, eos_id):
        base = int(messages[0]["content"])
        return "", list(range(base, base + 6)), [False, False, True, True, True, True]


@pytest.fixture
def sft_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(DSV4EncoderConfig, "build", lambda self, path: _StubEncoder())
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": str(base)}, {"role": "assistant", "content": "ok"}]})
            + "\n"
            for base in range(10, 90, 10)
        )
    )
    config = deepseek_v4_flash_sft(dataset_path=str(path))
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.shuffle = False
    config.dataloader.read_options = grain.ReadOptions(num_threads=2, prefetch_buffer_size=4)
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, FirstFitPackingConfig)
    assert isinstance(config.dataloader.dataset.dataset, SingleDatasetConfig)
    assert isinstance(config.dataloader.dataset.dataset.processor, DeepSeekV4ChatProcessor.Config)
    assert config.dataloader.dataset.dataset.processor.per_doc_alignment == 128
    return config.dataloader


def test_sft_recipe_builds_aligned_supervised_batches(sft_recipe):
    loader = sft_recipe.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/deepseek_v3"),
        max_context_length=128,
        num_tokens_per_microbatch=256,
    )
    try:
        microbatch = next(iter(loader))
    finally:
        loader.close()

    assert microbatch.input.tolist() == list(range(10, 15)) + [0] * 123 + list(range(20, 25)) + [0] * 123
    assert microbatch.labels.tolist() == [-100, 12, 13, 14, 15] + [-100] * 123 + [-100, 22, 23, 24, 25] + [-100] * 123
    assert microbatch.positions.tolist() == list(range(128)) * 2
    assert microbatch.padding_mask.tolist() == ([False] * 5 + [True] * 123) * 2
    assert int(microbatch.loss_token_counts) == 8


@pytest.mark.parametrize("rank", (0, 1), ids=("dp-rank-0", "dp-rank-1"))
def test_sft_grain_restores_packing_and_prefetch_across_epoch(sft_recipe, rank):
    kwargs = dict(
        dp_world_size=2,
        dp_rank=rank,
        tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/deepseek_v3"),
        max_context_length=128,
        num_tokens_per_microbatch=384,
    )
    with ExitStack() as cleanup:
        original = sft_recipe.build(**kwargs)
        cleanup.callback(original.close)
        restored = sft_recipe.build(**kwargs)
        cleanup.callback(restored.close)
        first = next(iter(original))
        assert first.input[::128].tolist() == ([10, 20, 30] if rank == 0 else [50, 60, 70])
        state = json.loads(json.dumps(deepcopy(original.state_dict())))
        restored.load_state_dict(state)
        for _ in range(4):
            expected = next(iter(original))
            actual = next(iter(restored))
            expected_inputs, inputs = expected.as_input_dict(), actual.as_input_dict()
            assert int(actual.loss_token_counts) == int(expected.loss_token_counts)
            assert inputs.keys() == expected_inputs.keys()
            for key in inputs:
                torch.testing.assert_close(
                    torch.as_tensor(inputs[key]), torch.as_tensor(expected_inputs[key]), rtol=0, atol=0
                )
