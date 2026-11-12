# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from copy import deepcopy

import numpy as np
import pytest
import torch

from tests.unit_tests.models.common.text_data_utils import IntegerTextTokenizer, aligned_text_config


def test_grain_loader_keeps_document_alignment_labels_and_positions(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text('{"text":"3 4"}\n{"text":"5"}\n')
    config = aligned_text_config(dataset, alignment=4)
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=IntegerTextTokenizer(),
        max_context_length=4,
        num_tokens_per_microbatch=8,
    )

    try:
        microbatch = next(iter(loader))
    finally:
        loader.close()

    torch.testing.assert_close(microbatch.input, torch.tensor([1, 3, 4, 0, 1, 5, 0, 0]))
    torch.testing.assert_close(microbatch.labels, torch.tensor([3, 4, 2, -100, 5, 2, -100, -100]))
    torch.testing.assert_close(microbatch.positions, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    torch.testing.assert_close(
        microbatch.padding_mask, torch.tensor([False, False, False, True, False, False, True, True])
    )
    assert int(microbatch.loss_token_counts) == 5


@pytest.mark.parametrize("token_budget", (4, 8), ids=("one-context", "two-contexts"))
def test_long_document_positions_follow_context_boundaries(tmp_path, token_budget):
    dataset = tmp_path / "data.json"
    dataset.write_text('{"text":"3"}\n{"text":"4 5 6 7 8"}\n')
    loader = aligned_text_config(dataset).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=IntegerTextTokenizer(),
        max_context_length=4,
        num_tokens_per_microbatch=token_budget,
    )
    try:
        batches = list(loader)
    finally:
        loader.close()

    expected_positions = [[0, 1, 0, 1], [0, 1, 0, 1]] if token_budget == 4 else [[0, 1, 0, 1, 2, 3, 0, 1]]
    expected_inputs = [[1, 3, 1, 4], [5, 6, 7, 8]] if token_budget == 4 else [[1, 3, 1, 4, 5, 6, 7, 8]]
    expected_labels = [[3, 2, 4, 5], [6, 7, 8, 2]] if token_budget == 4 else [[3, 2, 4, 5, 6, 7, 8, 2]]
    assert len(batches) == len(expected_positions)
    for microbatch, positions, tokens, targets in zip(
        batches, expected_positions, expected_inputs, expected_labels, strict=True
    ):
        assert microbatch.positions.tolist() == positions
        assert microbatch.input.tolist() == tokens
        assert microbatch.labels.tolist() == targets


@pytest.mark.parametrize("rank", (0, 1), ids=("dp-rank-0", "dp-rank-1"))
def test_map_loader_partitions_shards_and_restores_pending_pack(tmp_path, rank):
    dataset = tmp_path / "data.json"
    dataset.write_text("".join(f'{{"text":"{token}"}}\n' for token in range(10, 18)))
    config = aligned_text_config(dataset, repeat=True)
    kwargs = dict(
        dp_world_size=2,
        dp_rank=rank,
        tokenizer=IntegerTextTokenizer(),
        max_context_length=6,
        num_tokens_per_microbatch=6,
    )
    with ExitStack() as cleanup:
        original = config.build(**kwargs)
        cleanup.callback(original.close)
        restored = config.build(**kwargs)
        cleanup.callback(restored.close)
        iterator = iter(original)
        prefix = [next(iterator), next(iterator)]
        restored.load_state_dict(deepcopy(original.state_dict()))
        tail = [next(iterator), next(iterator)]
        resumed = iter(restored)
        for expected in tail:
            actual = next(resumed)
            expected_inputs, actual_inputs = expected.as_input_dict(), actual.as_input_dict()
            assert int(actual.loss_token_counts) == int(expected.loss_token_counts)
            assert actual_inputs.keys() == expected_inputs.keys()
            for key in expected_inputs:
                torch.testing.assert_close(
                    torch.as_tensor(actual_inputs[key]), torch.as_tensor(expected_inputs[key]), rtol=0, atol=0
                )

    shard = np.arange(10 + rank * 4, 14 + rank * 4)
    expected = np.tile(shard, 3)
    actual = []
    for microbatch in (*prefix, *tail):
        actual.extend(microbatch.input[1::2].tolist())
    assert actual == expected.tolist()
