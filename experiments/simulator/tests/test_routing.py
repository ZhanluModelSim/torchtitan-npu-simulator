# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan_simulator.compat.routing import balanced_splits


@pytest.mark.parametrize(
    "rank,expected_receives", [(0, [8, 8]), (1, [7, 7])], ids=["first-partition", "last-partition"]
)
def test_balanced_splits_match_uneven_round_robin_expert_assignments(rank, expected_receives):
    # Five tokens, three choices, four experts: [4, 4, 4, 3] routed rows.
    expert_ids = torch.arange(15) % 4
    histogram = torch.bincount(expert_ids, minlength=4).reshape(2, 2)

    sends, receives = balanced_splits(5, 3, 4, 2, rank)

    assert sends == histogram.sum(1).tolist() == [8, 7]
    assert receives == expected_receives
