# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torchtitan.components.data import TokenizedTrainingMicrobatch

from torchtitan_npu.models.deepseek_v4_1 import trainer as v41_trainer
from torchtitan_npu.models.deepseek_v4_1.indexer import FULL, Selector
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import asc


def test_step_denominator_counts_queries_across_accumulation_and_data_ranks(monkeypatch):
    config = Selector.Config(mode=FULL, compress_ratio=2, num_index_heads=2, index_head_dim=4, index_topk=4)
    selector, reference = asc(config).build(), Selector(config)

    engine = object.__new__(v41_trainer.DeepSeekV41TrainingEngine)
    engine.model_parts = [torch.nn.ModuleList([selector, reference])]
    engine.device = torch.device("cpu")
    mesh = object()
    requested_meshes = []
    engine.parallelism_context = SimpleNamespace(
        dp_enabled=True, get_mesh=lambda name: requested_meshes.append(name) or mesh
    )
    groups = [
        [
            TokenizedTrainingMicrobatch(
                input=torch.zeros(n, dtype=torch.int64),
                labels=torch.full((n,), -100),
                positions=torch.arange(n),
                padding_mask=torch.ones(n, dtype=torch.bool),
                loss_token_counts=torch.tensor(0),
                routing_token_counts=torch.tensor([0]),
            )
        ]
        for n in (4, 8)
    ]
    reductions = []
    loss_counts, routing_counts = torch.tensor(0), torch.tensor([0])

    def reduce_count(value, group):
        assert group is mesh
        reductions.append(value.clone())
        return value + 20

    def run_forward_backward(self, *, microbatch_groups, global_loss_token_counts, global_routing_token_counts):
        assert microbatch_groups is groups
        assert global_loss_token_counts is loss_counts
        assert global_routing_token_counts is routing_counts
        torch.testing.assert_close(selector.num_global_queries, torch.tensor(expected_queries))
        return "forward-backward-result"

    monkeypatch.setattr(v41_trainer.dist_utils, "dist_sum_tensor", reduce_count)
    monkeypatch.setattr(v41_trainer.TrainingEngineEx, "forward_backward", run_forward_backward)
    expected_queries = 32.0
    assert (
        engine.forward_backward(
            microbatch_groups=groups,
            global_loss_token_counts=loss_counts,
            global_routing_token_counts=routing_counts,
        )
        == "forward-backward-result"
    )
    # Cached selectors must still receive the new denominator on each step.
    groups = groups[:1]
    expected_queries = 24.0
    assert (
        engine.forward_backward(
            microbatch_groups=groups,
            global_loss_token_counts=loss_counts,
            global_routing_token_counts=routing_counts,
        )
        == "forward-backward-result"
    )
    assert reference.num_global_queries is None
    assert requested_meshes == ["dp", "dp"]
    assert len(reductions) == 2
    torch.testing.assert_close(reductions[0], torch.tensor(12.0))
    torch.testing.assert_close(reductions[1], torch.tensor(4.0))
