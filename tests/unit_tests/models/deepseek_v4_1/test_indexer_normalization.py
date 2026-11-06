# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch

from torchtitan_npu.models.deepseek_v4_1 import config_registry
from torchtitan_npu.models.deepseek_v4_1.indexer import FULL, Selector


class _FusedStandIn(Selector):
    """A stand-in for a fused port: the trainer discovers those by class, not by a flag.

    The real ports subclass :class:`Selector` (and the reference selector is the only module
    whose exact class is ``Selector``), which is what the engine keys the step denominator on.
    """


def test_step_denominator_counts_queries_across_accumulation_and_data_ranks(monkeypatch):
    config = Selector.Config(mode=FULL, compress_ratio=2, num_index_heads=2, index_head_dim=4, index_topk=4)
    selector, reference = _FusedStandIn(config), Selector(config)

    engine = object.__new__(config_registry.DeepSeekV41TrainingEngine)
    engine.model_parts = [torch.nn.ModuleList([selector, reference])]
    engine.device = torch.device("cpu")
    mesh = object()
    requested_meshes = []
    engine.parallelism_context = SimpleNamespace(
        dp_enabled=True, get_mesh=lambda name: requested_meshes.append(name) or mesh
    )
    from torchtitan.components.data import TokenizedTrainingMicrobatch

    groups = [
        [
            TokenizedTrainingMicrobatch(
                input=torch.zeros(n, dtype=torch.int64),
                labels=torch.full((n,), -100),
                positions=torch.arange(n),
                padding_mask=torch.ones(n, dtype=torch.bool),
                num_valid_tokens=0,
            )
        ]
        for n in (4, 8)
    ]
    reductions = []

    def reduce_count(value, group):
        assert group is mesh
        reductions.append(value.clone())
        return value + 20

    def run_forward_backward(self, *, microbatch_groups, global_valid_tokens):
        assert microbatch_groups is groups
        torch.testing.assert_close(selector.num_global_queries, torch.tensor(expected_queries))
        return "forward-backward-result"

    monkeypatch.setattr(config_registry.dist_utils, "dist_sum_tensor", reduce_count)
    monkeypatch.setattr(config_registry.NPUTrainingEngine, "forward_backward", run_forward_backward)
    expected_queries = 32.0
    assert engine.forward_backward(microbatch_groups=groups, global_valid_tokens=0) == "forward-backward-result"
    # Cached selectors must still receive the new denominator on each step.
    groups = groups[:1]
    expected_queries = 24.0
    assert engine.forward_backward(microbatch_groups=groups, global_valid_tokens=0) == "forward-backward-result"
    assert reference.num_global_queries is None
    assert requested_meshes == ["dp", "dp"]
    assert len(reductions) == 2
    torch.testing.assert_close(reductions[0], torch.tensor(12.0))
    torch.testing.assert_close(reductions[1], torch.tensor(4.0))
