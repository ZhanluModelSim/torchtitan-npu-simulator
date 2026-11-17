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
    whose exact class is ``Selector``), which is what ``TrainerEx`` keys the step denominator on.
    """


def test_step_denominator_counts_queries_across_accumulation_and_data_ranks(monkeypatch):
    config = Selector.Config(
        mode=FULL, compress_ratio=2, num_index_heads=2, index_head_dim=4, index_topk=4
    )
    selector, reference = _FusedStandIn(config), Selector(config)

    def initialize(self, config):
        self.model_parts = [torch.nn.ModuleList([selector, reference])]
        self.anticipatory_schedule = None

    monkeypatch.setattr(config_registry.TrainerEx, "__init__", initialize)
    job = SimpleNamespace(
        override=SimpleNamespace(imports=["torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc"])
    )
    trainer = config_registry.DeepSeekV41Trainer(job)
    trainer.device = torch.device("cpu")
    mesh = object()
    trainer.parallel_dims = SimpleNamespace(dp_enabled=True, get_mesh=lambda name: mesh)
    batches = [({"input": torch.zeros(1, n)}, torch.full((1, n), -100)) for n in (4, 8)]
    reductions = []

    def reduce_count(value, group):
        assert group is mesh
        reductions.append(value.clone())
        return value + 20  # Another data rank contributes 20 queries.

    def run_step(self, data_iterator):
        prefetched = list(data_iterator)
        for inputs, labels in prefetched:
            self.forward_backward_step(inputs, labels=labels, global_valid_tokens=torch.tensor(0))

    def run_microbatch(self, input_dict, **kwargs):
        assert "input" in input_dict
        torch.testing.assert_close(selector.num_global_queries, torch.tensor(32.0))

    monkeypatch.setattr(config_registry.dist_utils, "dist_sum_tensor", reduce_count)
    monkeypatch.setattr(config_registry.GradientClippingTrainer, "train_step", run_step)
    monkeypatch.setattr(config_registry.TrainerEx, "forward_backward_step", run_microbatch)
    trainer.train_step(iter(batches))
    assert reference.num_global_queries is None
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0], torch.tensor(12.0))
