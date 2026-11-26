# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 block passes each step's explicit denominator into its consumers."""

import pytest
import torch
from torchtitan.config import derive
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import preprocess, tiny_v41_model
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1.model import GraphTrainerDeepSeekV41Model


@pytest.mark.parametrize("graph_trainer", [False, True], ids=["eager", "graph"])
def test_explicit_step_denominator_reaches_all_reference_consumers(monkeypatch, graph_trainer):
    model, config = tiny_v41_model(monkeypatch)
    if graph_trainer:
        model = build_cpu_model(derive(config, GraphTrainerDeepSeekV41Model.Config))
    tokens = torch.arange(128).remainder(32)
    positions = torch.arange(128)
    losses = [module for module in model.modules() if isinstance(module, SparseIndexerLoss)]
    assert losses and all(loss.coeff == 1.0 and loss.mass_weighted and loss.reduce_mesh == "loss" for loss in losses)
    metrics = []
    for count in (128.0, 256.0):
        for loss in losses:
            loss._reset_accumulators()
        inputs, _, kwargs = preprocess(
            model, tokens, tokens, positions, global_routing_token_counts=torch.tensor([count])
        )
        with torch.no_grad():
            model(inputs, **kwargs)
        metrics.append(torch.stack([loss.instance_acc.clone() for loss in losses]))

    assert (metrics[0] > 0).all()
    torch.testing.assert_close(metrics[1] * 2, metrics[0])
