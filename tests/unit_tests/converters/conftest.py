# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torchtitan.distributed import ParallelDims


@pytest.fixture
def mock_job_config():
    def _create(model_name: str = "llama3", converters: list = None):
        config = MagicMock()
        config.model.name = model_name
        config.model.converters = converters or []
        return config

    return _create


@pytest.fixture
def mock_parallel_dims():
    return MagicMock(spec=ParallelDims)


@pytest.fixture
def simple_model():
    class SimpleRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = 1e-6

    class SimpleMethod(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = SimpleRMSNorm(64)
            self.linear = nn.Linear(64, 64)

    return SimpleMethod()


@pytest.fixture
def prepare_shared_expert_for_conversion():
    def _prepare(model, shared):
        linears = (shared.w1, shared.w2, shared.w3)
        shared.swiglu_limit = 2.0
        marker = torch.tensor(1.0)
        shared.register_buffer("conversion_marker", marker)
        shared.eval()
        hook_calls = []
        shared.register_forward_hook(lambda *args: hook_calls.append(True))
        return SimpleNamespace(
            linears=linears,
            marker=marker,
            hook_calls=hook_calls,
            state_dict_keys=tuple(model.state_dict()),
        )

    return _prepare
