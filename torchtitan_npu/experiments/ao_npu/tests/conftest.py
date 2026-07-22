# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import torch
import torch_npu  # noqa: F401


def pytest_configure(config):
    """Apply torchtitan_npu patches at session start.

    Replicates the hook in ``tests/conftest.py`` since this test tree is no
    longer under ``tests/`` and would otherwise not inherit it.
    """
    import torchtitan_npu  # noqa: F401


@pytest.fixture(autouse=True)
def _init_test():
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch._dynamo.reset()
    torch.compiler.reset()

    gc.collect()
    torch.npu.empty_cache()


@pytest.fixture
def mock_distributed_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12355")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
