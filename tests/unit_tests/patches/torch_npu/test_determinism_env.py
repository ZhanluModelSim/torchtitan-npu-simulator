# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the NPU deterministic-mode environment bridge."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "torchtitan_npu"
    / "patches"
    / "torch_npu"
    / "determinism.py"
)


@pytest.fixture
def determinism_patch(monkeypatch):
    original_global = torch.use_deterministic_algorithms
    spec = importlib.util.spec_from_file_location("determinism_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def fake_original(mode, *args, **kwargs):
        calls.append((mode, args, kwargs))

    module.original_use_deterministic_algorithms = fake_original
    monkeypatch.setattr(torch, "use_deterministic_algorithms", original_global)
    return module, calls


def test_enable_sets_npu_deterministic_environment(determinism_patch, monkeypatch):
    module, calls = determinism_patch
    for key in module._NPU_DETERMINISTIC_ENV:
        monkeypatch.delenv(key, raising=False)

    module.patched_use_deterministic_algorithms(True, warn_only=True)

    assert {key: os.environ.get(key) for key in module._NPU_DETERMINISTIC_ENV} == {
        "HCCL_DETERMINISTIC": "true",
        "CLOSE_MATMUL_K_SHIFT": "1",
    }
    assert calls == [(True, (), {"warn_only": True})]


def test_disable_removes_npu_deterministic_environment(determinism_patch, monkeypatch):
    module, calls = determinism_patch
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", "1")

    module.patched_use_deterministic_algorithms(False)

    for key in module._NPU_DETERMINISTIC_ENV:
        assert os.environ.get(key) is None
    assert calls == [(False, (), {})]
