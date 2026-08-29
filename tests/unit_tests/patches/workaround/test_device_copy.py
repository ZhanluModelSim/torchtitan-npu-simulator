# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the NPU-to-CPU DeviceCopy workaround.

The test exercises the small policy seam without constructing a real Inductor
graph or allocating an NPU tensor.  The workaround must pin only asynchronous
NPU-to-CPU copies; all other copies must retain the layout selected by
Inductor.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_PATCH_PATH = _REPO.parent / "torchtitan_npu" / "patches" / "workaround" / "device_copy.py"


class _FakeLayout:
    def __init__(self):
        self.is_pinned = False


class _FakeDeviceCopy:
    def __init__(self):
        self.layout = _FakeLayout()

    @classmethod
    def create(cls, x, device, non_blocking):
        return cls()


class _FakeNode:
    def __init__(self, device_type):
        self._device = None if device_type is None else types.SimpleNamespace(type=device_type)

    def get_device(self):
        return self._device


@pytest.fixture
def device_copy_patch(monkeypatch):
    """Load the patch module without importing the package-level plugin chain."""
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    fake_inductor_ir = types.SimpleNamespace(DeviceCopy=_FakeDeviceCopy, Layout=_FakeLayout)
    fake_inductor = types.ModuleType("torch._inductor")
    fake_inductor.ir = fake_inductor_ir
    monkeypatch.setitem(sys.modules, "torch._inductor", fake_inductor)
    monkeypatch.setitem(sys.modules, "torch._inductor.ir", fake_inductor_ir)

    spec = importlib.util.spec_from_file_location("device_copy_workaround_test", _PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.inductor_ir = fake_inductor_ir
    module._original_device_copy_create = (
        lambda x, device, non_blocking, *args, **kwargs: _FakeDeviceCopy()
    )
    return module


@pytest.mark.parametrize(
    ("source", "destination", "non_blocking", "expected_pinned"),
    [
        pytest.param("npu", "cpu", True, True, id="async-npu-to-cpu"),
        pytest.param("npu", "cpu", False, False, id="sync-npu-to-cpu"),
        pytest.param("cpu", "cpu", True, False, id="cpu-to-cpu"),
        pytest.param("npu", "npu", True, False, id="npu-to-npu"),
        pytest.param(None, "cpu", True, False, id="unknown-source"),
    ],
)
def test_device_copy_pins_only_async_npu_to_cpu(
    device_copy_patch, source, destination, non_blocking, expected_pinned
):
    result = device_copy_patch._patched_device_copy_create(
        _FakeDeviceCopy,
        _FakeNode(source),
        types.SimpleNamespace(type=destination),
        non_blocking,
    )
    assert result.layout.is_pinned is expected_pinned
