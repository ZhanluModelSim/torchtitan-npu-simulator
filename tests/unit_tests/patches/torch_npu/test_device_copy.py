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
_PATCH_PATH = _REPO.parent / "torchtitan_npu" / "patches" / "torch_npu" / "device_copy.py"


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
def device_copy_patch(monkeypatch, tmp_path):
    """Load the patch module without importing the package-level plugin chain."""
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu = types.SimpleNamespace(is_available=lambda: False)
    fake_torch_npu.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    saved_modules = {}
    for name in list(sys.modules):
        if name.startswith("torch_npu._inductor"):
            saved_modules[name] = sys.modules.pop(name)
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
    module._original_device_copy_create = lambda x, device, non_blocking, *args, **kwargs: _FakeDeviceCopy()
    yield module
    for name in list(sys.modules):
        if name.startswith("torch_npu._inductor"):
            del sys.modules[name]
    sys.modules.update(saved_modules)


def _write_backend(root, *, common_source=None):
    inductor = root / "_inductor"
    inductor.mkdir()
    (inductor / "__init__.py").write_text("raise ImportError('inductor initialization')\n")
    if common_source is None:
        return
    lowering = inductor / "ascendc" / "lowering"
    lowering.mkdir(parents=True)
    for package in (inductor, inductor / "ascendc", lowering):
        (package / "__init__.py").write_text("")
    (lowering / "common.py").write_text(common_source)


def test_apply_absent_ascendc_does_not_initialize_inductor(device_copy_patch, monkeypatch, tmp_path):
    _write_backend(tmp_path)
    monkeypatch.setattr(device_copy_patch.torch_npu.npu, "is_available", lambda: True)
    original = _FakeDeviceCopy.create.__func__

    device_copy_patch.apply()

    assert _FakeDeviceCopy.create.__func__ is original
    assert "torch_npu._inductor" not in sys.modules


def test_apply_present_ascendc_installs_device_copy_patch(device_copy_patch, monkeypatch, tmp_path):
    _write_backend(
        tmp_path,
        common_source=(
            "class _LoweringGuard:\n"
            "    calls = []\n"
            "    @classmethod\n"
            "    def support(cls, *args): cls.calls.append(args)\n"
            "def exclude(): return (torch.float32,)\n"
            "import torch\n"
        ),
    )
    monkeypatch.setattr(device_copy_patch.torch_npu.npu, "is_available", lambda: True)
    monkeypatch.setattr(_FakeDeviceCopy, "create", _FakeDeviceCopy.__dict__["create"])

    device_copy_patch.apply()

    from torch_npu._inductor.ascendc.lowering.common import _LoweringGuard

    assert _FakeDeviceCopy.create.__func__ is device_copy_patch._patched_device_copy_create
    assert _LoweringGuard.calls == [
        (device_copy_patch.torch.ops.prims.device_put, (device_copy_patch.torch.float32, device_copy_patch.torch.int64))
    ]
    result = _FakeDeviceCopy.create(_FakeNode("npu"), types.SimpleNamespace(type="cpu"), True)
    assert result.layout.is_pinned


@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
def test_apply_present_backend_propagates_import_error(device_copy_patch, monkeypatch, tmp_path, error_type):
    _write_backend(tmp_path, common_source=f"raise {error_type.__name__}('backend dependency failed')\n")
    monkeypatch.setattr(device_copy_patch.torch_npu.npu, "is_available", lambda: True)

    with pytest.raises(error_type, match="backend dependency failed"):
        device_copy_patch.apply()


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
def test_device_copy_pins_only_async_npu_to_cpu(device_copy_patch, source, destination, non_blocking, expected_pinned):
    result = device_copy_patch._patched_device_copy_create(
        _FakeDeviceCopy,
        _FakeNode(source),
        types.SimpleNamespace(type=destination),
        non_blocking,
    )
    assert result.layout.is_pinned is expected_pinned
