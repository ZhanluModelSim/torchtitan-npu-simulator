# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def peak_flops_ext():
    """Load the peak-flops extension against the real torchtitan checkout.

    Loading the module file directly (like ``test_ep_process_group.py``)
    avoids the NPU-bound package bootstrap while still exercising the
    ``torchtitan.tools.utils.get_peak_flops`` seam on the real upstream
    module.
    """
    from torchtitan.tools import utils as titan_utils

    original = titan_utils.get_peak_flops

    path = Path(__file__).resolve().parents[5] / "torchtitan_npu" / "extensions" / "torchtitan" / "tools" / "utils.py"
    spec = importlib.util.spec_from_file_location("peak_flops_test_impl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    yield module, titan_utils, original

    sys.modules.pop(spec.name, None)
    titan_utils.get_peak_flops = original


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("Ascend910_9362", 353.8944e12),
        ("Ascend910_9392", 353.8944e12),
        ("Ascend910B1", 373.88e12),
        ("Ascend910B2", 353.8944e12),
        ("Ascend910B3", 294.912e12),
        ("Ascend910B4", 245.76e12),
        ("Ascend950PR", 432e12),
        ("Ascend950PR_9582", 432e12),
        ("Ascend950DT_9591", 486e12),
        ("Ascend950DT_9582", 432e12),
    ],
)
def test_ascend_peak_flops_bypasses_upstream(peak_flops_ext, monkeypatch, device_name, expected):
    module, titan_utils, _ = peak_flops_ext

    def unexpected_upstream_call(device_name):
        raise AssertionError(f"unexpected upstream lookup for {device_name}")

    monkeypatch.setattr(
        module,
        "_upstream_get_peak_flops",
        unexpected_upstream_call,
    )

    assert titan_utils.get_peak_flops(device_name) == expected


def test_unknown_device_delegates_to_upstream(peak_flops_ext, monkeypatch):
    module, titan_utils, _ = peak_flops_ext
    calls = []

    def upstream_get_peak_flops(device_name):
        calls.append(device_name)
        return 1.0

    monkeypatch.setattr(
        module,
        "_upstream_get_peak_flops",
        upstream_get_peak_flops,
    )

    assert titan_utils.get_peak_flops("H100") == 1.0
    assert calls == ["H100"]


def test_metrics_processor_uses_installed_peak_flops_hook(peak_flops_ext):
    module, titan_utils, _ = peak_flops_ext
    import torchtitan.components.metrics as titan_metrics

    assert titan_utils.get_peak_flops is module.get_peak_flops
    assert titan_metrics.utils.get_peak_flops is module.get_peak_flops
