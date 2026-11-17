# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Private-interface drift fails before installing capture state or patches."""

import inspect

import pytest
import torch
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torchtitan_simulator import meta_env
from torchtitan_simulator.compat.fsdp import validate_fsdp_capture
from torchtitan_simulator.runtime import meta_runtime


def test_native_fsdp_preflight_reports_stack_without_mutating_runtime():
    original = FSDPParamGroup.unshard
    multithreading = torch.autograd.is_multithreading_enabled()
    report = validate_fsdp_capture()
    assert report["software"]["packages"]["torch"]["path"] == torch.__file__
    assert report["software"]["packages"]["torch"]["version"] == torch.__version__
    assert any(name.endswith("FSDPParamGroup.unshard") for name in report["signatures"])
    assert FSDPParamGroup.unshard is original
    assert torch.autograd.is_multithreading_enabled() == multithreading
    assert not meta_env._is_meta_simulation


@pytest.mark.parametrize("drift", ["missing", "required_argument", "keyword_only", "descriptor"])
def test_runtime_rejects_fsdp_drift_before_patch_installation(monkeypatch, drift):
    import torch_npu

    original_rotary = torch_npu.npu_rotary_mul
    original_device = torch.get_device_module
    multithreading = torch.autograd.is_multithreading_enabled()
    if drift == "missing":
        monkeypatch.delattr(FSDPParamGroup, "wait_for_unshard")
        symbol = "wait_for_unshard"
    elif drift == "required_argument":
        monkeypatch.setattr(FSDPParamGroup, "reshard", lambda self, required: None)
        symbol = "reshard"
    elif drift == "keyword_only":
        monkeypatch.setattr(FSDPParamGroup, "unshard", lambda self, *, async_op=False: None)
        symbol = "unshard"
    else:
        method = FSDPParamGroup._prefetch_unshard
        monkeypatch.setattr(FSDPParamGroup, "_prefetch_unshard", method)
        symbol = "_prefetch_unshard"
    with pytest.raises(RuntimeError, match=symbol) as error, meta_runtime():
        pytest.fail("An incompatible runtime must not activate capture")
    assert "software stack" in str(error.value)
    assert torch.__version__ in str(error.value)
    assert torch_npu.npu_rotary_mul is original_rotary
    assert torch.get_device_module is original_device
    assert torch.autograd.is_multithreading_enabled() == multithreading
    assert not meta_env._is_meta_simulation


def test_optional_signature_extension_is_not_rejected(monkeypatch):
    def extended_reshard(self, *, optional_policy=None):
        raise AssertionError("Preflight must not invoke native hooks")

    monkeypatch.setattr(FSDPParamGroup, "reshard", extended_reshard)
    report = validate_fsdp_capture()
    assert str(inspect.signature(extended_reshard)) in report["signatures"].values()


def test_runtime_preserves_preflight_evidence_in_capabilities():
    with meta_runtime() as runtime:
        report = runtime.capabilities["fsdp_capture_compatibility"]
        assert report["check"] == "fsdp-call-boundaries-v1"
        assert report["software"]["packages"]["torch"]["path"] == torch.__file__
