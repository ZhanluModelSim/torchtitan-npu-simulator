# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta, unpatch_device_type_to_meta


def test_patch_device_type_to_meta_rebinds_all_dependent_modules():
    import torchtitan.components.metrics as metrics_mod
    import torchtitan.distributed.parallel_dims as parallel_dims_mod
    import torchtitan.distributed.utils as dist_utils_mod
    import torchtitan.tools.utils as utils_mod

    try:
        patch_device_type_to_meta()
        assert utils_mod.device_type == "meta"
        assert metrics_mod.device_type == "meta"
        assert parallel_dims_mod.device_type == "meta"
        assert dist_utils_mod.device_type == "meta"
        assert utils_mod.device_module.get_device_name() == "Meta_Simulator"
    finally:
        unpatch_device_type_to_meta()


def test_unpatch_restores_original_device_type():
    import torchtitan.tools.utils as utils_mod

    original = utils_mod.device_type
    patch_device_type_to_meta()
    unpatch_device_type_to_meta()
    assert utils_mod.device_type == original


def test_patch_is_idempotent():
    import torchtitan.tools.utils as utils_mod

    try:
        patch_device_type_to_meta()
        patch_device_type_to_meta()  # must not raise, must not double-save originals
        assert utils_mod.device_type == "meta"
    finally:
        unpatch_device_type_to_meta()


def test_stub_device_module_methods_used_by_trainer_and_metrics_do_not_raise():
    try:
        patch_device_type_to_meta()
        import torchtitan.tools.utils as utils_mod

        stub = utils_mod.device_module
        stub.set_device(torch.device("meta:0"))
        assert stub.current_device() == 0
        assert stub.device_count() == 1
        stub.synchronize()
        stub.empty_cache()
        stub.reset_peak_memory_stats()
        props = stub.get_device_properties(torch.device("meta:0"))
        assert props.total_memory > 0  # non-zero: avoids DeviceMemoryMonitor's ZeroDivisionError
        stats = stub.memory_stats(torch.device("meta:0"))
        assert stats["active_bytes.all.peak"] == 0
    finally:
        unpatch_device_type_to_meta()


def test_meta_device_materialization_pattern_used_by_trainer_init_weights():
    # Mirrors Trainer.__init__'s `model.to_empty(device=init_device)` +
    # `nn.init.*` calls (trainer.py:407-411 in the pinned commit) -- this
    # must never raise once device_type/device_module are patched to meta.
    module = nn.Linear(4, 8, device="meta")
    module.to_empty(device="meta:0")
    with torch.no_grad():
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
    assert module.weight.device.type == "meta"


def test_patch_neutralizes_torch_npu_optimizer_device_probe_when_torch_npu_present():
    # Regression test for a real crash found under genuine torch_npu (CANN
    # container): torch_npu monkeypatches
    # torch.optim.optimizer._get_foreach_kernels_supported_devices to cache
    # a real device name the first time any optimizer runs
    # _default_to_fused_or_foreach(); filling that cache calls
    # torch_npu.npu.current_device(), which unconditionally runs a real
    # aclInit() and raises with no NPU hardware present. Skipped (not
    # failed) when torch_npu is not installed, e.g. this repo's CPU-only
    # unit-test sandbox.
    torch_npu = pytest.importorskip("torch_npu", reason="torch_npu-specific regression test")
    import torch_npu.utils._optim as npu_optim

    original = npu_optim._device_name
    npu_optim._device_name = None
    try:
        patch_device_type_to_meta()
        assert npu_optim._device_name is not None
    finally:
        unpatch_device_type_to_meta()
        npu_optim._device_name = original


def test_patch_redirects_swap_optimizer_get_device_info_to_meta_stub():
    # Regression test for a real crash found via the 16-layer
    # DeepSeek-V4-Pro smoke run (SwapOptimizersContainer.__init__, used
    # because the acceptance config sets optimizer.swap_optimizer=True):
    # torchtitan_npu.patches.optimizer.swap_optimizer.get_torch_device()
    # calls torchtitan.tools.utils.get_device_info() FRESH each time
    # (independently re-detecting the "live" device via
    # _get_available_device_type()), bypassing the module-level
    # device_type/device_module globals patched above entirely -- so
    # `get_torch_device().Stream()` resolved real torch.cuda.Stream(),
    # which unconditionally requires real CUDA hardware. Skipped (not
    # failed) when torch_npu is not installed.
    pytest.importorskip("torch_npu", reason="torch_npu-specific regression test")
    import torchtitan_npu.patches.optimizer.swap_optimizer as swap_optimizer_mod

    try:
        patch_device_type_to_meta()
        device_type, device_module = swap_optimizer_mod.get_device_info()
        assert device_type == "meta"
        stream = device_module.Stream()
        stream.synchronize()  # must not touch real hardware
    finally:
        unpatch_device_type_to_meta()


def test_patch_forces_mhc_converter_to_report_a5_device_type():
    # Regression test for a real crash found via the 16-layer
    # DeepSeek-V4-Pro smoke run (first forward pass reaching an mHC
    # layer): MHCPreConverter/MHCPostConverter select a Triton-JIT kernel
    # path unless get_npu_device_type() == "A5", and Triton kernels are
    # JIT-compiled AND LAUNCHED on a real accelerator at call time (no
    # meta-device mode exists for a Triton launch) -- crashing with a real
    # aclInit() hardware-init error. Scoped to mhc_prepost's own
    # by-value-imported name only (not the shared definition, and not
    # other converters' own by-value imports of the same function), so
    # other converters' already-working non-"A5" branches are undisturbed.
    # Skipped (not failed) when torch_npu is not installed.
    pytest.importorskip("torch_npu", reason="torch_npu-specific regression test")
    import torchtitan_npu.converters.kernels.mhc_prepost as mhc_prepost_mod
    import torchtitan_npu.tools.device as device_mod

    try:
        patch_device_type_to_meta()
        assert mhc_prepost_mod.get_npu_device_type() == "A5"
    finally:
        unpatch_device_type_to_meta()
        # the shared definition (used by npu_smla.py/mx_capability_check.py)
        # must be untouched by this scoped patch
        assert mhc_prepost_mod.get_npu_device_type is device_mod.get_npu_device_type


def test_patch_neutralizes_fsdp_meta_param_validation():
    # Regression test for a real crash found via the 16-layer
    # DeepSeek-V4-Pro smoke run (first forward pass through an
    # FSDP2-wrapped model): FSDPParamGroup._lazy_init() unconditionally
    # calls _validate_no_meta_params(), which raises RuntimeError if any
    # sharded parameter's device.type == "meta" -- always true under this
    # simulator by design (every parameter lives on meta forever; no real
    # memory is ever allocated). Pure PyTorch (torch.distributed.fsdp), so
    # this test runs regardless of torch_npu availability.
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    original = FSDPParamGroup._validate_no_meta_params
    try:
        patch_device_type_to_meta()
        # must not raise, regardless of self/sharded-param state:
        FSDPParamGroup._validate_no_meta_params(None)
    finally:
        unpatch_device_type_to_meta()
        assert FSDPParamGroup._validate_no_meta_params is original


def test_patch_registers_torch_meta_as_a_device_accessor_module():
    # Regression test for a real crash found via CANN-container validation:
    # torch.distributed.device_mesh._get_device_handle(device_type) does
    # `getattr(torch, device_type, None)` to resolve a device module (the
    # same mechanism used for "cuda"/"npu"/"xpu"), and FSDP2's
    # _get_device_from_mesh calls `.current_device()` on the result
    # unconditionally for any non-"cpu" device_type -- including "meta",
    # which torch does not register a module for by default, previously
    # crashing with `AttributeError: 'NoneType' object has no
    # attribute 'current_device'`.
    had_meta_before = hasattr(torch, "meta")
    try:
        patch_device_type_to_meta()
        assert torch.meta.current_device() == 0
        assert torch.meta.is_available() is True
        assert torch.meta.is_initialized() is True
        with torch.meta.stream(None):
            pass
        event = torch.meta.Event()
        event.record()
        event.synchronize()

        # FSDP2's foreach_all_gather/foreach_reduce (exercised on every
        # forward/backward pass) construct Stream()s via device_handle
        # (our torch.meta stub) with a `priority=` kwarg, then call
        # record_event()/wait_event()/wait_stream() on them and
        # device_handle.current_stream() directly -- found via the
        # 16-layer DeepSeek-V4-Pro smoke run.
        stream = torch.meta.Stream(priority=-1)
        recorded_event = stream.record_event()
        assert isinstance(recorded_event, torch.meta.Event)
        stream.wait_event(recorded_event)
        stream.wait_stream(torch.meta.current_stream())
        assert torch.meta.current_stream().query() is True
    finally:
        unpatch_device_type_to_meta()
        assert hasattr(torch, "meta") == had_meta_before

