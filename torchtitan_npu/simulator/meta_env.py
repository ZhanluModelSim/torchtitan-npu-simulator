# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forces torchtitan's device_type/device_module globals to `"meta"` so
`Trainer.__init__` builds, materializes (`to_empty`), and initializes
(`init_weights`) its model entirely on the meta device -- no real memory is
ever allocated. See design doc §5.1 for the verification this relies on."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch

# Sentinel: distinguishes "attribute did not exist before patching" (must be
# deleted on unpatch) from "attribute existed and was None" (must be restored
# to None on unpatch).
_MISSING = object()


class _MetaDeviceModule:
    """Minimal stand-in for `torch.cuda`/`torch_npu`, covering every method
    actually called on `device_module` by `torchtitan.trainer`,
    `torchtitan.components.metrics`, and `torchtitan.distributed.utils`
    (verified against the pinned torchtitan commit -- see design doc §5.1),
    plus the subset of the `torch.<device_type>` module API that
    `torch.distributed.device_mesh._get_device_handle`/FSDP2 resolve via
    `getattr(torch, device_type)` (`current_device`, `is_available`,
    `is_initialized`, `current_stream`, `stream`, `memory_summary`,
    `Event`, `Stream`)."""

    name = "Meta_Simulator"

    def set_device(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def reset_peak_memory_stats(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def get_device_name(self, *_args: Any, **_kwargs: Any) -> str:
        return self.name

    def get_device_properties(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        # total_memory must be non-zero: torchtitan's own
        # DeviceMemoryMonitor._to_pct() unconditionally divides by it
        # (`100 * memory / self.device_capacity`) when computing display
        # percentages. Since `memory_stats()` below always reports 0 bytes
        # under meta simulation (no real memory is ever allocated), the
        # resulting percentage is always 0% regardless of this
        # denominator's exact value -- it exists purely to avoid
        # ZeroDivisionError, not to represent a real capacity.
        return SimpleNamespace(total_memory=1, name=self.name)

    def memory_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {
            "active_bytes.all.peak": 0,
            "reserved_bytes.all.peak": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def current_stream(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @contextmanager
    def stream(self, *_args: Any, **_kwargs: Any):
        yield

    def memory_summary(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    class Event:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def record(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def elapsed_time(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

    class Stream:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait_stream(self, *_args: Any, **_kwargs: Any) -> None:
            return None


_PATCHED_MODULE_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchtitan.tools.utils", ("device_type", "device_module")),
    ("torchtitan.components.metrics", ("device_type", "device_module")),
    ("torchtitan.distributed.parallel_dims", ("device_type",)),
    ("torchtitan.distributed.utils", ("device_type", "device_module")),
)

# Any stable placeholder string works here: the only consumer is a string
# comparison against fixed version-name literals (see
# `torch_npu.utils._optim.patch_supported_devices`), never a real device query.
_DUMMY_NPU_DEVICE_NAME = "Ascend910B_Simulator"

_original_values: dict[tuple[str, str], Any] = {}
_original_fsdp_validate_no_meta_params: Any = _MISSING
_patched = False


def _neutralize_torch_npu_optimizer_device_probe() -> None:
    """Real `torch_npu` installs (found under a genuine CANN container)
    monkeypatch `torch.optim.optimizer._get_foreach_kernels_supported_devices`
    to lazily cache a real device name the first time any optimizer's
    `.step()` runs `_default_to_fused_or_foreach()`. That cache-fill calls
    `torch_npu.npu.current_device()`, which unconditionally calls
    `torch_npu.npu._lazy_init()` -> `torch_npu._C._npu_init()` -- a real
    `aclInit()` hardware call that raises (`"Failed to obtain the SOC
    version"`) with no NPU device present. Pre-filling the module-level
    cache (`torch_npu.utils._optim._device_name`) short-circuits that
    lazy-init path entirely; harmless/no-op when `torch_npu` is not
    installed (e.g. this repo's CPU-only unit-test sandbox)."""
    try:
        import torch_npu.utils._optim as npu_optim
    except ImportError:
        return

    if npu_optim._device_name is None:
        _original_values[("torch_npu.utils._optim", "_device_name")] = None
        npu_optim._device_name = _DUMMY_NPU_DEVICE_NAME


def _patch_swap_optimizer_get_device_info(stub: _MetaDeviceModule) -> None:
    """`torchtitan_npu.patches.optimizer.swap_optimizer` imports
    `get_device_info` BY VALUE (`from torchtitan.tools.utils import
    get_device_info`) and calls it fresh inside `get_torch_device()` to
    build `SwapOptimizersContainer.swap_to_device_stream = get_torch_device().Stream()`.
    `get_device_info()` independently re-detects the "live" device
    type/module via `_get_available_device_type()` every call -- bypassing
    the module-level `device_type`/`device_module` globals patched above
    entirely -- so under real torch_npu it resolves real `torch.cuda`
    (compiled into every torch build regardless of GPU presence), whose
    `.Stream()` unconditionally requires real CUDA hardware. Patches the
    by-value-imported name directly on `swap_optimizer`'s own module
    namespace; no-op if torch_npu (and therefore this torchtitan_npu
    submodule) is not importable, e.g. this repo's CPU-only unit-test
    sandbox."""
    try:
        import torchtitan_npu.patches.optimizer.swap_optimizer as swap_optimizer_mod
    except ImportError:
        return

    key = ("torchtitan_npu.patches.optimizer.swap_optimizer", "get_device_info")
    if key in _original_values:
        return
    _original_values[key] = swap_optimizer_mod.get_device_info
    swap_optimizer_mod.get_device_info = lambda: ("meta", stub)


def _neutralize_fsdp_meta_param_validation() -> None:
    """`FSDPParamGroup._lazy_init()` (triggered by the model's first
    forward call) unconditionally calls `_validate_no_meta_params()`,
    which raises `RuntimeError` if ANY sharded parameter's
    `.device.type == "meta"`. This check exists to catch real-training
    user errors (forgetting to materialize a meta-constructed model onto
    real hardware before training) -- but the simulator deliberately keeps
    every parameter on the meta device forever (that is the entire point:
    no real memory is ever allocated), so this defensive check is always a
    false positive here. Neutralizing it is pure PyTorch (`torch.distributed
    .fsdp`), not torchtitan/torchtitan_npu-specific, so this patch applies
    (and is tested) regardless of torch_npu availability.

    Tracked separately from `_original_values` (a class attribute, not a
    module attribute -- `importlib.import_module()` cannot resolve
    `"...FSDPParamGroup"` as a module path)."""
    global _original_fsdp_validate_no_meta_params
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    if _original_fsdp_validate_no_meta_params is not _MISSING:
        return
    _original_fsdp_validate_no_meta_params = FSDPParamGroup._validate_no_meta_params
    FSDPParamGroup._validate_no_meta_params = lambda self: None


def patch_device_type_to_meta() -> None:
    """Idempotently rebind `device_type="meta"` / `device_module=<stub>`
    across every module that imported them by value at load time, register
    the same stub as `torch.meta` (see `_MetaDeviceModule`'s docstring for
    why), neutralize `torch_npu`'s real-hardware optimizer device probe and
    swap-optimizer device stream construction (see
    `_neutralize_torch_npu_optimizer_device_probe` and
    `_patch_swap_optimizer_get_device_info`), and disable FSDP2's
    meta-param validation (see `_neutralize_fsdp_meta_param_validation`)."""
    global _patched
    if _patched:
        return

    stub = _MetaDeviceModule()
    for module_path, attr_names in _PATCHED_MODULE_ATTRS:
        module = importlib.import_module(module_path)
        for attr_name in attr_names:
            if not hasattr(module, attr_name):
                continue
            _original_values[(module_path, attr_name)] = getattr(module, attr_name)
            value: Any = "meta" if attr_name == "device_type" else stub
            setattr(module, attr_name, value)

    _original_values[("torch", "meta")] = getattr(torch, "meta", _MISSING)
    torch.meta = stub

    _neutralize_torch_npu_optimizer_device_probe()
    _patch_swap_optimizer_get_device_info(stub)
    _neutralize_fsdp_meta_param_validation()
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched, _original_fsdp_validate_no_meta_params
    for (module_path, attr_name), original in _original_values.items():
        module = importlib.import_module(module_path)
        if original is _MISSING:
            delattr(module, attr_name)
        else:
            setattr(module, attr_name, original)
    _original_values.clear()

    if _original_fsdp_validate_no_meta_params is not _MISSING:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        FSDPParamGroup._validate_no_meta_params = _original_fsdp_validate_no_meta_params
        _original_fsdp_validate_no_meta_params = _MISSING

    _patched = False
