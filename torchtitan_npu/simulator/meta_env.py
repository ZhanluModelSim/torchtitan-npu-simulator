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
from types import SimpleNamespace
from typing import Any


class _MetaDeviceModule:
    """Minimal stand-in for `torch.cuda`/`torch_npu`, covering every method
    actually called on `device_module` by `torchtitan.trainer`,
    `torchtitan.components.metrics`, and `torchtitan.distributed.utils`
    (verified against the pinned torchtitan commit -- see design doc §5.1)."""

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
        return SimpleNamespace(total_memory=0, name=self.name)

    def memory_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {
            "active_bytes.all.peak": 0,
            "reserved_bytes.all.peak": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }


_PATCHED_MODULE_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchtitan.tools.utils", ("device_type", "device_module")),
    ("torchtitan.components.metrics", ("device_type", "device_module")),
    ("torchtitan.distributed.parallel_dims", ("device_type",)),
    ("torchtitan.distributed.utils", ("device_type", "device_module")),
)

_original_values: dict[tuple[str, str], Any] = {}
_patched = False


def patch_device_type_to_meta() -> None:
    """Idempotently rebind `device_type="meta"` / `device_module=<stub>`
    across every module that imported them by value at load time."""
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
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched
    for (module_path, attr_name), original in _original_values.items():
        module = importlib.import_module(module_path)
        setattr(module, attr_name, original)
    _original_values.clear()
    _patched = False
