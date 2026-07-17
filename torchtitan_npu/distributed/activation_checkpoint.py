# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU extensions for TorchTitan activation checkpointing."""

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import cast

from torchtitan.distributed import activation_checkpoint as titan_ac

# TorchTitan constructs selective AC policies through a process-global helper.
_SAC_SAVE_OPS_LOCK = RLock()
_GET_SAVE_OPS_ATTR = "_get_save_ops"


@contextmanager
def extend_selective_ac_save_ops(additional_ops: Collection[object]) -> Iterator[None]:
    """Add save ops while TorchTitan constructs a selective AC policy."""
    if not additional_ops:
        yield
        return

    additional_ops = set(additional_ops)
    with _SAC_SAVE_OPS_LOCK:
        original_get_save_ops = getattr(titan_ac, _GET_SAVE_OPS_ATTR, None)
        if not callable(original_get_save_ops):
            raise RuntimeError("Installed TorchTitan does not expose the expected SAC save-op helper")

        get_native_save_ops = cast("Callable[[], Collection[object]]", original_get_save_ops)

        def get_save_ops() -> set[object]:
            return set(get_native_save_ops()) | additional_ops

        setattr(titan_ac, _GET_SAVE_OPS_ATTR, get_save_ops)
        try:
            yield
        finally:
            setattr(titan_ac, _GET_SAVE_OPS_ATTR, original_get_save_ops)
