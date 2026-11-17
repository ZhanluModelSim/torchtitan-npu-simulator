# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe actual native SAC cache mutations without changing its policy."""

from collections import defaultdict

import torch
from torch.utils._pytree import tree_leaves

from torchtitan_simulator.capture.dispatch_capture import get_active_capture


def _entry_tensors(entry):
    # Native cache entries hold version wrappers. Read their stored values,
    # never get_val(): invoking an unpack hook here would change execution.
    return [leaf.val for leaf in tree_leaves(entry) if isinstance(getattr(leaf, "val", None), torch.Tensor)]


class _ObservedEntries(dict):
    def __init__(self):
        super().__init__()
        self.slots: dict[int, list[int | None]] = {}

    def __setitem__(self, key, value):
        capture = get_active_capture()
        previous = self.get(key)
        super().__setitem__(key, value)
        slots = self.slots.pop(key, ())
        if capture is None:
            return
        if slots:
            for slot in slots:
                capture.record_autograd_saved_tensor_unpack(slot)
            capture.record_synthetic_ac_cache_hit("native_selective_cache", _entry_tensors(previous))
            for slot in slots:
                capture.record_autograd_saved_tensor_release(slot)
        tensors = _entry_tensors(value)
        if tensors:
            self.slots[key] = [
                capture.record_autograd_saved_tensor_pack(tensor, source="selective_cache") for tensor in tensors
            ]


def observe_selective_cache(forward_context, recompute_context):
    """Attach to the shared native invocation-indexed cache, if present.

    Full AC has no cache. An unfamiliar SAC cache layout must fail explicitly
    instead of silently pretending that policy intent proves retention.
    """
    storage = getattr(forward_context, "storage", None)
    if storage is None:
        return
    if storage is not getattr(recompute_context, "storage", None) or not isinstance(storage, dict) or storage:
        raise RuntimeError("Unsupported selective checkpoint cache ownership/layout")
    observed = defaultdict(_ObservedEntries)
    forward_context.storage = observed
    recompute_context.storage = observed
