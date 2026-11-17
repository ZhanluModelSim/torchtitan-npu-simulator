# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve actual save ownership independently of checkpoint heuristics."""

from torchtitan_simulator.memory.plugins import MemoryModelContext


def retain_saved_storages(context: MemoryModelContext) -> None:
    capture_end = max((event.seq_idx for event in context.events), default=0) + 1
    retained = 0
    for saved in context.autograd_saved_tensors:
        root = context.tensor_root(saved.tensor_id, saved.storage_key)
        lifetime = context.lifetimes_by_tensor_id.get(root)
        # Explicit FSDP events replace generic full-parameter views. A Python
        # saved view does not keep that explicitly freed allocation resident.
        if lifetime is None:
            continue
        release = saved.release_seq if saved.release_tracking else saved.unpack_seq
        last_use = max([release if release >= 0 else capture_end, *lifetime.consumer_seqs])
        if last_use > lifetime.death_seq:
            lifetime.death_seq = last_use
            retained += 1
        if saved.source == "selective_cache":
            lifetime.kind = "checkpoint_saved_for_recompute"
            lifetime.reason = "native_selective_cache"
    if retained:
        context.notes.append(f"Actual saved storage ownership extended {retained} use-def lifetimes.")
