# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Instance-local NPU adapters for torch_checkpointing 0.1.1."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
import torch_npu
from torch_checkpointing._state_dict_stager import StateDictStager, StorageManager
from torch_checkpointing.checkpoint_saver import AsyncCheckpointSaver
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.staging import DefaultStager


class _PinnedStoragePool(StorageManager):
    def delete_unused(self):
        for size, storages in self._size_to_storages.items():
            del storages[self._size_to_storages_next_index[size] :]

    def _new_storage(self, size):
        # Allocator-owned pinning needs no CUDA host-register/unregister calls.
        return torch.empty(size, dtype=torch.uint8, device="cpu").pin_memory(device="npu").untyped_storage()


class _NpuStateDictStager(StateDictStager):
    def __init__(self, config):
        # The upstream non-blocking default probes CUDA; retain the caller's setting.
        super().__init__(use_non_blocking_copy=config.use_non_blocking_copy)
        self._storage_manager = _PinnedStoragePool(pin_memory=True, share_memory=False, pin_memory_min_bytes=0)
        # Pinned-to-shared staging uses an extra host buffer and CPU copy.
        # TODO: Revisit with Ascend/pytorch #46095 after validating shared-memory pinning.
        self._shared_stager = StateDictStager(share_memory=config.use_shared_memory)

    def stage(self, state_dict):
        stream = torch_npu.npu.current_stream()
        try:
            host = super().stage(state_dict)
        finally:
            # Drain partial D2H copies before shared staging or error propagation.
            stream.synchronize()
        return self._shared_stager.stage(host)

    def close(self):
        super().close()
        self._shared_stager.close()

    def _stage_untyped_storage(self, storage):
        if storage.device.type == "cpu":
            return storage
        return super()._stage_untyped_storage(storage)


class _NpuStager(DefaultStager):
    def __init__(self, config):
        super().__init__(replace(config, use_pinned_memory=False))
        # A negative index makes the inherited CUDA-only device guards no-ops.
        self._cuda_device = -1
        self._npu_device = torch_npu.npu.current_device() if self._accelerator_available else None
        if self._accelerator_available and config.use_pinned_memory:
            self._state_dict_stager.close()
            self._state_dict_stager = _NpuStateDictStager(config)

    def stage(self, state_dict, keys_not_requiring_copy=()):
        with torch.accelerator.device_index(self._npu_device):
            return super().stage(state_dict, keys_not_requiring_copy)

    def _init_worker_thread(self):
        if self._npu_device is not None:
            torch_npu.npu.set_device(self._npu_device)


class _NpuMetadataManager(DefaultMetadataManager):
    def _all_gather_pickled_data(self, data, process_group=None):
        # Let the process group choose the collective device (NPU for HCCL).
        gathered = [None] * dist.get_world_size(process_group)
        dist.all_gather_object(gathered, data, group=process_group)
        result = {}
        for group_rank, value in enumerate(gathered):
            rank = dist.get_global_rank(process_group, group_rank) if process_group is not None else group_rank
            result[rank] = value
        return result


def adapt_checkpoint_manager(manager) -> None:
    """Replace device leaves before first use; retain upstream ownership."""
    # pylint: disable=protected-access
    # The fixed backend exposes no stager/metadata injection point.
    # Its saver and loader must share the same replacement metadata cache.
    metadata = manager._metadata_manager
    if metadata is not None:
        replacement = _NpuMetadataManager(rank_info=metadata._rank_info)
        manager._metadata_manager = replacement
        manager._saver._metadata_manager = replacement
        manager._loader._metadata_manager = replacement
        metadata.close()
    if isinstance(manager._saver, AsyncCheckpointSaver):
        stager = _NpuStager(manager._config.save.staging_config)
        manager._saver.stager.close()
        manager._saver._checkpoint_stager = stager
