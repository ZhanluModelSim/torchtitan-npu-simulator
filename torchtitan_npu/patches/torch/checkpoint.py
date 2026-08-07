# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch target:
# - torch.distributed.checkpoint.filesystem.FileSystemWriter
# - torchtitan.components.checkpoint.CheckpointManager
#
# Why:
# Large NPU training checkpoints can leave hundreds of GiB in Linux page cache
# and may retain temporary NPU allocator blocks after save returns. This patch
# keeps upstream DCP layout unchanged while adding NPU-specific resource controls
# from torchtitan_npu.config.configs.CheckpointConfig.
#
# Maintenance:
# _patched_checkpoint_manager_dcp_save is aligned with torchtitan upstream
# commit ac13e536c84e7f6647b14fa9375c3c8a8a2b8578. Re-check this patch whenever
# docs/community/versioning_policy.md advances the torchtitan baseline.

import gc
import os
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torchtitan.components.checkpoint import AsyncMode, CheckpointManager
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

_ORIGINAL_FILESYSTEM_WRITER_WRITE_DATA = FileSystemWriter._write_data
_ORIGINAL_CHECKPOINT_MANAGER_INIT = CheckpointManager.__init__
_ORIGINAL_CHECKPOINT_MANAGER_SAVE = CheckpointManager.save
_CHECKPOINT_MANAGER_INIT_SIGNATURE = signature(_ORIGINAL_CHECKPOINT_MANAGER_INIT)


def drop_page_cache_for_path(path: str | os.PathLike[str]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return

    root = Path(path)
    candidates = [root] if root.is_file() else root.rglob("*")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        fd = None
        try:
            fd = os.open(candidate, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            logger.debug("Failed to drop page cache for %s: %s", candidate, exc)
        finally:
            if fd is not None:
                os.close(fd)


def _apply_filesystem_writer_options(manager: CheckpointManager, writer: FileSystemWriter) -> None:
    writer.sync_files = getattr(manager, "_npu_checkpoint_sync_files", True)
    writer_with_options: Any = writer
    writer_with_options._npu_drop_page_cache_after_save = bool(
        getattr(manager, "_npu_checkpoint_drop_page_cache_after_save", False)
    )


def _create_filesystem_writer(manager: CheckpointManager, checkpoint_id: str) -> FileSystemWriter:
    writer = FileSystemWriter(checkpoint_id)
    _apply_filesystem_writer_options(manager, writer)
    return writer


def _disable_device_prefetch_for_async_writer(writer: Any) -> None:
    writer_with_options: Any = writer
    if hasattr(writer_with_options, "per_thread_copy_ahead"):
        writer_with_options.per_thread_copy_ahead = 0


def _upload_completion_for_async_save(result: Any) -> Any:
    if isinstance(result, AsyncSaveResponse):
        return result.upload_completion
    return result


@wraps(_ORIGINAL_FILESYSTEM_WRITER_WRITE_DATA)
def _patched_filesystem_writer_write_data(self, planner, file_queue):
    result = _ORIGINAL_FILESYSTEM_WRITER_WRITE_DATA(self, planner, file_queue)
    if getattr(self, "_npu_drop_page_cache_after_save", False):
        drop_page_cache_for_path(self.path)
    return result


@wraps(_ORIGINAL_CHECKPOINT_MANAGER_INIT)
def _patched_checkpoint_manager_init(self, *args, **kwargs):
    bound_args = _CHECKPOINT_MANAGER_INIT_SIGNATURE.bind_partial(self, *args, **kwargs)
    checkpoint_config = bound_args.arguments.get("config")
    if checkpoint_config is None:
        checkpoint_config = bound_args.arguments.get("checkpoint_config")

    _ORIGINAL_CHECKPOINT_MANAGER_INIT(self, *args, **kwargs)

    self._npu_checkpoint_sync_files = getattr(checkpoint_config, "sync_files", True)
    self._npu_checkpoint_drop_page_cache_after_save = getattr(checkpoint_config, "drop_page_cache_after_save", False)
    self._npu_checkpoint_empty_cache_after_save = getattr(checkpoint_config, "empty_cache_after_save", True)


@torch.no_grad()
def _patched_checkpoint_manager_dcp_save(
    self,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    async_mode: AsyncMode,
    enable_garbage_collection: bool = False,
    to_hf: bool = False,
):
    ret = None
    hf_state_dict_metadata: dict[str, Any] | None = None

    storage_writer = None
    checkpoint_save_id = None
    fqn_to_index_mapping = None
    if to_hf:
        assert self.sd_adapter is not None, (
            "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
        )
        state_dict = self.sd_adapter.to_hf(state_dict)

        fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
        if fqn_to_index_mapping:
            storage_writer = HuggingFaceStorageWriter(
                path=os.path.join(checkpoint_id, "sharded"),
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=False,
            )
        else:
            storage_writer = HuggingFaceStorageWriter(
                path=checkpoint_id,
                save_distributed=True,
                enable_consolidation=True,
            )
        _apply_filesystem_writer_options(self, storage_writer)
    else:
        checkpoint_save_id = checkpoint_id
        storage_writer = _create_filesystem_writer(self, checkpoint_id)

    if async_mode == AsyncMode.ASYNC:
        _disable_device_prefetch_for_async_writer(storage_writer)
        ret = dcp.async_save(
            state_dict,
            storage_writer=storage_writer,
            checkpoint_id=checkpoint_save_id,
            process_group=self.pg,
            async_stager=DefaultStager(StagingOptions(False, False, False, False)),
        )
        ret = _upload_completion_for_async_save(ret)
    elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
        _disable_device_prefetch_for_async_writer(storage_writer)
        ret = dcp.async_save(
            state_dict,
            storage_writer=storage_writer,
            checkpoint_id=checkpoint_save_id,
            process_group=self.pg,
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,
            async_stager=self.stager,
        )
    else:
        ret = dcp.save(
            state_dict,
            storage_writer=storage_writer,
            checkpoint_id=checkpoint_save_id,
        )
        if to_hf:
            hf_state_dict_metadata = ret.state_dict_metadata

    if to_hf and fqn_to_index_mapping:
        if hf_state_dict_metadata is not None:
            exported_fqns = hf_state_dict_metadata.keys()
            dropped_fqns = fqn_to_index_mapping.keys() - exported_fqns
            if dropped_fqns:
                logger.warning(
                    "Dropping %d HF shard mapping keys not present in the exported state dict. "
                    "Verify they are intentionally disabled: %s",
                    len(dropped_fqns),
                    ", ".join(sorted(dropped_fqns)[:5]),
                )
            fqn_to_index_mapping = {key: index for key, index in fqn_to_index_mapping.items() if key in exported_fqns}
        if fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

    if enable_garbage_collection:
        GarbageCollection.collect("GC collection invoked by checkpointer.")

    return ret


@wraps(_ORIGINAL_CHECKPOINT_MANAGER_SAVE)
def _patched_checkpoint_manager_save(self, curr_step: int, last_step: bool = False):
    should_clear_cache = bool(
        getattr(self, "enable", False)
        and getattr(self, "_npu_checkpoint_empty_cache_after_save", True)
        and self._should_save(curr_step, last_step)
    )

    result = _ORIGINAL_CHECKPOINT_MANAGER_SAVE(self, curr_step, last_step)

    if should_clear_cache:
        gc.collect()
        try:
            torch.npu.empty_cache()  # pyrefly: ignore[missing-attribute]
        except Exception as exc:
            logger.debug("Failed to clear NPU cache after checkpoint save: %s", exc)
    return result


FileSystemWriter._write_data = _patched_filesystem_writer_write_data
CheckpointManager.__init__ = _patched_checkpoint_manager_init
CheckpointManager.dcp_save = _patched_checkpoint_manager_dcp_save
CheckpointManager.save = _patched_checkpoint_manager_save
