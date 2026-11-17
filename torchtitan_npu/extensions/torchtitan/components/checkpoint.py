# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint manager extension with NPU-aware save and hash verification."""

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.components.checkpoint
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
from torchtitan.components.checkpoint import (
    AsyncMode,
)
from torchtitan.components.checkpoint import (
    CheckpointManager as _CheckpointManager,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.utils import GarbageCollection

from torchtitan_npu.extensions.components.validation import (
    mark_checkpoint_manifest_pending,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)


@dataclass(kw_only=True, slots=True)
class CheckpointExtensions:
    verify_hash_manifest: bool = False
    """
    ``verify_hash_manifest`` enables per-file SHA-256 integrity verification.
    On save, a ``_checkpoint_hash_manifest.json`` is written containing the
    SHA-256 hash of every file in the checkpoint directory.  On load,
    the manifest is verified before any weight is materialised; a hash
    mismatch raises ``CheckpointManifestError`` and the load is rejected.
    If the manifest is absent (e.g. the checkpoint was saved without
    this option), the load proceeds silently. Only rank 0 performs
    the file I/O; the verdict is broadcast to all ranks.
    """


class CheckpointManager(_CheckpointManager):
    """Checkpoint manager with NPU-specific extensions."""

    @dataclass(kw_only=True, slots=True)
    class Config(_CheckpointManager.Config):
        extensions: CheckpointExtensions = field(default_factory=CheckpointExtensions)

    def __init__(self, config: Config, **kwargs):
        optimizers = kwargs.get("optimizers")
        # NovaSwap views share live CPU swap buffers. Pinned-memory staging
        # overlaps their copy with training, which can reuse those buffers.
        if (
            not getattr(optimizers, "supports_async_with_pinned_mem", True)
            and config.enable
            and config.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM.value
        ):
            raise ValueError(
                "checkpoint.async_mode='async_with_pinned_mem' is unsupported "
                "with swap_optimizer; use 'disabled' or 'async'"
            )

        self.verify_hash_manifest = config.extensions.verify_hash_manifest
        super().__init__(config=config, **kwargs)

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        if async_mode != AsyncMode.DISABLED or to_hf:
            return super().dcp_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_id,
                async_mode=async_mode,
                enable_garbage_collection=enable_garbage_collection,
                to_hf=to_hf,
            )

        dcp.save(
            state_dict,
            storage_writer=dcp.FileSystemWriter(
                checkpoint_id,
                per_thread_copy_ahead=0,
            ),
        )
        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        return None

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        if self.verify_hash_manifest:
            mark_checkpoint_manifest_pending(self, curr_step, last_step)

        result = super().save(curr_step, last_step)

        if self.verify_hash_manifest:
            write_checkpoint_manifest(self, curr_step, last_step)

        return result

    def dcp_load(self, state_dict, checkpoint_id, from_hf=False, from_quantized=False):
        if self.verify_hash_manifest:
            verify_checkpoint_manifest(checkpoint_id)

        return super().dcp_load(
            state_dict,
            checkpoint_id,
            from_hf,
            from_quantized,
        )


torchtitan.components.checkpoint.CheckpointManager = CheckpointManager
