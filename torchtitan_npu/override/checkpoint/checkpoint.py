# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NPU-aware checkpoint loading with configurable verify_hash_manifest."""

from dataclasses import dataclass

import torch
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.observability import structured_logger as sl

from torchtitan_npu.override.common.optimizer import VirtualCheckpointManager

from .validation import (
    mark_checkpoint_manifest_pending,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)


class NPUCheckpointManager(CheckpointManager):
    """CheckpointManager with a configurable verify_hash_manifest option for dcp_load."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        verify_hash_manifest: bool = False
        """
        Enable per-file SHA-256 integrity verification for checkpoints.
        On save, a ``_checkpoint_hash_manifest.json`` is written containing the
        SHA-256 hash of every file in the checkpoint directory.  On load,
        the manifest is verified before any weight is materialised; a hash
        mismatch raises ``CheckpointManifestError`` and the load is rejected.
        If the manifest is absent (e.g. the checkpoint was saved without
        this option), the load proceeds silently.  Only rank 0 performs
        the file I/O; the verdict is broadcast to all ranks.
        """

    def __init__(self, config: Config, **kwargs):
        self.verify_hash_manifest = config.verify_hash_manifest
        super().__init__(config=config, **kwargs)

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


class NPUVirtualCheckpointManager(NPUCheckpointManager, VirtualCheckpointManager):
    """Hash-verifying checkpoint manager compatible with Virtual Optimizer states."""

    @dataclass(kw_only=True, slots=True)
    class Config(NPUCheckpointManager.Config):  # pyrefly: ignore[bad-override]
        pass
