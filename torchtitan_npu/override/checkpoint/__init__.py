# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CheckpointManager overrides (registry-facing module)."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import derive, override

from .checkpoint import NPUCheckpointManager, NPUVirtualCheckpointManager


@override(
    target=CheckpointManager.Config,
    description="NPU checkpoint manager with configurable verify_hash_manifest",
    exact=True,
)
def npu(
    cfg: CheckpointManager.Config,
    *,
    verify_hash_manifest: bool = False,
) -> NPUCheckpointManager.Config:
    return derive(cfg, NPUCheckpointManager.Config, verify_hash_manifest=verify_hash_manifest)


@override(
    target=CheckpointManager.Config,
    description="NPU checkpoint manager with hash verification and Virtual Optimizer compatibility",
    exact=True,
)
def npu_virtual(
    cfg: CheckpointManager.Config,
    *,
    verify_hash_manifest: bool = False,
) -> NPUVirtualCheckpointManager.Config:
    return derive(cfg, NPUVirtualCheckpointManager.Config, verify_hash_manifest=verify_hash_manifest)
