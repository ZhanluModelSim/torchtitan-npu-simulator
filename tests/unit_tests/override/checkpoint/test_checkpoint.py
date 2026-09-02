# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for checkpoint override selection and composition."""

from __future__ import annotations

import inspect
import sys
import types
from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest


class _AsyncMode(str, Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class _CheckpointManager:
    @dataclass(kw_only=True, slots=True)
    class Config:
        interval: int = 1

    def __init__(self, config, **kwargs):
        self.config = config

    def save(self, curr_step, last_step=False):
        return True

    def load(self, step=-1):
        return True


class _VirtualCheckpointManager(_CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(_CheckpointManager.Config):
        pass

    def dcp_save(self, state_dict, checkpoint_id, async_mode):
        return "virtual-checkpoint-save"


registrations = {}
derive_calls = []


def _record_override(**kwargs):
    def decorator(function):
        registrations[function.__name__] = kwargs
        return function

    return decorator


def _record_derive(cfg, target, **deltas):
    derive_calls.append((cfg, target, deltas))
    return target


checkpoint_stub = types.ModuleType("torchtitan.components.checkpoint")
checkpoint_stub.AsyncMode = _AsyncMode
checkpoint_stub.CheckpointManager = _CheckpointManager

config_stub = types.ModuleType("torchtitan.config")
config_stub.derive = _record_derive
config_stub.override = _record_override

observability_stub = types.ModuleType("torchtitan.observability")
observability_stub.structured_logger = types.SimpleNamespace(
    log_trace_span=lambda name: lambda function: function
)

filesystem_stub = types.ModuleType("torchtitan.tools.filesystem")
filesystem_stub.exists = MagicMock(return_value=False)
tools_stub = types.ModuleType("torchtitan.tools")
tools_stub.filesystem = filesystem_stub

virtual_checkpoint_stub = types.ModuleType("torchtitan_npu.override.common.optimizer")
virtual_checkpoint_stub.VirtualCheckpointManager = _VirtualCheckpointManager

with (
    patch.dict(
        sys.modules,
        {
            "torchtitan_npu.compile": MagicMock(),
            "torchtitan_npu.patches": MagicMock(),
            "torchtitan.components.checkpoint": checkpoint_stub,
            "torchtitan.config": config_stub,
            "torchtitan.observability": observability_stub,
            "torchtitan.tools": tools_stub,
            "torchtitan.tools.filesystem": filesystem_stub,
            "torchtitan_npu.override.common.optimizer": virtual_checkpoint_stub,
        },
    ),
):
    from torchtitan_npu.override import checkpoint as product_checkpoint


pytestmark = pytest.mark.cpu


def test_npu_override_does_not_claim_specialized_checkpoint_configs():
    assert registrations["npu"]["exact"] is True


def test_npu_save_marks_manifest_pending_before_checkpoint_save(monkeypatch):
    events = []
    manager = object.__new__(product_checkpoint.NPUCheckpointManager)
    manager.verify_hash_manifest = True
    save_globals = inspect.unwrap(product_checkpoint.NPUCheckpointManager.save).__globals__
    monkeypatch.setitem(
        save_globals,
        "mark_checkpoint_manifest_pending",
        lambda *args: events.append("mark pending"),
    )
    monkeypatch.setattr(
        _CheckpointManager,
        "save",
        lambda *args: events.append("save checkpoint") or True,
    )
    monkeypatch.setitem(
        save_globals,
        "write_checkpoint_manifest",
        lambda *args: events.append("write manifest"),
    )

    result = manager.save(curr_step=1)

    assert result is True
    assert events == ["mark pending", "save checkpoint", "write manifest"]


def test_npu_virtual_override_builds_one_combined_checkpoint_config():
    source_config = object()

    result = product_checkpoint.npu_virtual(source_config, verify_hash_manifest=True)

    assert registrations["npu_virtual"]["exact"] is True
    assert result is product_checkpoint.NPUVirtualCheckpointManager.Config
    assert issubclass(
        product_checkpoint.NPUVirtualCheckpointManager,
        product_checkpoint.NPUCheckpointManager,
    )
    assert issubclass(
        product_checkpoint.NPUVirtualCheckpointManager,
        _VirtualCheckpointManager,
    )
    manager = object.__new__(product_checkpoint.NPUVirtualCheckpointManager)
    assert manager.dcp_save({}, "checkpoint", _AsyncMode.DISABLED) == "virtual-checkpoint-save"
    assert derive_calls[-1] == (
        source_config,
        product_checkpoint.NPUVirtualCheckpointManager.Config,
        {"verify_hash_manifest": True},
    )
