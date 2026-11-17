# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the NPU checkpoint manager extension."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


class _AsyncMode(str, Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class _CheckpointManager:
    @dataclass(kw_only=True, slots=True)
    class Config:
        interval: int = 1
        enable: bool = True
        async_mode: str = "disabled"

    def __init__(self, config, **kwargs):
        self.config = config

    def save(self, curr_step, last_step=False):
        return True

    def dcp_save(self, **kwargs):
        return "upstream-save"

    def dcp_load(self, *args, **kwargs):
        return "upstream-load"


torchtitan_stub = types.ModuleType("torchtitan")
torchtitan_stub.__path__ = []
components_stub = types.ModuleType("torchtitan.components")
components_stub.__path__ = []
torchtitan_stub.components = components_stub
checkpoint_stub = types.ModuleType("torchtitan.components.checkpoint")
checkpoint_stub.AsyncMode = _AsyncMode
checkpoint_stub.CheckpointManager = _CheckpointManager
components_stub.checkpoint = checkpoint_stub
observability_stub = types.ModuleType("torchtitan.observability")
observability_stub.structured_logger = types.SimpleNamespace(log_trace_span=lambda name: lambda function: function)
torchtitan_stub.observability = observability_stub
tools_stub = types.ModuleType("torchtitan.tools")
tools_stub.__path__ = []
torchtitan_stub.tools = tools_stub
utils_stub = types.ModuleType("torchtitan.tools.utils")
utils_stub.GarbageCollection = MagicMock()
tools_stub.utils = utils_stub
validation_stub = types.ModuleType("torchtitan_npu.extensions.components.validation")
validation_stub.mark_checkpoint_manifest_pending = MagicMock()
validation_stub.verify_checkpoint_manifest = MagicMock()
validation_stub.write_checkpoint_manifest = MagicMock()

module_path = Path(__file__).resolve().parents[5] / "torchtitan_npu/extensions/torchtitan/components/checkpoint.py"
module_name = "torchtitan_npu.extensions.torchtitan.components.checkpoint"
module_spec = importlib.util.spec_from_file_location(module_name, module_path)
product_checkpoint = importlib.util.module_from_spec(module_spec)

with patch.dict(
    sys.modules,
    {
        "torchtitan": torchtitan_stub,
        "torchtitan.components": components_stub,
        "torchtitan.components.checkpoint": checkpoint_stub,
        "torchtitan.observability": observability_stub,
        "torchtitan.tools": tools_stub,
        "torchtitan.tools.utils": utils_stub,
        "torchtitan_npu.extensions.components.validation": validation_stub,
    },
):
    module_spec.loader.exec_module(product_checkpoint)


pytestmark = pytest.mark.cpu


def test_config_defaults_hash_manifest_verification_to_disabled():
    config = product_checkpoint.CheckpointManager.Config()

    assert config.extensions.verify_hash_manifest is False


def test_manager_reads_hash_manifest_extension_config():
    config = product_checkpoint.CheckpointManager.Config(
        extensions=product_checkpoint.CheckpointExtensions(verify_hash_manifest=True)
    )

    manager = product_checkpoint.CheckpointManager(config)

    assert manager.verify_hash_manifest is True


def test_manager_rejects_pinned_async_for_swap_optimizer():
    config = product_checkpoint.CheckpointManager.Config(async_mode="async_with_pinned_mem")
    optimizers = types.SimpleNamespace(supports_async_with_pinned_mem=False)

    with pytest.raises(
        ValueError,
        match=r"async_with_pinned_mem.*unsupported.*swap_optimizer",
    ):
        product_checkpoint.CheckpointManager(config, optimizers=optimizers)


def test_manager_keeps_pinned_async_for_native_optimizer():
    config = product_checkpoint.CheckpointManager.Config(async_mode="async_with_pinned_mem")

    manager = product_checkpoint.CheckpointManager(config, optimizers=object())

    assert manager.config is config


def test_save_marks_pending_before_save_and_writes_manifest_afterward():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = True
    events = []

    with (
        patch.object(
            product_checkpoint,
            "mark_checkpoint_manifest_pending",
            side_effect=lambda *args: events.append("mark pending"),
        ),
        patch.object(
            _CheckpointManager,
            "save",
            side_effect=lambda *args: events.append("save checkpoint") or True,
        ),
        patch.object(
            product_checkpoint,
            "write_checkpoint_manifest",
            side_effect=lambda *args: events.append("write manifest"),
        ),
    ):
        result = manager.save(curr_step=1)

    assert result is True
    assert events == ["mark pending", "save checkpoint", "write manifest"]


def test_save_failure_keeps_pending_and_skips_manifest_write():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = True

    with (
        patch.object(product_checkpoint, "mark_checkpoint_manifest_pending") as mark,
        patch.object(
            _CheckpointManager,
            "save",
            side_effect=RuntimeError("checkpoint save failed"),
        ),
        patch.object(product_checkpoint, "write_checkpoint_manifest") as write,
        pytest.raises(RuntimeError, match="checkpoint save failed"),
    ):
        manager.save(curr_step=1)

    mark.assert_called_once_with(manager, 1, False)
    write.assert_not_called()


def test_save_skips_manifest_operations_when_disabled():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = False

    with (
        patch.object(product_checkpoint, "mark_checkpoint_manifest_pending") as mark,
        patch.object(_CheckpointManager, "save", return_value=True),
        patch.object(product_checkpoint, "write_checkpoint_manifest") as write,
    ):
        assert manager.save(curr_step=1) is True

    mark.assert_not_called()
    write.assert_not_called()


def test_dcp_load_verifies_manifest_before_upstream_load():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = True
    events = []

    with (
        patch.object(
            product_checkpoint,
            "verify_checkpoint_manifest",
            side_effect=lambda *args: events.append("verify manifest"),
        ),
        patch.object(
            _CheckpointManager,
            "dcp_load",
            side_effect=lambda *args, **kwargs: events.append("load checkpoint") or "loaded",
        ),
    ):
        result = manager.dcp_load({}, "checkpoint")

    assert result == "loaded"
    assert events == ["verify manifest", "load checkpoint"]


def test_checkpoint_manager_uses_synchronous_writer():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    state_dict = {"state": torch.tensor(1)}
    writer = MagicMock()

    with (
        patch.object(product_checkpoint.dcp, "FileSystemWriter", return_value=writer) as make_writer,
        patch.object(product_checkpoint.dcp, "save") as save,
        patch.object(product_checkpoint.GarbageCollection, "collect") as collect,
    ):
        result = manager.dcp_save(
            state_dict,
            "checkpoint",
            product_checkpoint.AsyncMode.DISABLED,
            enable_garbage_collection=True,
        )

    make_writer.assert_called_once_with("checkpoint", per_thread_copy_ahead=0)
    save.assert_called_once_with(state_dict, storage_writer=writer)
    collect.assert_called_once_with("GC collection invoked by checkpointer.")
    assert result is None


@pytest.mark.parametrize(
    ("async_mode", "to_hf"),
    [(_AsyncMode.ASYNC, False), (_AsyncMode.DISABLED, True)],
)
def test_checkpoint_manager_delegates_async_and_hf_paths(async_mode, to_hf):
    manager = object.__new__(product_checkpoint.CheckpointManager)

    with patch.object(
        _CheckpointManager,
        "dcp_save",
        return_value="upstream-save",
    ) as base_save:
        result = manager.dcp_save(
            {},
            "checkpoint",
            async_mode,
            to_hf=to_hf,
        )

    assert result == "upstream-save"
    base_save.assert_called_once_with(
        state_dict={},
        checkpoint_id="checkpoint",
        async_mode=async_mode,
        enable_garbage_collection=False,
        to_hf=to_hf,
    )
