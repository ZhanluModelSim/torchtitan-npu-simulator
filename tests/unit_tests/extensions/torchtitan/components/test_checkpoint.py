# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the NPU checkpoint manager extension."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchtitan.components.checkpointer.dcp import CheckpointManager as _CheckpointManager

from torchtitan_npu.extensions.torchtitan.components import checkpoint as product_checkpoint

_AsyncMode = product_checkpoint.AsyncMode


pytestmark = pytest.mark.cpu


def test_config_defaults_hash_manifest_verification_to_disabled():
    config = product_checkpoint.CheckpointManager.Config()

    assert config.extensions.verify_hash_manifest is False


def test_manager_reads_hash_manifest_extension_config():
    config = product_checkpoint.CheckpointManager.Config(
        extensions=product_checkpoint.CheckpointExtensions(verify_hash_manifest=True)
    )

    with patch.object(_CheckpointManager, "__init__", return_value=None) as initialize:
        manager = product_checkpoint.CheckpointManager(config)

    initialize.assert_called_once_with(config=config)
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

    optimizers = object()
    with patch.object(_CheckpointManager, "__init__", return_value=None) as initialize:
        product_checkpoint.CheckpointManager(config, optimizers=optimizers)

    initialize.assert_called_once_with(config=config, optimizers=optimizers)


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


def test_load_checkpoint_verifies_manifest_before_upstream_load():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = True
    manager._metadata_pg = object()
    events = []

    with (
        patch.object(
            product_checkpoint,
            "verify_checkpoint_manifest",
            side_effect=lambda *args: events.append("verify manifest"),
        ),
        patch.object(
            _CheckpointManager,
            "_load_checkpoint",
            side_effect=lambda *args, **kwargs: events.append("load checkpoint"),
        ),
    ):
        result = manager._load_checkpoint({}, "checkpoint", from_hf=True, from_quantized=False)

    assert result is None
    assert events == ["verify manifest", "load checkpoint"]


def test_load_checkpoint_hash_failure_prevents_restore():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = True
    with (
        patch.object(product_checkpoint, "verify_checkpoint_manifest", side_effect=ValueError("hash mismatch")),
        patch.object(_CheckpointManager, "_load_checkpoint") as restore,
        pytest.raises(ValueError, match="hash mismatch"),
    ):
        manager._load_checkpoint({}, "checkpoint", from_hf=False, from_quantized=False)
    restore.assert_not_called()


def test_load_checkpoint_disabled_verification_preserves_states_and_flags():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.verify_hash_manifest = False
    states = {"model": object()}
    with (
        patch.object(product_checkpoint, "verify_checkpoint_manifest") as verify,
        patch.object(_CheckpointManager, "_load_checkpoint") as restore,
    ):
        manager._load_checkpoint(states, "checkpoint", from_hf=True, from_quantized=True)
    verify.assert_not_called()
    restore.assert_called_once_with(states, "checkpoint", from_hf=True, from_quantized=True)


def _run_isolated_patch(script, *args):
    import subprocess

    module_path = (
        Path(__file__).resolve().parents[5] / "torchtitan_npu/patches/torchtitan/distributed/checkpoint/filesystem.py"
    )
    subprocess.run([sys.executable, "-c", script, str(module_path), *map(str, args)], check=True)


def test_bounded_reader_does_not_read_past_blob():
    script = """
import importlib.util
import io
import sys

spec = importlib.util.spec_from_file_location("dcp_mmap_filesystem_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base = io.BytesIO(b"prefix" + b"blob" + b"trailing")
view = module._BoundedReader(base, 6, 4)

view.seek(0)
assert view.read() == b"blob"

view.seek(5)
assert view.read() == b""

buffer = bytearray(8)
assert view.readinto(buffer) == 0
"""
    _run_isolated_patch(script)


def test_filesystem_reader_mmap_loads_local_dcp(tmp_path):
    script = """
import importlib.util
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

spec = importlib.util.spec_from_file_location("dcp_mmap_filesystem_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = {
    "weight": torch.arange(4096, dtype=torch.float32).reshape(64, 64),
    "bias": torch.arange(257, dtype=torch.bfloat16),
}
checkpoint_id = Path(sys.argv[2])
dcp.save(source, checkpoint_id=checkpoint_id)
target = {
    "weight": torch.empty_like(source["weight"]),
    "bias": torch.empty_like(source["bias"]),
}
reader = dcp.FileSystemReader(checkpoint_id, mmap=True)
assert reader.mmap is True
dcp.load(target, storage_reader=reader)
torch.testing.assert_close(target["weight"], source["weight"])
torch.testing.assert_close(target["bias"], source["bias"])
del reader
del target
"""
    _run_isolated_patch(script, tmp_path / "checkpoint")


def test_checkpoint_manager_uses_synchronous_writer():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    state_dict = {"state": torch.tensor(1)}
    writer = MagicMock()

    with (
        patch.object(product_checkpoint.dcp, "FileSystemWriter", return_value=writer) as make_writer,
        patch.object(product_checkpoint.dcp, "save") as save,
        patch.object(product_checkpoint.GarbageCollector, "collect") as collect,
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


@pytest.mark.parametrize("to_hf", [False, True], ids=["dcp", "hf"])
def test_last_step_dtype_conversion_preserves_row_shards(to_hf):
    manager = object.__new__(product_checkpoint.CheckpointManager)
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    counter = torch.tensor(3, dtype=torch.int64)
    metadata = {
        "global_shape": (8, 3),
        "global_offsets": ((4, 0),),
        "local_offsets": ((0, 0),),
        "local_sizes": ((4, 3),),
    }
    for name, value in metadata.items():
        setattr(weight, name, value)
    cached_state = {"table": weight, "counter": counter}

    def get_state_dict():
        return cached_state

    def checkpoint_id(step):
        return f"step-{step}"

    manager.states = {"model": types.SimpleNamespace(state_dict=get_state_dict)}
    manager.last_save_model_only = True
    manager.last_save_in_hf = to_hf
    manager.export_dtype = torch.bfloat16
    manager._create_checkpoint_id = checkpoint_id
    manager.dcp_save = MagicMock()

    manager._save_last_step(5)

    saved_state = manager.dcp_save.call_args.args[0]
    assert saved_state is not cached_state
    assert cached_state["table"] is weight
    saved = saved_state["table"]
    assert saved.dtype == torch.bfloat16
    assert saved is not weight
    torch.testing.assert_close(saved, weight.to(torch.bfloat16), rtol=0, atol=0)
    for name, expected in metadata.items():
        assert getattr(saved, name) == expected
    assert saved_state["counter"] is counter
    assert weight.dtype == torch.float32
    assert manager.dcp_save.call_args.kwargs == {
        "checkpoint_id": "step-5",
        "async_mode": _AsyncMode.DISABLED,
        "enable_garbage_collection": True,
        "to_hf": to_hf,
    }


def test_last_step_full_checkpoint_delegates_to_upstream():
    manager = object.__new__(product_checkpoint.CheckpointManager)
    manager.last_save_model_only = False
    manager.last_save_in_hf = False

    with patch.object(_CheckpointManager, "_save_last_step") as base_save_last_step:
        manager._save_last_step(5)

    base_save_last_step.assert_called_once_with(5)
