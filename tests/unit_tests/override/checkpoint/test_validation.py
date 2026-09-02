# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for checkpoint hash manifests."""

from __future__ import annotations

import importlib.util
import json
import os
import posixpath
import sys
import threading
import types
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from unittest import mock
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from fsspec.core import url_to_fs


class _AsyncMode(str, Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class _FilesystemAdapter:
    @staticmethod
    def is_remote(path):
        return "://" in str(path)

    @staticmethod
    def _resolve(path):
        return url_to_fs(path)

    @staticmethod
    def exists(path):
        if _FilesystemAdapter.is_remote(path):
            fs, resolved_path = url_to_fs(path)
            return fs.exists(resolved_path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path):
        if _FilesystemAdapter.is_remote(path):
            fs, resolved_path = url_to_fs(path)
            return fs.isfile(resolved_path)
        return os.path.isfile(path)

    @staticmethod
    def listdir(path):
        if _FilesystemAdapter.is_remote(path):
            fs, resolved_path = url_to_fs(path)
            return [posixpath.basename(entry) for entry in fs.ls(resolved_path, detail=False)]
        return os.listdir(path)

    @staticmethod
    def join(base, name):
        if _FilesystemAdapter.is_remote(base):
            return posixpath.join(str(base), name)
        return os.path.join(base, name)


class _AsyncCheckpointManager:
    async_mode = _AsyncMode.ASYNC

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.save_future = Future()

    def _should_save(self, curr_step, last_step=False):
        return True

    def _create_checkpoint_id(self, curr_step):
        return self.checkpoint_path


checkpoint_stub = types.ModuleType("torchtitan.components.checkpoint")
checkpoint_stub.AsyncMode = _AsyncMode
tools_stub = types.ModuleType("torchtitan.tools")
tools_stub.filesystem = _FilesystemAdapter

module_path = Path(__file__).resolve().parents[4] / "torchtitan_npu/override/checkpoint/validation.py"
module_spec = importlib.util.spec_from_file_location("checkpoint_verified_under_test", module_path)
product_validation = importlib.util.module_from_spec(module_spec)

with patch.dict(
    sys.modules,
    {
        "torchtitan.components.checkpoint": checkpoint_stub,
        "torchtitan.tools": tools_stub,
    },
):
    module_spec.loader.exec_module(product_validation)


pytestmark = pytest.mark.cpu


@pytest.fixture
def remote_checkpoint():
    checkpoint_uri = f"memory://checkpoint-tests-{uuid.uuid4().hex}/step-1"
    fs, checkpoint_path = url_to_fs(checkpoint_uri)
    fs.makedirs(checkpoint_path)
    yield checkpoint_uri, fs, checkpoint_path
    fs.rm(checkpoint_path, recursive=True)


def test_write_manifest_creates_hash_mapping_for_remote_checkpoint(remote_checkpoint):
    checkpoint_uri, fs, checkpoint_path = remote_checkpoint
    with fs.open(posixpath.join(checkpoint_path, "data.bin"), "wb") as checkpoint_file:
        checkpoint_file.write(b"checkpoint payload")

    product_validation._write_manifest(checkpoint_uri)

    manifest_path = posixpath.join(checkpoint_path, product_validation._MANIFEST_FILENAME)
    assert fs.exists(manifest_path)
    with fs.open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)
    assert set(manifest["hash"]) == {"data.bin"}


def test_verify_manifest_detects_modified_remote_checkpoint_file(remote_checkpoint):
    checkpoint_uri, fs, checkpoint_path = remote_checkpoint
    data_path = posixpath.join(checkpoint_path, "data.bin")
    with fs.open(data_path, "wb") as checkpoint_file:
        checkpoint_file.write(b"original payload")
    product_validation._write_manifest(checkpoint_uri)

    with fs.open(data_path, "wb") as checkpoint_file:
        checkpoint_file.write(b"modified payload")

    with pytest.raises(product_validation.CheckpointManifestError, match="hash mismatch"):
        product_validation.verify_checkpoint_manifest(checkpoint_uri)


def test_async_save_future_finishes_only_after_manifest_is_written(tmp_path, monkeypatch):
    manager = _AsyncCheckpointManager(tmp_path)
    checkpoint_save_future = manager.save_future
    manifest_started = threading.Event()
    allow_manifest_to_finish = threading.Event()
    write_manifest = product_validation._write_manifest

    def write_manifest_after_release(checkpoint_path):
        manifest_started.set()
        assert allow_manifest_to_finish.wait(timeout=5)
        write_manifest(checkpoint_path)

    monkeypatch.setattr(product_validation, "_write_manifest", write_manifest_after_release)
    product_validation.mark_checkpoint_manifest_pending(manager, curr_step=1)
    product_validation.write_checkpoint_manifest(manager, curr_step=1)

    save_thread = threading.Thread(target=lambda: checkpoint_save_future.set_result(None))
    save_thread.start()
    assert manifest_started.wait(timeout=2)
    with pytest.raises(FutureTimeoutError):
        manager.save_future.result(timeout=0.01)

    allow_manifest_to_finish.set()
    save_thread.join(timeout=2)
    manager.save_future.result(timeout=2)
    assert not save_thread.is_alive()
    assert (tmp_path / product_validation._MANIFEST_FILENAME).is_file()
    assert not (tmp_path / product_validation._PENDING_MANIFEST_FILENAME).exists()


def test_async_manifest_failure_is_reported_by_save_future(tmp_path, monkeypatch):
    manager = _AsyncCheckpointManager(tmp_path)
    checkpoint_save_future = manager.save_future
    product_validation.mark_checkpoint_manifest_pending(manager, curr_step=1)

    def fail_manifest_write(checkpoint_path):
        raise OSError("manifest write failed")

    monkeypatch.setattr(product_validation, "_write_manifest", fail_manifest_write)
    product_validation.write_checkpoint_manifest(manager, curr_step=1)
    checkpoint_save_future.set_result(None)

    with pytest.raises(OSError, match="manifest write failed"):
        manager.save_future.result()
    assert (tmp_path / product_validation._PENDING_MANIFEST_FILENAME).is_file()


def test_verify_manifest_rejects_checkpoint_with_pending_marker(tmp_path):
    manager = _AsyncCheckpointManager(tmp_path)
    product_validation.mark_checkpoint_manifest_pending(manager, curr_step=1)

    with pytest.raises(product_validation.CheckpointManifestError, match="incomplete"):
        product_validation.verify_checkpoint_manifest(tmp_path)


def _write_local_manifest(checkpoint_path, hash_mapping):
    manifest = {
        "granularity": "file",
        "algorithm": "sha256",
        "hash": hash_mapping,
    }
    manifest_path = checkpoint_path / product_validation._MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest))
