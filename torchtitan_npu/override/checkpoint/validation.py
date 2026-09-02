# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-file checkpoint hash verification via SHA-256 manifests."""

import hashlib
import json
import logging
import os
from concurrent.futures import Future
from typing import Any

import torch.distributed as dist
from torchtitan.components.checkpoint import AsyncMode
from torchtitan.tools import filesystem

logger = logging.getLogger(__name__)

# Fixed manifest filename.  Prefixed with ``_`` so DCP / safetensors
# loaders that glob for known extensions simply ignore it.
_MANIFEST_FILENAME = "_checkpoint_hash_manifest.json"

# Marker left behind when a checkpoint save does not finish its manifest.
_PENDING_MANIFEST_FILENAME = "_checkpoint_hash_manifest.pending"

# Stream-read chunk size for large-file hashing.  Keeps memory usage
# constant regardless of file size.
_READ_CHUNK_SIZE = 1 << 20  # 1 MiB


class CheckpointManifestError(Exception):
    """Raised when a checkpoint manifest is incomplete or does not match its files."""

    pass


def _open_file(filepath: str | os.PathLike, mode: str):
    if filesystem.is_remote(filepath):
        fs, path = filesystem._resolve(filepath)
        return fs.open(path, mode)
    return open(filepath, mode)


def _make_directory(directory: str | os.PathLike) -> None:
    if filesystem.is_remote(directory):
        fs, path = filesystem._resolve(directory)
        fs.makedirs(path, exist_ok=True)
    else:
        os.makedirs(directory, exist_ok=True)


def _is_writer_and_reader_rank() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _clear_manifest_pending(directory: str | os.PathLike) -> None:
    pending_path = filesystem.join(directory, _PENDING_MANIFEST_FILENAME)
    if filesystem.exists(pending_path):
        if filesystem.is_remote(pending_path):
            fs, path = filesystem._resolve(pending_path)
            fs.rm(path)
        else:
            os.remove(pending_path)


def _compute_file_hash(filepath: str | os.PathLike) -> str:
    """Stream-hash a single file with SHA-256, returning a hex digest."""
    h = hashlib.sha256()
    with _open_file(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_directory_hashes(directory: str | os.PathLike) -> dict[str, str]:
    """Hash every file in *directory* (non-recursive, sorted by name)."""
    entries: list[tuple[str, str]] = []
    for filename in sorted(filesystem.listdir(directory)):
        filepath = filesystem.join(directory, filename)
        if filesystem.isfile(filepath) and filename not in {_MANIFEST_FILENAME, _PENDING_MANIFEST_FILENAME}:
            entries.append((filename, _compute_file_hash(filepath)))
    return dict(entries)


def _write_manifest(directory: str | os.PathLike) -> None:
    """Compute SHA-256 hashes for every file in *directory* and write the manifest."""
    if _is_writer_and_reader_rank():
        hashes = _compute_directory_hashes(directory)
        payload: dict[str, Any] = {
            "granularity": "file",
            "algorithm": "sha256",
            "hash": hashes,
        }
        manifest_path = filesystem.join(directory, _MANIFEST_FILENAME)
        with _open_file(manifest_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        _clear_manifest_pending(directory)
        logger.info("Wrote checkpoint hash manifest to %s (%d hashes).", manifest_path, len(hashes))


def mark_checkpoint_manifest_pending(manager: Any, curr_step: int, last_step: bool = False) -> None:
    """Mark a checkpoint as incomplete before its save starts."""
    if manager._should_save(curr_step, last_step) and _is_writer_and_reader_rank():
        directory = manager._create_checkpoint_id(curr_step)
        _make_directory(directory)
        pending_path = filesystem.join(directory, _PENDING_MANIFEST_FILENAME)
        with _open_file(pending_path, "w") as pending_file:
            pending_file.write("pending\n")


def write_checkpoint_manifest(manager: Any, curr_step: int, last_step: bool = False) -> None:
    """Write the integrity manifest for *checkpoint_id*, handling both sync and async save modes."""
    if not manager._should_save(curr_step, last_step):
        return
    checkpoint_id = manager._create_checkpoint_id(curr_step)
    if last_step or manager.async_mode == AsyncMode.DISABLED:
        _write_manifest(checkpoint_id)
    else:
        if manager.save_future is None:
            return

        save_future = manager.save_future
        manifest_future = Future()

        def _write_manifest_cb(fut: Future) -> None:
            if fut.cancelled():
                manifest_future.cancel()
                return
            try:
                fut.result()
                _write_manifest(checkpoint_id)
            except Exception as exc:
                logger.exception("Failed to write checkpoint manifest for %s.", checkpoint_id)
                manifest_future.set_exception(exc)
            else:
                manifest_future.set_result(None)

        save_future.add_done_callback(_write_manifest_cb)
        manager.save_future = manifest_future


def _load_manifest(directory: str | os.PathLike) -> dict[str, Any]:
    """Load and parse the manifest for *directory*.

    Raises ``FileNotFoundError`` if the manifest is absent.
    Raises ``CheckpointManifestError`` if manifest creation is incomplete.
    Raises ``ValueError`` if the manifest is incompatible.
    """
    pending_path = filesystem.join(directory, _PENDING_MANIFEST_FILENAME)
    if filesystem.exists(pending_path):
        raise CheckpointManifestError(
            f"Checkpoint hash manifest is incomplete at {directory}; found pending marker {pending_path}."
        )

    manifest_path = filesystem.join(directory, _MANIFEST_FILENAME)
    if not filesystem.exists(manifest_path):
        raise FileNotFoundError(
            f"Checkpoint hash manifest not found at {manifest_path}. "
            f"Checkpoint was likely saved without verify_hash_manifest=True. "
            f"If you want to load this checkpoint, you can set verify_hash_manifest=False."
        )
    with _open_file(manifest_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("hash"), dict):
        raise ValueError(f"Manifest at {manifest_path} is missing a valid 'hash' mapping.")
    if payload.get("granularity") != "file":
        raise ValueError(
            f"Manifest granularity is '{payload.get('granularity')}', but this verifier only supports 'file'."
        )
    return payload


def verify_checkpoint_manifest(directory: str | os.PathLike) -> None:
    """Re-hash every file listed in the manifest and compare."""
    if _is_writer_and_reader_rank():
        manifest_data = _load_manifest(directory)
        expected: dict[str, str] = manifest_data.get("hash", {})

        mismatches: list[str] = []
        for filename, expected_hash in expected.items():
            filepath = filesystem.join(directory, filename)
            actual_hash = _compute_file_hash(filepath)

            if actual_hash != expected_hash:
                mismatches.append(
                    f"  {filename}: hash mismatch (expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
                )

        if mismatches:
            raise CheckpointManifestError(
                f"Checkpoint hash manifest verification failed for {directory}:\n" + "\n".join(mismatches)
            )
        logger.info("Checkpoint hash manifest verified for %s", directory)
