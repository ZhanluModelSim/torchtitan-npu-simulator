# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchtitan_npu.patches.torch import clip_grad


@pytest.mark.parametrize("async_mode", ["disabled", "async", "async_with_pinned_mem"])
def test_checkpoint_manager_dcp_save_builds_writer_for_all_async_modes(monkeypatch, tmp_path, async_mode):
    from torch.distributed.checkpoint.filesystem import FileSystemWriter
    from torchtitan.components.checkpoint import AsyncMode

    from torchtitan_npu.patches.torch import checkpoint

    mode = AsyncMode(async_mode)
    stager = object()
    upload_future = object()
    async_response = checkpoint.AsyncSaveResponse(
        staging_completion=object(),
        upload_completion=upload_future,
    )
    captured = {}

    def fake_dcp_save(*args, storage_writer=None, **kwargs):
        captured["writer"] = storage_writer
        captured["kwargs"] = kwargs
        return "saved"

    def fake_dcp_async_save(*args, storage_writer=None, async_stager=None, **kwargs):
        captured["writer"] = storage_writer
        captured["async_stager"] = async_stager
        captured["kwargs"] = kwargs
        return async_response

    manager = SimpleNamespace(
        pg=None,
        stager=stager,
        _npu_checkpoint_sync_files=False,
        _npu_checkpoint_drop_page_cache_after_save=True,
    )

    monkeypatch.setattr(checkpoint.dcp, "save", fake_dcp_save)
    monkeypatch.setattr(checkpoint.dcp, "async_save", fake_dcp_async_save)

    result = checkpoint._patched_checkpoint_manager_dcp_save(manager, {}, str(tmp_path / "checkpoint"), mode)
    writer = captured["writer"]

    assert isinstance(writer, FileSystemWriter)
    assert writer.sync_files is False
    writer_with_options: Any = writer
    assert writer_with_options._npu_drop_page_cache_after_save is True

    restored_writer = pickle.loads(pickle.dumps(writer))
    restored_writer_with_options: Any = restored_writer
    assert restored_writer.sync_files is False
    assert restored_writer_with_options._npu_drop_page_cache_after_save is True

    if mode == AsyncMode.DISABLED:
        assert result == "saved"
        assert "async_stager" not in captured
        assert writer.per_thread_copy_ahead > 0
    elif mode == AsyncMode.ASYNC:
        assert result is upload_future
        assert writer.per_thread_copy_ahead == 0
        assert captured["async_stager"] is not None
        assert captured["async_stager"] is not writer
        assert (
            captured["async_stager"]._config.use_pinned_memory,
            captured["async_stager"]._config.use_shared_memory,
            captured["async_stager"]._config.use_async_staging,
            captured["async_stager"]._config.use_non_blocking_copy,
        ) == (False, False, False, False)
    else:
        assert result is async_response
        assert writer.per_thread_copy_ahead == 0
        assert captured["async_stager"] is stager
        assert captured["kwargs"]["async_checkpointer_type"] is checkpoint.AsyncCheckpointerType.PROCESS


def test_checkpoint_manager_dcp_save_uses_global_hf_shard_mapping(monkeypatch, tmp_path, caplog):
    from torchtitan.components.checkpoint import AsyncMode

    from torchtitan_npu.patches.torch import checkpoint

    source_mapping = {
        "kept.weight": 2,
        "remote_expert.weight": 3,
        "unused_mtp.weight": 4,
    }
    adapter = SimpleNamespace(
        fqn_to_index_mapping=source_mapping.copy(),
        to_hf=lambda state_dict: {
            "kept.weight": torch.ones(1),
        },
    )
    captured = {}

    def capture_writer(*args, fqn_to_index_mapping, **kwargs):
        captured["writer_mapping"] = fqn_to_index_mapping
        return SimpleNamespace()

    def capture_consolidation(*args, fqn_to_index_mapping, **kwargs):
        captured["consolidation_mapping"] = fqn_to_index_mapping

    metadata = SimpleNamespace(
        state_dict_metadata={
            "kept.weight": object(),
            "remote_expert.weight": object(),
        }
    )

    monkeypatch.setattr(checkpoint, "HuggingFaceStorageWriter", capture_writer)
    monkeypatch.setattr(checkpoint.dcp, "save", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(checkpoint, "consolidate_safetensors_files_on_every_rank", capture_consolidation)

    manager = SimpleNamespace(
        sd_adapter=adapter,
        _npu_checkpoint_sync_files=True,
        _npu_checkpoint_drop_page_cache_after_save=False,
    )

    result = checkpoint._patched_checkpoint_manager_dcp_save(
        manager,
        {},
        str(tmp_path / "checkpoint"),
        AsyncMode.DISABLED,
        to_hf=True,
    )

    expected_mapping = {
        "kept.weight": 2,
        "remote_expert.weight": 3,
    }
    assert result is metadata
    assert captured["writer_mapping"] == source_mapping
    assert captured["consolidation_mapping"] == expected_mapping
    assert adapter.fqn_to_index_mapping == source_mapping
    assert "unused_mtp.weight" in caplog.text


def test_group_dtensors_by_layout_groups_non_dtensors_together():
    tensor_a = torch.randn(2, 2)
    tensor_b = torch.randn(2, 2)

    grouped = clip_grad.group_dtensors_by_layout([tensor_a, tensor_b])

    assert len(grouped) == 1
    assert ("non_dtensor", None) in grouped
    assert grouped[("non_dtensor", None)] == [tensor_a, tensor_b]


def test_group_dtensors_by_layout_handles_empty_input():
    grouped = clip_grad.group_dtensors_by_layout([])

    assert grouped == {}
