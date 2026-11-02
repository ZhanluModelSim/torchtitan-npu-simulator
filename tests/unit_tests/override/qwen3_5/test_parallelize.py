# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the Qwen context-parallel metadata boundary."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# Metadata conversion is CPU-only; the remaining CP tests await the folded migration.
_requires_folded_cp = pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
)

_MODULE_PATH = Path(__file__).resolve().parents[4] / "torchtitan_npu" / "override" / "qwen3_5" / "parallelize.py"


class _Mesh:
    def __init__(self, size=2, rank=1):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


@pytest.fixture
def qwen_cp(monkeypatch):
    compile_module = types.ModuleType("torchtitan.distributed.compile")
    compile_module.apply_compile = lambda *args, **kwargs: None
    fsdp_module = types.ModuleType("torchtitan.distributed.fsdp")
    fsdp_module.resolve_fsdp_mesh = lambda *args, **kwargs: None
    fsdp_module.resolve_sparse_fsdp_mesh = lambda *args, **kwargs: None
    fsdp_module.apply_fsdp_to_decoder = lambda *args, **kwargs: None
    fsdp_module.apply_fsdp_to_vision_encoder = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchtitan.distributed.compile", compile_module)
    monkeypatch.setitem(sys.modules, "torchtitan.distributed.fsdp", fsdp_module)

    spec = importlib.util.spec_from_file_location("qwen_cp_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_sequence_metadata_caches_cpu_lengths(qwen_cp):
    varlen = types.SimpleNamespace(cu_seq_q=torch.tensor([0, 3, 8], dtype=torch.int32))
    metadata = qwen_cp.build_sequence_metadata(varlen)

    assert metadata.cu_seqlens.dtype == torch.int64
    assert metadata.cu_seqlens.device.type == "cpu"
    assert torch.equal(metadata.cu_seqlens, torch.tensor([0, 3, 8], dtype=torch.int64))
    assert torch.equal(metadata.cu_seqlens_cpu, metadata.cu_seqlens)
    assert torch.equal(metadata.actual_seq_qlen, torch.tensor([3, 8], dtype=torch.int64))


def test_prepare_sequence_metadata_replaces_varlen_input(qwen_cp):
    varlen = types.SimpleNamespace(cu_seq_q=torch.tensor([0, 2, 5], dtype=torch.int32))
    args = (torch.empty(1, 5, 4),)
    module = types.SimpleNamespace()

    returned_args, returned_kwargs = qwen_cp.prepare_sequence_metadata(module, args, {"attention_masks": varlen})

    assert returned_args is args
    assert isinstance(returned_kwargs["attention_masks"], qwen_cp.QwenCPMetadata)
    assert torch.equal(returned_kwargs["attention_masks"].actual_seq_qlen, torch.tensor([2, 5]))


@_requires_folded_cp
def test_shard_local_heads_selects_rank_chunk(qwen_cp):
    tensor = torch.arange(8).view(4, 2)
    shard = qwen_cp.shard_local_heads(tensor, _Mesh(size=2, rank=1))
    assert torch.equal(shard, tensor[2:])


def _plain_cp_dims():
    return types.SimpleNamespace(cp_enabled=True, ep_enabled=False, tp_enabled=False)


def _plain_cp_model():
    return types.SimpleNamespace(vision_encoder=None)


@_requires_folded_cp
def test_cp_parallelizer_refuses_unmigrated_folded_path(qwen_cp):
    with pytest.raises(NotImplementedError, match="not migrated to the folded token layout"):
        qwen_cp.parallelize_qwen3_5_cp(
            _plain_cp_model(),
            parallel_dims=_plain_cp_dims(),
            training=types.SimpleNamespace(),
            parallelism=types.SimpleNamespace(),
            compile_config=types.SimpleNamespace(),
            ac_config=None,
            dump_folder="unused",
        )


@_requires_folded_cp
def test_npu_dispatcher_refuses_cp_before_touching_the_model(qwen_cp):
    # The refusal must fire before any FSDP wrap/parallelize call: the fake
    # model exposes neither, so reaching them would raise AttributeError.
    with pytest.raises(NotImplementedError, match="not migrated to the folded token layout"):
        qwen_cp.parallelize_qwen3_5_npu(
            _plain_cp_model(),
            parallel_dims=_plain_cp_dims(),
            training=types.SimpleNamespace(),
            parallelism=types.SimpleNamespace(spmd_backend="spmd_types"),
            compile_config=types.SimpleNamespace(enable=False, components=()),
            ac_config=None,
            dump_folder="unused",
        )
