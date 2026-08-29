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
    metadata = qwen_cp.build_sequence_metadata(varlen, _Mesh(), batch=1, length=8)

    assert metadata.cu_seqlens.dtype == torch.int64
    assert metadata.cu_seqlens.device.type == "cpu"
    assert torch.equal(metadata.cu_seqlens, torch.tensor([0, 3, 8], dtype=torch.int64))
    assert torch.equal(metadata.cu_seqlens_cpu, metadata.cu_seqlens)
    assert torch.equal(metadata.actual_seq_qlen, torch.tensor([3, 8], dtype=torch.int64))


def test_prepare_sequence_metadata_replaces_varlen_input(qwen_cp):
    varlen = types.SimpleNamespace(cu_seq_q=torch.tensor([0, 2, 5], dtype=torch.int32))
    args = (torch.empty(1, 5, 4),)
    module = types.SimpleNamespace(context_parallel_mesh=_Mesh(size=1, rank=0))

    returned_args, returned_kwargs = qwen_cp.prepare_sequence_metadata(module, args, {"attention_masks": varlen})

    assert returned_args is args
    assert isinstance(returned_kwargs["attention_masks"], qwen_cp.QwenCPMetadata)
    assert torch.equal(returned_kwargs["attention_masks"].actual_seq_qlen, torch.tensor([2, 5]))


def test_shard_local_heads_selects_rank_chunk(qwen_cp):
    tensor = torch.arange(8).view(4, 2)
    shard = qwen_cp.shard_local_heads(tensor, _Mesh(size=2, rank=1))
    assert torch.equal(shard, tensor[2:])
