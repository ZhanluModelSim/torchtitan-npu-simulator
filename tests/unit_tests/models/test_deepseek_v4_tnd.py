# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts for the current DSV4 packed-token metadata.

The old master tests targeted the removed global-TND data loader and MTP
sidecar.  These tests preserve the useful boundary and compression contracts
against the current ``deepseek_v4.packed`` API instead.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKED_PATH = REPO_ROOT / "torchtitan_npu" / "models" / "deepseek_v4" / "packed.py"


def _load_packed_module(monkeypatch: pytest.MonkeyPatch):
    attention = types.ModuleType("torchtitan.models.common.attention")

    class VarlenMetadata:
        def __init__(self, *, cu_seq_q, cu_seq_k, max_q, max_k):
            self.cu_seq_q = cu_seq_q
            self.cu_seq_k = cu_seq_k
            self.max_q = max_q
            self.max_k = max_k

    attention.VarlenMetadata = VarlenMetadata
    monkeypatch.setitem(sys.modules, "torchtitan", types.ModuleType("torchtitan"))
    monkeypatch.setitem(sys.modules, "torchtitan.models", types.ModuleType("torchtitan.models"))
    monkeypatch.setitem(
        sys.modules, "torchtitan.models.common", types.ModuleType("torchtitan.models.common")
    )
    monkeypatch.setitem(sys.modules, "torchtitan.models.common.attention", attention)

    spec = importlib.util.spec_from_file_location("dsv4_packed_model_test", PACKED_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu
def test_packed_metadata_preserves_request_boundaries_and_ratio_four_blocks(monkeypatch):
    packed = _load_packed_module(monkeypatch)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    metadata = packed.build_dsv4_packed_metadata(positions, [4, 4])

    assert metadata.lengths.tolist() == [4, 4]
    assert metadata.sequence_ranges == ((0, 4), (4, 8))
    assert metadata.compression_for_ratio(4).lengths.tolist() == [1, 1]
    assert metadata.compression_for_ratio(4).block_starts.tolist() == [0, 4]
    assert metadata.compression_for_ratio(4).residual.tolist() == [0, 0]


@pytest.mark.cpu
def test_packed_metadata_maps_partial_blocks_and_valid_token_padding(monkeypatch):
    packed = _load_packed_module(monkeypatch)
    positions = torch.tensor([[0, 1, 2, 0, 1, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    metadata = packed.build_dsv4_packed_metadata(positions, [1, 2, 2], valid_tokens=valid)

    compressed = metadata.compression_for_ratio(2)
    assert metadata.lengths.tolist() == [3, 2]
    assert compressed.lengths.tolist() == [1, 1]
    assert compressed.residual.tolist() == [1, 0]
    assert compressed.storage_indices.tolist() == [0, 1]

    source = torch.arange(6).view(1, 6)
    compact = packed.compact_token_tensor(source, metadata)
    assert compact.tolist() == [0, 1, 2, 3, 4]
    restored = packed.restore_token_tensor(compact, metadata, fill_value=-1)
    assert restored.tolist() == [[0, 1, 2, 3, 4, -1]]


@pytest.mark.cpu
def test_packed_metadata_rejects_noncontiguous_document_positions(monkeypatch):
    packed = _load_packed_module(monkeypatch)
    with pytest.raises(ValueError, match="reset to 0 and increase contiguously"):
        packed.build_dsv4_packed_metadata(torch.tensor([[0, 2, 0, 1]]), [2])
