# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protect checkpoint tensor semantics at the export serialization boundary."""

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


class IdentityHFAdapter:
    """Already-mapped HF keys, including the adapter's float32 tid2eid format."""

    fqn_to_index_mapping = None

    @staticmethod
    def to_hf(state_dict):
        return state_dict


@pytest.fixture
def export_module():
    root = Path(__file__).resolve().parents[4]
    path = root / "torchtitan_npu/tools/checkpoint_conversion/convert_to_hf.py"
    spec = importlib.util.spec_from_file_location("checkpoint_export_dtype", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_exported_tensors(output):
    files = list(output.glob("*.safetensors"))
    assert files, "The real HF/DCP writer must produce safetensors files"
    restored = {}
    for file in files:
        restored.update(load_file(file))
    return restored


DTYPES = [torch.bfloat16, torch.float16, torch.float32]
DTYPE_IDS = ["bf16", "fp16", "fp32"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_export_dtype_preserves_hash_routing_expert_ids(export_module, tmp_path, dtype):
    # DeepSeek-V4's to_hf stores integer-valued routing IDs as float32.
    original = torch.tensor([[255.0, 257.0, 383.0]], dtype=torch.float32)
    key = "layers.0.ffn.gate.tid2eid"
    output = tmp_path / "hash-router"

    export_module._save_as_hf({key: original}, IdentityHFAdapter(), dtype, output)
    stored = read_exported_tensors(output)[key]
    expert_ids = stored.to(torch.int64)

    torch.testing.assert_close(expert_ids, torch.tensor([[255, 257, 383]]), rtol=0, atol=0)
    torch.testing.assert_close(stored, original, rtol=0, atol=0)
    expert_outputs = torch.arange(384, dtype=torch.float32)
    torch.testing.assert_close(expert_outputs[expert_ids], torch.tensor([[255.0, 257.0, 383.0]]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_export_dtype_still_converts_floating_weights(export_module, tmp_path, dtype):
    source = {"layers.0.ffn.gate.weight": torch.tensor([[1.25, -2.5, 3.125]])}
    output = tmp_path / "weights"

    export_module._save_as_hf(source, IdentityHFAdapter(), dtype, output)
    stored = read_exported_tensors(output)

    assert stored["layers.0.ffn.gate.weight"].dtype == dtype
    torch.testing.assert_close(
        stored["layers.0.ffn.gate.weight"].float(), torch.tensor([[1.25, -2.5, 3.125]]), rtol=0, atol=0
    )
