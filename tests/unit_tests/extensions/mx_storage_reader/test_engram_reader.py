# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU row/block MX Engram reads through shared safetensors I/O."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list

from torchtitan_npu.extensions.mx_storage_reader.engram import EngramHuggingFaceStorageReader


class EngramSlicePlanner(DefaultLoadPlanner):
    def lookup_tensor(self, index):
        return self.state_dict[index.fqn]

    def create_local_plan(self):
        requests = []
        for name, value in self.state_dict.items():
            region = ChunkStorageMetadata(torch.Size((17, 7)), value.shape)
            requests.extend(create_read_items_for_chunk_list(name, self.metadata.state_dict_metadata[name], [region]))
        return LoadPlan(requests)


def test_engram_mx_sidecars_and_unknown_keys(tmp_path):
    weights = {
        "layers.1.engram.embed.weight": torch.ones(41, 32).to(torch.float8_e4m3fn),
        "layers.1.engram.wkv.weight": torch.full((64, 64), 2.0).to(torch.float8_e4m3fn),
    }
    scales = {
        "layers.1.engram.embed.scale": torch.full((41, 1), 4.0).to(torch.float8_e8m0fnu),
        "layers.1.engram.wkv.scale": torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu),
    }
    save_file(weights, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(scales, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    **dict.fromkeys(weights, "model-00001-of-00002.safetensors"),
                    **dict.fromkeys(scales, "model-00002-of-00002.safetensors"),
                },
            }
        )
    )
    reader = EngramHuggingFaceStorageReader(
        str(tmp_path), table_shapes={"layers.1.engram.embed.weight": (41, 32)}, from_quantized=True
    )
    target = {key: torch.empty(value.shape) for key, value in weights.items()}
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["layers.1.engram.embed.weight"], torch.full((41, 32), 4.0), rtol=0, atol=0)
    expected = torch.tensor([[2.0, 4.0], [8.0, 16.0]]).repeat_interleave(32, 0).repeat_interleave(32, 1)
    torch.testing.assert_close(target["layers.1.engram.wkv.weight"], expected, rtol=0, atol=0)
    slices = {"layers.1.engram.embed.weight": torch.empty(20, 17), "layers.1.engram.wkv.weight": torch.empty(40, 41)}
    dcp.load(slices, storage_reader=reader, planner=EngramSlicePlanner())
    for name, value in slices.items():
        torch.testing.assert_close(
            value, target[name][17 : 17 + value.shape[0], 7 : 7 + value.shape[1]], rtol=0, atol=0
        )
    # Do not silently discard future or orphan Engram quantization keys.
    save_file(
        {"layers.1.engram.unknown.scale": torch.ones(1, 1).to(torch.float8_e8m0fnu)},
        str(tmp_path / "unknown.safetensors"),
    )
    with pytest.raises(ValueError, match="Unknown or orphan Engram quantization scale"):
        reader.read_metadata()
