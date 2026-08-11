# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint._hf_utils import CUSTOM_METADATA_KEY, SAVED_OFFSETS_KEY
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem

from torchtitan_npu.checkpoint import MXFP8HuggingFaceStorageReader


def test_mxfp8_loader_end_to_end(monkeypatch, tmp_path):
    shape = (2, 128)
    qdata = torch.linspace(-3, 3, 256).reshape(shape).to(torch.float8_e4m3fn)
    scale_bits = torch.tensor([[126, 127, 128, 129], [129, 128, 127, 126]], dtype=torch.uint8)
    scale = scale_bits.view(torch.float8_e8m0fnu)
    norm = torch.tensor([0.5, 1.5])
    qdata_files = ("qdata-0.safetensors", "qdata-1.safetensors")
    scale_files = ("scale-0.safetensors", "scale-1.safetensors")

    def save_shard(file_name, tensors, offsets):
        sharding_info = {
            key: {SAVED_OFFSETS_KEY: list(offsets[key])}
            for key in tensors
        }
        save_file(
            tensors,
            tmp_path / file_name,
            metadata={CUSTOM_METADATA_KEY: json.dumps(sharding_info)},
        )

    save_shard(
        qdata_files[0],
        {"linear.weight": qdata[:, :32].contiguous(), "norm.weight": norm},
        {"linear.weight": (0, 0), "norm.weight": (0,)},
    )
    save_shard(
        qdata_files[1],
        {"linear.weight": qdata[:, 32:].contiguous()},
        {"linear.weight": (0, 32)},
    )
    save_shard(
        scale_files[0],
        {"linear.scale": scale[:, :1].contiguous()},
        {"linear.scale": (0, 0)},
    )
    save_shard(
        scale_files[1],
        {"linear.scale": scale[:, 1:].contiguous()},
        {"linear.scale": (0, 1)},
    )
    target = {
        "linear.weight": torch.empty(shape, dtype=torch.bfloat16),
        "norm.weight": torch.empty_like(norm, dtype=torch.bfloat16),
    }
    anti_mx_quant_calls = []

    def fake_npu_anti_mx_quant(qdata, mxscale, *, axis, dst_type, src_type):
        anti_mx_quant_calls.append((axis, dst_type, src_type))
        scale_u8 = mxscale.view(torch.uint8).flatten(-2)[..., : (qdata.shape[-1] + 31) // 32]
        decoded_scale = torch.ldexp(
            torch.ones_like(scale_u8, dtype=torch.float32),
            scale_u8.to(torch.int32) - 127,
        )
        expanded_scale = torch.repeat_interleave(decoded_scale, 32, dim=-1)[..., : qdata.shape[-1]]
        return qdata.to(torch.float32) * expanded_scale

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            float8_e8m0fnu=torch.float8_e8m0fnu,
            npu_anti_mx_quant=fake_npu_anti_mx_quant,
        ),
    )
    fake_device_module = SimpleNamespace(
        Stream=lambda *, device: object(),
        Event=lambda: SimpleNamespace(
            record=lambda _stream: None,
            synchronize=lambda: None,
        ),
        stream=lambda _stream: nullcontext(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: fake_device_module)

    reader = MXFP8HuggingFaceStorageReader(str(tmp_path), thread_count=1)
    dcp.load(target, storage_reader=reader)

    decoded_scale = torch.ldexp(
        torch.ones_like(scale_bits, dtype=torch.float32),
        scale_bits.to(torch.int32) - 127,
    )
    expected = qdata.to(torch.float32) * torch.repeat_interleave(decoded_scale, 32, dim=-1)
    torch.testing.assert_close(target["linear.weight"], expected.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(target["norm.weight"], norm.to(torch.bfloat16), rtol=0, atol=0)
    assert anti_mx_quant_calls == [
        (-1, torch.float32, torch.float8_e4m3fn),
        (-1, torch.float32, torch.float8_e4m3fn),
    ]

    dest_index = MetadataIndex("linear.weight", offset=torch.Size([0, 0]))
    storage_index = MetadataIndex("linear.weight", offset=torch.Size([0, 32]))
    request = ReadItem(
        LoadItemType.TENSOR,
        dest_index,
        torch.Size([0, 0]),
        storage_index,
        torch.Size([1, 33]),
        torch.Size([1, 20]),
    )
    descriptor = reader._mxfp8_tensors["linear.weight"]
    qdata_path = tmp_path / qdata_files[1]
    with safe_open(qdata_path, framework="pt") as qdata_handle:
        aligned_qdata, local_scale, crop_start = reader._read_mxfp8_npu_inputs(
            qdata_handle,
            str(qdata_path),
            request,
            descriptor,
        )
    torch.testing.assert_close(aligned_qdata, qdata[1:2, 64:128])
    torch.testing.assert_close(local_scale.view(torch.uint8), scale_bits[1:2, 2:4])
    assert crop_start == 1
