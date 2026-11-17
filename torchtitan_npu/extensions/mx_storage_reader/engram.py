# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engram row/block MXFP8 loading with CPU dequantization and shared HF I/O."""

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.quantized_hf_storage import QuantizedHuggingFaceStorageReader

from .common import SafetensorsReaderMixin, weight_scale_pairs


def dequantize_engram_weight(weight, scale, *, row_block):
    """Decode official row-MX embeddings or block-MX gate projection weights."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("Engram quantized weights and scales must be matrices")
    expected = ((weight.shape[0] + row_block - 1) // row_block, (weight.shape[1] + 31) // 32)
    if tuple(scale.shape) != expected:
        raise ValueError(f"Engram scale shape {tuple(scale.shape)} does not match {expected}")
    result = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], 4096):
        end = min(start + 4096, weight.shape[0])
        scales = scale[start // row_block : (end + row_block - 1) // row_block].float()
        scales = scales.repeat_interleave(row_block, 0).repeat_interleave(32, 1)
        result[start:end] = weight[start:end].float() * scales[: end - start, : weight.shape[1]]
    return result


class EngramHuggingFaceStorageReader(SafetensorsReaderMixin, QuantizedHuggingFaceStorageReader):
    """Extend the upstream reader with Engram's official `.scale` tensors."""

    def __init__(self, path, *, table_shapes, from_quantized):
        super().__init__(path, target_dtype=torch.float32, thread_count=4)
        self.table_shapes = table_shapes
        self.from_quantized = from_quantized

    def read_metadata(self):
        metadata = (
            self._read_quantized_metadata() if self.from_quantized else HuggingFaceStorageReader.read_metadata(self)
        )
        for name, shape in self.table_shapes.items():
            stored = metadata.state_dict_metadata.get(name)
            if stored is not None and (not isinstance(stored, TensorStorageMetadata) or tuple(stored.size) != shape):
                raise ValueError(f"HF Engram table {name} metadata does not match expected shape {shape}")
        return metadata

    def _read_quantized_metadata(self):
        self._load_quantization_metadata()
        metadata = self.read_raw_metadata()
        self._raw_metadata = self._raw_tensor_metadata.copy()
        recognized_scales = set()
        for name, stored in self._raw_metadata.items():
            self._tensor_full_shapes[name] = stored.size
            if ".engram." not in name:
                continue
            if name.endswith(".scale"):
                weight_name = name.removesuffix("scale") + "weight"
                if (
                    self._weight_scale_mapping.get(weight_name) != name
                    or self._raw_metadata[weight_name].properties.dtype != torch.float8_e4m3fn
                ):
                    raise ValueError(f"Unknown or orphan Engram quantization scale: {name}")
            elif stored.properties.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                if (
                    stored.properties.dtype != torch.float8_e4m3fn
                    or not name.endswith((".engram.embed.weight", ".engram.wkv.weight"))
                    or name not in self._weight_scale_mapping
                ):
                    raise ValueError(f"Unknown or missing scale for quantized Engram weight: {name}")
                scale_name = self._weight_scale_mapping[name]
                scale = self._raw_metadata.get(scale_name)
                row_block = 1 if name.endswith(".embed.weight") else 32
                if len(stored.size) != 2 or scale is None:
                    raise ValueError(f"Invalid Engram MX weight/scale pair: {name}")
                expected = ((stored.size[0] + row_block - 1) // row_block, (stored.size[1] + 31) // 32)
                if tuple(scale.size) != expected or scale.properties.dtype != torch.float8_e8m0fnu:
                    raise ValueError(f"Invalid Engram MX scale shape or dtype: {scale_name}, expected {expected}")
                recognized_scales.add(scale_name)
        for name in recognized_scales:
            metadata.state_dict_metadata.pop(name)
        return metadata

    def _build_weight_scale_mapping(self, weight_map):
        super()._build_weight_scale_mapping(weight_map)
        for weight_name, scale_name in weight_scale_pairs(weight_map).items():
            if weight_name.endswith((".engram.embed.weight", ".engram.wkv.weight")):
                self._weight_scale_mapping[weight_name] = scale_name

    def _process_read_request(self, f, req, planner):
        name = req.storage_index.fqn
        scale = name.removesuffix("weight") + "scale"
        if ".engram." in name and f.get_slice(name).get_dtype() == "F8_E4M3" and name not in self._weight_scale_mapping:
            raise ValueError(f"Quantized Engram weight {name} requires from_quantized=True and its scale tensor")
        if scale in self._weight_map and name not in self._weight_scale_mapping:
            raise NotImplementedError(
                f"The upstream HF reader does not support the quantization format of {name}; "
                "convert non-Engram weights to a supported HF format first."
            )
        return super()._process_read_request(f, req, planner)

    def _read_quantized_tensor_with_block_alignment(self, req, safetensor_file):
        name = req.storage_index.fqn
        if not name.endswith((".engram.embed.weight", ".engram.wkv.weight")):
            return super()._read_quantized_tensor_with_block_alignment(req, safetensor_file)
        row_block = 1 if name.endswith(".embed.weight") else 32
        row, col = (a + b for a, b in zip(req.storage_index.offset, req.storage_offsets, strict=True))
        height, width = req.lengths
        r0, c0 = row // row_block * row_block, col // 32 * 32
        r1 = (row + height + row_block - 1) // row_block * row_block
        c1 = (col + width + 31) // 32 * 32
        shape = self._raw_metadata[name].size
        r1, c1 = min(r1, shape[0]), min(c1, shape[1])
        storage_path = self._raw_storage_data[req.storage_index].relative_path
        weight = self._read_tensor_region(
            name,
            self._raw_metadata[name],
            ChunkStorageMetadata(torch.Size((r0, c0)), torch.Size((r1 - r0, c1 - c0))),
            storage_path,
            safetensor_file,
        )
        scale_name = self._weight_scale_mapping[name]
        scale = self._read_tensor_region(
            scale_name,
            self._raw_metadata[scale_name],
            ChunkStorageMetadata(
                torch.Size((r0 // row_block, c0 // 32)),
                torch.Size(((r1 + row_block - 1) // row_block - r0 // row_block, (c1 + 31) // 32 - c0 // 32)),
            ),
            storage_path,
            safetensor_file,
        )
        values = dequantize_engram_weight(weight, scale, row_block=row_block)
        return values[row - r0 : row - r0 + height, col - c0 : col - c0 + width]
