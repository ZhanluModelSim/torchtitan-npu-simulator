# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/v2.12.0/torch/distributed/checkpoint/hf_storage.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared safetensors metadata and cross-file region I/O for MX readers."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import _getdtype
from torch.distributed.checkpoint._hf_utils import CUSTOM_METADATA_KEY, SAVED_OFFSETS_KEY, SUFFIX, _HFStorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)

if TYPE_CHECKING:
    from torch.distributed.checkpoint.planner import ReadItem
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list


def get_safetensors_dtype(dtype: str) -> torch.dtype:
    if dtype == "F8_E8M0":
        return torch.float8_e8m0fnu
    return _getdtype(dtype)


def make_slices(offsets: torch.Size, lengths: torch.Size) -> tuple[slice, ...]:
    return tuple(slice(int(offset), int(offset + length)) for offset, length in zip(offsets, lengths, strict=True))


def weight_scale_pairs(keys):
    """Discover explicit .weight/.scale sidecars without hiding unknown keys."""
    return {
        key: key.removesuffix("weight") + "scale"
        for key in keys
        if key.endswith(".weight") and key.removesuffix("weight") + "scale" in keys
    }


class SafetensorsReaderMixin:
    """Use with a HuggingFaceStorageReader; retain raw metadata for sidecars."""

    fs: Any
    path: Any
    load_id: str

    def read_raw_metadata(self) -> Metadata:
        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        for safetensors_file in self.fs.ls(self.path):
            if not safetensors_file.endswith(SUFFIX):
                continue
            with safe_open(safetensors_file, framework="pt") as file:
                extra_metadata = file.metadata()
                dcp_sharding_info = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(extra_metadata[CUSTOM_METADATA_KEY])

                for key in file.keys():  # noqa: SIM118
                    tensor_slice = file.get_slice(key)
                    shape = tensor_slice.get_shape()
                    dtype = get_safetensors_dtype(tensor_slice.get_dtype())
                    offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY] if dcp_sharding_info else [0] * len(shape)
                    chunk = ChunkStorageMetadata(offsets=torch.Size(offset), sizes=torch.Size(shape))

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=dtype),
                            size=torch.Size(saved + start for saved, start in zip(shape, offset, strict=True)),
                            chunks=[chunk],
                        )
                    else:
                        tensor_metadata = state_dict_metadata[key]
                        if tensor_metadata.properties.dtype != dtype:
                            raise ValueError(f"Inconsistent shard dtype for {key}")
                        tensor_metadata.chunks.append(chunk)
                        tensor_metadata.size = torch.Size(
                            max(saved, shard + start)
                            for saved, shard, start in zip(tensor_metadata.size, shape, offset, strict=True)
                        )

                    index = MetadataIndex(fqn=key, offset=offset)
                    storage_data[index] = _HFStorageInfo(
                        relative_path=safetensors_file,
                        shape=torch.Size(shape),
                        dtype=dtype,
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # pyrefly: ignore [bad-argument-type]
            storage_data=storage_data,
        )
        if metadata.storage_meta is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        self._raw_tensor_metadata = state_dict_metadata
        self._raw_storage_data = storage_data.copy()
        return metadata

    def _read_tensor_region(
        self,
        fqn: str,
        tensor_metadata: TensorStorageMetadata,
        region: ChunkStorageMetadata,
        open_file_path: str,
        open_file: Any,
    ) -> torch.Tensor:
        read_items = create_read_items_for_chunk_list(fqn, tensor_metadata, [region])
        expected_numel = math.prod(region.sizes)
        read_numel = sum(math.prod(item.lengths) for item in read_items)
        if read_numel != expected_numel:
            raise ValueError(
                f"Incomplete checkpoint shards for {fqn}: region offset={region.offsets}, size={region.sizes}"
            )

        def read_source(item: ReadItem) -> torch.Tensor:
            storage_info = self._raw_storage_data[item.storage_index]
            source_slices = make_slices(item.storage_offsets, item.lengths)
            if storage_info.relative_path == open_file_path:
                return open_file.get_slice(fqn)[source_slices]
            with safe_open(storage_info.relative_path, framework="pt", device="cpu") as source_file:
                return source_file.get_slice(fqn)[source_slices]

        if len(read_items) == 1:
            item = read_items[0]
            if all(offset == 0 for offset in item.dest_offsets) and item.lengths == region.sizes:
                return read_source(item)

        result = torch.empty(region.sizes, dtype=tensor_metadata.properties.dtype)
        for item in read_items:
            result[make_slices(item.dest_offsets, item.lengths)].copy_(read_source(item))
        return result
