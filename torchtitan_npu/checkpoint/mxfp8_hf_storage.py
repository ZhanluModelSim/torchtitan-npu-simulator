# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/v2.12.0/torch/distributed/checkpoint/hf_storage.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import os  # noqa: TC003
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import _getdtype
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    SUFFIX,
    _HFStorageInfo,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem  # noqa: TC002
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.futures import Future

from .mxfp8_dequant_backend import dequantize_mxfp8_on_npu

MXFP8_BLOCK_SIZE = 32
MXFP8_SCALE_PAIR_SIZE = MXFP8_BLOCK_SIZE * 2


def get_safetensors_dtype(dtype: str) -> torch.dtype:
    if dtype == "F8_E8M0":
        return torch.float8_e8m0fnu
    return _getdtype(dtype)


@dataclass(frozen=True, slots=True)
class MXFP8TensorDescriptor:
    qdata_key: str
    scale_key: str
    qdata_metadata: TensorStorageMetadata
    scale_metadata: TensorStorageMetadata


def get_mxfp8_regions(
    req: ReadItem,
    descriptor: MXFP8TensorDescriptor,
) -> tuple[ChunkStorageMetadata, ChunkStorageMetadata, int]:
    saved_offsets = req.storage_index.offset
    if saved_offsets is None:
        raise ValueError(f"Missing saved offsets for MXFP8 tensor {descriptor.qdata_key}")

    global_start = torch.Size(
        int(saved + local) for saved, local in zip(saved_offsets, req.storage_offsets, strict=True)
    )
    global_k_end = int(global_start[-1] + req.lengths[-1])
    aligned_k_start = (int(global_start[-1]) // MXFP8_SCALE_PAIR_SIZE) * MXFP8_SCALE_PAIR_SIZE
    aligned_k_end = min(
        int(descriptor.qdata_metadata.size[-1]),
        ((global_k_end + MXFP8_SCALE_PAIR_SIZE - 1) // MXFP8_SCALE_PAIR_SIZE) * MXFP8_SCALE_PAIR_SIZE,
    )

    qdata_region = ChunkStorageMetadata(
        offsets=torch.Size((*global_start[:-1], aligned_k_start)),
        sizes=torch.Size((*req.lengths[:-1], aligned_k_end - aligned_k_start)),
    )
    scale_k_start = aligned_k_start // MXFP8_BLOCK_SIZE
    scale_k_end = (aligned_k_end + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    scale_region = ChunkStorageMetadata(
        offsets=torch.Size((*global_start[:-1], scale_k_start)),
        sizes=torch.Size((*req.lengths[:-1], scale_k_end - scale_k_start)),
    )
    return qdata_region, scale_region, int(global_start[-1]) - aligned_k_start


def make_slices(offsets: torch.Size, lengths: torch.Size) -> tuple[slice, ...]:
    return tuple(slice(int(offset), int(offset + length)) for offset, length in zip(offsets, lengths, strict=True))


class MXFP8HuggingFaceStorageReader(HuggingFaceStorageReader):
    """Load fixed-format MXFP8 safetensors through the standard DCP flow."""

    def __init__(
        self,
        path: str,
        thread_count: int = 4,
        npu_max_inflight: int = 2,
    ) -> None:
        super().__init__(path=path, thread_count=thread_count)
        if npu_max_inflight <= 0:
            raise ValueError("npu_max_inflight must be positive")

        self.npu_max_inflight = npu_max_inflight
        self._mxfp8_tensors: dict[str, MXFP8TensorDescriptor] = {}

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        super().reset(checkpoint_id)
        self._mxfp8_tensors = {}

    def read_metadata(self) -> Metadata:
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
        self._mxfp8_tensors = self._discover_mxfp8_tensors(metadata)
        return metadata

    @staticmethod
    def _discover_mxfp8_tensors(metadata: Metadata) -> dict[str, MXFP8TensorDescriptor]:
        tensors: dict[str, MXFP8TensorDescriptor] = {}
        for qdata_key, qdata_md in metadata.state_dict_metadata.items():
            if not isinstance(qdata_md, TensorStorageMetadata) or qdata_md.properties.dtype != torch.float8_e4m3fn:
                continue
            if not qdata_key.endswith(".weight"):
                raise ValueError(f"MXFP8 qdata key must end with .weight: {qdata_key}")
            scale_key = f"{qdata_key[: -len('.weight')]}.scale"
            scale_md = metadata.state_dict_metadata.get(scale_key)
            if not isinstance(scale_md, TensorStorageMetadata):
                raise ValueError(f"Missing MXFP8 scale tensor for {qdata_key}: expected {scale_key}")
            num_blocks = (qdata_md.size[-1] + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
            expected_scale_shape = torch.Size((*qdata_md.size[:-1], num_blocks))
            if scale_md.size != expected_scale_shape:
                raise ValueError(
                    f"Invalid MXFP8 scale shape for {scale_key}: {scale_md.size}, expected {expected_scale_shape}"
                )
            tensors[qdata_key] = MXFP8TensorDescriptor(qdata_key, scale_key, qdata_md, scale_md)
        return tensors

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
            storage_info = self.storage_data[item.storage_index]
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

    def _read_mxfp8_npu_inputs(
        self,
        qdata_file: Any,
        qdata_file_path: str,
        req: ReadItem,
        descriptor: MXFP8TensorDescriptor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        qdata_region, scale_region, crop_start = get_mxfp8_regions(req, descriptor)
        qdata = self._read_tensor_region(
            descriptor.qdata_key,
            descriptor.qdata_metadata,
            qdata_region,
            qdata_file_path,
            qdata_file,
        )
        scale = self._read_tensor_region(
            descriptor.scale_key,
            descriptor.scale_metadata,
            scale_region,
            qdata_file_path,
            qdata_file,
        )
        return qdata, scale, crop_start

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        has_mxfp8 = any(req.storage_index.fqn in self._mxfp8_tensors for req in plan.items)
        if not has_mxfp8:
            return super().read_data(plan, planner)
        return self._read_data_npu(plan, planner)

    def _read_data_npu(
        self,
        plan: LoadPlan,
        planner: LoadPlanner,
    ) -> Future[None]:
        requests = list(plan.items)
        targets = {id(req): planner.resolve_tensor(req).detach() for req in requests}
        device = next(iter(targets.values())).device
        device_module = torch.get_device_module(device)
        stream = device_module.Stream(device=device)
        inflight: deque[tuple[Any, ReadItem, torch.Tensor, tuple[torch.Tensor, ...]]] = deque()

        def finish_oldest() -> None:
            event, request, target, _keepalive = inflight.popleft()
            event.synchronize()
            planner.commit_tensor(request, target)

        per_file: dict[str, list[ReadItem]] = {}
        for req in requests:
            storage_info = self.storage_data[req.storage_index]
            per_file.setdefault(storage_info.relative_path, []).append(req)

        for file_name, file_requests in per_file.items():
            with safe_open(filename=file_name, framework="pt", device="cpu") as qdata_file:
                for req in file_requests:
                    target = targets[id(req)]
                    descriptor = self._mxfp8_tensors.get(req.storage_index.fqn)
                    with device_module.stream(stream):
                        if descriptor is None:
                            tensor = qdata_file.get_slice(req.storage_index.fqn)[
                                make_slices(req.storage_offsets, req.lengths)
                            ]
                            target.copy_(tensor, non_blocking=True)
                            keepalive = (tensor,)
                        else:
                            qdata, scale, crop_start = self._read_mxfp8_npu_inputs(
                                qdata_file,
                                file_name,
                                req,
                                descriptor,
                            )
                            resources = dequantize_mxfp8_on_npu(
                                qdata,
                                scale,
                                crop_start=crop_start,
                                requested_length=int(req.lengths[-1]),
                                target=target,
                            )
                            keepalive = (qdata, scale, *resources)
                        event = device_module.Event()
                        event.record(stream)
                    inflight.append((event, req, target, keepalive))
                    if len(inflight) >= self.npu_max_inflight:
                        finish_oldest()

        while inflight:
            finish_oldest()
        future = Future()
        future.set_result(None)
        return future
