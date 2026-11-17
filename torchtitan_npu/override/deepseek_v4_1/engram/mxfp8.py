# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 CANN Engram lookup backed by an authoritative FP32 Host table."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.profiler import record_function

from .ascendc import HostOffloadEngramTable

_MXFP8_BLOCK_SIZE = 32
_E4M3_MAX = 448.0
_E8M0_MIN_EXPONENT = -127
_E8M0_MAX_EXPONENT = 127


@torch.no_grad()
def _quantize_mxfp8_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows to the E4M3 data and E8M0 per-32 scale format."""
    if rows.ndim != 2 or rows.shape[1] % _MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"MXFP8 Engram rows must be 2D with width divisible by {_MXFP8_BLOCK_SIZE}, got {tuple(rows.shape)}."
        )

    blocks = rows.detach().float().unflatten(1, (-1, _MXFP8_BLOCK_SIZE))
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(amax / _E4M3_MAX))
    exponent = exponent.clamp(min=_E8M0_MIN_EXPONENT, max=_E8M0_MAX_EXPONENT)
    scale_f32 = torch.exp2(exponent)
    scale = scale_f32.to(torch.float8_e8m0fnu)
    # Use the stored E8M0 value for quantization as well, so dequantization sees
    # exactly the same power-of-two scale even at the representable boundaries.
    quantized = (blocks / scale.float().unsqueeze(-1)).clamp(min=-_E4M3_MAX, max=_E4M3_MAX)
    return quantized.flatten(1).to(torch.float8_e4m3fn), scale


def _dequantize_mxfp8_rows(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Match the official V4.1 Engram lookup: block multiply, then BF16."""
    if values.shape[-1] != scale.shape[-1] * _MXFP8_BLOCK_SIZE:
        raise RuntimeError(
            f"MXFP8 Engram Fetch returned incompatible data and scale widths: {values.shape[-1]} and {scale.shape[-1]}."
        )
    blocks = values.float().unflatten(-1, (-1, _MXFP8_BLOCK_SIZE))
    # NPU Cast/InplaceCopy does not support E8M0 -> FP32 in the current stack.
    # E8M0 stores the biased FP32 exponent directly; code 0 is 2**-127
    # (a FP32 subnormal), and code 255 denotes NaN.
    exponent = scale.view(torch.uint8).to(torch.int32)
    bits = exponent << 23
    bits = torch.where(exponent == 0, 0x00400000, bits)
    bits = torch.where(exponent == 255, 0x7FC00000, bits)
    scale_f32 = bits.view(torch.float32)
    return (blocks * scale_f32.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class MXFP8HostOffloadEngramTable(HostOffloadEngramTable):
    """Fetch MXFP8 rows while SparseAdam updates an FP32 Host master table."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostOffloadEngramTable.Config):
        quantization_chunk_rows: int = 32768

    def __init__(self, config: Config):
        super().__init__(config)
        if self.embedding_dim % _MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"MXFP8 Engram requires embedding_dim divisible by {_MXFP8_BLOCK_SIZE}, got {self.embedding_dim}."
            )
        if config.quantization_chunk_rows <= 0:
            raise ValueError(f"quantization_chunk_rows must be positive, got {config.quantization_chunk_rows}.")
        self.quantization_chunk_rows = config.quantization_chunk_rows
        self._quantized_storage: torch.Tensor | None = None
        self._quantized_scale: torch.Tensor | None = None
        self.register_load_state_dict_post_hook(self._refresh_quantized_storage_after_load)

    def _fetch_storage_dtype(self, *, param_dtype: torch.dtype) -> torch.dtype:
        return torch.float8_e4m3fn

    def _require_quantized_storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._quantized_storage is None or self._quantized_scale is None:
            raise RuntimeError("MXFP8 Engram storage is not initialized; initialize model weights before lookup.")
        return self._quantized_storage, self._quantized_scale

    def _fetch_storage(self) -> torch.Tensor:
        return self._require_quantized_storage()[0]

    def _fetch_scale(self) -> torch.Tensor:
        return self._require_quantized_storage()[1]

    def _initialize_fetch_storage(self) -> None:
        pin_memory = self.weight.is_pinned()
        self._quantized_storage = torch.empty(
            self.weight.shape,
            dtype=torch.float8_e4m3fn,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._quantized_scale = torch.empty(
            (self.weight.shape[0], self.embedding_dim // _MXFP8_BLOCK_SIZE),
            dtype=torch.float8_e8m0fnu,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._refresh_quantized_storage()

    @torch.no_grad()
    def _refresh_quantized_storage(self, row_ids: torch.Tensor | None = None) -> None:
        storage, scale = self._require_quantized_storage()
        if row_ids is not None:
            row_ids = row_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
            if row_ids.numel() == 0:
                return
            quantized, row_scale = _quantize_mxfp8_rows(self.weight.index_select(0, row_ids))
            # CPU index_copy does not dispatch for float8 yet. Both formats use
            # one byte per element, so update their storage through byte views.
            storage.view(torch.uint8).index_copy_(0, row_ids, quantized.view(torch.uint8))
            scale.view(torch.uint8).index_copy_(0, row_ids, row_scale.view(torch.uint8))
            return

        for start in range(0, self.weight.shape[0], self.quantization_chunk_rows):
            end = min(start + self.quantization_chunk_rows, self.weight.shape[0])
            quantized, row_scale = _quantize_mxfp8_rows(self.weight[start:end])
            storage[start:end].copy_(quantized)
            scale[start:end].copy_(row_scale)

    def _refresh_quantized_storage_after_load(self, module, incompatible_keys) -> None:
        if self._quantized_storage is not None:
            self._refresh_quantized_storage()

    def refresh_lookup_storage(self, row_ids: torch.Tensor) -> None:
        """Requantize the rows selected by the Host sparse optimizer step."""
        self._refresh_quantized_storage(row_ids)

    def _lookup_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _fetch_rows(self, row_ids: torch.Tensor):
        storage, scale = self._require_quantized_storage()
        buffer = self._require_elastic_buffer(storage, row_ids.numel(), scale)
        with record_function("engram::fetch_mxfp8"):
            indices = row_ids.reshape(-1).to(dtype=torch.int32).contiguous()
            fetched, fetched_scale, fetch = buffer.engram_fetch(indices)()
            return _dequantize_mxfp8_rows(fetched, fetched_scale), fetch


__all__ = ["MXFP8HostOffloadEngramTable"]
