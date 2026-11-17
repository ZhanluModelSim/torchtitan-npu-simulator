# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KVCompressEpilogV2 cache dequantization primitives.

The operator stores each compressed KV row as a low-precision data region,
followed by per-group BF16 scales and 32-byte-alignment padding. This module
decodes the ``mxfp8_bf16`` and ``mxfp4_bf16`` layouts.
"""

__all__ = ["dequantize_mx_bf16", "kv_cache_layout"]

import torch

from torchao_npu.quantization.quant_primitives.mx import _get_fp4_e2m1_pair_lut

_QUANT_MODE_IDS = {
    "mxfp8_bf16": 2,
    "mxfp4_bf16": 4,
}
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _mode_id(quant_mode: int | str) -> int:
    """Normalize a public KVCompressEpilogV2 quantization mode."""

    if isinstance(quant_mode, str):
        mode = _QUANT_MODE_IDS.get(quant_mode.strip().lower())
    elif type(quant_mode) is int:
        mode = quant_mode
    else:
        mode = None
    if mode not in (2, 4):
        raise ValueError("quant_mode must be 2/4 or mxfp8_bf16/mxfp4_bf16")
    return mode


def kv_cache_layout(
    d: int,
    quant_mode: int | str,
    group_size: int = 32,
) -> tuple[int, int, int, int]:
    """Calculate the byte layout of one KVCompressEpilogV2 cache row.

    This decoder does not impose the encoder's ``d <= 8192`` limit because it
    only interprets an existing cache buffer.

    Args:
        d: Number of values in an uncompressed cache row.
        quant_mode: ``2``/``"mxfp8_bf16"`` or ``4``/``"mxfp4_bf16"``.
        group_size: Number of values sharing one BF16 scale. MXFP8 requires
            32; MXFP4 accepts 16 or 32.

    Returns:
        ``(num_groups, data_bytes, payload_bytes, aligned_row_bytes)``.

    Raises:
        ValueError: If the layout arguments are unsupported or inconsistent.
    """

    mode = _mode_id(quant_mode)
    if type(group_size) is not int:
        raise ValueError("group_size must be an integer")
    if mode == 2 and group_size != 32:
        raise ValueError("MXFP8 requires group_size=32")
    if mode == 4 and group_size not in (16, 32):
        raise ValueError("MXFP4 requires group_size=16 or group_size=32")
    if type(d) is not int or d <= 0:
        raise ValueError("d must be a positive integer")
    if d % group_size != 0:
        raise ValueError("d must be divisible by group_size")

    num_groups = d // group_size
    data_bytes = d if mode == 2 else d // 2
    payload_bytes = data_bytes + 2 * num_groups
    aligned_row_bytes = (payload_bytes + 31) // 32 * 32
    return num_groups, data_bytes, payload_bytes, aligned_row_bytes


@torch.no_grad()
def dequantize_mx_bf16(
    cache: torch.Tensor,
    d: int,
    quant_mode: int | str,
    group_size: int = 32,
    fp8_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a KVCompressEpilogV2 cache.

    Padding and bytes beyond the aligned row payload are ignored. Every input
    row is decoded; the caller remains responsible for selecting initialized
    cache rows. ``round_scale`` is unnecessary because the stored BF16 scales
    fully determine reconstruction.

    Args:
        cache: Cache tensor shaped ``[num_rows, row_bytes]``. MXFP4 requires
            ``torch.uint8``. MXFP8 accepts ``torch.uint8`` or a native FP8 dtype.
        d: Number of values in each reconstructed row.
        quant_mode: ``2``/``"mxfp8_bf16"`` or ``4``/``"mxfp4_bf16"``.
        group_size: Number of values sharing one BF16 scale.
        fp8_dtype: FP8 element dtype for a uint8 MXFP8 cache. For a native FP8
            cache it is inferred, and a conflicting explicit value is rejected.
        output_dtype: Element dtype of the reconstructed tensor.

    Returns:
        A tensor shaped ``[num_rows, d]`` with dtype ``output_dtype``.

    Raises:
        ValueError: If the cache tensor or layout metadata is invalid.
    """

    mode = _mode_id(quant_mode)
    num_groups, data_bytes, payload_bytes, row_bytes = kv_cache_layout(d, mode, group_size)
    _validate_cache(cache, row_bytes)

    if mode == 2:
        fp8_dtype = _validate_mxfp8_cache(cache, fp8_dtype)
    else:
        _validate_mxfp4_cache(cache, fp8_dtype)

    # Both supported cache dtypes occupy one byte. Reinterpret native FP8 as
    # bytes instead of numerically converting it, which would corrupt scales.
    raw = cache.detach().view(torch.uint8).contiguous()
    num_rows = raw.shape[0]
    scales = _decode_bf16_scales(raw, data_bytes, payload_bytes)

    if mode == 2:
        assert fp8_dtype is not None
        values = raw[:, :data_bytes].contiguous().view(fp8_dtype).float()
    else:
        values = _decode_mxfp4_values(raw[:, :data_bytes], num_rows, d)

    restored = values.reshape(num_rows, num_groups, group_size) * scales.float().unsqueeze(-1)
    return restored.reshape(num_rows, d).to(output_dtype)


def _validate_cache(cache: torch.Tensor, row_bytes: int) -> None:
    """Validate the common cache shape and storage width."""

    if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
        raise ValueError("cache must be a two-dimensional torch.Tensor")
    if cache.shape[1] < row_bytes:
        raise ValueError(f"cache row requires at least {row_bytes} bytes")


def _validate_mxfp8_cache(cache: torch.Tensor, fp8_dtype: torch.dtype | None) -> torch.dtype:
    """Validate MXFP8 storage and return the dtype used to decode its bytes."""

    if cache.dtype in _FP8_DTYPES:
        if fp8_dtype is not None and fp8_dtype != cache.dtype:
            raise ValueError("fp8_dtype conflicts with cache.dtype")
        return cache.dtype
    if cache.dtype != torch.uint8:
        raise ValueError("MXFP8 cache must use uint8, float8_e4m3fn, or float8_e5m2")
    if fp8_dtype not in _FP8_DTYPES:
        raise ValueError("fp8_dtype must be specified for a uint8 MXFP8 cache")
    assert fp8_dtype is not None
    return fp8_dtype


def _validate_mxfp4_cache(cache: torch.Tensor, fp8_dtype: torch.dtype | None) -> None:
    """Validate packed MXFP4 storage arguments."""

    if cache.dtype != torch.uint8:
        raise ValueError("packed MXFP4 cache must use torch.uint8")
    if fp8_dtype is not None:
        raise ValueError("fp8_dtype does not apply to MXFP4")


def _decode_bf16_scales(raw: torch.Tensor, data_bytes: int, payload_bytes: int) -> torch.Tensor:
    """Reinterpret contiguous scale bytes as BF16."""

    scale_bytes = raw[:, data_bytes:payload_bytes].contiguous()
    return scale_bytes.view(torch.bfloat16)


def _decode_mxfp4_values(
    packed: torch.Tensor,
    num_rows: int,
    d: int,
) -> torch.Tensor:
    """Decode low-nibble-first E2M1 values into FP32."""

    # Reuse MXFP4's lookup table for the packed E2M1 representation.
    # pyrefly: ignore [bad-assignment, bad-argument-type, bad-argument-count]
    lut: torch.Tensor = _get_fp4_e2m1_pair_lut(packed.device, torch.bfloat16, True)
    indices = packed.contiguous().reshape(-1).long()
    values = torch.index_select(lut, dim=0, index=indices)
    return values.reshape(num_rows, d).float()
