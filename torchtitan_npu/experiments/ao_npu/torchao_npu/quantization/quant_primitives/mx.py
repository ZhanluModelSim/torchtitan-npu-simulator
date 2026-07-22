# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 low-level primitives for NPU.

These are pure tensor helpers (no autograd, no matmul) used by higher-level
ops in :mod:`torchtitan_npu.experiments.ao_npu.torchao_npu.ops.mx_ops`
(e.g., ``MXFP4FakeQuantize.forward`` calls :func:`mxfp4_dequantize`).
"""

import functools

import torch


def mxfp4_dequantize(
    data: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    block_size: int,
    output_shape: torch.Size,
    output_dtype: torch.dtype,
    low_first: bool = True,
) -> torch.Tensor:
    """Dequantize MXFP4 data from ``torch_npu.npu_dynamic_mx_quant`` output.

    Args:
        data: uint8 tensor (output y from npu_dynamic_mx_quant).
              Last dim is halved (2 FP4 values packed per byte).
        scale: uint8 E8M0 tensor (output mxscale_out).
               ndim = data.ndim + 1, with a trailing 2 dim packing scale pairs.
        axis: Quantization axis used in npu_dynamic_mx_quant.
        block_size: Block size used in npu_dynamic_mx_quant.
        output_dtype: Target dtype for the dequantized output.
        low_first: If True, low nibble is the first element in each packed byte.
    """
    assert output_shape[axis] % block_size == 0, f"quant dim must be divisible by block_size ({block_size})"

    # Use cached 256-entry LUT (indexed by full byte, no nibble splitting)
    lut = _get_fp4_e2m1_pair_lut(data.device, torch.bfloat16, low_first)
    idx = data.to(torch.uint8).reshape(-1).to(torch.long)
    values = torch.index_select(lut, dim=0, index=idx)

    # Reconstruct original input shape (last dim doubled after unpacking)
    values = values.reshape(*data.shape[:-1], data.shape[-1] * 2)
    if data.shape[-1] * 2 != output_shape[-1]:
        values = values.narrow(-1, 0, output_shape[-1])

    orig_shape = values.shape
    orig_ndim = values.ndim
    pos_axis = axis if axis >= 0 else axis + orig_ndim
    qdim = orig_shape[pos_axis]
    num_blocks = qdim // block_size

    # Unpack scale: NPU packed format → [..., num_blocks, ...]
    # The trailing 2 dim stores scale pairs; move it next to the packed-block dim
    scale = scale.to(torch.uint8)
    scale = scale.movedim(-1, pos_axis + 1)
    scale = scale.flatten(pos_axis, pos_axis + 1)  # [..., packed_blocks*2, ...]
    scale = scale.narrow(pos_axis, 0, num_blocks)  # trim padding when num_blocks is odd

    # E8M0 uint8 → bf16 scale: 2^(e - 127)
    scale = torch.exp2(scale.to(torch.bfloat16) - 127.0)

    # Block-wise broadcast multiply
    # Values: [..., qdim, ...] → [..., num_blocks, block_size, ...]
    values = values.unflatten(pos_axis, (num_blocks, block_size))

    # Scale: [..., num_blocks, ...] → [..., num_blocks, 1, ...]
    scale = scale.unsqueeze(pos_axis + 1)

    result = values * scale  # broadcasts over block_size
    result = result.reshape(*orig_shape)

    return result.to(output_dtype)


@functools.cache
def _get_fp4_e2m1_pair_lut(device, dtype=torch.bfloat16, low_first: bool = True):
    """LUT mapping a packed uint8 byte to a pair of decoded FP4 values.

    Returns shape [256, 2], indexed by the full byte value.
    low nibble → index 0, high nibble → index 1 (or swapped if low_first=False).
    """
    fp4_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    if low_first:
        lut = [[fp4_vals[p & 0x0F], fp4_vals[p >> 4]] for p in range(256)]
    else:
        lut = [[fp4_vals[p >> 4], fp4_vals[p & 0x0F]] for p in range(256)]

    return torch.tensor(lut, device=device, dtype=dtype)
