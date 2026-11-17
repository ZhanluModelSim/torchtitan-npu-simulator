# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block MX low-level primitives for NPU.

Pure tensor helpers (no autograd, no matmul) used by the ops in
:mod:`torchao_npu.ops.block_mx_ops`.
"""

import importlib

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig

importlib.import_module("cann_ops_nn")


def block_mx_quantize(
    tensor: torch.Tensor,
    config: BlockMXQuantizeConfig,
    axis: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply block MX quantization to ``tensor``, inserting an MXFP4
    fake-quantization when ``config.mxfp4_fake_quantize_config`` is set.

    A real transpose (the ``Contiguous/Transpose`` copy the NPU ops insert for
    non-dense input) is avoided with best effort, but stays inevitable when:

    - ``tensor`` is not a pure permutation view (e.g. an ``as_strided`` view with gaps);
    - restoring density would cross the leading / trailing-two-dims boundary,
      which would change the quantized 32x32 blocks;
    - ``axis`` is not the innermost-contiguous of the two trailing dims --
      mxfp4-fake-quantization path only.

    Returns:
        ``(q, s1, s2)``: the quantized tensor and the dim -1 / dim -2 scales,
        shaped as ``npu_dynamic_block_mx_quant`` would produce for ``tensor``.
        Exception: direct FP4 on a transposed layout packs ``q`` along dim -2
        instead of dim -1 -- equivalent under the low-precision matmul, not
        byte-identical to the raw op.
    """

    assert tensor.ndim >= 2, f"tensor must be at least 2D, got {tensor.ndim}D"

    if config.mxfp4_fake_quantize_config is not None:
        assert axis is not None, "``axis`` must be set when MXFP4 fake-quantization is enabled."
        return _mxfp4_fake_quantize_then_block_mx_quantize(tensor, config, axis)
    else:
        return _direct_block_mx_quantize(tensor, config)


def _mxfp4_fake_quantize_then_block_mx_quantize(
    tensor: torch.Tensor,
    config: BlockMXQuantizeConfig,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 fake-quantize ``tensor`` along ``axis``, then convert to block MX.

    The fused ``mx_to_block_mx_quant`` kernel only accepts an mxfp4 tensor
    packed along the last dim, so ``axis`` is permuted there and back.
    """

    fp4_tensor, mxscale, perm_indices = quantize_mxfp4_payload(tensor, config, axis)
    q_p, s1_p, s2_p = mxfp4_to_block_mx(fp4_tensor, mxscale, config)
    return from_block_quant_layout(tensor.shape, perm_indices, q_p, s1_p, s2_p)


def quantize_mxfp4_payload(
    tensor: torch.Tensor, config: BlockMXQuantizeConfig, axis: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return packed MXFP4, scales and the permutation into kernel layout."""
    mxfp4_config = config.mxfp4_fake_quantize_config
    assert mxfp4_config is not None, "config.mxfp4_fake_quantize_config must be set, None passed."

    assert axis in (-2, -1, tensor.ndim - 2, tensor.ndim - 1), (
        f"axis must point to one of the last two dims, got axis={axis} for ndim={tensor.ndim}"
    )

    # ``axis`` must land at -1; whether that reverses [-2, -1] decides the swap.
    swap = axis in (-2, tensor.ndim - 2)
    tensor_f, perm_indices = _to_block_quant_layout(tensor, swap)

    fp4_tensor, mxscale = torch_npu.npu_dynamic_mx_quant(
        tensor_f,
        axis=-1,
        dst_type=mxfp4_config.npu_elem_dtype,
        block_size=mxfp4_config.block_size,
        round_mode=mxfp4_config.round_mode,
        scale_alg=mxfp4_config.scale_alg,
        dst_type_max=mxfp4_config.dst_type_max,
    )

    return fp4_tensor, mxscale, perm_indices


def mxfp4_to_block_mx(
    fp4_tensor: torch.Tensor, mxscale: torch.Tensor, config: BlockMXQuantizeConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert kernel-layout MXFP4 into Block FP8 without an intermediate BF16 tensor."""
    mxfp4_config = config.mxfp4_fake_quantize_config
    assert mxfp4_config is not None
    q_p, s1_p, s2_p = torch.ops.cann_ops_nn.mx_to_block_mx_quant(
        fp4_tensor,
        mxscale,
        dst_type=config.npu_elem_dtype,
        x_type=mxfp4_config.npu_elem_dtype,
    )

    return q_p, s1_p, s2_p


def _direct_block_mx_quantize(
    tensor: torch.Tensor,
    config: BlockMXQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block MX quantize the two trailing dims of ``tensor`` in 32x32 blocks.

    ``npu_dynamic_block_mx_quant`` returns ``(q, s1, s2)``, ``s1`` along dim -1
    and ``s2`` along dim -2. For FP4, ``q`` packs two values per byte along the
    innermost-contiguous quant dim: a dense input matches the raw op
    byte-for-byte, a swapped layout gives a transposed-packed ``q`` -- the same
    values, consumable by the matmuls (which detect transposed packed weights),
    but not the raw op's byte layout. Scales are unaffected.
    """

    # Descending stride puts the innermost-contiguous quant dim at -1; whether
    # that reverses [-2, -1] decides the swap.
    swap = tensor.stride(-1) > tensor.stride(-2)
    tensor_f, perm_indices = _to_block_quant_layout(tensor, swap)

    q_p, s1_p, s2_p = torch_npu.npu_dynamic_block_mx_quant(
        tensor_f,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    # FP4 comes back as uint8 with the last dim halved; relabel it. No-op for FP8.
    q_p = q_p.view(config.elem_dtype)

    return from_block_quant_layout(tensor.shape, perm_indices, q_p, s1_p, s2_p)


def _to_block_quant_layout(tensor: torch.Tensor, swap: bool) -> tuple[torch.Tensor, list[int]]:
    """Permute ``tensor`` into the dense 2D/3D layout the block MX kernels need.

    The kernels have no stride handling, so for non-dense input they silently
    insert a ``Contiguous/Transpose`` copy. To avoid it we permute within two
    parts only -- the leading dims, and the two quant dims -- since moving a dim
    across that boundary would change which 32x32 blocks are quantized. A pure
    permutation view becomes dense and no copy happens; otherwise the kernel
    copies internally, no worse than being called directly. Ranks above 3 are
    flattened to 3D, free when the permuted tensor is dense.

    Returns:
        ``(tensor_f, perm_indices)``, the latter to be handed to
        :func:`from_block_quant_layout`.
    """

    # Leading dims: outermost-first by stride. Quant dims: reversed iff ``swap``.
    strides = tensor.stride()
    lead = sorted(range(tensor.ndim - 2), key=lambda d: strides[d], reverse=True)
    quant_order = [tensor.ndim - 1, tensor.ndim - 2] if swap else [tensor.ndim - 2, tensor.ndim - 1]
    perm_indices = [*lead, *quant_order]

    tensor_p = tensor.permute(perm_indices)
    tensor_f = tensor_p.reshape(-1, *tensor_p.shape[-2:]) if tensor.ndim > 3 else tensor_p
    return tensor_f, perm_indices


def from_block_quant_layout(
    shape: tuple[int, ...] | torch.Size,
    perm_indices: list[int],
    q_p: torch.Tensor,
    s1_p: torch.Tensor,
    s2_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore quantized layout using metadata only, without allocating a weight template."""
    ndim = len(shape)
    shape_p = [shape[i] for i in perm_indices]

    if ndim > 3:
        # ``q_p``'s last dim is halved for FP4, so it is taken from ``q_p`` itself.
        q_p = q_p.reshape(*shape_p[:-1], q_p.shape[-1])
        s1_p = s1_p.reshape(*shape_p[:-2], *s1_p.shape[1:])
        s2_p = s2_p.reshape(*shape_p[:-2], *s2_p.shape[1:])

    # Inverse permutation, with the scales' trailing pack dim appended.
    perm_back_indices = [0] * ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j

    q = q_p.permute(perm_back_indices)
    s1 = s1_p.permute([*perm_back_indices, ndim])
    s2 = s2_p.permute([*perm_back_indices, ndim])

    # The kernel's s1 is along the permuted -1. If that dim is the original -2,
    # swap so s1 tracks the original -1 and s2 the original -2.
    if perm_indices[-1] == ndim - 2:
        s1, s2 = s2, s1
    return q, s1, s2
