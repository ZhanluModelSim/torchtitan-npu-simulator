# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 low-level primitives for NPU.

These are pure tensor helpers (no matmul) used by higher-level ops in
:mod:`torchao_npu.ops.mx_ops`
(e.g., ``MXFP4FakeQuantize.forward`` calls :func:`mxfp4_dequantize`).
"""

import functools
from math import ceil

import torch
import torch_npu

from torchao_npu import normalize_dim
from torchao_npu.quantization import (
    _FP4_DTYPES,
    _FP8_DTYPES,
    _SUPPORTED_HP_DTYPES,
)
from torchao_npu.quantization.quant_configs import MXQuantizeConfig


def mx_quantize(
    tensor: torch.Tensor,
    axis: int,
    config: MXQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MX quantize ``tensor`` along ``axis``, avoiding a real transpose when possible.

    ``npu_dynamic_mx_quant`` requires dense (``is_contiguous()``) input and,
    given a non-dense tensor, silently inserts a ``Contiguous/Transpose`` copy
    before quantizing -- even when the quant axis is already contiguous in
    memory (the op checks ``is_contiguous()`` on the whole tensor, not per-axis).
    This is an inherent limitation of the underlying operator; the kernel has no
    stride handling, so it can only consume a dense layout.

    How a real transpose can be avoided: if the quant axis is the innermost-contiguous
    dimension (``stride(axis) == 1``), reorder dims so the quant axis lands at -1
    (a view, no copy), quantize along -1, and permute both outputs back to the
    original layout. For a pure permutation view this restores dense storage;
    a sliced tensor with gaps may still require a copy inside the NPU operator.

    When a real transpose is inevitable, no view can make the tensor dense and
    the raw op performs a real copy internally. (We still permute the outputs
    back, so the result is correct regardless.) This happens when any of:

    - the quant axis is strided (``stride(axis) != 1``);
    - the tensor is not a pure permutation view, so the permuted tensor stays
      non-dense.

    Dense inputs with unit quant-axis stride take the same permutation path.
    Avoid checking whole-tensor contiguity here: a sliced gradient with an
    unbacked symbolic row count can require a data-dependent guard for that check.

    Warning:
        The transpose-avoiding optimization assumes every dim has a positive
        stride; it does not consider tensors with a stride-0 dim (e.g. a
        broadcast dim) yet.

    Args:
        tensor: Tensor to quantize, shape ``(..., K, ...)``.
        axis: Dimension to quantize along.
        config: MX quantization parameters.

    Returns:
        ``(y, scale)``; ``y`` uses ``config.elem_dtype`` (packed for FP4), and
        ``scale`` has the canonical shape produced by ``npu_dynamic_mx_quant``.
    """

    axis = normalize_dim(axis, tensor.ndim)

    # A strided quant axis goes directly to the raw op. Do not query global
    # contiguity: AOTAutograd may supply a sliced gradient with unbacked sizes.
    if tensor.stride(axis) != 1:
        y, scale = torch_npu.npu_dynamic_mx_quant(
            tensor,
            axis=axis,
            dst_type=config.npu_elem_dtype,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        return y.view(config.elem_dtype), scale

    else:
        # Unit-stride quant axis (dense or non-dense input): permute it to -1.
        # ``stride(axis) == 1`` here, so ordering the other dims by descending
        # stride and appending ``axis`` reproduces the dense layout (a pure view)
        # whenever the tensor is a pure permutation view. Appending ``axis``
        # explicitly -- instead of relying on a descending-stride sort to place
        # it last -- keeps the quant axis at -1 even when another dim ties on
        # stride 1, e.g. a size-1 dim.
        perm_indices = sorted(
            (d for d in range(tensor.ndim) if d != axis),
            key=lambda d: tensor.stride(d),
            reverse=True,
        )
        perm_indices.append(axis)
        tensor_p = tensor.permute(perm_indices)

        y_p, scale_p = torch_npu.npu_dynamic_mx_quant(
            tensor_p,
            axis=-1,
            dst_type=config.npu_elem_dtype,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        y_p = y_p.view(config.elem_dtype)

        # Inverse permutation: y_p dim j corresponds to tensor dim perm_indices[j].
        perm_back_indices = [0] * tensor.ndim
        for j, p in enumerate(perm_indices):
            perm_back_indices[p] = j
        y = y_p.permute(perm_back_indices)

        # scale_p has tensor.ndim+1 dims: the non-axis dims (in permuted order),
        # then the block dim, then the trailing pack-2 dim. The inverse
        # permutation already maps the quant axis to the block slot, so just
        # append tensor.ndim to restore the canonical layout, [.. axis .., block, .. , 2].
        perm_back_indices.append(tensor.ndim)
        scale = scale_p.permute(perm_back_indices)

        return y, scale


def to_scale_of_block32(scale: torch.Tensor, block_size: int, quant_dim_size: int) -> torch.Tensor:
    """Re-key an MX scale from ``block_size`` blocks to 32-sized blocks.

    ``torch_npu.npu_anti_mx_quant`` only accepts a scale keyed at ``block_size == 32``,
    while ``npu_dynamic_mx_quant`` can emit one keyed at any multiple of 32. Each coarse
    block scale is duplicated ``block_size // 32`` times to fill the finer blocks it
    covers. The quant dim is always assumed to be the last dim.

    Args:
        scale: MX scale of shape ``(..., packed_blocks, 2)``, keyed at ``block_size``.
        block_size: Block size of ``scale``; assumed to be a multiple of 32 by the caller.
        quant_dim_size: Size of the quantized dimension, used to size the 32-keyed
            packing (including its padding).

    Returns:
        ``scale`` unchanged when it is already 32-keyed, otherwise the 32-keyed layout
        of shape ``(..., ceil(ceil(quant_dim_size / 32) / 2), 2)``.
    """
    if block_size <= 32:
        return scale

    flat = scale.reshape(*scale.shape[:-2], -1)  # merge pair dim
    flat = flat.repeat_interleave(block_size // 32, dim=-1)  # duplicate scales

    # because block_size >= 32, there is always enough padding rows for us to reuse
    sdim_padded = ceil(ceil(quant_dim_size / 32) / 2) * 2
    flat = flat[..., :sdim_padded]  # trim the padded tail
    return flat.reshape(*flat.shape[:-1], sdim_padded // 2, 2)  # re-form pack dim


@torch.library.custom_op("torchao_npu::mx_last_dim_fake_quantize", mutates_args=(), device_types="npu")
def mx_last_dim_fake_quantize(
    x: torch.Tensor,
    quant_elem_dtype: int,
    block_size: int,
    round_mode: str,
    scale_alg: int,
    quant_elem_dtype_max: float,
) -> torch.Tensor:
    """Quantize then immediately dequantize ``x`` along its trailing dimension.

    the result is differentiable via a straight-through estimator.

    Args:
        x: Tensor to fake-quantize.
        quant_elem_dtype: Element dtype of the quantized tensor, in the encoding
            ``npu_dynamic_mx_quant`` expects; ``npu_anti_mx_quant`` decodes the same
            dtype.
        block_size: MX block size for both ops.
        round_mode: Cast mode of the quantizer.
        scale_alg: Scale algorithm of the quantizer.
        quant_elem_dtype_max: Maximum of the target dtype; ``0.0`` infers it.

    Returns:
        The fake-quantized tensor: fresh, contiguous, with ``x``'s shape and dtype.

    Raises:
        AssertionError: If the dequantized values are not contiguous, which the NPU
            kernels otherwise guarantee.
    """
    qdata, scale = torch_npu.npu_dynamic_mx_quant(
        x,
        axis=-1,
        dst_type=quant_elem_dtype,
        block_size=block_size,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=quant_elem_dtype_max,
    )

    scale = to_scale_of_block32(scale, block_size, x.shape[-1])

    dequant = torch_npu.npu_anti_mx_quant(qdata, scale, axis=-1, dst_type=x.dtype, src_type=quant_elem_dtype)
    assert dequant.is_contiguous(), "``dequant`` is expected to be contiguous."
    return dequant


@mx_last_dim_fake_quantize.register_fake
def _(x, quant_elem_dtype, block_size, round_mode, scale_alg, quant_elem_dtype_max):
    """Metadata only: dense output mirroring ``x``'s shape and dtype."""
    del quant_elem_dtype, block_size, round_mode, scale_alg, quant_elem_dtype_max
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def mx_last_dim_fake_quantize_backward(ctx, grad_output):
    """Straight-through: the tensor arg keeps its gradient, the scalars get none."""
    return grad_output, None, None, None, None, None


torch.library.register_autograd(mx_last_dim_fake_quantize, mx_last_dim_fake_quantize_backward)


def mx_fake_quantize(
    tensor: torch.Tensor,
    axis: int,
    config: MXQuantizeConfig,
) -> torch.Tensor:
    """MX fake-quantize ``tensor`` along ``axis`` (differentiable via a straight-through estimator).

    Quantizes with ``npu_dynamic_mx_quant`` and immediately dequantizes the result
    with ``npu_anti_mx_quant``, so the values are those of a real MX
    quantization/dequantization round trip while the returned tensor keeps
    ``tensor.dtype``.

    Unlike :func:`mx_quantize`, the quant axis is always permuted to -1 first,
    because ``npu_anti_mx_quant`` only consumes quantized data whose quant axis is
    trailing. The permutation is a view, so it introduces no copy of its own: for a
    tensor whose quant axis is already trailing it is the identity, and otherwise
    the quant op copies internally when it cannot consume the permuted layout as-is.

    Args:
        tensor: Tensor to quantize, shape ``(..., K, ...)``.
        axis: Dimension to quantize along.
        config: MX quantization parameters.

    Returns:
        The fake-quantized tensor, with the same shape and dtype as ``tensor``.
    """

    axis = normalize_dim(axis, tensor.ndim)

    perm_indices = sorted(
        (d for d in range(tensor.ndim) if d != axis),
        key=lambda d: tensor.stride(d),
        reverse=True,
    )
    perm_indices.append(axis)
    tensor_p = tensor.permute(perm_indices)

    y_fake_p = mx_last_dim_fake_quantize(
        tensor_p,
        config.npu_elem_dtype,
        config.block_size,
        config.round_mode,
        config.scale_alg,
        config.dst_type_max,
    )

    # Inverse permutation: y_p dim j corresponds to tensor dim perm_indices[j].
    perm_back_indices = [0] * tensor.ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j
    y_fake = y_fake_p.permute(perm_back_indices)

    return y_fake


def mx_dequantize(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    block_size: int,
    src_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MX data, avoiding a real transpose when possible.

    The inverse of :func:`mx_quantize`: given the ``(qdata, scale)`` pair that op
    returns and the ``axis`` it quantized along, restore the unquantized values as
    ``output_dtype``.

    Args:
        qdata: Quantized data. Its quant axis must be innermost-contiguous.
        scale: uint8 E8M0 block scale of ``qdata``. Keyed at ``block_size``.
        axis: Dimension ``qdata`` was quantized along.
        block_size: MX block size ``scale`` is keyed at; a positive multiple of 32.
        src_dtype: Element dtype of ``qdata``.
        output_dtype: dtype of the dequantized values.

    Returns:
        The dequantized tensor, in ``output_dtype`` and ``qdata``'s layout, with the
        quant axis restored to its unquantized size.

    Raises:
        ValueError: If ``src_dtype`` is not an MX element dtype
            (``torch.float8_e4m3fn``, ``torch.float8_e5m2`` or
            ``torch.float4_e2m1fn_x2``).

        AssertionError: If any of the following holds:

            - ``qdata.dtype`` is neither ``torch.uint8`` nor ``src_dtype``: only those
              two say how the stored bytes are to be read, ``uint8`` being the raw
              storage the quantizer emits and ``src_dtype`` a typed view of it;
            - ``output_dtype`` is not a high-precision dtype (``torch.float32``,
              ``torch.bfloat16`` or ``torch.float16``);
            - the quant axis is not unit stride. ``npu_anti_mx_quant`` requires
              quantized data laid out as the quantizer emits it, which has the quant
              axis innermost-contiguous; a strided quant axis (e.g. a transposed
              tensor) can be neither consumed nor permuted into such a layout.
    """
    if qdata.dtype is not torch.uint8:
        assert qdata.dtype is src_dtype, (
            "``qdata.dtype`` must match ``src_dtype`` if ``qdata.dtype`` is not ``torch.uint8``."
        )

    assert output_dtype in _SUPPORTED_HP_DTYPES, (
        f"``output_dtype`` only supports {_SUPPORTED_HP_DTYPES}, ``output_dtype``={output_dtype} passed."
    )
    axis = normalize_dim(axis, qdata.ndim)
    assert qdata.stride(axis) == 1, (
        f"``npu_anti_mx_quant`` consumes only quantized data whose quant axis is unit stride, "
        f"got stride({axis})={qdata.stride(axis)}"
    )

    # Same rule as ``mx_quantize``: order the non-axis dims by descending stride and
    # append the quant axis, so the quant axis lands at -1 while the permutation stays
    # a view of the dense layout whenever ``qdata`` is a pure permutation view.
    # torch_npu.npu_anti_mx_quant only accepts last-dim quantized data.
    perm_indices = sorted(
        (d for d in range(qdata.ndim) if d != axis),
        key=lambda d: qdata.stride(d),
        reverse=True,
    )
    perm_indices.append(axis)
    qdata_p = qdata.permute(perm_indices)

    if src_dtype in _FP4_DTYPES:
        # An FP4 quant dim is packed, so its unquantized size is twice the stored one.
        quant_dim_size = qdata_p.shape[-1] * 2

        # torch_npu.npu_anti_mx_quant requires qdata_p.dtype is uint8 when src_type is FP4.
        qdata_p = qdata_p.view(torch.uint8)

    elif src_dtype in _FP8_DTYPES:
        quant_dim_size = qdata_p.shape[-1]

        # torch_npu.npu_anti_mx_quant requires qdata_p.dtype is the same as src_type in
        # the case of FP8.
        qdata_p = qdata_p.view(src_dtype)

    else:
        raise ValueError(
            f"``src_dtype`` only supports {_FP4_DTYPES} and {_FP8_DTYPES}, ``src_dtype``={src_dtype} passed."
        )

    # ``scale`` has one dim more than ``qdata``: its first ``qdata.ndim`` dims follow
    # the data (the block dim riding along the quant axis), the pack-2 dim stays last.
    scale_p = scale.permute([*perm_indices, qdata.ndim])

    # ``to_scale_of_block32`` needs that size to derive the 32-keyed scale, the only
    # layout ``npu_anti_mx_quant`` accepts.
    scale_p = to_scale_of_block32(scale_p, block_size, quant_dim_size)

    dequant_p = torch_npu.npu_anti_mx_quant(
        qdata_p,
        scale_p,
        axis=-1,
        dst_type=output_dtype,
        src_type=src_dtype,
    )

    # Inverse permutation: dequant_p dim j corresponds to qdata dim perm_indices[j].
    perm_back_indices = [0] * qdata.ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j

    return dequant_p.permute(perm_back_indices)


def mx_quantize_dual_axis(
    tensor: torch.Tensor,
    config: MXQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize ``tensor`` along its two trailing dims, avoiding real transposes when possible.

    ``npu_dynamic_mx_quant_with_dual_axis`` quantizes the two trailing dims of a
    dense tensor and returns ``(y1, s1, y2, s2)`` (``y1``/``s1`` along dim -1,
    ``y2``/``s2`` along dim -2). Given a non-dense tensor it silently inserts a
    ``Contiguous/Transpose`` copy before quantizing, because the kernel has no
    stride handling and can only consume a dense layout. This is an inherent
    limitation of the underlying operator.

    How a real transpose can be avoided: the dims fall into two parts -- the two
    quant dims (-2, -1) and the leading dims. Crossing a dim between parts would
    quantize a different pair and change the semantics, so we only permute within
    each part. We permute each part by descending stride and quantize the result;
    this reproduces the raw op exactly (the op only ever quantizes the trailing
    two dims). When the two quant dims are the two innermost dims the permutation
    is dense, so no copy is needed; otherwise the op copies internally, no worse
    than calling it directly. If the op quantized the two dims in the opposite
    order, we swap ``q1``/``q2`` so the result matches the raw op.

    Args:
        tensor: Tensor to quantize, shape ``(..., row, col)`` (``ndim >= 2``).
        config: MX quantization parameters.

    Returns:
        ``(y1, s1, y2, s2)`` with the same shapes ``npu_dynamic_mx_quant_with_dual_axis``
        would produce for ``tensor``.
    """
    # Permute each part so that, within each part, the strides run descending.
    strides = tensor.stride()
    # Part 1 (leading dims): order them by stride so they run outermost-first.
    lead = sorted(range(tensor.ndim - 2), key=lambda d: strides[d], reverse=True)
    # Part 2 (the two quant dims): order them descending so the innermost one
    # lands at -1. Whether this reverses the original [-2, -1] decides the swap.
    swap = strides[tensor.ndim - 1] > strides[tensor.ndim - 2]
    quant_order = [tensor.ndim - 2, tensor.ndim - 1] if not swap else [tensor.ndim - 1, tensor.ndim - 2]
    perm_indices = [*lead, *quant_order]
    tensor_p = tensor.permute(perm_indices)

    y1_p, s1_p, y2_p, s2_p = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        tensor_p,
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    y1_p = y1_p.view(config.elem_dtype)
    y2_p = y2_p.view(config.elem_dtype)

    # Inverse permutation of the tensor dims. Applying it to any op output --
    # appending the extra block/pack dim for a scale -- returns the tensor to the
    # original layout.
    perm_back_indices = [0] * tensor.ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j

    y1 = y1_p.permute(perm_back_indices)
    y2 = y2_p.permute(perm_back_indices)
    s1 = s1_p.permute([*perm_back_indices, tensor.ndim])
    s2 = s2_p.permute([*perm_back_indices, tensor.ndim])

    # The op's q1 is along tensor_p's -1. When the two quant dims were permuted
    # in the opposite order, that dim is the original -2: swap the outputs so q1
    # tracks the original -1 (and q2 the original -2).
    if swap:
        y1, y2 = y2, y1
        s1, s2 = s2, s1
    return y1, s1, y2, s2


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
    lut: torch.Tensor = _get_fp4_e2m1_pair_lut(data.device, torch.bfloat16, low_first)
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
def _get_fp4_e2m1_pair_lut(device, dtype=torch.bfloat16, low_first: bool = True) -> torch.Tensor:
    """LUT mapping a packed uint8 byte to a pair of decoded FP4 values.

    Returns shape [256, 2], indexed by the full byte value.
    low nibble → index 0, high nibble → index 1 (or swapped if low_first=False).
    """
    fp4_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=dtype,
    )
    p = torch.arange(256, device=device)
    # Each byte packs two FP4 values: low nibble (bits 0-3) and high nibble (bits 4-7).
    # Index fp4_vals (16 entries) by these nibbles.
    low_nibble = p & 0x0F  # mask out high nibble → 0..15
    high_nibble = p >> 4  # shift down high nibble → 0..15
    idx0, idx1 = (low_nibble, high_nibble) if low_first else (high_nibble, low_nibble)
    return torch.stack([fp4_vals[idx0], fp4_vals[idx1]], dim=1)
