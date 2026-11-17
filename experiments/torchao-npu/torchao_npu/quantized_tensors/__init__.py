# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen, quantized NPU tensor subclasses (inference/rollout weight format).

Quantized tensors (:class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`,
``DualAxisMXTensor``, ``BlockMXTensor``) hold **already-quantized** low-precision
data plus its scale(s). They are the inference/rollout counterpart of the
training-time wrapper tensors in :mod:`torchao_npu.wrapper_tensors`, which keep a
high-precision master weight and quantize on the fly inside matmuls.

Design:

- ``__torch_function__`` is disabled so every op falls through to
  ``__torch_dispatch__``
  (:class:`~torchao_npu.quantized_tensors.base_quantized_tensor.BaseQuantizedTensor`'s
  classmethod), bypassing the Python-level function layer entirely.
- **Values are frozen.** ``__torch_dispatch__`` rejects every in-place
  (mutable-schema) op; the stored quantization is immutable.
"""

import torch

from torchao_npu import normalize_dim


def permutation_from_transpose(dim0: int, dim1: int, ndim: int) -> list[int]:
    """Full permutation index list equivalent to ``transpose(dim0, dim1)``."""
    perm = list(range(ndim))
    d0, d1 = normalize_dim(dim0, ndim), normalize_dim(dim1, ndim)
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return perm


def resolve_pack_axis(qdata: torch.Tensor, *candidate_axes: int) -> int:
    """Locate the axis an FP4 ``qdata`` packs its values along, among ``candidate_axes``.

    FP4 stores two values per element along a single axis, which is the qdata's
    unit-stride dim. NPU ops produce qdata in which the pack axis is the last dim. But
    later permutations may move the pack axis. This function will search the unique axis in
    ``candidate_axes`` whose stride is 1 and report it as the pack axis. It requires there
    is one and only one axis in ``candidate_axes`` that has a unit-stride.

    Args:
        qdata: FP4-packed tensor (dtype ``torch.float4_e2m1fn_x2``).

        *candidate_axes: Dims the pack axis may be, (negative indices allowed).
                         One and only one candidate is allowed to have a unit-stride.
                         Every dim is searched when omitted.

    Returns:
        The pack axis, as a non-negative index.

    Raises:
        ValueError: If no candidate axis -- or more than one -- is unit-stride.
    """

    assert qdata.dtype is torch.float4_e2m1fn_x2, (
        f"The dtype of ``qdata`` must be {torch.float4_e2m1fn_x2}, a {qdata.dtype} tensor passed."
    )

    if not candidate_axes:
        candidate_axes = tuple(range(qdata.ndim))

    pack_axis = None
    for axis in {normalize_dim(dim, qdata.ndim) for dim in candidate_axes}:
        if qdata.stride(axis) == 1:
            if pack_axis is None:
                pack_axis = axis
            else:
                raise ValueError(
                    f"More than one axis {(pack_axis, axis)} in ``candidate_axes`` has a unit-stride, "
                    "the pack axis cannot be determined."
                )

    if pack_axis is None:
        axes = ", ".join(f"stride({a})={qdata.stride(a)}" for a in candidate_axes)
        raise ValueError(
            f"Invalid strides for FP4 quantized tensors: {axes}; expected one of them "
            f"to be 1 (the pack axis must be unit-stride)."
        )

    return pack_axis


def logical_qdata_shape(qdata: torch.Tensor, pack_axis: int | None) -> torch.Size:
    """Compute the logical (high-precision) shape a quantized ``qdata`` represents.

    FP4 stores two values per element along ``pack_axis``, so that axis is doubled;
    FP8 stores one value per element, so ``qdata.shape`` is already the logical shape.

    Args:
        qdata: Quantized data, either FP8 (``float8_e4m3fn`` / ``float8_e5m2``) or FP4
            (``float4_e2m1fn_x2``).

        pack_axis: Dim ``qdata`` packs its two values per element along, required for
            FP4 and ignored for FP8 (negative indices allowed). When set, it has to be a
            unit-stride dim, whatever the dtype.

    Returns:
        The logical shape ``qdata`` represents.

    Raises:
        IndexError: If ``pack_axis`` is out of range for ``qdata``.
        ValueError: If ``pack_axis`` is ``None`` for FP4, or is set but not unit-stride.
    """
    if pack_axis is not None:
        pack_axis = normalize_dim(pack_axis, qdata.ndim)

        if qdata.stride(pack_axis) != 1:
            raise ValueError(
                f"The stride of pack_axis is {qdata.stride(pack_axis)}, a unit-stride pack_axis is expected."
            )

    if qdata.dtype is torch.float4_e2m1fn_x2:
        if pack_axis is None:
            raise ValueError(
                f"When ``qdata.dtype``=``torch.float4_e2m1fn_x2``, pack_axis should be set,  {pack_axis} passed."
            )

        shape = list(qdata.shape)
        shape[pack_axis] *= 2
        return torch.Size(shape)

    else:
        return qdata.shape


# Imported after the helpers: the submodules below import them back from this
# (then partially initialized) package.
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.dual_axis_mx_tensor import DualAxisMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

__all__ = [
    "BlockMXTensor",
    "DualAxisMXTensor",
    "MXTensor",
]
