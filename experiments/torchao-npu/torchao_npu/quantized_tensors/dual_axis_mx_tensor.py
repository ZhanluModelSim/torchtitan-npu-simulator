# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dual-axis MX quantized tensor"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao_npu import normalize_dim
from torchao_npu.quantization import _FP4_DTYPES, _SUPPORTED_MX_ELEM_DTYPES
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_quantize_dual_axis
from torchao_npu.quantized_tensors import (
    logical_qdata_shape,
    permutation_from_transpose,
    resolve_pack_axis,
)
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

aten = torch.ops.aten


class DualAxisMXTensor(BaseQuantizedTensor):
    """A weight MX-quantized along both trailing dims, matching ``mx_quantize_dual_axis``.

    Stores two quantized copies of the same logical tensor and their E8M0 block
    scales: ``qdata1``/``scale1`` quantize along the last dim, ``qdata2``/``scale2``
    along the second-to-last.

    Shape support is limited to the axis-preserving ops -- transpose / t /
    permute / movedim -- and, because both quant axes are the trailing two dims,
    they must keep those two dims in the trailing two slots; only a swap between
    them is allowed. A swap transposes both ``qdata`` and ``scale``, then
    exchanges the two ``qdata``/scale pairs, so each pair keeps covering the dim
    it was quantized along. Every other op raises ``NotImplementedError``, except
    the handlers inherited from
    :class:`~torchao_npu.quantized_tensors.base_quantized_tensor.BaseQuantizedTensor`.

    Attributes:
        qdata1: Quantized values of dtype ``quant_config.elem_dtype`` along the last
            dim. FP8 (``float8_e4m3fn`` / ``float8_e5m2``) stores one value per byte
            with ``qdata1.shape == logical shape``. FP4 (``float4_e2m1fn_x2``) packs
            two values per element along ``pack_axis``, halving that axis in
            ``qdata1.shape``.

        scale1: uint8 E8M0 scales along the last dim, for a logical
            ``[..., R, C]``: ``[..., R, ceil(ceil(C/32)/2), 2]`` with a
            trailing pack-2 dim.

        qdata2: Quantized values along the second-to-last dim. Same dtype and layout
            rules as ``qdata1``: both outputs of
            ``npu_dynamic_mx_quant_with_dual_axis`` pack the input's last dim, so the
            two ``qdata`` shapes agree while their layouts need not.

        scale2: uint8 E8M0 scales along the second-to-last dim:
            ``[..., ceil(ceil(R/32)/2), C, 2]``.

        orig_dtype: The original high-precision dtype before quantization; also
            this tensor's logical (outer) dtype.

        quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            this tensor was quantized with.

        act_quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            activations should be quantized with when multiplied against this
            tensor, or ``None`` if unknown/absent.

        pack_axis: Non-negative dim both ``qdata`` pack their two values per element
            along, or ``None`` for FP8. Shape ops keep it in sync with the data.
    """

    tensor_data_names = ["qdata1", "scale1", "qdata2", "scale2"]
    tensor_attribute_names = [
        "orig_dtype",
        "quant_config",
        "act_quant_config",
    ]
    optional_tensor_attribute_names = ["pack_axis"]

    def __new__(
        cls,
        qdata1: torch.Tensor,
        scale1: torch.Tensor,
        qdata2: torch.Tensor,
        scale2: torch.Tensor,
        orig_dtype: torch.dtype,
        quant_config: MXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
        pack_axis: int | None = None,
    ):
        # Neither qdata1 nor qdata2 can be preferred for the initialization of
        # strides/storage_offset/layout. So their default values are used.
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_qdata_shape(qdata1, pack_axis),
            dtype=orig_dtype,
            device=qdata1.device,
            requires_grad=False,
        )
        return self

    def __init__(
        self,
        qdata1: torch.Tensor,
        scale1: torch.Tensor,
        qdata2: torch.Tensor,
        scale2: torch.Tensor,
        orig_dtype: torch.dtype,
        quant_config: MXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
        pack_axis: int | None = None,
    ):
        if qdata2.dtype is not qdata1.dtype:
            raise ValueError(f"``qdata1`` and ``qdata2`` must share a dtype, got {qdata1.dtype} and {qdata2.dtype}.")

        if qdata1.dtype not in _SUPPORTED_MX_ELEM_DTYPES:
            raise ValueError(f"{type(self).__name__} supports FP8 or FP4 qdata, got {qdata1.dtype}.")

        if qdata1.dtype is not quant_config.elem_dtype:
            raise ValueError(
                f"The dtypes of ``qdata1``/``qdata2`` ({qdata1.dtype}) does not match "
                f"``quant_config.elem_dtype`` ({quant_config.elem_dtype})."
            )

        if qdata1.ndim < 2:
            raise ValueError(f"{type(self).__name__} requires qdata.ndim >= 2, got {qdata1.ndim}.")

        # Both qdata describe one logical tensor. They are quantized along different
        # axes, so their block layouts differ, but they pack the same axis and have to
        # describe the same shape.
        logical1 = logical_qdata_shape(qdata1, pack_axis)
        logical2 = logical_qdata_shape(qdata2, pack_axis)
        if logical1 != logical2:
            raise ValueError(
                f"qdata1 and qdata2 must represent the same logical shape, got {tuple(logical1)} and {tuple(logical2)}."
            )

        if scale1.ndim != qdata1.ndim + 1 or scale2.ndim != qdata1.ndim + 1:
            raise ValueError(
                f"scale1/scale2 ndim must be qdata.ndim + 1 = {qdata1.ndim + 1}, got {scale1.ndim} and {scale2.ndim}."
            )

        if qdata1.dtype not in _FP4_DTYPES and pack_axis is not None:
            raise ValueError(
                f"Values in a non-FP4 tensor are not packed. ``pack_axis`` should not be set, "
                f"``pack_axis``={pack_axis} passed."
            )

        self.qdata1 = qdata1
        self.scale1 = scale1
        self.qdata2 = qdata2
        self.scale2 = scale2
        self.act_quant_config = act_quant_config
        self.quant_config = quant_config
        self.orig_dtype = orig_dtype
        self.pack_axis = normalize_dim(pack_axis, qdata1.ndim) if pack_axis is not None else None

    @classmethod
    def from_hp(
        cls,
        tensor: torch.Tensor,
        quant_config: MXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
    ) -> "DualAxisMXTensor":
        """Dual-axis MX quantize a high-precision ``tensor`` (its two trailing dims).

        Args:
            tensor: High-precision tensor to quantize; its two trailing dims are
                quantized in 32-element groups, so ``tensor.ndim >= 2`` is required.

            quant_config: MX quantization parameters.

            act_quant_config: The config activations should be quantized with when
                multiplied against the returned tensor, or ``None`` if unknown/absent.

        Returns:
            A :class:`~torchao_npu.quantized_tensors.dual_axis_mx_tensor.DualAxisMXTensor`
            holding the quantized ``qdata1``/``qdata2`` and their scales.

        Raises:
            ValueError: If ``tensor.ndim < 2``.
        """
        if tensor.ndim < 2:
            raise ValueError(f"{cls.__name__}.from_hp requires tensor.ndim >= 2, got {tensor.ndim}.")

        qdata1, scale1, qdata2, scale2 = mx_quantize_dual_axis(tensor, quant_config)
        pack_axis = resolve_pack_axis(qdata1) if qdata1.dtype is torch.float4_e2m1fn_x2 else None
        return cls(qdata1, scale1, qdata2, scale2, tensor.dtype, quant_config, act_quant_config, pack_axis)

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize back to ``orig_dtype``, using whichever quant axis is innermost-contiguous.

        The tensor stores one quantization per trailing dim, and only a pair whose quant axis is
        innermost-contiguous (``stride == 1``) can back the dequantization op. The pair covering
        ``dim=-1`` (``qdata1``/``scale1``) is used when ``qdata1.stride(-1) == 1``, otherwise the
        pair covering ``dim=-2`` (``qdata2``/``scale2``) when ``qdata2.stride(-2) == 1``.

        The chosen pair is converted with :meth:`to_mx_tensor` and dequantized as an
        :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`.

        Args:
            output_dtype: The dtype of the returned tensor, ``orig_dtype`` when ``None``;

        Returns:
            A plain tensor of dtype ``output_dtype``

        Raises:
            RuntimeError: If neither quant axis is innermost-contiguous.
        """
        if self.qdata1.stride(-1) == 1:
            quant_axis = self.qdata1.ndim - 1
        elif self.qdata2.stride(-2) == 1:
            quant_axis = self.qdata2.ndim - 2
        else:
            raise RuntimeError(
                f"Neither quant axis of this {type(self).__name__} is innermost-contiguous: "
                f"stride(-1)={self.qdata1.stride(-1)} for ``qdata1`` and "
                f"stride(-2)={self.qdata2.stride(-2)} for ``qdata2``; one of them must be 1 "
                "for the dequantization op to consume the data."
            )

        return self.to_mx_tensor(quant_axis).dequantize(output_dtype)

    def to_mx_tensor(self, quant_axis: int) -> MXTensor:
        """Convert to the single-axis :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`
        quantized along ``quant_axis``.

        ``qdata`` and the matching scale are used directly. No additional re-quantization or copy
        are used. The returned
        :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` shares this tensor's
        :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig` and inherits its
        ``act_quant_config``.

        Args:
            quant_axis: The dim the returned
                :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` is quantized along;
                must be one of the two trailing dims.

        Returns:
            An :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` sharing this tensor's
            ``qdata1``/``scale1`` (``quant_axis == ndim - 1``) or ``qdata2``/``scale2``
            (``quant_axis == ndim - 2``).

        Raises:
            ValueError: If ``quant_axis`` is not one of the two trailing dims.
        """
        quant_axis = normalize_dim(quant_axis, self.qdata1.ndim)
        if quant_axis not in (self.qdata1.ndim - 2, self.qdata1.ndim - 1):
            raise ValueError(
                f"Only the last two dims are supported for ``quant_axis``, quant_axis={quant_axis} passed."
            )

        qdata, scale = (self.qdata1, self.scale1) if quant_axis == self.qdata1.ndim - 1 else (self.qdata2, self.scale2)
        return MXTensor(
            qdata, scale, self.orig_dtype, quant_axis, self.quant_config, self.act_quant_config, self.pack_axis
        )


implements = DualAxisMXTensor.implements


def _permute(x: DualAxisMXTensor, perm: list[int]) -> DualAxisMXTensor:
    """Permute both ``qdata`` and both scales; the last two dims must stay in the trailing two slots.

    Leading dims permute freely; a swap of the trailing two exchanges the two
    quantized copies and the two scales (each with its two spatial dims swapped).
    Both ``qdata`` pack the same axis, which is tracked onto its permuted position.

    Args:
        x: The source :class:`DualAxisMXTensor`.
        perm: Full permutation of ``x``'s dims, as non-negative indices.

    Returns:
        A new :class:`DualAxisMXTensor` with permuted components.

    Raises:
        ValueError: If the permutation moves a dim across the leading /
            trailing-two boundary.
    """
    if {perm[-2], perm[-1]} != {x.ndim - 2, x.ndim - 1}:
        raise ValueError(
            f"{type(x).__name__} only supports permutations that keep the last two dims "
            f"trailing (optionally swapped); got {perm}."
        )

    swap = perm[-2] == x.ndim - 1
    new_pack_axis = perm.index(x.pack_axis) if x.pack_axis is not None else None

    # ``scale1``/``scale2`` carry an extra trailing pack-2 dim, which stays last.
    scale_perm = [*perm, x.scale1.ndim - 1]

    if not swap:
        new_qdata1 = x.qdata1.permute(perm)
        new_s1 = x.scale1.permute(scale_perm)

        new_qdata2 = x.qdata2.permute(perm)
        new_s2 = x.scale2.permute(scale_perm)

    else:
        # Permute individual tensors of the two (``qdata``, ``scale``) pairs
        # as well as swap the role of the two pairs.

        new_qdata1 = x.qdata2.permute(perm)
        new_s1 = x.scale2.permute(scale_perm)

        new_qdata2 = x.qdata1.permute(perm)
        new_s2 = x.scale1.permute(scale_perm)

    return DualAxisMXTensor(
        new_qdata1,
        new_s1,
        new_qdata2,
        new_s2,
        x.orig_dtype,
        x.quant_config,
        x.act_quant_config,
        new_pack_axis,
    )


@implements(aten.permute.default)
def _(func, types, args, kwargs):
    x, dims = args[0], args[1]
    perm = [normalize_dim(d, x.ndim) for d in dims]
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    x, dim0, dim1 = args[0], args[1], args[2]
    perm = permutation_from_transpose(dim0, dim1, x.ndim)
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


@implements(aten.t.default)
def _(func, types, args, kwargs):
    x = args[0]
    # Match plain torch semantics: the <=2D check lives in the C++ native impl,
    # which subclass dispatch bypasses, so enforce it here. For 2D it is
    # transpose(0, 1); for <=1D a no-op.
    if x.ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {x.ndim}D")
    perm = permutation_from_transpose(0, 1, x.ndim) if x.ndim == 2 else list(range(x.ndim))
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([DualAxisMXTensor])
