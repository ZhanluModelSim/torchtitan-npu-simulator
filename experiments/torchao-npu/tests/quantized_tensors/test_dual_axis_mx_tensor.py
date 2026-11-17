# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`torchao_npu.quantized_tensors.dual_axis_mx_tensor.DualAxisMXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu import normalize_dim
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize_dual_axis
from torchao_npu.quantized_tensors.dual_axis_mx_tensor import DualAxisMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

# =========================================================================
# Construction and validation
# =========================================================================


@pytest.mark.parametrize(
    "qdata1, qdata2, pack_axis",
    [
        # distinguishable content: the two pairs trade roles on a swap
        (
            torch.full((4, 64), 1.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.full((4, 64), 2.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            None,
        ),
        (
            torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            -1,
        ),
    ],
    ids=["fp8", "fp4"],
)
def test_construction(qdata1, qdata2, pack_axis):
    config = MXQuantizeConfig(elem_dtype=qdata1.dtype)
    scale1 = torch.full((*qdata1.shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)  # [..., R, R_blocks, 2]
    scale2 = torch.full((*qdata1.shape[:-2], 1, qdata1.shape[-1], 2), 4.0, dtype=torch.float8_e8m0fnu)

    expected_pack_axis = normalize_dim(pack_axis, qdata1.ndim) if pack_axis is not None else None

    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config, pack_axis=pack_axis)

    assert x.shape == torch.Size([4, 64]), f"logical shape {tuple(x.shape)} != (4, 64)"
    assert x.dtype is torch.bfloat16, f"logical dtype {x.dtype} != torch.bfloat16"
    assert x.pack_axis == expected_pack_axis, f"pack axis tracked as {x.pack_axis}, expected {expected_pack_axis}"
    assert x.quant_config is config, "the config must be stored as passed"


@pytest.mark.parametrize(
    "qdata1, qdata2, pack_axis, expected_shape, expected_pack_axis",
    [
        (
            torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            -1,
            (4, 64),
            1,
        ),
        (
            torch.randint(0, 256, (128, 8), dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t(),
            torch.randint(0, 256, (128, 8), dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t(),
            0,
            (16, 128),
            0,
        ),
    ],
    ids=["packed-on-the-last-dim", "packed-on-the-first-dim"],
)
def test_fp4_construction_doubles_the_packed_dim(qdata1, qdata2, pack_axis, expected_shape, expected_pack_axis):
    """Both qdata pack the same dim, which need not be the last one."""
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    scale1 = torch.full((*qdata1.shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((*qdata1.shape[:-2], 1, qdata1.shape[-1], 2), 4.0, dtype=torch.float8_e8m0fnu)

    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config, pack_axis=pack_axis)

    assert x.shape == torch.Size(expected_shape), f"logical shape {tuple(x.shape)} != {tuple(expected_shape)}"
    assert x.pack_axis == expected_pack_axis, f"pack axis tracked as {x.pack_axis}, expected {expected_pack_axis}"


def test_construction_rejects_mismatched_qdata_dtypes():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.randn(4, 64).to(torch.float8_e4m3fn)
    qdata2 = qdata1.to(torch.float8_e5m2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="must share a dtype"):
        DualAxisMXTensor(qdata1, scale, qdata2, scale, torch.bfloat16, config)


def test_construction_requires_qdata_to_agree_on_the_logical_shape():
    """Both qdata describe one logical tensor, so their shapes must match."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.randn(4, 64).to(torch.float8_e4m3fn)
    qdata2 = torch.randn(4, 32).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((4, 1, 2), 4.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="same logical shape"):
        DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)


def test_construction_rejects_scale_with_wrong_ndim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 1, 2), 4.0, dtype=torch.float8_e8m0fnu)  # one rank too deep

    with pytest.raises(ValueError, match="scale1/scale2 ndim must be"):
        DualAxisMXTensor(qdata, scale1, qdata, scale2, torch.bfloat16, config)


def test_construction_rejects_1d_qdata():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64).to(torch.float8_e4m3fn)
    scale = torch.full((64, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match=r"requires qdata.ndim >= 2"):
        DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)


def test_construction_rejects_pack_axis_on_non_fp4():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="non-FP4 tensor are not packed"):
        DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config, pack_axis=1)


# =========================================================================
# Axis-preserving shape ops
# =========================================================================


def test_permute_keeps_content_and_leading_dims():
    """Leading dims permute freely; the trailing two stay trailing."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.randn(2, 3, 4, 64).to(torch.float8_e4m3fn)
    qdata2 = torch.randn(2, 3, 4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((2, 3, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((2, 3, 1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)
    perm = [1, 0, 2, 3]

    y = x.permute(perm)

    expected_qdata1 = x.qdata1.permute(perm)
    assert y.qdata1.shape == expected_qdata1.shape, (
        f"permuted qdata1 shape {tuple(y.qdata1.shape)} != the expected {tuple(expected_qdata1.shape)}"
    )
    assert torch.equal(y.qdata1.view(torch.uint8), expected_qdata1.view(torch.uint8)), (
        "permuted qdata1 values differ from the permuted original"
    )

    expected_qdata2 = x.qdata2.permute(perm)
    assert y.qdata2.shape == expected_qdata2.shape, (
        f"permuted qdata2 shape {tuple(y.qdata2.shape)} != the expected {tuple(expected_qdata2.shape)}"
    )
    assert torch.equal(y.qdata2.view(torch.uint8), expected_qdata2.view(torch.uint8)), (
        "permuted qdata2 values differ from the permuted original"
    )

    expected_scale1 = x.scale1.permute([*perm, x.scale1.ndim - 1])
    assert y.scale1.shape == expected_scale1.shape, (
        f"permuted scale1 shape {tuple(y.scale1.shape)} != the expected {tuple(expected_scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), expected_scale1.view(torch.uint8)), (
        "permuted scale1 values differ from the permuted original"
    )

    expected_scale2 = x.scale2.permute([*perm, x.scale2.ndim - 1])
    assert y.scale2.shape == expected_scale2.shape, (
        f"permuted scale2 shape {tuple(y.scale2.shape)} != the expected {tuple(expected_scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), expected_scale2.view(torch.uint8)), (
        "permuted scale2 values differ from the permuted original"
    )


def test_swap_exchanges_the_two_pairs():
    """A swap trades the (qdata, scale) pairs so each keeps the dim it was quantized along."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.full((4, 64), 1.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    qdata2 = torch.full((4, 64), 2.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)

    y = x.transpose(-1, -2)

    assert y.shape == torch.Size([64, 4]), f"swapped shape {tuple(y.shape)} != torch.Size([64, 4])"

    expected_qdata1 = x.qdata2.transpose(-1, -2)
    assert y.qdata1.shape == expected_qdata1.shape, (
        f"swapped qdata1 shape {tuple(y.qdata1.shape)} != the original qdata2's {tuple(expected_qdata1.shape)}"
    )
    assert torch.equal(y.qdata1.view(torch.uint8), expected_qdata1.view(torch.uint8)), (
        "swapped qdata1 values differ from the original qdata2"
    )

    expected_scale1 = x.scale2.transpose(0, 1)
    assert y.scale1.shape == expected_scale1.shape, (
        f"swapped scale1 shape {tuple(y.scale1.shape)} != the original scale2's {tuple(expected_scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), expected_scale1.view(torch.uint8)), (
        "swapped scale1 values differ from the original scale2"
    )

    expected_qdata2 = x.qdata1.transpose(-1, -2)
    assert y.qdata2.shape == expected_qdata2.shape, (
        f"swapped qdata2 shape {tuple(y.qdata2.shape)} != the original qdata1's {tuple(expected_qdata2.shape)}"
    )
    assert torch.equal(y.qdata2.view(torch.uint8), expected_qdata2.view(torch.uint8)), (
        "swapped qdata2 values differ from the original qdata1"
    )

    expected_scale2 = x.scale1.transpose(0, 1)
    assert y.scale2.shape == expected_scale2.shape, (
        f"swapped scale2 shape {tuple(y.scale2.shape)} != the original scale1's {tuple(expected_scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), expected_scale2.view(torch.uint8)), (
        "swapped scale2 values differ from the original scale1"
    )


def test_swap_then_to_mx_tensor_pairs_the_quantized_dims():
    """After a swap the pairs must still match the dims they were quantized on."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.full((4, 64), 1.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    qdata2 = torch.full((4, 64), 2.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)

    y = x.transpose(-1, -2)

    # axis -1 of the swapped tensor is the original -2, which qdata2 covered
    expected = x.qdata2.transpose(-1, -2)
    assert y.to_mx_tensor(-1).qdata.shape == expected.shape, (
        f"swapped to_mx_tensor(-1) qdata shape {tuple(y.to_mx_tensor(-1).qdata.shape)} != {tuple(expected.shape)}"
    )
    assert torch.equal(y.to_mx_tensor(-1).qdata.view(torch.uint8), expected.view(torch.uint8)), (
        "swapped to_mx_tensor(-1) must pair with the original qdata2"
    )

    expected = x.qdata1.transpose(-1, -2)
    assert y.to_mx_tensor(-2).qdata.shape == expected.shape, (
        f"swapped to_mx_tensor(-2) qdata shape {tuple(y.to_mx_tensor(-2).qdata.shape)} != {tuple(expected.shape)}"
    )
    assert torch.equal(y.to_mx_tensor(-2).qdata.view(torch.uint8), expected.view(torch.uint8)), (
        "swapped to_mx_tensor(-2) must pair with the original qdata1"
    )


def test_permute_rejects_moving_the_last_two_dims_away():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    with pytest.raises(ValueError, match="keep the last two dims trailing"):
        x.permute([2, 0, 1])


def test_t_is_a_swap_on_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata1 = torch.randn(4, 64).to(torch.float8_e4m3fn)
    qdata2 = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)

    y = x.t()
    expected = x.transpose(0, 1)

    assert y.qdata1.shape == expected.qdata1.shape, (
        f"t() gave qdata1 shape {tuple(y.qdata1.shape)}, transpose(0, 1) gave {tuple(expected.qdata1.shape)}"
    )
    assert torch.equal(y.qdata1.view(torch.uint8), expected.qdata1.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different qdata1 values"
    )

    assert y.qdata2.shape == expected.qdata2.shape, (
        f"t() gave qdata2 shape {tuple(y.qdata2.shape)}, transpose(0, 1) gave {tuple(expected.qdata2.shape)}"
    )
    assert torch.equal(y.qdata2.view(torch.uint8), expected.qdata2.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different qdata2 values"
    )


def test_t_rejects_more_than_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    with pytest.raises(RuntimeError, match="expects a tensor with <= 2 dimensions"):
        x.t()


# =========================================================================
# Conversion to the single-axis MXTensor
# =========================================================================


@pytest.mark.parametrize(
    "qdata, pack_axis",
    [
        (torch.randn(4, 64).to(torch.float8_e4m3fn), None),
        (torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2), -1),
    ],
    ids=["fp8", "fp4"],
)
@pytest.mark.parametrize("quant_axis, qdata_name", [(1, "qdata1"), (0, "qdata2")])
def test_to_mx_tensor_shares_components(qdata, pack_axis, quant_axis, qdata_name):
    config = MXQuantizeConfig(elem_dtype=qdata.dtype)
    scale1 = torch.full((*qdata.shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((*qdata.shape[:-2], 1, qdata.shape[-1], 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale1, qdata, scale2, torch.bfloat16, config, pack_axis=pack_axis)

    y = x.to_mx_tensor(quant_axis)

    assert y.quant_axis == quant_axis, f"quantized along {y.quant_axis}, expected {quant_axis}"
    assert y.pack_axis == x.pack_axis, f"pack axis {y.pack_axis} != the source's {x.pack_axis}"
    assert y.qdata is getattr(x, qdata_name), f"quantized along {quant_axis} must reuse {qdata_name}"
    assert y.orig_dtype is x.orig_dtype, f"orig_dtype {y.orig_dtype} != the source's {x.orig_dtype}"


def test_to_mx_tensor_rejects_other_axes():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    with pytest.raises(ValueError, match="Only the last two dims are supported"):
        x.to_mx_tensor(0)


# =========================================================================
# Inherited behavior
# =========================================================================


def test_to_dtype_changes_only_the_logical_dtype():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    y = x.to(torch.float16)

    assert y.dtype is torch.float16, f".to(torch.float16) left the logical dtype at {y.dtype}"
    assert y.qdata1 is x.qdata1, ".to(dtype) cast the stored qdata1"


def test_values_are_frozen():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    with pytest.raises(RuntimeError, match="immutable"):
        x.zero_()


def test_matmul_is_not_supported():
    """Matmul is whitelisted for MXTensor only."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata, scale, qdata, scale, torch.bfloat16, config)

    with pytest.raises(NotImplementedError):
        torch.mm(x, x)


# =========================================================================
# Quantization against the primitives (NPU kernels)
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_from_hp_matches_mx_quantize_dual_axis(elem_dtype):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = DualAxisMXTensor.from_hp(x, config)
    qdata1, scale1, qdata2, scale2 = mx_quantize_dual_axis(x, config)

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"

    # the op packs both qdata along the input's last dim; FP8 has no packing at all
    expected_pack_axis = y.ndim - 1 if elem_dtype is torch.float4_e2m1fn_x2 else None
    assert y.pack_axis == expected_pack_axis, f"pack axis {y.pack_axis}, expected {expected_pack_axis}"

    assert y.qdata1.shape == qdata1.shape, (
        f"qdata1 shape {tuple(y.qdata1.shape)} != the primitive's {tuple(qdata1.shape)}"
    )
    assert torch.equal(y.qdata1.view(torch.uint8), qdata1.view(torch.uint8)), (
        "qdata1 values differ from the primitive's"
    )

    assert y.scale1.shape == scale1.shape, (
        f"scale1 shape {tuple(y.scale1.shape)} != the primitive's {tuple(scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), scale1.view(torch.uint8)), (
        "scale1 values differ from the primitive's"
    )

    assert y.qdata2.shape == qdata2.shape, (
        f"qdata2 shape {tuple(y.qdata2.shape)} != the primitive's {tuple(qdata2.shape)}"
    )
    assert torch.equal(y.qdata2.view(torch.uint8), qdata2.view(torch.uint8)), (
        "qdata2 values differ from the primitive's"
    )

    assert y.scale2.shape == scale2.shape, (
        f"scale2 shape {tuple(y.scale2.shape)} != the primitive's {tuple(scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), scale2.view(torch.uint8)), (
        "scale2 values differ from the primitive's"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_to_mx_tensor_equals_single_axis_from_hp(elem_dtype):
    """The pair stored for one quant axis must match a single-axis quantization."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = DualAxisMXTensor.from_hp(x, config).to_mx_tensor(-2)
    y_ref = MXTensor.from_hp(x, config, axis=-2)

    assert y.pack_axis == y_ref.pack_axis, (
        f"pack axis {y.pack_axis} != the single-axis quantization's {y_ref.pack_axis}"
    )

    assert y.qdata.shape == y_ref.qdata.shape, (
        f"qdata shape {tuple(y.qdata.shape)} != the single-axis quantization's {tuple(y_ref.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), y_ref.qdata.view(torch.uint8)), (
        "qdata values differ from the single-axis quantization's"
    )

    assert y.scale.shape == y_ref.scale.shape, (
        f"scale shape {tuple(y.scale.shape)} != the single-axis quantization's {tuple(y_ref.scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), y_ref.scale.view(torch.uint8)), (
        "scale values differ from the single-axis quantization's"
    )


# =========================================================================
# Dequantize
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_uses_the_last_dim_pair(elem_dtype):
    """For a freshly quantized tensor the innermost-contiguous quant axis is the last dim.

    The pair that covers it, ``qdata1``/``scale1``, is the one whose dequantization the result
    must match; using the other pair would give different values, so this pins the choice.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = DualAxisMXTensor.from_hp(tensor, config)

    dequantized = x.dequantize()
    expected = mx_dequantize(
        x.qdata1,
        x.scale1,
        x.qdata1.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == expected.stride(), f"stride mismatch: {dequantized.stride()} vs {expected.stride()}"
    assert torch.equal(dequantized, expected), "the result is not the last-dim pair's"


def test_dequantize_after_a_swap_dequantizes_the_same_axis():
    """A swap moves the two pairs, and with them the trailing dim each one covers.

    ``t()`` makes the unit-stride dim second-to-last, so ``dequantize`` has to fall through to
    the second pair -- which now covers the last dim, the axis the first pair covered before the
    swap. The values are therefore the same quantization, transposed.
    """
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = DualAxisMXTensor.from_hp(tensor, config)

    y = x.t()
    dequantized = y.dequantize()
    expected = mx_dequantize(
        x.qdata1,
        x.scale1,
        x.qdata1.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == y.shape, f"shape mismatch: {dequantized.shape} vs {y.shape}"
    assert dequantized.stride() == expected.t().stride(), (
        f"stride mismatch: {dequantized.stride()} vs {expected.t().stride()}"
    )
    assert torch.equal(dequantized, expected.t()), "the swapped tensor dequantized the wrong pair"


def test_dequantize_rejects_qdata_without_a_unit_stride_quant_axis():
    """Neither trailing dim is innermost-contiguous: there is no pair to dequantize.

    ``from_hp`` and the shape ops cannot produce such a tensor -- they leave one of the two dims
    unit-stride -- but the constructor does not forbid it, so the error is checked here.
    """
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    # A permutation that puts the unit-stride dim first: both trailing strides are > 1.
    qdata1 = torch.randn(4, 8, 8).to(torch.float8_e4m3fn).permute(2, 1, 0)
    qdata2 = torch.randn(4, 8, 8).to(torch.float8_e4m3fn).permute(2, 1, 0)
    scale1 = torch.full((8, 8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((8, 1, 4, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = DualAxisMXTensor(qdata1, scale1, qdata2, scale2, torch.bfloat16, config)

    with pytest.raises(RuntimeError, match="innermost-contiguous"):
        x.dequantize()
