# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`torchao_npu.quantized_tensors.block_mx_tensor.BlockMXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor

# =========================================================================
# Construction and validation
# =========================================================================


def test_construction():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)  # [..., R, R_blocks, 2]
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)  # [..., C_blocks, C, 2]

    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)

    assert x.shape == torch.Size([4, 64])
    assert x.dtype is torch.bfloat16
    assert x.pack_axis is None
    assert x.quant_config is config


@pytest.mark.parametrize(
    "qdata, pack_axis, expected_shape, expected_pack_axis",
    [
        (
            torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            -1,
            (4, 64),
            1,
        ),
        (
            torch.randint(0, 256, (128, 8), dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t(),
            0,
            (16, 128),
            0,
        ),
    ],
    ids=["packed-on-the-last-dim", "packed-on-the-first-dim"],
)
def test_fp4_construction_doubles_the_packed_dim(qdata, pack_axis, expected_shape, expected_pack_axis):
    """The packed dim is whichever the qdata is unit-stride on, not necessarily the last one."""
    config = BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    scale1 = torch.full((*qdata.shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((*qdata.shape[:-2], 1, qdata.shape[-1], 2), 4.0, dtype=torch.float8_e8m0fnu)

    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config, pack_axis=pack_axis)

    assert x.shape == torch.Size(expected_shape), f"logical shape {tuple(x.shape)} != {tuple(expected_shape)}"
    assert x.pack_axis == expected_pack_axis, f"pack axis tracked as {x.pack_axis}, expected {expected_pack_axis}"


def test_construction_rejects_scale_with_wrong_ndim():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 1, 2), 4.0, dtype=torch.float8_e8m0fnu)  # one rank too deep

    with pytest.raises(ValueError, match="scale1/scale2 ndim must be"):
        BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)


def test_construction_rejects_1d_qdata():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64).to(torch.float8_e4m3fn)
    scale = torch.full((64, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match=r"requires qdata.ndim >= 2"):
        BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)


@pytest.mark.parametrize(
    "qdata_dtype, config_dtype, expected_error, expected_message",
    [
        # qdata dtypes this class does not store
        (torch.bfloat16, torch.float8_e4m3fn, ValueError, "supports FP8 or FP4 qdata"),
        (torch.float32, torch.float4_e2m1fn_x2, ValueError, "supports FP8 or FP4 qdata"),
        # supported dtypes that disagree with the config
        (torch.float8_e4m3fn, torch.float8_e5m2, AssertionError, "is not consistent with"),
        (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, AssertionError, "is not consistent with"),
        (torch.float8_e5m2, torch.float8_e4m3fn, AssertionError, "is not consistent with"),
        (torch.float8_e5m2, torch.float4_e2m1fn_x2, AssertionError, "is not consistent with"),
        (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, AssertionError, "is not consistent with"),
        (torch.float4_e2m1fn_x2, torch.float8_e5m2, AssertionError, "is not consistent with"),
    ],
)
def test_construction_rejects_qdata_dtype_mismatch(qdata_dtype, config_dtype, expected_error, expected_message):
    """Every (qdata dtype, config elem_dtype) pair outside the three matching ones."""
    if qdata_dtype is torch.float4_e2m1fn_x2:
        qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        pack_axis = -1  # FP4 data cannot even be described without its pack axis
    else:
        qdata = torch.randn(4, 64).to(qdata_dtype)
        pack_axis = None
    config = BlockMXQuantizeConfig(elem_dtype=config_dtype)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(expected_error, match=expected_message):
        BlockMXTensor(qdata, scale, scale, torch.bfloat16, config, pack_axis=pack_axis)


def test_construction_rejects_pack_axis_on_non_fp4():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="non-FP4 tensor are not packed"):
        BlockMXTensor(qdata, scale, scale, torch.bfloat16, config, pack_axis=1)


# =========================================================================
# Axis-preserving shape ops
# =========================================================================


def test_permute_keeps_content_and_leading_dims():
    """Leading dims permute freely; the trailing two stay trailing."""
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 3, 4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((2, 3, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((2, 3, 1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)
    perm = [1, 0, 2, 3]

    y = x.permute(perm)

    expected_qdata = x.qdata.permute(perm)
    assert y.qdata.shape == expected_qdata.shape, (
        f"permuted qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "permuted qdata values differ from the permuted original"
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


def test_swap_transposes_qdata_and_exchanges_the_two_scales():
    """A swap transposes qdata and swaps the scales' roles -- content, not just shape."""
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)

    y = x.transpose(-1, -2)

    assert y.shape == torch.Size([64, 4]), f"swapped shape {tuple(y.shape)} != torch.Size([64, 4])"

    expected_qdata = x.qdata.transpose(-1, -2)
    assert y.qdata.shape == expected_qdata.shape, (
        f"swapped qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "swapped qdata values differ from the transposed original"
    )

    expected_scale1 = x.scale2.transpose(0, 1)  # the two scales trade roles
    assert y.scale1.shape == expected_scale1.shape, (
        f"swapped scale1 shape {tuple(y.scale1.shape)} != the original scale2's {tuple(expected_scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), expected_scale1.view(torch.uint8)), (
        "swapped scale1 values differ from the original scale2"
    )

    expected_scale2 = x.scale1.transpose(0, 1)
    assert y.scale2.shape == expected_scale2.shape, (
        f"swapped scale2 shape {tuple(y.scale2.shape)} != the original scale1's {tuple(expected_scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), expected_scale2.view(torch.uint8)), (
        "swapped scale2 values differ from the original scale1"
    )


def test_permute_rejects_moving_the_last_two_dims_away():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

    with pytest.raises(ValueError, match="keep the last two dims trailing"):
        x.permute([2, 0, 1])


def test_t_is_a_swap_on_2d():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale1 = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((1, 64, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)

    y = x.t()
    expected = x.transpose(0, 1)

    assert y.qdata.shape == expected.qdata.shape, (
        f"t() gave qdata shape {tuple(y.qdata.shape)}, transpose(0, 1) gave {tuple(expected.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different qdata values"
    )

    assert y.scale1.shape == expected.scale1.shape, (
        f"t() gave scale1 shape {tuple(y.scale1.shape)}, transpose(0, 1) gave {tuple(expected.scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), expected.scale1.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different scale1 values"
    )


def test_t_rejects_more_than_2d():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

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
@pytest.mark.parametrize("quant_axis, use_scale1", [(1, True), (0, False)])
def test_to_mx_tensor_shares_components(qdata, pack_axis, quant_axis, use_scale1):
    config = BlockMXQuantizeConfig(elem_dtype=qdata.dtype)
    scale1 = torch.full((*qdata.shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((*qdata.shape[:-2], 1, qdata.shape[-1], 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config, pack_axis=pack_axis)

    y = x.to_mx_tensor(quant_axis)

    assert y.quant_axis == quant_axis, f"quantized along {y.quant_axis}, expected {quant_axis}"
    assert y.pack_axis == x.pack_axis, f"pack axis {y.pack_axis} != the source's {x.pack_axis}"
    assert y.qdata is x.qdata, "the converted tensor must share qdata, not copy it"

    expected_scale = x.scale1 if use_scale1 else x.scale2
    assert y.scale is expected_scale, f"quantized along {quant_axis} must use the matching scale"

    assert type(y.quant_config) is MXQuantizeConfig, (
        f"quant_config type {type(y.quant_config).__name__}, expected MXQuantizeConfig"
    )


def test_to_mx_tensor_rejects_other_axes():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

    with pytest.raises(ValueError, match="Only the last two dims are supported"):
        x.to_mx_tensor(0)


# =========================================================================
# Inherited behavior
# =========================================================================


def test_to_dtype_changes_only_the_logical_dtype():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

    y = x.to(torch.float32)

    assert y.dtype is torch.float32, f".to(torch.float32) left the logical dtype at {y.dtype}"
    assert y.qdata is x.qdata, ".to(dtype) cast the stored qdata"


def test_values_are_frozen():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

    with pytest.raises(RuntimeError, match="immutable"):
        x.zero_()


def test_matmul_is_not_supported():
    """Matmul is whitelisted for MXTensor only."""
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale, scale, torch.bfloat16, config)

    with pytest.raises(NotImplementedError):
        torch.mm(x, x)


# =========================================================================
# Quantization against the primitive (NPU kernels)
# =========================================================================


def test_from_hp_matches_block_mx_quantize():
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    y = BlockMXTensor.from_hp(x, config)
    qdata, scale1, scale2 = block_mx_quantize(x, config)

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"
    assert y.pack_axis is None, f"FP8 needs no pack axis, got {y.pack_axis}"

    assert y.qdata.shape == qdata.shape, f"qdata shape {tuple(y.qdata.shape)} != the primitive's {tuple(qdata.shape)}"
    assert torch.equal(y.qdata.view(torch.uint8), qdata.view(torch.uint8)), "qdata values differ from the primitive's"

    assert y.scale1.shape == scale1.shape, (
        f"scale1 shape {tuple(y.scale1.shape)} != the primitive's {tuple(scale1.shape)}"
    )
    assert torch.equal(y.scale1.view(torch.uint8), scale1.view(torch.uint8)), (
        "scale1 values differ from the primitive's"
    )

    assert y.scale2.shape == scale2.shape, (
        f"scale2 shape {tuple(y.scale2.shape)} != the primitive's {tuple(scale2.shape)}"
    )
    assert torch.equal(y.scale2.view(torch.uint8), scale2.view(torch.uint8)), (
        "scale2 values differ from the primitive's"
    )


# =========================================================================
# Dequantize
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_uses_the_last_dim_scale(elem_dtype):
    """For a freshly quantized tensor the innermost-contiguous quant axis is the last dim.

    Its scale, ``scale1``, is the one whose dequantization the result must match; using the
    other scale would give different values, so this pins the choice.
    """
    config = BlockMXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = BlockMXTensor.from_hp(tensor, config)

    dequantized = x.dequantize()
    expected = mx_dequantize(
        x.qdata,
        x.scale1,
        x.qdata.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == expected.stride(), f"stride mismatch: {dequantized.stride()} vs {expected.stride()}"
    assert torch.equal(dequantized, expected), "the result is not the last-dim scale's"


def test_dequantize_after_a_swap_dequantizes_the_same_axis():
    """A swap exchanges the two scales, and with them the trailing dim each one covers.

    ``t()`` makes the unit-stride dim second-to-last, so ``dequantize`` has to fall through to
    the second scale -- which now covers the last dim, the axis ``scale1`` covered before the
    swap. The values are therefore the same quantization, transposed.
    """
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = BlockMXTensor.from_hp(tensor, config)

    y = x.t()
    dequantized = y.dequantize()
    expected = mx_dequantize(
        x.qdata,
        x.scale1,
        x.qdata.ndim - 1,
        block_size=config.block_size,
        src_dtype=config.elem_dtype,
        output_dtype=tensor.dtype,
    )

    assert dequantized.shape == y.shape, f"shape mismatch: {dequantized.shape} vs {y.shape}"
    assert dequantized.stride() == expected.t().stride(), (
        f"stride mismatch: {dequantized.stride()} vs {expected.t().stride()}"
    )
    assert torch.equal(dequantized, expected.t()), "the swapped tensor dequantized the wrong scale"


def test_dequantize_rejects_qdata_without_a_unit_stride_quant_axis():
    """Neither trailing dim is innermost-contiguous: there is no scale to dequantize.

    ``from_hp`` and the shape ops cannot produce such a tensor -- they leave one of the two dims
    unit-stride -- but the constructor does not forbid it, so the error is checked here.
    """
    config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    # A permutation that puts the unit-stride dim first: both trailing strides are > 1.
    qdata = torch.randn(4, 8, 8).to(torch.float8_e4m3fn).permute(2, 1, 0)
    scale1 = torch.full((8, 8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    scale2 = torch.full((8, 1, 4, 2), 4.0, dtype=torch.float8_e8m0fnu)
    x = BlockMXTensor(qdata, scale1, scale2, torch.bfloat16, config)

    with pytest.raises(RuntimeError, match="innermost-contiguous"):
        x.dequantize()
