# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`torchao_npu.quantized_tensors.mx_tensor.MXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.ops.mx_ops import to_mx_then_bmm, to_mx_then_grouped_mm, to_mx_then_mm
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_fake_quantize
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

# =========================================================================
# Construction and validation
# =========================================================================


def test_fp8_construction():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert x.shape == torch.Size([4, 64])
    assert x.dtype is torch.bfloat16
    assert x.quant_axis == 1  # normalized from -1
    assert x.pack_axis is None
    assert x.quant_config is config


def test_fp4_construction_doubles_the_logical_shape():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    x = MXTensor(qdata, scale, torch.bfloat16, -2, config, pack_axis=-1)

    assert x.shape == torch.Size([4, 64])  # qdata is (4, 32); packing doubles axis 1
    assert x.quant_axis == 0
    assert x.pack_axis == 1  # normalized from -1
    assert x.qdata.dtype is torch.float4_e2m1fn_x2


def test_construction_rejects_qdata_dtype_mismatch():
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    fp4_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    with pytest.raises(AssertionError, match="is not consistent with"):
        MXTensor(qdata, scale, torch.bfloat16, -1, fp4_config)


def test_construction_rejects_scale_with_wrong_ndim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)  # ndim = qdata.ndim + 2

    with pytest.raises(ValueError, match=r"scale.ndim must be"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config)


def test_construction_rejects_pack_axis_on_non_fp4():
    """Values of an FP8 tensor are not packed, so a pack axis is meaningless."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="non-FP4 tensor are not packed"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=1)


def test_construction_rejects_fp4_without_pack_axis():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="pack_axis should be set"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config)


# =========================================================================
# Axis-preserving shape ops
# =========================================================================


def test_permute_tracks_quant_axis_and_keeps_content():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    perm = [0, 2, 1]

    y = x.permute(perm)

    assert y.quant_axis == 1  # was 2, tracked through the permutation
    assert y.pack_axis is None
    expected_qdata = x.qdata.permute(perm)
    assert y.qdata.shape == expected_qdata.shape, (
        f"permuted qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "permuted qdata values differ from the permuted original"
    )

    expected_scale = x.scale.permute([*perm, x.scale.ndim - 1])
    assert y.scale.shape == expected_scale.shape, (
        f"permuted scale shape {tuple(y.scale.shape)} != the expected {tuple(expected_scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected_scale.view(torch.uint8)), (
        "permuted scale values differ from the permuted original"
    )


def test_transpose_swaps_axes_and_content():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.transpose(-1, -2)

    assert y.shape == torch.Size([64, 4])
    assert y.quant_axis == 0
    expected_qdata = x.qdata.transpose(-1, -2)
    assert y.qdata.shape == expected_qdata.shape, (
        f"transposed qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "transposed qdata values differ from the transposed original"
    )

    expected_scale = x.scale.transpose(0, 1)
    assert y.scale.shape == expected_scale.shape, (
        f"transposed scale shape {tuple(y.scale.shape)} != the expected {tuple(expected_scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected_scale.view(torch.uint8)), (
        "transposed scale values differ from the transposed original"
    )


def test_t_matches_transpose_on_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.t()
    expected = x.transpose(0, 1)

    assert y.qdata.shape == expected.qdata.shape, (
        f"t() gave qdata shape {tuple(y.qdata.shape)}, transpose(0, 1) gave {tuple(expected.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different qdata values"
    )
    assert y.quant_axis == expected.quant_axis, (
        f"t() tracked quant_axis {y.quant_axis}, transpose(0, 1) tracked {expected.quant_axis}"
    )


def test_t_rejects_more_than_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(RuntimeError, match="expects a tensor with <= 2 dimensions"):
        x.t()


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize(
    "shape",
    [(2, 64), (2, 3, 64), (2, 3, 4, 64), (2, 3, 4, 5, 64)],  # ..., 64: the quantized dim
    ids=["2d", "3d", "4d", "5d"],
)
@pytest.mark.parametrize(
    "source, destination",
    [(-1, 0), (0, 1), (0, -1)],
    ids=["last-to-first", "first-to-second", "first-to-last"],
)
def test_movedim_matches_the_equivalent_permute(elem_dtype, shape, source, destination):
    """``movedim`` decomposes into a permute, so the axis tracking has to follow it."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    if elem_dtype is torch.float4_e2m1fn_x2:
        # FP4 stores two values per byte along the last dim, so the stored dim is halved
        qdata = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        pack_axis = -1
    else:
        qdata = torch.randn(shape).to(torch.float8_e4m3fn)
        pack_axis = None
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=pack_axis)

    y = torch.movedim(x, source, destination)

    # movedim is defined as taking `source` out of the dims and putting it at `destination`
    perm = list(range(len(shape)))
    perm.insert(destination % len(shape), perm.pop(source))
    expected = x.permute(perm)
    expected_quant_axis = perm.index(len(shape) - 1)

    assert y.shape == expected.shape, (
        f"movedim({source}, {destination}) gave shape {tuple(y.shape)}, permute({perm}) gave {tuple(expected.shape)}"
    )
    assert y.quant_axis == expected_quant_axis, (
        f"movedim({source}, {destination}) tracked quant_axis {y.quant_axis}, expected {expected_quant_axis}"
    )
    assert y.pack_axis == (expected_quant_axis if elem_dtype is torch.float4_e2m1fn_x2 else None), (
        f"movedim({source}, {destination}) tracked pack_axis {y.pack_axis}, expected "
        f"{expected_quant_axis if elem_dtype is torch.float4_e2m1fn_x2 else None}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)), (
        "movedim and the equivalent permute produced different qdata"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected.scale.view(torch.uint8)), (
        "movedim and the equivalent permute produced different scale"
    )


def test_fp4_shape_ops_track_the_pack_axis():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=-1)

    y = x.t()

    expected_qdata = x.qdata.t()

    assert y.pack_axis == 0, f"t() tracked pack_axis {y.pack_axis}, expected the packed dim to move to 0"
    assert y.quant_axis == 0, f"t() tracked quant_axis {y.quant_axis}, expected 0"
    assert y.qdata.shape == expected_qdata.shape, (
        f"transposed FP4 qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "transposed FP4 qdata values differ from the transposed original"
    )


# =========================================================================
# Frozen values
# =========================================================================


@pytest.mark.parametrize("op", ["zero_", "fill_", "copy_"])
def test_in_place_ops_are_rejected(op):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    qdata_before = x.qdata.clone()

    with pytest.raises(RuntimeError, match="immutable"):
        if op == "copy_":
            x.copy_(x)
        elif op == "zero_":
            x.zero_()
        else:
            x.fill_(1.0)

    assert torch.equal(x.qdata, qdata_before)


# =========================================================================
# Matmul operand rules (no kernel involved)
# =========================================================================


def test_mm_rejects_non_mx_right_operand():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="must be a MXTensor"):
        torch.mm(x, torch.randn(64, 8, dtype=torch.bfloat16))


def test_mm_requires_right_operand_quantized_along_dim_minus_2():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(64, 8).to(torch.float8_e4m3fn),
        torch.full((64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,  # must be -2 for a right operand
        config,
        act_quant_config=config,
    )

    with pytest.raises(ValueError, match="quantized along dim=-2"):
        torch.mm(activation, weight)


def test_mm_requires_left_operand_quantized_along_dim_minus_1():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,  # must be -1 for a left operand
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="quantized along dim=-1"):
        torch.mm(activation, weight)


def test_mm_requires_matching_activation_config():
    """The left operand must be quantized with the right operand's act_quant_config."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    other_config = MXQuantizeConfig(elem_dtype=torch.float8_e5m2)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e5m2),  # stored [N, K]
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        other_config,
        act_quant_config=other_config,
    ).transpose(-2, -1)

    with pytest.raises(ValueError, match="act_quant_config"):
        torch.mm(activation, weight)


def test_mm_rejects_a_plain_left_operand_without_an_activation_config():
    """A high-precision left operand needs the right operand's act_quant_config."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,  # no act_quant_config
    ).transpose(-2, -1)

    with pytest.raises(RuntimeError, match="carries no"):
        torch.mm(torch.randn(4, 64, dtype=torch.bfloat16), weight)


def test_mm_rejects_a_left_operand_that_requires_grad():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(RuntimeError, match="inference-only"):
        torch.mm(activation, weight)


def test_addmm_checks_operand_ranks():
    """``addmm`` is 2D-only, like the ``mm`` it wraps (plus a bias)."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = torch.randn(1, 4, 64, dtype=torch.bfloat16)  # 3D: rejected before anything else
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K]
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    )
    bias = torch.zeros(8, dtype=torch.float32)

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.addmm(bias, activation, weight.transpose(-2, -1))


def test_mm_checks_operand_ranks_and_shapes():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.mm(activation, weight)


def test_mm_out_dtype_checks_operand_ranks():
    """``torch.mm(..., out_dtype=...)`` is the ``aten.mm.dtype`` overload: same operand rules."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.mm(activation, weight, out_dtype=torch.float16)


def test_addmm_out_dtype_checks_operand_ranks():
    """``out_dtype`` is the fourth positional argument of the ``aten.addmm.dtype`` overload."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives
    bias = torch.zeros(8, dtype=torch.float32)

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.addmm(bias, activation, weight, out_dtype=torch.float16)


def test_bmm_out_dtype_checks_operand_ranks():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives
    activation = torch.randn(4, 64, dtype=torch.bfloat16)  # 2D: bmm's operands must be 3D

    with pytest.raises(ValueError, match="expects 3D operands"):
        torch.bmm(activation, weight, out_dtype=torch.float16)


def test_matmul_rejects_a_batched_weight():
    """``matmul``'s handler supports a 2D right operand only: a batched weight needs ``expand``."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(2, 8, 64).to(torch.float8_e4m3fn),  # stored [B, N, K], quantized along K
        torch.full((2, 8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [B, K, N]: what a matmul receives
    activation = torch.randn(4, 64, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="2D right operand"):
        MXTensor.__torch_dispatch__(torch.ops.aten.matmul.default, (MXTensor,), (activation, weight), {})


# =========================================================================
# Matmul against the training path (NPU kernels)
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_mm_matches_to_mx_then_mm(elem_dtype):
    """A pre-quantized weight must give what the training path computes.

    ``to_mx_then_mm`` quantizes both operands itself; the MXTensor path reuses
    the weight's stored quantization. Matching them is the train/inference
    consistency this class exists for.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.dtype is torch.bfloat16
    assert torch.equal(y, y_ref)


def test_bmm_matches_to_mx_then_bmm():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_bmm(a, w.transpose(-2, -1), config, config)

    assert torch.equal(y, y_ref)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_out_dtype_is_honored(elem_dtype, out_dtype):
    """``torch.mm(..., out_dtype=...)`` takes the ``aten.mm.dtype`` overload into the kernel."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1), out_dtype=out_dtype)
    y_default = torch.mm(a, weight.transpose(-2, -1))

    assert y.dtype is out_dtype, f"out_dtype {out_dtype} not honored: got {y.dtype}"
    if out_dtype is y_default.dtype:
        assert torch.equal(y, y_default), "asking for the dtype the product already has changed it"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_bmm_out_dtype_is_honored(elem_dtype):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_addmm_out_dtype_is_honored(elem_dtype):
    """``out_dtype`` is the fourth positional argument of the ``aten.addmm.dtype`` overload."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize(("beta", "alpha"), [(2, 1), (1, 3), (2, 3)])
def test_addmm_scales_the_bias_and_the_product(beta, alpha):
    """``beta``/``alpha`` scale the kernel's output: the fused epilogue has no scaling terms."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), beta=beta, alpha=alpha)
    y_ref = alpha * to_mx_then_mm(a, w.transpose(-2, -1), config, config) + beta * bias

    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "scaled addmm differs from the scaled training path"
    )


# =========================================================================
# Grouped matmul
# =========================================================================


def test_grouped_mm_checks_the_contracting_dim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    offs = torch.tensor([2, 4], dtype=torch.int32)

    weight_mismatched_k = MXTensor(
        torch.randn(2, 32, 8).to(torch.float8_e4m3fn),  # K = 32, activation's K = 64
        torch.full((2, 1, 8, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
        act_quant_config=config,
    )

    with pytest.raises(ValueError, match="Contracting dim mismatch"):
        torch._grouped_mm(activation, weight_mismatched_k, offs=offs)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_grouped_mm_matches_to_mx_then_grouped_mm(elem_dtype):
    """A pre-quantized expert weight must give what the training path computes."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    m, k, n, experts = 64, 128, 64, 4
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(experts, n, k, device="npu", dtype=torch.bfloat16)  # stored [E, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    offs = torch.tensor([8, 32, 48, 64], device="npu", dtype=torch.int32)  # uneven groups

    y = torch._grouped_mm(a, weight.transpose(-2, -1), offs=offs)
    y_ref = to_mx_then_grouped_mm(a, w.transpose(-2, -1), offs, config, config)

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert torch.equal(y, y_ref), "grouped matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_matches_to_mx_then_mm(elem_dtype, shape):
    """``F.linear`` takes the stored ``[N, K]`` weight; that must give the training path's result.

    A 3D activation goes through the path ``F.linear`` itself flattens, so the shape it restores
    is checked along with the values. Every trailing dim is a multiple of 32, as the training
    path's dual-axis quantization of either operand requires.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.nn.functional.linear(a, weight)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "linear differs from the training path's"


@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_with_bias_matches_to_mx_then_mm(shape):
    """With a bias, ``F.linear`` routes through ``addmm``, which fuses the bias into the kernel."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(n, device="npu", dtype=torch.float32)

    y = torch.nn.functional.linear(a, weight, bias)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config) + bias

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "linear with a bias differs from the training path's plus the bias"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.linear`` handler itself, which ``F.linear`` never reaches.

    ``aten::linear`` is a composite op, decomposed into ``t``/``addmm`` before dispatch, so the
    handler runs only when something dispatches ``linear`` itself -- a backend registering it,
    as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way. The
    activation flattens to a multiple of 32 rows for the training path's sake.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.linear.default, (MXTensor,), (a, weight, None), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the linear handler differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_matches_to_mx_then_mm(elem_dtype, shape):
    """``torch.matmul`` against a stored ``[N, K]`` weight must give the training path's result.

    A 3D activation is folded into the rows by the composite ``matmul`` before its ``mm``, so the
    shape it restores is checked along with the values.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.matmul(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.matmul`` handler itself, which ``torch.matmul`` reaches only through its leaves.

    ``aten::matmul`` is a composite op, decomposed into ``mm``/``bmm``/``expand`` before dispatch,
    so the handler runs only when something dispatches ``matmul`` itself -- a backend registering
    it, as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.matmul.default, (MXTensor,), (a, weight.transpose(-2, -1)), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the matmul handler differs from the training path's"


# =========================================================================
# Dequantize
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_matches_mx_fake_quantize(elem_dtype):
    """``dequantize`` undoes ``from_hp``'s quantization, reproducing ``mx_fake_quantize``.

    ``mx_fake_quantize`` quantizes and dequantizes the permuted tensor in one op, so for a
    tensor both permute the same way they see the same ``qdata``/``scale`` and must agree bit
    for bit.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    dequantized = x.dequantize()
    fake_quantized = mx_fake_quantize(tensor, -1, config)

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == fake_quantized.stride(), (
        f"stride mismatch: {dequantized.stride()} vs {fake_quantized.stride()}"
    )
    assert torch.equal(dequantized, fake_quantized), "dequantized values differ from mx_fake_quantize"


def test_dequantize_honors_output_dtype():
    """Every MX value fits in bfloat16, so both output dtypes agree bit for bit."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    as_bfloat16 = x.dequantize()
    as_float32 = x.dequantize(torch.float32)

    assert as_bfloat16.dtype is torch.bfloat16, f"dtype {as_bfloat16.dtype} != torch.bfloat16"
    assert as_float32.dtype is torch.float32, f"dtype {as_float32.dtype} != torch.float32"
    assert as_float32.stride() == as_bfloat16.stride(), (
        f"stride mismatch: {as_float32.stride()} vs {as_bfloat16.stride()}"
    )
    assert torch.equal(as_float32, as_bfloat16.to(torch.float32)), "the two output dtypes disagree"


def test_dequantize_rejects_a_strided_quant_axis():
    """A pass-through quantization keeps the quant axis strided, which cannot be dequantized."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(4, 256, 128, device="npu", dtype=torch.bfloat16)

    x = MXTensor.from_hp(tensor, config, axis=1)  # dense, non-trailing axis: no permutation
    assert x.qdata.stride(x.quant_axis) != 1, "the pass-through route should keep the quant axis strided"

    with pytest.raises(AssertionError, match="unit stride"):
        x.dequantize()
