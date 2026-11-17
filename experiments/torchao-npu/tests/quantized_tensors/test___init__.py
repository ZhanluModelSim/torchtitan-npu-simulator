# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the helpers shared by the quantized tensor classes."""

import itertools

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantized_tensors import (
    logical_qdata_shape,
    permutation_from_transpose,
    resolve_pack_axis,
)


@pytest.mark.parametrize(
    "dim0, dim1, ndim, expected",
    [
        (0, 1, 3, [1, 0, 2]),
        (1, 2, 3, [0, 2, 1]),
        (-1, -2, 3, [0, 2, 1]),
        (0, 1, 5, [1, 0, 2, 3, 4]),
        (1, 3, 5, [0, 3, 2, 1, 4]),
        (-1, -3, 5, [0, 1, 4, 3, 2]),
    ],
)
def test_permutation_from_transpose(dim0, dim1, ndim, expected):
    perm = permutation_from_transpose(dim0, dim1, ndim)

    assert perm == expected, (
        f"permutation_from_transpose(dim0={dim0}, dim1={dim1}, ndim={ndim}) returned {perm}, expected {expected}"
    )


@pytest.mark.parametrize("ndim", [2, 3, 5])
def test_permutation_from_transpose_is_its_own_inverse(ndim):
    """A dims list produced by a swap is its own inverse: applying it twice is the identity."""
    identity = list(range(ndim))

    for dim0, dim1 in itertools.combinations(range(ndim), 2):
        perm = permutation_from_transpose(dim0, dim1, ndim)
        composed = [perm[i] for i in perm]

        assert composed == identity, (
            f"permutation_from_transpose(dim0={dim0}, dim1={dim1}, ndim={ndim}) "
            f"returned {perm}; applying it twice gives {composed}, expected {identity}"
        )


def test_permutation_from_transpose_out_of_range():
    with pytest.raises(IndexError):
        permutation_from_transpose(0, 3, 3)


@pytest.mark.parametrize("shape", [(64,), (4, 64), (2, 4, 64)])
def test_logical_qdata_shape_fp8_is_the_stored_shape(shape):
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)

    assert logical_qdata_shape(qdata, None) == shape


@pytest.mark.parametrize(
    "qdata_shape, pack_axis, expected",
    [
        ((32,), -1, (64,)),
        ((4, 32), 1, (4, 64)),
        ((4, 32), -1, (4, 64)),
        ((2, 4, 32), -1, (2, 4, 64)),
    ],
)
def test_logical_qdata_shape_fp4_doubles_the_pack_axis(qdata_shape, pack_axis, expected):
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    assert logical_qdata_shape(qdata, pack_axis) == expected


@pytest.mark.parametrize("qdata_shape", [(32,), (4, 32), (2, 4, 32)])
def test_logical_qdata_shape_fp4_requires_a_pack_axis(qdata_shape):
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    with pytest.raises(ValueError, match="pack_axis should be set"):
        logical_qdata_shape(qdata, None)


@pytest.mark.parametrize(
    "qdata_shape, pack_axis",
    [
        ((4, 32), 0),
        ((2, 4, 32), 0),
        ((2, 4, 32), 1),
    ],
)
def test_logical_qdata_shape_rejects_non_unit_stride_pack_axis(qdata_shape, pack_axis):
    """Doubling a dim only describes packing if that dim is the unit-stride one."""
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    with pytest.raises(ValueError, match="unit-stride pack_axis is expected"):
        logical_qdata_shape(qdata, pack_axis)


@pytest.mark.parametrize(
    "qdata_shape, pack_axis",
    [
        ((32,), 1),
        ((4, 32), 2),
        ((2, 4, 32), 3),
    ],
)
def test_logical_qdata_shape_rejects_out_of_range_pack_axis(qdata_shape, pack_axis):
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    with pytest.raises(IndexError):
        logical_qdata_shape(qdata, pack_axis)


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_logical_qdata_shape_ignores_pack_axis_for_fp8(shape):
    """FP8 stores one value per element, so a set pack_axis only has to be valid."""
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)

    assert logical_qdata_shape(qdata, -1) == shape
    with pytest.raises(ValueError, match="unit-stride pack_axis is expected"):
        logical_qdata_shape(qdata, 0)


@pytest.mark.parametrize(
    "qdata_shape, expected",
    [
        ((32,), 0),
        ((4, 32), 1),
        ((2, 4, 32), 2),
    ],
)
def test_resolve_pack_axis_finds_the_unit_stride_dim(qdata_shape, expected):
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    assert resolve_pack_axis(qdata) == expected


@pytest.mark.parametrize("qdata_shape", [(4, 32), (2, 4, 32)])
def test_resolve_pack_axis_honors_candidate_axes(qdata_shape):
    qdata = torch.randint(0, 256, qdata_shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    assert resolve_pack_axis(qdata, -1) == qdata.ndim - 1
    with pytest.raises(ValueError, match="expected one of them to be 1"):
        resolve_pack_axis(qdata, 0)


@pytest.mark.parametrize("shape", [(64,), (4, 64), (2, 4, 64)])
def test_resolve_pack_axis_requires_fp4(shape):
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)

    with pytest.raises(AssertionError, match="must be"):
        resolve_pack_axis(qdata)
