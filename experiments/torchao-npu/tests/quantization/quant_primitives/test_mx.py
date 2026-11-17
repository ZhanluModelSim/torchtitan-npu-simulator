# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.library
import torch_npu
from torchao_npu.ops.mx_ops import mxfp8_dequantize
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import (
    _get_fp4_e2m1_pair_lut,
    mx_dequantize,
    mx_fake_quantize,
    mx_last_dim_fake_quantize,
    mx_quantize,
    mx_quantize_dual_axis,
    mxfp4_dequantize,
)


def test_fp4_pair_lut_decodes_both_nibble_orders():
    low_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, True)
    high_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, False)

    assert low_first.shape == (256, 2)
    assert torch.equal(low_first[0x21], torch.tensor([0.5, 1.0]))
    assert torch.equal(high_first[0x21], torch.tensor([1.0, 0.5]))


def test_mxfp4_dequantize_unpacks_values_and_e8m0_scale():
    data = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scale = torch.full((1, 1, 2), 127, dtype=torch.uint8)

    result = mxfp4_dequantize(
        data,
        scale,
        axis=-1,
        block_size=32,
        output_shape=torch.Size((1, 32)),
        output_dtype=torch.float32,
    )

    expected = torch.tensor([0.5, 1.0] * 16).reshape(1, 32)
    assert torch.equal(result, expected)


def test_mxfp4_dequantize_rejects_partial_blocks():
    with pytest.raises(AssertionError, match="divisible by block_size"):
        mxfp4_dequantize(
            torch.zeros(1, 15, dtype=torch.uint8),
            torch.zeros(1, 1, 2, dtype=torch.uint8),
            axis=-1,
            block_size=32,
            output_shape=torch.Size((1, 30)),
            output_dtype=torch.float32,
        )


# =========================================================================
# Tests for mx_quantize (permutation-aware wrapper over npu_dynamic_mx_quant)
# =========================================================================


@pytest.mark.parametrize(
    "layout",
    ["dense", "strided_axis", "perm_view", "non_pure", "cat_gate", "cat_up", "single_row"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mx_quantize_matches_raw_op_for_all_layouts(layout, elem_dtype):
    """Compare wrapper values/scales with the raw NPU op, including cat gradients.

    These are value-equivalence checks: they verify correctness, not that a real
    transpose was avoided (that would require profiling). Each tested input
    layout must produce identical y/scale to the raw op.
    """

    # Allocate within the test so RNG/device state is managed by the fixture.
    axis = -1
    if layout == "dense":
        tensor = torch.randn(256, 128, device="npu", dtype=torch.bfloat16)
    elif layout == "strided_axis":
        tensor = torch.randn(128, 256, device="npu", dtype=torch.bfloat16).T
    elif layout == "perm_view":
        tensor = torch.randn(4, 128, 256, device="npu", dtype=torch.bfloat16).transpose(1, 2)
        axis = 1
    elif layout == "non_pure":
        tensor = torch.randn(4, 256, 129, device="npu", dtype=torch.bfloat16)[..., :128]
    else:
        rows = 1 if layout == "single_row" else 3
        packed = torch.randn(rows, 256, device="npu", dtype=torch.bfloat16)
        tensor = packed[:, 128:] if layout == "cat_up" else packed[:, :128]
        assert tensor.stride() == (256, 1)

    def as_uint8(t):
        # FP8 tensors can't be compared element-wise with torch.equal on NPU;
        # compare exact bytes instead. Each value is at most 1 byte (FP8, or FP4
        # packed two per byte), so a uint8 view reinterprets the storage without
        # a copy. contiguous() makes the view legal on permuted (non-contiguous)
        # outputs.
        return t.contiguous().view(torch.uint8)

    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    y, scale = mx_quantize(tensor, axis, config)
    y_ref, scale_ref = torch_npu.npu_dynamic_mx_quant(
        tensor,
        axis=axis,
        dst_type=config.npu_elem_dtype,
        block_size=config.block_size,
        round_mode=config.round_mode,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    assert y.shape == y_ref.shape, f"y shape mismatch: {y.shape} vs {y_ref.shape}"
    assert y.dtype == y_ref.dtype, f"y dtype mismatch: {y.dtype} vs {y_ref.dtype}"
    assert torch.equal(as_uint8(y), as_uint8(y_ref)), "y values mismatch"
    assert scale.shape == scale_ref.shape, f"scale shape mismatch: {scale.shape} vs {scale_ref.shape}"
    assert torch.equal(as_uint8(scale), as_uint8(scale_ref)), "scale values mismatch"


@pytest.mark.parametrize("which", ["left", "right"])
@pytest.mark.parametrize("case_b", ["dense", "strided_axis", "perm_view", "non_pure"])
@pytest.mark.parametrize("case_a", ["dense", "strided_axis", "perm_view", "non_pure"])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_quantize_quant_matmul_matches_raw_op(elem_dtype, case_a, case_b, which):
    """npu_quant_matmul output should be identical whether an operand is quantized by
    mx_quantize or by the raw op.
    """

    def matmul_operand(case, batch, rows, cols):
        """Build a 3D ``(batch, rows, cols)`` bf16 batched matmul operand in one
        of four layouts:

        - ``dense``: contiguous.
        - ``strided_axis`` / ``perm_view``: a 2-way transposed view (non-contiguous).
        - ``non_pure``: an as_strided view with a row gap (non-contiguous, in-bounds).
        """
        if case == "strided_axis" or case == "perm_view":
            return torch.randn(batch, cols, rows, device="npu", dtype=torch.bfloat16).transpose(-1, -2)

        elif case == "non_pure":
            base = torch.randn(batch, rows, cols + 1, device="npu", dtype=torch.bfloat16)
            return torch.as_strided(base, (batch, rows, cols), (rows * (cols + 1), cols + 1, 1))

        elif case == "dense":
            return torch.randn(batch, rows, cols, device="npu", dtype=torch.bfloat16)

        else:
            raise ValueError(f"`case` only support {['dense', 'strided_axis', 'perm_view', 'non_pure']}")

    b, m, k, n = 2, 64, 128, 256
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    left = matmul_operand(case_a, b, m, k)
    right = matmul_operand(case_b, b, k, n)

    def quant_operand(t, axis, use_mx):
        if use_mx:
            return mx_quantize(t, axis, config)

        else:
            return torch_npu.npu_dynamic_mx_quant(
                t,
                axis=axis,
                dst_type=config.npu_elem_dtype,
                block_size=config.block_size,
                round_mode=config.round_mode,
                scale_alg=config.scale_alg,
                dst_type_max=config.dst_type_max,
            )

    def matmul(a_q, a_s, b_q, b_s):
        return torch_npu.npu_quant_matmul(
            a_q,
            b_q,
            b_s,
            pertoken_scale=a_s,
            output_dtype=left.dtype,
            group_sizes=[1, 1, config.block_size],
            scale_dtype=config.npu_scale_dtype,
            pertoken_scale_dtype=config.npu_scale_dtype,
            x1_dtype=config.npu_matmul_dtype,
            x2_dtype=config.npu_matmul_dtype,
        )

    # mx_quantize on the selected operand; the other operand via the raw op.
    a_q, a_s = quant_operand(left, -1, use_mx=(which == "left"))
    b_q, b_s = quant_operand(right, -2, use_mx=(which == "right"))
    # Both-raw reference.
    a_q_r, a_s_r = quant_operand(left, -1, use_mx=False)
    b_q_r, b_s_r = quant_operand(right, -2, use_mx=False)

    y = matmul(a_q, a_s, b_q, b_s)
    y_ref = matmul(a_q_r, a_s_r, b_q_r, b_s_r)

    assert y.shape == y_ref.shape
    assert torch.equal(y, y_ref), "npu_quant_matmul result differs between mx_quantize and raw op"


# =========================================================================
# Tests for mx_quantize_dual_axis (permutation-aware wrapper over
# npu_dynamic_mx_quant_with_dual_axis)
# =========================================================================


@pytest.mark.parametrize(
    "tensor",
    [
        # 1. Dense (is_contiguous() True) -> identity permutation, no swap.
        torch.randn(256, 128, device="npu", dtype=torch.bfloat16),
        # 2. 2D transpose -> the two quant dims land reversed, so swap happens.
        torch.randn(128, 256, device="npu", dtype=torch.bfloat16).t(),
        # 3. Pure permutation view with reordered leading dims (the two quant
        #    dims keep their order) -> leading-only permutation, no swap.
        torch.randn(5, 3, 64, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2, 3),
        # 4. Non-pure strided view (strides don't match a permutation) -> no free
        #    permutation.
        torch.as_strided(torch.randn(4, 4, device="npu", dtype=torch.bfloat16), (2, 4), (2, 1)),
        # 5. Pure permutation view that crosses the leading/trailing-two
        #    boundary: restoring density would move the leading dim between the
        #    two quant dims, so the permuted tensor stays non-dense (inevitable
        #    copy), but outputs must still match.
        torch.randn(64, 3, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2),
    ],
    ids=["dense", "transposed", "perm_view", "non_pure", "cross_boundary"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mx_quantize_dual_axis_matches_raw_op_for_all_layouts(tensor, elem_dtype):
    """mx_quantize_dual_axis matches npu_dynamic_mx_quant_with_dual_axis on all layouts.

    These are value-equivalence checks: they verify correctness, not that a real
    transpose was avoided (that would require profiling). The point is each of the
    five branch inputs must produce identical y1/s1/y2/s2 to the raw op.
    """

    def as_uint8(t):
        # FP8 tensors can't be compared element-wise with torch.equal on NPU;
        # compare exact bytes instead. Each value is at most 1 byte, so a uint8
        # view reinterprets the storage without a copy. contiguous() makes the
        # view legal on permuted (non-contiguous) outputs.
        return t.contiguous().view(torch.uint8)

    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    y1, s1, y2, s2 = mx_quantize_dual_axis(tensor, config)
    r1, rs1, r2, rs2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        tensor,
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    assert y1.shape == r1.shape, f"y1 shape mismatch: {y1.shape} vs {r1.shape}"
    assert y2.shape == r2.shape, f"y2 shape mismatch: {y2.shape} vs {r2.shape}"
    assert s1.shape == rs1.shape, f"s1 shape mismatch: {s1.shape} vs {rs1.shape}"
    assert s2.shape == rs2.shape, f"s2 shape mismatch: {s2.shape} vs {rs2.shape}"
    assert torch.equal(as_uint8(y1), as_uint8(r1)), "y1 values mismatch"
    assert torch.equal(as_uint8(y2), as_uint8(r2)), "y2 values mismatch"
    assert torch.equal(as_uint8(s1), as_uint8(rs1)), "s1 values mismatch"
    assert torch.equal(as_uint8(s2), as_uint8(rs2)), "s2 values mismatch"


@pytest.mark.parametrize("which", ["left", "right"])
@pytest.mark.parametrize("case_b", ["dense", "transposed", "perm_view", "non_pure"])
@pytest.mark.parametrize("case_a", ["dense", "transposed", "perm_view", "non_pure"])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_quantize_dual_axis_quant_matmul_matches_raw_op(elem_dtype, case_a, case_b, which):
    """npu_quant_matmul output should be identical whether an operand is dual-axis
    quantized by mx_quantize_dual_axis or by the raw op.
    """

    def matmul_operand(case, batch, rows, cols):
        """Build a 3D ``(batch, rows, cols)`` bf16 operand in one of four layouts."""
        if case == "transposed" or case == "perm_view":
            return torch.randn(batch, cols, rows, device="npu", dtype=torch.bfloat16).transpose(-1, -2)
        elif case == "non_pure":
            base = torch.randn(batch, rows, cols + 1, device="npu", dtype=torch.bfloat16)
            return torch.as_strided(base, (batch, rows, cols), (rows * (cols + 1), cols + 1, 1))
        elif case == "dense":
            return torch.randn(batch, rows, cols, device="npu", dtype=torch.bfloat16)
        else:
            raise ValueError(f"`case` only support {['dense', 'transposed', 'perm_view', 'non_pure']}")

    b, m, k, n = 2, 64, 128, 256
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    left = matmul_operand(case_a, b, m, k)
    right = matmul_operand(case_b, b, k, n)

    def quant_operand(t, use_mx):
        if use_mx:
            return mx_quantize_dual_axis(t, config)
        else:
            return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                t,
                round_mode=config.round_mode,
                dst_type=config.npu_elem_dtype,
                scale_alg=config.scale_alg,
                dst_type_max=config.dst_type_max,
            )

    def matmul(a_q1, a_s1, b_q2, b_s2):
        # Replicates _MXQuantBMM.forward: x1 = A's k-dim quant (a_q1/a_s1),
        # x2 = b's k-dim quant (b_q2/b_s2).
        return torch_npu.npu_quant_matmul(
            a_q1,
            b_q2,
            b_s2,
            pertoken_scale=a_s1,
            output_dtype=left.dtype,
            group_sizes=[1, 1, config.block_size],
            scale_dtype=config.npu_scale_dtype,
            pertoken_scale_dtype=config.npu_scale_dtype,
            x1_dtype=config.npu_matmul_dtype,
            x2_dtype=config.npu_matmul_dtype,
        )

    # mx_quantize_dual_axis on the selected operand; the other operand via the raw op.
    a_q1, a_s1, _, _ = quant_operand(left, use_mx=(which == "left"))
    _, _, b_q2, b_s2 = quant_operand(right, use_mx=(which == "right"))
    # Both-raw reference.
    a_q1_r, a_s1_r, _, _ = quant_operand(left, use_mx=False)
    _, _, b_q2_r, b_s2_r = quant_operand(right, use_mx=False)

    y = matmul(a_q1, a_s1, b_q2, b_s2)
    y_ref = matmul(a_q1_r, a_s1_r, b_q2_r, b_s2_r)

    assert y.shape == y_ref.shape
    assert torch.equal(y, y_ref), "npu_quant_matmul result differs between mx_quantize_dual_axis and raw op"


# =========================================================================
# Tests for mx_fake_quantize (quantize then dequantize round trip)
# =========================================================================


@pytest.mark.parametrize(
    "tensor, axis",
    [
        # 1. Dense (is_contiguous() True) -> if branch, the ops see the tensor as-is.
        (torch.randn(256, 128, device="npu", dtype=torch.bfloat16), -1),
        # 2. Quant axis strided (stride(axis) != 1): a transposed 2D tensor, which
        #    no view can undo, so the ops see the tensor as-is.
        (torch.randn(128, 256, device="npu", dtype=torch.bfloat16).transpose(0, 1), 1),
        # 3. Pure perm view with innermost-contiguous quant axis (like wo_a): the
        #    permuting branch, where FP4 packs a different dim than the raw ops' call.
        (torch.randn(4, 128, 256, device="npu", dtype=torch.bfloat16).transpose(1, 2), 1),
        # 4. Non-pure strided view, stride(axis) == 1 but no permutation makes it
        #    dense, so the permuting branch's permutation is the identity and the
        #    op copies internally. The quant dim is a whole number of blocks, which
        #    the dequantize helper the reference uses requires.
        (torch.as_strided(torch.randn(2, 65, device="npu", dtype=torch.bfloat16), (2, 64), (65, 1)), 1),
    ],
    ids=["dense", "strided_axis", "perm_view", "non_pure"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_fake_quantize_matches_quantize_then_dequantize_for_all_layouts(tensor, axis, elem_dtype):
    """mx_fake_quantize reproduces quantize-then-dequantize on all four layouts.

    The reference quantizes with the raw ``npu_dynamic_mx_quant`` and dequantizes
    with ``mxfp8_dequantize``/``mxfp4_dequantize``, the helpers training's fake
    quantization has used so far (``MXFP4FakeQuantize.forward``). The raw
    ``npu_anti_mx_quant`` cannot serve as the reference: it only consumes quantized
    data whose quant axis is trailing, which the ``perm_view`` layout breaks.

    These are value-equivalence checks: they verify correctness, not that a real
    transpose was avoided (that would require profiling). The point is each of the
    four branch inputs must dequantize to what the raw ops produce for the original
    tensor. Values are compared rather than quantized bytes: for FP4 the two calls
    may pack different dims (two values per byte), which does not change what each
    value dequantizes to.

    The result is then fake quantized a second time: the dequantized tensor's layout
    need not match the original's (``non_pure`` comes back dense), so the second call
    may take the other branch, and it must still reproduce the values exactly.
    """

    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    y = mx_fake_quantize(tensor, axis, config)
    y_q, scale = torch_npu.npu_dynamic_mx_quant(
        tensor,
        axis=axis,
        dst_type=config.npu_elem_dtype,
        block_size=config.block_size,
        round_mode=config.round_mode,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    dequantize = mxfp4_dequantize if elem_dtype is torch.float4_e2m1fn_x2 else mxfp8_dequantize
    y_ref = dequantize(
        y_q,
        scale,
        axis=axis,
        block_size=config.block_size,
        output_shape=tensor.shape,
        output_dtype=tensor.dtype,
    )

    assert y.shape == tensor.shape, f"shape mismatch: {y.shape} vs {tensor.shape}"
    assert y.dtype == tensor.dtype, f"dtype mismatch: {y.dtype} vs {tensor.dtype}"
    assert torch.equal(y, y_ref), "fake quantized values differ from the reference dequantize"

    # Idempotency: quantize -> dequantize -> quantize -> dequantize is a no-op.
    y2 = mx_fake_quantize(y, axis, config)
    assert torch.equal(y, y2), "fake quantization is not idempotent"


@pytest.mark.parametrize("block_size", [32, 64, 96, 128])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_fake_quantize_is_idempotent(elem_dtype, block_size):
    """Fake quantizing an already fake-quantized tensor reproduces it exactly.

    Reference-free by design: ``mxfp8_dequantize``/``mxfp4_dequantize`` only
    implement ``block_size=32``, so above 32 the round trip is checked against
    itself. ``scale_alg=0`` is the only algorithm that supports both FP4 and FP8
    together with a ``block_size`` other than 32.
    """

    config = MXQuantizeConfig(elem_dtype=elem_dtype, block_size=block_size, scale_alg=0)
    x = torch.randn(4, 384, device="npu", dtype=torch.bfloat16)

    y = mx_fake_quantize(x, -1, config)
    y2 = mx_fake_quantize(y, -1, config)

    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert y.dtype == x.dtype, f"dtype mismatch: {y.dtype} vs {x.dtype}"
    assert not torch.equal(y, x), "the round trip must actually change values"
    assert torch.equal(y, y2), "fake quantization is not idempotent"


def test_mx_fake_quantize_is_straight_through_differentiable():
    """The fake-quantized span keeps identity gradients instead of detaching."""

    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = torch.randn(4, 64, device="npu", dtype=torch.bfloat16)
    x = torch.randn(4, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)

    y = mx_fake_quantize(x, -1, config)

    assert y.dtype == x.dtype
    assert not torch.equal(y, x), "the round trip must actually change values"

    (y * weight).sum().backward()

    # Identity through the quantized span: the gradient is the (unquantized) weight.
    assert x.grad is not None, "the fake-quantized span must not detach the graph"
    assert torch.equal(x.grad, weight)


def test_mx_last_dim_fake_quantize_custom_op_passes_opcheck():
    """Schema, mutation contract, autograd formula and fake metadata of the op."""

    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(4, 64, device="npu", dtype=torch.bfloat16)

    torch.library.opcheck(
        mx_last_dim_fake_quantize,
        (
            x,
            config.npu_elem_dtype,
            config.block_size,
            config.round_mode,
            config.scale_alg,
            config.dst_type_max,
        ),
    )


# =========================================================================
# Tests for mx_dequantize (permutation-aware wrapper over npu_anti_mx_quant)
# =========================================================================


@pytest.mark.parametrize(
    "tensor, axis",
    [
        # 1. Dense with a trailing quant axis: the permutation is the identity.
        (torch.randn(256, 128, device="npu", dtype=torch.bfloat16), -1),
        # 2. Pure permutation view whose quant axis is innermost-contiguous (like
        #    wo_a): permuting it to -1 restores dense storage, avoiding the copy.
        (torch.randn(4, 128, 256, device="npu", dtype=torch.bfloat16).transpose(1, 2), 1),
        # 3. The same, with the unit-stride quant axis leading instead of second.
        (torch.randn(4, 256, 128, device="npu", dtype=torch.bfloat16).permute(2, 0, 1), 0),
        # 4. Non-pure strided view: no permutation makes it dense, so the op inserts
        #    its own copy. The quant axis is the trailing one, the permutation the
        #    identity.
        (torch.as_strided(torch.randn(2, 129, device="npu", dtype=torch.bfloat16), (2, 128), (129, 1)), 1),
    ],
    ids=["dense", "perm_view", "perm_view_leading", "non_pure"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_mx_dequantize_matches_fake_quantize(tensor, axis, elem_dtype, block_size):
    """mx_dequantize undoes mx_quantize, reproducing mx_fake_quantize exactly.

    mx_fake_quantize quantizes the permuted tensor and dequantizes it in one op, so for a
    layout both functions permute the same way they hand the very same qdata/scale to the
    same dequantization and have to agree bit for bit. Every layout below has a unit-stride
    quant axis, which is what mx_dequantize requires.
    """

    # scale_alg=0 is the only algorithm that covers FP4 and FP8 together with a
    # block_size other than 32, i.e. the re-keying to 32-sized blocks.
    config = MXQuantizeConfig(elem_dtype=elem_dtype, block_size=block_size, scale_alg=0)

    qdata, scale = mx_quantize(tensor, axis, config)
    dequantized = mx_dequantize(qdata, scale, axis, block_size, elem_dtype, tensor.dtype)
    fake_quantized = mx_fake_quantize(tensor, axis, config)

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype == tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert torch.equal(dequantized, fake_quantized), "dequantized values differ from mx_fake_quantize"


def test_mx_dequantize_rejects_a_strided_quant_axis():
    """A quant axis that is not innermost-contiguous cannot be dequantized.

    A dense tensor quantized along a non-trailing axis takes mx_quantize's pass-through
    route, so the pair it returns keeps a strided quant axis: the layout npu_anti_mx_quant
    cannot consume, and one no permutation can repair.
    """

    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(4, 256, 128, device="npu", dtype=torch.bfloat16)

    qdata, scale = mx_quantize(tensor, 1, config)
    assert qdata.stride(1) != 1, "the pass-through route should keep the quant axis strided"

    with pytest.raises(AssertionError, match="unit stride"):
        mx_dequantize(qdata, scale, 1, config.block_size, config.elem_dtype, tensor.dtype)


def test_mx_dequantize_rejects_unsupported_dtypes():
    """The dtype checks run before any data is touched, so the tensors below are never consumed.

    Only uint8 ``qdata`` gets past them: ``src_dtype`` must name a format, and a typed ``qdata``
    must agree with it. ``output_dtype`` is checked last.
    """

    raw = torch.zeros(8, 32, device="npu", dtype=torch.uint8)
    typed = torch.zeros(8, 32, device="npu", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(8, 1, 2, device="npu", dtype=torch.uint8)

    with pytest.raises(ValueError, match="src_dtype"):
        mx_dequantize(raw, scale, -1, 32, torch.uint8, torch.bfloat16)

    with pytest.raises(AssertionError, match="must match"):
        mx_dequantize(typed, scale, -1, 32, torch.uint8, torch.bfloat16)

    with pytest.raises(AssertionError, match="output_dtype"):
        mx_dequantize(typed, scale, -1, 32, torch.float8_e4m3fn, torch.float64)


def test_mx_dequantize_honors_output_dtype():
    """Dequantizing to float32 and to bfloat16 must agree, bit for bit.

    A dequantized MX value is an FP8/FP4 mantissa times a power-of-two E8M0 scale, so it
    carries fewer significand bits than bfloat16 has: both output dtypes are exact, making
    their difference a pure dtype question.
    """

    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    qdata, scale = mx_quantize(tensor, -1, config)

    as_bfloat16 = mx_dequantize(qdata, scale, -1, config.block_size, config.elem_dtype, torch.bfloat16)
    as_float32 = mx_dequantize(qdata, scale, -1, config.block_size, config.elem_dtype, torch.float32)

    assert as_bfloat16.dtype == torch.bfloat16
    assert as_float32.dtype == torch.float32
    assert torch.equal(as_float32, as_bfloat16.to(torch.float32))
