# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize


def _as_uint8(t):
    # FP8 tensors can't be compared element-wise with torch.equal on NPU;
    # compare exact bytes instead. Each value is at most 1 byte, so a uint8
    # view reinterprets the storage without a copy. contiguous() makes the
    # view legal on permuted (non-contiguous) outputs.
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("shape", [(1024, 2048), (4, 1024, 2048)])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_block_mx_quantize_without_mxfp4_matches_direct_npu_quant(shape, elem_dtype):
    """No-mxfp4 branch forwards the config to npu_dynamic_block_mx_quant unchanged."""
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(elem_dtype=elem_dtype)

    b_q, b_s1, b_s2 = block_mx_quantize(tensor, axis=-2, config=config)

    ref_q, ref_s1, ref_s2 = torch_npu.npu_dynamic_block_mx_quant(
        tensor,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    assert b_q.dtype == elem_dtype
    assert b_q.shape == tensor.shape
    assert torch.equal(b_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(b_s1.view(torch.uint8), ref_s1.view(torch.uint8))
    assert torch.equal(b_s2.view(torch.uint8), ref_s2.view(torch.uint8))


@pytest.mark.parametrize("shape", [(1024, 2048), (4, 1024, 2048)])
def test_block_mx_quantize_mxfp4_last_axis_matches_manual_fusion(shape):
    """axis=-1 mxfp4 branch equals composing npu_dynamic_mx_quant with the fused kernel."""
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    fq_cfg = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    config = BlockMXQuantizeConfig(mxfp4_fake_quantize_config=fq_cfg)

    b_q, b_s1, b_s2 = block_mx_quantize(tensor, axis=-1, config=config)

    fp4_tensor, mxscale = torch_npu.npu_dynamic_mx_quant(
        tensor,
        axis=-1,
        dst_type=fq_cfg.npu_elem_dtype,
        block_size=fq_cfg.block_size,
        round_mode=fq_cfg.round_mode,
        scale_alg=fq_cfg.scale_alg,
        dst_type_max=fq_cfg.dst_type_max,
    )
    ref_q, ref_s1, ref_s2 = torch.ops.cann_ops_nn.mx_to_block_mx_quant(
        fp4_tensor,
        mxscale,
        dst_type=config.npu_elem_dtype,
        x_type=fq_cfg.npu_elem_dtype,
    )

    assert b_q.dtype == config.elem_dtype
    assert b_q.shape == tensor.shape
    assert torch.equal(b_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(b_s1.view(torch.uint8), ref_s1.view(torch.uint8))
    assert torch.equal(b_s2.view(torch.uint8), ref_s2.view(torch.uint8))


@pytest.mark.parametrize("shape", [(1024, 2048), (4, 1024, 2048)])
def test_block_mx_quantize_mxfp4_second_last_axis_is_transposed_last_axis(shape):
    """axis=-2 mxfp4 branch equals the axis=-1 result of the transposed input,
    transposed back with the two scale tensors' roles swapped.
    """
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    b_q, b_s1, b_s2 = block_mx_quantize(tensor, axis=-2, config=config)
    q_p, s1_p, s2_p = block_mx_quantize(tensor.transpose(-2, -1), axis=-1, config=config)

    assert b_q.shape == tensor.shape
    assert torch.equal(b_q.view(torch.uint8), q_p.transpose(-2, -1).contiguous().view(torch.uint8))
    assert torch.equal(b_s1.view(torch.uint8), s2_p.transpose(-3, -2).contiguous().view(torch.uint8))
    assert torch.equal(b_s2.view(torch.uint8), s1_p.transpose(-3, -2).contiguous().view(torch.uint8))


@pytest.mark.parametrize("axis", [0, -3, 3])
def test_block_mx_quantize_rejects_unsupported_axis_with_mxfp4(axis):
    """axis must point to one of the last two dims; for ndim=3 that is -2, -1, 1 or 2."""
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    with pytest.raises(AssertionError, match="axis"):
        block_mx_quantize(torch.randn(4, 1024, 2048, device="npu", dtype=torch.bfloat16), axis=axis, config=config)


# =========================================================================
# Direct FP4 tests. Decided policy: no byte-equality assertions against the
# raw op for FP4 -- shape/dtype contracts here; values are pinned only by
# the matmul-output equivalence test below (and the ops-level SQNR tests).
# =========================================================================


@pytest.mark.parametrize("shape", [(1024, 2048), (4, 1024, 2048), (5, 3, 64, 128)])
def test_block_mx_quantize_direct_fp4_output_contract(shape):
    """Direct FP4: q is fp4-typed with the last dim halved (two values per
    byte); scale shapes are dtype-independent, so they must equal an FP8
    config's. The 4D case additionally covers the leading-dim flattening.
    """
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    q, s1, s2 = block_mx_quantize(tensor, axis=-2, config=config)
    _, s1_ref, s2_ref = block_mx_quantize(tensor, axis=-2, config=BlockMXQuantizeConfig())

    assert q.dtype == torch.float4_e2m1fn_x2
    assert q.shape == (*tensor.shape[:-1], tensor.shape[-1] // 2)
    assert s1.shape == s1_ref.shape
    assert s2.shape == s2_ref.shape


def test_block_mx_quantize_direct_fp4_transposed_view_packs_dim_minus_2():
    """Direct FP4 of a transposed view returns a transposed-packed q -- dim -2
    halved instead of dim -1 (the documented exception). Scale shapes keep the
    raw-op layout regardless.
    """
    tensor = torch.randn(64, 256, device="npu", dtype=torch.bfloat16).t()  # [256, 64] view
    config = BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    q, s1, s2 = block_mx_quantize(tensor, axis=-2, config=config)
    _, s1_ref, s2_ref = block_mx_quantize(tensor.contiguous(), axis=-2, config=config)

    assert q.dtype == torch.float4_e2m1fn_x2
    assert q.shape == (tensor.shape[-2] // 2, tensor.shape[-1])
    assert s1.shape == s1_ref.shape
    assert s2.shape == s2_ref.shape


# =========================================================================
# Layout tests: block_mx_quantize avoids real transposes when possible.
# These are value-equivalence checks: they verify correctness, not that a
# real transpose was avoided (that would require profiling).
# =========================================================================


@pytest.mark.parametrize(
    "tensor",
    [
        # 1. Dense (is_contiguous() True) -> identity permutation, no swap.
        torch.randn(256, 128, device="npu", dtype=torch.bfloat16),
        # 2. 2D transpose -> the two quant dims land reversed, so swap happens.
        torch.randn(128, 256, device="npu", dtype=torch.bfloat16).t(),
        # 3. 3D batched transpose (the grouped/bmm weight layout) -> swap with
        #    a leading batch dim present.
        torch.randn(3, 128, 64, device="npu", dtype=torch.bfloat16).transpose(-1, -2),
        # 4. Non-pure strided view (row gap; strides don't match a permutation)
        #    -> no free permutation.
        torch.as_strided(torch.randn(64, 160, device="npu", dtype=torch.bfloat16), (64, 128), (160, 1)),
        # 5. Pure permutation view that crosses the leading/trailing-two
        #    boundary: restoring density would move the leading dim between the
        #    two quant dims, so the permuted tensor stays non-dense (inevitable
        #    copy), but outputs must still match.
        torch.randn(64, 3, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2),
    ],
    ids=["dense", "transposed", "transposed_3d", "non_pure", "cross_boundary"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_block_mx_quantize_without_mxfp4_matches_raw_op_for_all_layouts(tensor, elem_dtype):
    """Direct path matches npu_dynamic_block_mx_quant on all input layouts."""
    config = BlockMXQuantizeConfig(elem_dtype=elem_dtype)

    q, s1, s2 = block_mx_quantize(tensor, axis=-2, config=config)
    ref_q, ref_s1, ref_s2 = torch_npu.npu_dynamic_block_mx_quant(
        tensor,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    assert q.shape == ref_q.shape, f"q shape mismatch: {q.shape} vs {ref_q.shape}"
    assert q.dtype == ref_q.dtype, f"q dtype mismatch: {q.dtype} vs {ref_q.dtype}"
    assert s1.shape == ref_s1.shape, f"s1 shape mismatch: {s1.shape} vs {ref_s1.shape}"
    assert s2.shape == ref_s2.shape, f"s2 shape mismatch: {s2.shape} vs {ref_s2.shape}"
    assert torch.equal(_as_uint8(q), _as_uint8(ref_q)), "q values mismatch"
    assert torch.equal(_as_uint8(s1), _as_uint8(ref_s1)), "s1 values mismatch"
    assert torch.equal(_as_uint8(s2), _as_uint8(ref_s2)), "s2 values mismatch"


@pytest.mark.parametrize(
    "tensor",
    [
        # Same layouts as the direct-path test above, plus a 4D pure
        # permutation view exercising the leading-dim flattening.
        torch.randn(256, 128, device="npu", dtype=torch.bfloat16),
        torch.randn(128, 256, device="npu", dtype=torch.bfloat16).t(),
        torch.randn(3, 128, 64, device="npu", dtype=torch.bfloat16).transpose(-1, -2),
        torch.randn(5, 3, 64, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2, 3),
        torch.as_strided(torch.randn(64, 160, device="npu", dtype=torch.bfloat16), (64, 128), (160, 1)),
        torch.randn(64, 3, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2),
    ],
    ids=["dense", "transposed", "transposed_3d", "perm_view_4d", "non_pure", "cross_boundary"],
)
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_block_mx_quantize_mxfp4_matches_contiguous_for_all_layouts(tensor, axis, elem_dtype):
    """mxfp4 path: quantizing a view equals quantizing its contiguous copy.

    Layout must not affect values. Combined with the dense-input fusion tests
    above, this pins the permute-back correctness of the mxfp4 path.
    """
    config = BlockMXQuantizeConfig(
        elem_dtype=elem_dtype,
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    q, s1, s2 = block_mx_quantize(tensor, axis=axis, config=config)
    ref_q, ref_s1, ref_s2 = block_mx_quantize(tensor.contiguous(), axis=axis, config=config)

    assert q.shape == ref_q.shape, f"q shape mismatch: {q.shape} vs {ref_q.shape}"
    assert s1.shape == ref_s1.shape, f"s1 shape mismatch: {s1.shape} vs {ref_s1.shape}"
    assert s2.shape == ref_s2.shape, f"s2 shape mismatch: {s2.shape} vs {ref_s2.shape}"
    assert torch.equal(_as_uint8(q), _as_uint8(ref_q)), "q values mismatch"
    assert torch.equal(_as_uint8(s1), _as_uint8(ref_s1)), "s1 values mismatch"
    assert torch.equal(_as_uint8(s2), _as_uint8(ref_s2)), "s2 values mismatch"


@pytest.mark.parametrize("layout", ["dense", "perm_view"])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_block_mx_quantize_without_mxfp4_flattens_leading_dims(layout, elem_dtype):
    """ndim > 3 equals applying the raw op to every trailing 2D slice.

    Per-trailing-slice independence is the defining semantics of block MX
    quantization at any rank, so looping the (2D-capable) raw op over the
    leading indices constructs the 4D ground truth without flattening or
    permuting -- no index bookkeeping is shared with the implementation.
    """
    tensor = torch.randn(5, 3, 64, 128, device="npu", dtype=torch.bfloat16)
    if layout == "perm_view":
        tensor = tensor.permute(1, 0, 2, 3)

    config = BlockMXQuantizeConfig(elem_dtype=elem_dtype)
    q, s1, s2 = block_mx_quantize(tensor, axis=-2, config=config)

    assert q.shape == tensor.shape
    assert s1.shape[:2] == tensor.shape[:2]
    assert s2.shape[:2] == tensor.shape[:2]
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            ref_q, ref_s1, ref_s2 = torch_npu.npu_dynamic_block_mx_quant(
                tensor[i, j].contiguous(),
                dst_type=config.npu_elem_dtype,
                scale_alg=config.scale_alg,
                dst_type_max=config.dst_type_max,
            )
            assert torch.equal(_as_uint8(q[i, j]), _as_uint8(ref_q)), f"q mismatch at slice [{i}, {j}]"
            assert torch.equal(_as_uint8(s1[i, j]), _as_uint8(ref_s1)), f"s1 mismatch at slice [{i}, {j}]"
            assert torch.equal(_as_uint8(s2[i, j]), _as_uint8(ref_s2)), f"s2 mismatch at slice [{i}, {j}]"


@pytest.mark.parametrize("axis", [-1, -2])
def test_block_mx_quantize_mxfp4_flattens_leading_dims(axis):
    """mxfp4 path, ndim > 3: equals the 2D path on every trailing slice.

    The reference is the wrapper itself on each contiguous 2D slice -- that is
    the dense 2D path, which the fusion tests above pin against the manually
    composed kernels, so it serves as the trusted base.
    """
    tensor = torch.randn(5, 3, 64, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2, 3)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    q, s1, s2 = block_mx_quantize(tensor, axis=axis, config=config)

    assert q.shape == tensor.shape
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            ref_q, ref_s1, ref_s2 = block_mx_quantize(tensor[i, j].contiguous(), axis=axis, config=config)
            assert torch.equal(_as_uint8(q[i, j]), _as_uint8(ref_q)), f"q mismatch at slice [{i}, {j}]"
            assert torch.equal(_as_uint8(s1[i, j]), _as_uint8(ref_s1)), f"s1 mismatch at slice [{i}, {j}]"
            assert torch.equal(_as_uint8(s2[i, j]), _as_uint8(ref_s2)), f"s2 mismatch at slice [{i}, {j}]"


@pytest.mark.parametrize("pos_axis, neg_axis", [(0, -2), (1, -1)])
def test_block_mx_quantize_mxfp4_positive_axis_matches_negative(pos_axis, neg_axis):
    """Positive axis spellings (ndim-2, ndim-1) behave exactly like -2 / -1."""
    tensor = torch.randn(128, 256, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    out_pos = block_mx_quantize(tensor, axis=pos_axis, config=config)
    out_neg = block_mx_quantize(tensor, axis=neg_axis, config=config)
    for a, b in zip(out_pos, out_neg, strict=True):
        assert a.shape == b.shape
        assert torch.equal(_as_uint8(a), _as_uint8(b))


@pytest.mark.parametrize("case", ["dense", "transposed", "non_pure", "cross_boundary"])
@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
        (
            MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
            BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
        (
            MXQuantizeConfig(),
            BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
    ],
    ids=["mxfp8", "mxfp4", "mxfp4-qat"],
)
def test_block_mx_quantize_quant_matmul_matches_contiguous(case, config_a, config_b):
    """npu_quant_matmul output is identical whether b was quantized from a view
    or from its contiguous copy.

    Replicates _BlockMXQuantBMM.forward's calling convention; the critical
    coverage is the matmul consuming the wrapper's permuted-view q/scales. For
    mxfp4 this is the value pin: in the transposed case the two b_q operands
    are packed along different dims (transposed-packed vs the raw layout), and
    equal outputs prove both layouts encode the same quantized values. The
    mxfp4 case pairs FP4 activations with the FP4 weights -- the quant matmul
    kernel does not support FP8 x FP4.
    """
    b, m, k, n = 2, 64, 128, 256

    if case == "dense":
        right = torch.randn(b, k, n, device="npu", dtype=torch.bfloat16)
    elif case == "transposed":
        right = torch.randn(b, n, k, device="npu", dtype=torch.bfloat16).transpose(-1, -2)
    elif case == "non_pure":
        base = torch.randn(b, k, n + 32, device="npu", dtype=torch.bfloat16)
        right = torch.as_strided(base, (b, k, n), (k * (n + 32), n + 32, 1))
    elif case == "cross_boundary":
        # Pure permutation view crossing the leading/trailing-two boundary.
        right = torch.randn(k, b, n, device="npu", dtype=torch.bfloat16).permute(1, 0, 2)
    else:
        raise ValueError(f"unknown case {case}")

    left = torch.randn(b, m, k, device="npu", dtype=torch.bfloat16)
    a_q1, a_s1, _, _ = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        left,
        round_mode=config_a.round_mode,
        dst_type=config_a.npu_elem_dtype,
        scale_alg=config_a.scale_alg,
        dst_type_max=config_a.dst_type_max,
    )

    def matmul(b_q, b_s2):
        # Replicates _BlockMXQuantBMM.forward's matmul call.
        return torch_npu.npu_quant_matmul(
            a_q1,
            b_q,
            b_s2,
            pertoken_scale=a_s1,
            output_dtype=left.dtype,
            scale_dtype=config_b.npu_scale_dtype,
            pertoken_scale_dtype=config_a.npu_scale_dtype,
            x1_dtype=config_a.npu_matmul_dtype,
            x2_dtype=config_b.npu_matmul_dtype,
            group_sizes=[1, 1, config_b.block_size],
        )

    b_q, _, b_s2 = block_mx_quantize(right, axis=-2, config=config_b)
    b_q_r, _, b_s2_r = block_mx_quantize(right.contiguous(), axis=-2, config=config_b)

    y = matmul(b_q, b_s2)
    y_ref = matmul(b_q_r, b_s2_r)

    assert y.shape == y_ref.shape
    assert torch.equal(y, y_ref), "npu_quant_matmul result differs between view-quantized and contiguous-quantized b"
