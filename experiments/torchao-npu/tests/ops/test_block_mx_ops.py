# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchao.float8.float8_utils import compute_error
from torchao_npu.ops.block_mx_ops import (
    to_block_mx_then_bmm,
    to_block_mx_then_grouped_mm,
    to_block_mx_then_mm,
)
from torchao_npu.ops.mx_ops import (
    mxfp4_fake_quantize,
    mxfp8_dequantize,
)
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize

# (shape, config_a, config_b, sqnr_threshold) cases shared by the forward and
# gradient SQNR tests below.
sqnr_cases = [
    ((128, 64, 256), MXQuantizeConfig(), BlockMXQuantizeConfig(), 17.0),
    ((256, 128, 128), MXQuantizeConfig(), BlockMXQuantizeConfig(), 17.0),
    (
        (128, 64, 256),
        MXQuantizeConfig(),
        BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        12.0,
    ),
    (
        (128, 64, 256),
        MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        9.0,
    ),
]


@pytest.mark.parametrize(
    "m, k, n, config_a, config_b",
    [
        (128, 64, 256, MXQuantizeConfig(), BlockMXQuantizeConfig()),
        (256, 128, 128, MXQuantizeConfig(), BlockMXQuantizeConfig()),
        (64, 256, 64, MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_forward_shape_and_dtype(m, k, n, config_a, config_b):
    """Output shape and dtype match input expectations.

    b is passed as a transposed column-major tensor, simulating the common
    pattern where a linear weight is stored as [n, k] and ``.T`` is applied.
    """
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    # weight stored [n, k] (out_features, in_features), use .T to get [k, n]
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    b = weight.T  # [k, n] with column-major strides

    out = to_block_mx_then_mm(a, b, config_a, config_b)

    assert out.shape == (m, n), f"Expected ({m}, {n}), got {out.shape}"
    assert out.dtype == a.dtype, f"Expected {a.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.parametrize("shape, config_a, config_b, sqnr_threshold", sqnr_cases)
def test_sqnr_forward(shape, config_a, config_b, sqnr_threshold):
    """Block MX forward output has acceptable SQNR vs high-precision matmul."""
    m, k, n = shape
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    b = weight.T

    out_ref = a @ b
    out_fp8 = to_block_mx_then_mm(a, b, config_a, config_b)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"Forward SQNR too low: {sqnr:.2f} db"


@pytest.mark.parametrize("shape, config_a, config_b, sqnr_threshold", sqnr_cases)
def test_sqnr_gradients(shape, config_a, config_b, sqnr_threshold):
    """Block MX backward gradients have acceptable SQNR vs high-precision backward.

    Checks both da (grad wrt a) and db (grad wrt b).
    """
    m, k, n = shape
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    b = weight.T

    # --- Reference: high-precision forward + backward ---
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = a_ref @ b_ref
    out_ref.sum().backward()

    # --- Block MX forward + backward ---
    a_fp8 = a.clone().detach().requires_grad_(True)
    b_fp8 = b.clone().detach().requires_grad_(True)
    out_fp8 = to_block_mx_then_mm(a_fp8, b_fp8, config_a, config_b)
    out_fp8.sum().backward()

    # SQNR of da (grad wrt a)
    sqnr_da = compute_error(a_ref.grad.float(), a_fp8.grad.float()).item()
    assert sqnr_da > sqnr_threshold, f"da SQNR too low: {sqnr_da:.2f} db"

    # SQNR of db (grad wrt b)
    sqnr_db = compute_error(b_ref.grad.float(), b_fp8.grad.float()).item()
    assert sqnr_db > sqnr_threshold, f"db SQNR too low: {sqnr_db:.2f} db"


@pytest.mark.parametrize(
    "m, k, n, config_a, config_b",
    [
        (32, 64, 128, MXQuantizeConfig(), BlockMXQuantizeConfig()),
        (128, 32, 64, MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_backward_finiteness(m, k, n, config_a, config_b):
    """Gradients are finite (no NaN/Inf) and non-zero."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    b = weight.T

    out = to_block_mx_then_mm(a, b, config_a, config_b)
    out.sum().backward()

    for name, g in [("a", a.grad), ("weight", weight.grad)]:
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_non_2d_input(config_a, config_b):
    """3D + 2D inputs produce correct output; 1D b raises."""
    a_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = weight.T

    out = to_block_mx_then_mm(a_3d, b, config_a, config_b)
    assert out.shape == (4, 32, 128)

    a = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    b_1d = torch.randn(64, device="npu", dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        to_block_mx_then_mm(a, b_1d, config_a, config_b)


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_contracting_dim_mismatch(config_a, config_b):
    """Mismatched contracting dimensions raise an error."""
    a = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    # weight is [64, 256], so its transpose is [256, 64] and the contracting dims 64 and 256 differ
    weight = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    b = weight.T

    with pytest.raises((AssertionError, RuntimeError)):
        to_block_mx_then_mm(a, b, config_a, config_b)


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_no_requires_grad(config_a, config_b):
    """Gradient tracking is not required on either operand."""
    a = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = weight.T

    out = to_block_mx_then_mm(a, b, config_a, config_b)
    assert out.shape == (32, 128)
    assert not out.requires_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_dtype_preservation(dtype, config_a, config_b):
    """Output dtype matches a's dtype."""
    a = torch.randn(64, 128, device="npu", dtype=dtype)
    weight = torch.randn(32, 128, device="npu", dtype=dtype)
    b = weight.T
    out = to_block_mx_then_mm(a, b, config_a, config_b)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# --- helpers for grouped mm tests ---


def _group_list_from_sizes(group_sizes: list[int], device: str = "npu") -> torch.Tensor:
    """Build a cumsum group_list for the given per-group sizes.

    The API expects ``group_list`` to have one entry per group, containing
    cumulative token counts: ``[s0, s0+s1, ..., m]``.
    Use ``group_list_type=0`` (cumsum format).
    """
    gs = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return gs.cumsum(0)


# (shape, group_sizes, config_a, config_b, sqnr_threshold) cases shared by the
# grouped forward and gradient SQNR tests. The direct-FP4 row exists only for
# the forward test: npu_grouped_matmul does not support mxfp4 with group type
# 2, which the backward wgrad (db) call needs.
grouped_sqnr_cases = [
    (
        (192, 64, 128, 3),
        [64, 64, 64],
        MXQuantizeConfig(),
        BlockMXQuantizeConfig(),
        17.0,
    ),
    (
        (192, 64, 128, 3),
        [64, 64, 64],
        MXQuantizeConfig(),
        BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        12.0,
    ),
]
grouped_sqnr_forward_cases = [
    *grouped_sqnr_cases,
    (
        (192, 64, 128, 3),
        [64, 64, 64],
        MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        9.0,
    ),
]


@pytest.mark.parametrize(
    "shape, group_sizes, config_a, config_b",
    [
        ((128, 64, 128, 2), [64, 64], MXQuantizeConfig(), BlockMXQuantizeConfig()),
        ((192, 64, 64, 3), [64, 64, 64], MXQuantizeConfig(), BlockMXQuantizeConfig()),
        ((128, 64, 64, 4), [32, 32, 32, 32], MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_forward_shape_and_dtype(shape, group_sizes, config_a, config_b):
    """Output shape and dtype match expectations for grouped matmul."""
    m, k, n, num_experts = shape
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(num_experts, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    out = to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)

    assert out.shape == (m, n), f"Expected ({m}, {n}), got {out.shape}"
    assert out.dtype == a.dtype, f"Expected {a.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.parametrize(
    "shape, group_sizes, config_a, config_b, sqnr_threshold",
    grouped_sqnr_forward_cases,
)
def test_grouped_sqnr_forward(shape, group_sizes, config_a, config_b, sqnr_threshold):
    """Grouped block MX forward output has acceptable SQNR."""
    m, k, n, num_experts = shape
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(num_experts, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    # Reference: group-by-group high-precision matmul
    out_ref = []
    for i in range(num_experts):
        s = group_list[i - 1].item() if i > 0 else 0
        e = group_list[i].item()
        out_ref.append(a[s:e] @ b[i])
    out_ref = torch.cat(out_ref, dim=0)

    out_fp8 = to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"Forward SQNR too low: {sqnr:.2f} db"


@pytest.mark.parametrize(
    "shape, group_sizes, config_a, config_b, sqnr_threshold",
    grouped_sqnr_cases,
)
def test_grouped_sqnr_gradients(shape, group_sizes, config_a, config_b, sqnr_threshold):
    """Grouped block MX backward gradients have acceptable SQNR."""
    m, k, n, num_experts = shape
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(num_experts, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)

    # --- Reference ---
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = []
    for i in range(num_experts):
        s = group_list[i - 1].item() if i > 0 else 0
        e = group_list[i].item()
        out_ref.append(a_ref[s:e] @ b_ref[i])
    out_ref = torch.cat(out_ref, dim=0)
    out_ref.sum().backward()

    # --- Block MX ---
    a_fp8 = a.clone().detach().requires_grad_(True)
    b_fp8 = b.clone().detach().requires_grad_(True)
    out_fp8 = to_block_mx_then_grouped_mm(a_fp8, b_fp8, group_list, config_a, config_b)
    out_fp8.sum().backward()

    sqnr_da = compute_error(a_ref.grad.float(), a_fp8.grad.float()).item()
    assert sqnr_da > sqnr_threshold, f"da SQNR too low: {sqnr_da:.2f} db"

    sqnr_db = compute_error(b_ref.grad.float(), b_fp8.grad.float()).item()
    assert sqnr_db > sqnr_threshold, f"db SQNR too low: {sqnr_db:.2f} db"


@pytest.mark.parametrize(
    "shape, group_sizes, config_a, config_b",
    [
        ((128, 64, 128, 2), [64, 64], MXQuantizeConfig(), BlockMXQuantizeConfig()),
        ((192, 64, 128, 3), [64, 64, 64], MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_backward_finiteness(shape, group_sizes, config_a, config_b):
    """Grouped backward gradients are finite and non-zero."""
    m, k, n, num_experts = shape
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(num_experts, k, n, device="npu", dtype=torch.bfloat16, requires_grad=True)
    group_list = _group_list_from_sizes(group_sizes)

    out = to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)
    out.sum().backward()

    for name, g in [("a", a.grad), ("b", b.grad)]:
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_non_2d_input(config_a, config_b):
    """Non-2D / non-3D inputs raise an error in grouped matmul."""
    # a 3D
    a_3d = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_mx_then_grouped_mm(a_3d, b, group_list, config_a, config_b)

    # b 2D (must be 3D)
    a = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    b_2d = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_mx_then_grouped_mm(a, b_2d, group_list, config_a, config_b)


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_contracting_dim_mismatch(config_a, config_b):
    """Mismatched contracting dimensions raise an error in grouped matmul."""
    a = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    # b is [2, 128, 64] while a's last dim is 64, so the contracting dims differ
    b = torch.randn(2, 128, 64, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([32, 64], dtype=torch.int32, device="npu")
    with pytest.raises((AssertionError, RuntimeError)):
        to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)


@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_no_requires_grad(config_a, config_b):
    """Gradient tracking not required on grouped operands."""
    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")

    out = to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)
    assert out.shape == (128, 128)
    assert not out.requires_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "config_a, config_b",
    [
        (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_grouped_dtype_preservation(dtype, config_a, config_b):
    """Output dtype matches a's dtype in grouped matmul."""
    a = torch.randn(128, 128, device="npu", dtype=dtype)
    b = torch.randn(2, 128, 64, device="npu", dtype=dtype)
    group_list = torch.tensor([64, 128], dtype=torch.int32, device="npu")

    out = to_block_mx_then_grouped_mm(a, b, group_list, config_a, config_b)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# =========================================================================
# block_mx_quantize
# =========================================================================


@pytest.mark.parametrize(
    "k, n, axis, config_b",
    [
        (64, 128, -2, BlockMXQuantizeConfig()),
        (
            64,
            128,
            -2,
            BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
        (128, 64, -2, BlockMXQuantizeConfig()),
        (
            128,
            64,
            -2,
            BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
    ],
)
def test_block_mx_quantize_shape_and_dtype_2d(k, n, axis, config_b):
    """2D b: returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)

    b_q, _, _ = block_mx_quantize(b, axis=axis, config=config_b)

    assert b_q.dtype == config_b.elem_dtype
    assert b_q.shape == b.shape


@pytest.mark.parametrize(
    "e, k, n, axis, config_b",
    [
        (2, 64, 128, -2, BlockMXQuantizeConfig()),
        (
            2,
            64,
            128,
            -2,
            BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
        ),
        (3, 128, 64, -2, BlockMXQuantizeConfig()),
    ],
)
def test_block_mx_quantize_shape_and_dtype_3d(e, k, n, axis, config_b):
    """3D b (grouped): returns 3 tensors with expected dtype; output shape preserved."""
    torch.manual_seed(42)
    b = torch.randn(e, k, n, device="npu", dtype=torch.bfloat16)

    b_q, _, _ = block_mx_quantize(b, axis=axis, config=config_b)

    assert b_q.dtype == config_b.elem_dtype
    assert b_q.shape == b.shape


# =========================================================================
# to_block_mx_then_bmm (block MX batched matmul)
# =========================================================================


# (batch, m, k, n) shape and the (config_a, config_b) pairs shared by the bmm
# SQNR tests; the forward and gradient thresholds differ.
bmm_shape = (4, 2048, 4096, 2048)
bmm_config_pairs = [
    (MXQuantizeConfig(), BlockMXQuantizeConfig()),
    (
        MXQuantizeConfig(),
        BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)),
    ),
    (
        MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    ),
]
bmm_sqnr_forward_cases = [
    (bmm_shape, *config_pair, threshold)
    for config_pair, threshold in zip(bmm_config_pairs, (27.5, 17.5, 14.0), strict=True)
]
bmm_sqnr_gradients_cases = [
    (bmm_shape, *config_pair, threshold)
    for config_pair, threshold in zip(bmm_config_pairs, (30.5, 17.5, 14.5), strict=True)
]


@pytest.mark.parametrize(
    "shape, config_a, config_b",
    [
        ((4, 2048, 4096, 2048), MXQuantizeConfig(), BlockMXQuantizeConfig()),
        ((4, 4096, 2048, 4096), MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_bmm_forward_shape_and_dtype(shape, config_a, config_b):
    """Block MX batched matmul output shape and dtype match expectations."""
    batch, m, k, n = shape
    act = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(batch, n, k, device="npu", dtype=torch.bfloat16).transpose(-1, -2)  # [batch, k, n]

    out = to_block_mx_then_bmm(act, weight, config_a, config_b)

    assert out.shape == (batch, m, n), f"Expected ({batch}, {m}, {n}), got {out.shape}"
    assert out.dtype == act.dtype, f"Expected {act.dtype}, got {out.dtype}"
    assert out.device.type == "npu"


@pytest.mark.parametrize("shape, config_a, config_b, sqnr_threshold", bmm_sqnr_forward_cases)
def test_bmm_sqnr_forward(shape, config_a, config_b, sqnr_threshold):
    """Block MX batched matmul forward output has acceptable SQNR."""
    batch, m, k, n = shape
    torch.manual_seed(42)
    act = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(batch, n, k, device="npu", dtype=torch.bfloat16).transpose(-1, -2)  # [batch, k, n]

    out_ref = torch.bmm(act, weight)
    out_fp8 = to_block_mx_then_bmm(act, weight, config_a, config_b)

    sqnr = compute_error(out_ref.float(), out_fp8.float()).item()
    assert sqnr > sqnr_threshold, f"BMM forward SQNR too low: {sqnr:.2f} db"


@pytest.mark.parametrize("shape, config_a, config_b, sqnr_threshold", bmm_sqnr_gradients_cases)
def test_bmm_sqnr_gradients(shape, config_a, config_b, sqnr_threshold):
    """Block MX batched matmul backward gradients have acceptable SQNR.

    The weight is stored as ``[batch, n, k]`` and transposed for the bmm
    (matching model usage); gradients are checked on the stored leaf parameter.
    """
    batch, m, k, n = shape
    torch.manual_seed(42)
    act = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(batch, n, k, device="npu", dtype=torch.bfloat16, requires_grad=True)

    # --- Reference ---
    act_ref = act.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    torch.bmm(act_ref, weight_ref.transpose(-1, -2)).sum().backward()

    # --- Block MX ---
    act_fp8 = act.clone().detach().requires_grad_(True)
    weight_fp8 = weight.clone().detach().requires_grad_(True)
    to_block_mx_then_bmm(act_fp8, weight_fp8.transpose(-1, -2), config_a, config_b).sum().backward()

    sqnr_da = compute_error(act_ref.grad.float(), act_fp8.grad.float()).item()
    assert sqnr_da > sqnr_threshold, f"da SQNR too low: {sqnr_da:.2f} db"

    sqnr_db = compute_error(weight_ref.grad.float(), weight_fp8.grad.float()).item()
    assert sqnr_db > sqnr_threshold, f"db SQNR too low: {sqnr_db:.2f} db"


@pytest.mark.parametrize(
    "shape, config_a, config_b",
    [
        ((4, 1024, 2048, 1024), MXQuantizeConfig(), BlockMXQuantizeConfig()),
    ],
)
def test_bmm_backward_finiteness(shape, config_a, config_b):
    """Block MX batched matmul gradients are finite and non-zero."""
    batch, m, k, n = shape
    torch.manual_seed(42)
    act = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(batch, n, k, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight_b = weight.transpose(-1, -2)  # [batch, k, n]

    to_block_mx_then_bmm(act, weight_b, config_a, config_b).sum().backward()

    for name, g in [("act", act.grad), ("weight", weight.grad)]:
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has non-finite values"
        assert g.norm().item() > 0, f"{name}.grad is all zeros"


# =========================================================================
# old vs new quantization path helpers
# =========================================================================


def _quant_matmul(a_quant, b_quant, a, scale_dtypes):
    """npu_quant_matmul with the a-side per-token scale; reshaped for ndim > 2."""
    (a_q1, a_s1), (b_q, b_s2) = a_quant, b_quant
    y = torch_npu.npu_quant_matmul(
        a_q1,
        b_q,
        b_s2,
        pertoken_scale=a_s1,
        output_dtype=a.dtype,
        scale_dtype=scale_dtypes[0],
        pertoken_scale_dtype=scale_dtypes[1],
        group_sizes=[1, 1, 32],
    )
    if a.ndim != 2:
        y = y.reshape(*a.shape[:-1], *y.shape[1:])
    return y


def _quant_a_grouped(a, config_a):
    """a-side dynamic MX quant used by the grouped equivalence tests."""
    return torch_npu.npu_dynamic_mx_quant(
        a,
        axis=-1,
        round_mode=config_a.round_mode,
        dst_type=config_a.npu_elem_dtype,
        block_size=config_a.block_size,
        scale_alg=config_a.scale_alg,
        dst_type_max=config_a.dst_type_max,
    )


def _grouped_quant_matmul(a_quant, b_quant, a, scale_dtypes, group_list):
    """group_type-0 npu_grouped_matmul with the a-side per-token scale."""
    (a_q1, a_s1), (b_q, b_s2) = a_quant, b_quant
    return torch_npu.npu_grouped_matmul(
        [a_q1],
        [b_q],
        scale=[b_s2],
        per_token_scale=[a_s1],
        group_list=group_list.to(torch.int64),
        group_type=0,
        output_dtype=a.dtype,
        group_list_type=0,
        scale_dtype=scale_dtypes[0],
        per_token_scale_dtype=scale_dtypes[1],
        split_item=3,
    )[0]


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((64, 128), -2),
        ((64, 128), -1),
        ((2, 64, 128), -2),
        ((2, 64, 128), -1),
        ((4, 128, 256), -2),
        ((4, 128, 256), -1),
        ((1024, 2048), -2),
        ((1024, 2048), -1),
    ],
)
def test_mxfp4_fused_op_equivalence(shape, axis):
    """
    Old (mxfp4_fake_quantize + dynamic_block_mx_quant) and new
    (fused cann_ops_nn.mx_to_block_mx_quant) paths produce identical results.
    """

    torch.manual_seed(42)
    b = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    # --- Old path: two-step ---
    hp_tensor = mxfp4_fake_quantize(b, config.mxfp4_fake_quantize_config, axis=axis)
    b_q_old, b_s1_old, b_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp_tensor,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: fused op ---
    b_q_new, b_s1_new, b_s2_new = block_mx_quantize(b, axis=axis, config=config)

    # --- Compare: dequantize block MX to bf16, then torch.equal ---
    # Block MX scale is broadcast across each 32×32 block (MXFP8 format).
    # Dequant along k-dim using B_s2 (forward scale) or n-dim using b_s1 (backward scale).
    block_size = 32
    for axis, s_old, s_new, label in [(-2, b_s2_old, b_s2_new, "s2"), (-1, b_s1_old, b_s1_new, "s1")]:
        dq_old = mxfp8_dequantize(
            b_q_old, s_old, axis=axis, block_size=block_size, output_shape=b.shape, output_dtype=b.dtype
        )
        dq_new = mxfp8_dequantize(
            b_q_new, s_new, axis=axis, block_size=block_size, output_shape=b.shape, output_dtype=b.dtype
        )
        assert torch.equal(dq_old, dq_new), f"Dequantized values differ with {label}"


@pytest.mark.parametrize(
    "m, k, n, axis",
    [
        (128, 64, 256, -2),
        (64, 128, 64, -1),
    ],
)
def test_mxfp4_matmul_equivalence(m, k, n, axis):
    """Matmul output using old vs new quantization paths has acceptable SQNR."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_a = MXQuantizeConfig()

    # Quantize a once (shared)
    a_q1, a_s1, _, _ = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        a.reshape(-1, a.shape[-1]),
        round_mode=config_a.round_mode,
        dst_type=config_a.npu_elem_dtype,
        scale_alg=config_a.scale_alg,
        dst_type_max=config_a.dst_type_max,
    )

    # --- Old path: quantize b ---
    hp = mxfp4_fake_quantize(b, config.mxfp4_fake_quantize_config, axis=axis)
    b_q_old, _, b_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    scale_dtypes = (config.npu_scale_dtype, config_a.npu_scale_dtype)
    y_old = _quant_matmul((a_q1, a_s1), (b_q_old, b_s2_old), a, scale_dtypes)

    # --- New path: quantize b ---
    b_q_new, _, b_s2_new = block_mx_quantize(b, axis=axis, config=config)
    y_new = _quant_matmul((a_q1, a_s1), (b_q_new, b_s2_new), a, scale_dtypes)

    assert torch.equal(y_old, y_new), "Matmul results differ between old and new quantization paths"


@pytest.mark.parametrize(
    "m, k, n, e, group_sizes",
    [
        (192, 64, 128, 3, [64, 64, 64]),
    ],
)
def test_mxfp4_grouped_matmul_equivalence(m, k, n, e, group_sizes):
    """Grouped matmul output using old vs new quantization paths is identical."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(e, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_a = MXQuantizeConfig()

    # Quantize a once (shared)
    a_q1, a_s1 = _quant_a_grouped(a, config_a)

    # --- Old path: quantize b ---
    hp = mxfp4_fake_quantize(b, config.mxfp4_fake_quantize_config, axis=-2)
    b_q_old, _, b_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    scale_dtypes = (config.npu_scale_dtype, config_a.npu_scale_dtype)
    y_old = _grouped_quant_matmul((a_q1, a_s1), (b_q_old, b_s2_old), a, scale_dtypes, group_list)

    # --- New path: quantize b ---
    b_q_new, _, b_s2_new = block_mx_quantize(b, axis=-2, config=config)
    y_new = _grouped_quant_matmul((a_q1, a_s1), (b_q_new, b_s2_new), a, scale_dtypes, group_list)

    assert torch.equal(y_old, y_new), "Grouped matmul results differ between old and new quantization paths"


@pytest.mark.parametrize(
    "m, k, n, axis",
    [
        (128, 64, 256, -2),
        (64, 128, 64, -1),
    ],
)
def test_mxfp4_matmul_backward_equivalence(m, k, n, axis):
    """Backward gradients using old vs new quantization paths are identical."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_a = MXQuantizeConfig()

    a_flat = a.reshape(-1, a.shape[-1])

    # Quantize a once (shared)
    a_q1, a_s1, a_q2, a_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        a_flat,
        round_mode=config_a.round_mode,
        dst_type=config_a.npu_elem_dtype,
        scale_alg=config_a.scale_alg,
        dst_type_max=config_a.dst_type_max,
    )

    # --- Old path: quantize b ---
    hp = mxfp4_fake_quantize(b, config.mxfp4_fake_quantize_config, axis=axis)
    b_q_old, b_s1_old, b_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: quantize b ---
    b_q_new, b_s1_new, b_s2_new = block_mx_quantize(b, axis=axis, config=config)

    def _backward(dy, a_q2, a_s2, b_q, b_s1):
        dy_q1, dy_s1, dy_q2, dy_s2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            dy.reshape(-1, dy.shape[-1]),
            round_mode=config_a.round_mode,
            dst_type=config_a.npu_elem_dtype,
            scale_alg=config_a.scale_alg,
            dst_type_max=config_a.dst_type_max,
        )
        da = torch_npu.npu_quant_matmul(
            dy_q1,
            b_q.t(),
            b_s1.transpose(0, 1),
            pertoken_scale=dy_s1,
            output_dtype=a.dtype,
            scale_dtype=config.npu_scale_dtype,
            pertoken_scale_dtype=config_a.npu_scale_dtype,
            group_sizes=[1, 1, 32],
        )
        db = torch_npu.npu_quant_matmul(
            a_q2.t(),
            dy_q2,
            dy_s2,
            pertoken_scale=a_s2.transpose(0, 1),
            output_dtype=a.dtype,
            # wgrad: both operands are a-side quantized (dy_s2, a_s2)
            scale_dtype=config_a.npu_scale_dtype,
            pertoken_scale_dtype=config_a.npu_scale_dtype,
            group_sizes=[1, 1, 32],
        )
        if dy.ndim != 2:
            da = da.reshape(*dy.shape[:-1], *da.shape[1:])
        return da, db

    # Forward matmul + backward for both paths
    scale_dtypes = (config.npu_scale_dtype, config_a.npu_scale_dtype)
    y_old = _quant_matmul((a_q1, a_s1), (b_q_old, b_s2_old), a, scale_dtypes)
    dy = torch.randn_like(y_old)
    da_old, db_old = _backward(dy, a_q2, a_s2, b_q_old, b_s1_old)

    y_new = _quant_matmul((a_q1, a_s1), (b_q_new, b_s2_new), a, scale_dtypes)
    assert torch.equal(y_old, y_new), "Forward results differ between old and new quantization paths"
    da_new, db_new = _backward(dy, a_q2, a_s2, b_q_new, b_s1_new)

    assert torch.equal(da_old, da_new), "da differs between old and new quantization paths"
    assert torch.equal(db_old, db_new), "db differs between old and new quantization paths"


@pytest.mark.parametrize(
    "m, k, n, e, group_sizes",
    [
        (192, 64, 128, 3, [64, 64, 64]),
    ],
)
def test_mxfp4_grouped_matmul_backward_equivalence(m, k, n, e, group_sizes):
    """Grouped backward gradients using old vs new quantization paths are identical."""
    torch.manual_seed(42)
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(e, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _group_list_from_sizes(group_sizes)
    config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    config_a = MXQuantizeConfig()

    # Quantize a once (shared)
    a_q1, a_s1 = _quant_a_grouped(a, config_a)
    a_q2, a_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
        a,
        group_list.to(torch.int32),
        round_mode=config_a.round_mode,
        dst_type=config_a.npu_elem_dtype,
        blocksize=config_a.block_size,
        scale_alg=config_a.scale_alg,
    )

    # --- Old path: quantize b ---
    hp = mxfp4_fake_quantize(b, config.mxfp4_fake_quantize_config, axis=-2)
    b_q_old, b_s1_old, b_s2_old = torch_npu.npu_dynamic_block_mx_quant(
        hp,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    # --- New path: quantize b ---
    b_q_new, b_s1_new, b_s2_new = block_mx_quantize(b, axis=-2, config=config)

    def _grouped_backward(dy, a_q2, a_s2, b_q, b_s1):
        dy_q1, dy_s1 = torch_npu.npu_dynamic_mx_quant(
            dy,
            axis=-1,
            round_mode=config_a.round_mode,
            dst_type=config_a.npu_elem_dtype,
            block_size=config_a.block_size,
            scale_alg=config_a.scale_alg,
            dst_type_max=config_a.dst_type_max,
        )
        dy_q2, dy_s2 = torch_npu.npu_grouped_dynamic_mx_quant(
            dy,
            group_list.to(torch.int32),
            round_mode=config_a.round_mode,
            dst_type=config_a.npu_elem_dtype,
            blocksize=config_a.block_size,
            scale_alg=config_a.scale_alg,
        )
        da = torch_npu.npu_grouped_matmul(
            [dy_q1],
            [b_q.transpose(-1, -2)],
            scale=[b_s1.transpose(1, 2)],
            per_token_scale=[dy_s1],
            group_list=group_list.to(torch.int64),
            group_type=0,
            output_dtype=a.dtype,
            group_list_type=0,
            scale_dtype=config.npu_scale_dtype,
            per_token_scale_dtype=config_a.npu_scale_dtype,
            split_item=3,
        )[0]
        db = torch_npu.npu_grouped_matmul(
            [a_q2.t()],
            [dy_q2],
            scale=[dy_s2],
            per_token_scale=[a_s2.transpose(0, 1)],
            group_list=group_list.to(torch.int64),
            group_type=2,
            output_dtype=a.dtype,
            group_list_type=0,
            # wgrad: both operands are a-side quantized (dy_s2, a_s2)
            scale_dtype=config_a.npu_scale_dtype,
            per_token_scale_dtype=config_a.npu_scale_dtype,
            split_item=3,
        )[0]
        return da, db

    # Forward + backward for both paths
    scale_dtypes = (config.npu_scale_dtype, config_a.npu_scale_dtype)
    y_old = _grouped_quant_matmul((a_q1, a_s1), (b_q_old, b_s2_old), a, scale_dtypes, group_list)
    dy = torch.randn_like(y_old)
    da_old, db_old = _grouped_backward(dy, a_q2, a_s2, b_q_old, b_s1_old)

    y_new = _grouped_quant_matmul((a_q1, a_s1), (b_q_new, b_s2_new), a, scale_dtypes, group_list)
    assert torch.equal(y_old, y_new), "Forward results differ between old and new quantization paths"
    da_new, db_new = _grouped_backward(dy, a_q2, a_s2, b_q_new, b_s1_new)

    assert torch.equal(da_old, da_new), "da differs between old and new quantization paths"
    assert torch.equal(db_old, db_new), "db differs between old and new quantization paths"
