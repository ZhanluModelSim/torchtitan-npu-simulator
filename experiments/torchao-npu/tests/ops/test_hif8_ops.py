# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU tests for the per-tensor HiF8 MM and Grouped MM kernels."""

import pytest
import torch
import torch_npu
from torchao.float8.float8_utils import compute_error
from torchao_npu.ops.hif8_ops import (
    to_hif8_then_bmm,
    to_hif8_then_grouped_mm,
    to_hif8_then_mm,
)
from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig

_SQNR_THRESHOLD = 12.0


def _group_list(group_sizes: list[int]) -> torch.Tensor:
    return torch.tensor(group_sizes, dtype=torch.int32, device="npu").cumsum(0)


def _sqnr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return compute_error(reference.float(), actual.float()).item()


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(torch.bfloat16, (128, 64, 256), id="bf16"),
        pytest.param(torch.float16, (64, 128, 128), id="fp16"),
    ],
)
def test_mm_forward_preserves_shape_and_dtype(dtype, shape):
    M, K, N = shape
    config = HiF8QuantizeConfig()
    activation = torch.randn(M, K, device="npu", dtype=dtype)
    weight = torch.randn(N, K, device="npu", dtype=dtype)

    output = to_hif8_then_mm(activation, weight.T, config, config)

    assert output.shape == (M, N)
    assert output.dtype == dtype
    assert output.device.type == "npu"


def test_mm_forward_and_backward_match_independent_dense_reference():
    torch.manual_seed(42)
    config = HiF8QuantizeConfig()
    activation = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(256, 64, device="npu", dtype=torch.bfloat16)

    activation_ref = activation.detach().clone().requires_grad_()
    weight_ref = weight.detach().T.clone().requires_grad_()
    reference = activation_ref @ weight_ref
    reference.sum().backward()

    activation_hif8 = activation.detach().clone().requires_grad_()
    weight_hif8 = weight.detach().T.clone().requires_grad_()
    actual = to_hif8_then_mm(activation_hif8, weight_hif8, config, config)
    actual.sum().backward()

    assert _sqnr(reference, actual) > _SQNR_THRESHOLD
    assert _sqnr(activation_ref.grad, activation_hif8.grad) > _SQNR_THRESHOLD
    assert _sqnr(weight_ref.grad, weight_hif8.grad) > _SQNR_THRESHOLD


def test_mm_quantizes_transposed_weight_view_without_extra_copy():
    """``quantize_hifloat8``'s permute-avoids-copy path must be bit-identical.

    ``weight.t()`` (no ``.contiguous()``) is exactly the layout a plain
    ``nn.Linear`` weight hits once transposed to ``[K, N]`` for the matmul.
    Per-tensor quantization is elementwise under one scalar scale, so
    quantizing the transposed view (dense base permuted, codes permuted back)
    must give the same bytes as quantizing an explicitly materialized
    contiguous copy of the same values.
    """
    torch.manual_seed(5)
    config = HiF8QuantizeConfig()
    activation = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)  # nn.Linear layout [N, K]

    def _run(make_weight_KN):
        act = activation.detach().clone().requires_grad_()
        w = make_weight_KN().requires_grad_()
        out = to_hif8_then_mm(act, w, config, config)
        out.sum().backward()
        return out.detach(), act.grad, w.grad

    contiguous_result = _run(lambda: weight.detach().clone().t().contiguous())
    view_result = _run(lambda: weight.detach().clone().t())

    assert torch.equal(view_result[0], contiguous_result[0]), "forward is not bit-identical"
    assert torch.equal(view_result[1], contiguous_result[1]), "dA is not bit-identical"
    assert torch.equal(view_result[2], contiguous_result[2]), "dB is not bit-identical"


def test_mm_supports_leading_dimensions_and_casts_fp32():
    config = HiF8QuantizeConfig()
    activation = torch.randn(4, 32, 64, device="npu", dtype=torch.float32)
    weight = torch.randn(128, 64, device="npu", dtype=torch.float32)

    output = to_hif8_then_mm(activation, weight.T, config, config)

    assert output.shape == (4, 32, 128)
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(((32, 64), (64,)), id="one-dimensional-weight"),
        pytest.param(((32, 64), (128, 32)), id="contracting-dimension"),
    ],
)
def test_mm_rejects_invalid_shapes(shapes):
    config = HiF8QuantizeConfig()
    activation = torch.randn(*shapes[0], device="npu", dtype=torch.bfloat16)
    weight = torch.randn(*shapes[1], device="npu", dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_mm(activation, weight, config, config)


def test_grouped_mm_forward_and_backward_match_independent_reference():
    torch.manual_seed(42)
    config = HiF8QuantizeConfig()
    group_sizes = [32, 64, 32]
    group_list = _group_list(group_sizes)
    activation = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(3, 64, 128, device="npu", dtype=torch.bfloat16)

    activation_ref = activation.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    reference_parts = []
    start = 0
    for expert, size in enumerate(group_sizes):
        end = start + size
        reference_parts.append(activation_ref[start:end] @ weight_ref[expert])
        start = end
    reference = torch.cat(reference_parts)
    reference.sum().backward()

    activation_hif8 = activation.detach().clone().requires_grad_()
    weight_hif8 = weight.detach().clone().requires_grad_()
    actual = to_hif8_then_grouped_mm(activation_hif8, weight_hif8, group_list, config, config)
    actual.sum().backward()

    assert actual.shape == (128, 128)
    assert _sqnr(reference, actual) > _SQNR_THRESHOLD
    assert _sqnr(activation_ref.grad, activation_hif8.grad) > _SQNR_THRESHOLD
    assert _sqnr(weight_ref.grad, weight_hif8.grad) > _SQNR_THRESHOLD


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(((4, 32, 64), (2, 64, 128)), id="three-dimensional-input"),
        pytest.param(((64, 64), (64, 128)), id="two-dimensional-weight"),
        pytest.param(((64, 64), (2, 128, 64)), id="contracting-dimension"),
    ],
)
def test_grouped_mm_rejects_invalid_shapes(shapes):
    config = HiF8QuantizeConfig()
    activation = torch.randn(*shapes[0], device="npu", dtype=torch.bfloat16)
    weight = torch.randn(*shapes[1], device="npu", dtype=torch.bfloat16)
    group_list = _group_list([32, 64])

    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_grouped_mm(activation, weight, group_list, config, config)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(torch.bfloat16, (8, 128, 64, 256), id="bf16"),
        pytest.param(torch.float16, (4, 64, 128, 128), id="fp16"),
    ],
)
def test_bmm_forward_preserves_shape_and_dtype(dtype, shape):
    batch, M, K, N = shape
    config = HiF8QuantizeConfig()
    activation = torch.randn(batch, M, K, device="npu", dtype=dtype)
    weight = torch.randn(batch, K, N, device="npu", dtype=dtype)

    output = to_hif8_then_bmm(activation, weight, config, config)

    assert output.shape == (batch, M, N)
    assert output.dtype == dtype
    assert output.device.type == "npu"


def test_bmm_forward_and_backward_match_independent_dense_reference():
    torch.manual_seed(42)
    config = HiF8QuantizeConfig()
    activation = torch.randn(6, 128, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(6, 64, 256, device="npu", dtype=torch.bfloat16)

    activation_ref = activation.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    reference = torch.bmm(activation_ref, weight_ref)
    reference.sum().backward()

    activation_hif8 = activation.detach().clone().requires_grad_()
    weight_hif8 = weight.detach().clone().requires_grad_()
    actual = to_hif8_then_bmm(activation_hif8, weight_hif8, config, config)
    actual.sum().backward()

    assert actual.shape == (6, 128, 256)
    assert _sqnr(reference, actual) > _SQNR_THRESHOLD
    assert _sqnr(activation_ref.grad, activation_hif8.grad) > _SQNR_THRESHOLD
    assert _sqnr(weight_ref.grad, weight_hif8.grad) > _SQNR_THRESHOLD


def test_bmm_quantizes_transposed_activation_view_without_extra_copy():
    """``quantize_hifloat8``'s permute-avoids-copy path must be
    bit-identical for BMM's activation operand.

    The operand is exactly what ``BatchedLinear.forward`` builds: a
    ``[T, H, K]`` contiguous tensor viewed as ``[H, T, K]``. Per-tensor
    quantization is elementwise under one scalar scale, so quantizing that
    transposed view (dense base permuted, codes permuted back) must give the
    same bytes as quantizing an explicitly materialized contiguous copy of
    the same values.
    """
    torch.manual_seed(7)
    T, H, K, N = 64, 4, 128, 96
    config = HiF8QuantizeConfig()
    x_THK = torch.randn(T, H, K, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(H, K, N, device="npu", dtype=torch.bfloat16)

    def _run(make_activation):
        activation = make_activation().requires_grad_()
        w = weight.detach().clone().requires_grad_()
        out = to_hif8_then_bmm(activation, w, config, config)
        out.sum().backward()
        return out.detach(), activation.grad, w.grad

    contiguous_result = _run(lambda: x_THK.detach().clone().transpose(0, 1).contiguous())
    view_result = _run(lambda: x_THK.detach().clone().transpose(0, 1))

    assert torch.equal(view_result[0], contiguous_result[0]), "forward is not bit-identical"
    assert torch.equal(view_result[1], contiguous_result[1]), "dA is not bit-identical"
    assert torch.equal(view_result[2], contiguous_result[2]), "dB is not bit-identical"


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(((32, 64), (32, 64, 128)), id="two-dimensional-input"),
        pytest.param(((2, 32, 64), (3, 64, 128)), id="batch-dimension"),
        pytest.param(((2, 32, 64), (2, 128, 64)), id="contracting-dimension"),
    ],
)
def test_bmm_rejects_invalid_shapes(shapes):
    config = HiF8QuantizeConfig()
    activation = torch.randn(*shapes[0], device="npu", dtype=torch.bfloat16)
    weight = torch.randn(*shapes[1], device="npu", dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        to_hif8_then_bmm(activation, weight, config, config)


@pytest.mark.parametrize("num_experts", [2, 3, 5], ids=lambda value: f"experts-{value}")
def test_grouped_mm_uses_weight_expert_count_for_scale_broadcast(num_experts):
    config = HiF8QuantizeConfig()
    num_tokens, K, N = 64, 32, 32
    base = num_tokens // num_experts
    group_sizes = [base] * (num_experts - 1) + [num_tokens - base * (num_experts - 1)]
    group_list = _group_list(group_sizes)
    activation = torch.randn(num_tokens, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(num_experts, K, N, device="npu", dtype=torch.bfloat16, requires_grad=True)

    output = to_hif8_then_grouped_mm(activation, weight, group_list, config, config)
    output.sum().backward()

    assert output.shape == (num_tokens, N)
    assert torch.isfinite(activation.grad.float()).all()
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    assert torch.isfinite(weight.grad.float()).all()


# =========================================================================
# Current/constant tensor scaling (CTS)
# =========================================================================


def _count_calls(monkeypatch, target, name):
    """Wrap ``torch_npu.<name>`` with a call counter; still calls the real op."""
    calls = {"n": 0}
    real = getattr(target, name)

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, spy)
    return calls


def test_quantize_always_recomputes_scale_via_dynamic_quant(monkeypatch):
    """Every call quantizes via npu_dynamic_quant; npu_quantize (static-scale
    quantize) is never used -- there is no persisted scale to reuse.
    """
    config = HiF8QuantizeConfig()
    activation = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)

    dynamic_calls = _count_calls(monkeypatch, torch_npu, "npu_dynamic_quant")
    static_calls = _count_calls(monkeypatch, torch_npu, "npu_quantize")

    for _ in range(3):
        to_hif8_then_mm(activation, weight.T, config, config)

    assert dynamic_calls["n"] == 3 * 2
    assert static_calls["n"] == 0
