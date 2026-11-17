# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the HiF8 parameter wrapper and its operation dispatch."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torchao.float8.float8_utils import compute_error
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase
from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig
from torchao_npu.wrapper_tensors import HiF8TrainingWeightWrapperTensor

from ..testing_utils import target_devices

_SQNR_THRESHOLD = 12.0


def _config() -> HiF8QuantizeConfig:
    return HiF8QuantizeConfig()


def _wrapper(shape, device="cpu", *, dtype=torch.float32):
    config = _config()
    return HiF8TrainingWeightWrapperTensor(
        torch.randn(*shape, device=device, dtype=dtype),
        weight_config=config,
        activation_config=config,
    )


@pytest.mark.parametrize("device", target_devices, ids=lambda value: str(value))
def test_init_stores_data_and_configs(device):
    weight_config = _config()
    activation_config = _config()
    data = torch.randn(64, 128, device=device)

    wrapper = HiF8TrainingWeightWrapperTensor(
        data,
        weight_config=weight_config,
        activation_config=activation_config,
    )

    assert getattr(wrapper, "_data") is data  # noqa: B009
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is activation_config


def test_init_rejects_missing_or_wrong_config():
    data = torch.randn(64, 128)
    config = _config()

    with pytest.raises(ValueError, match=r"^`weight_config` is required"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=None, activation_config=config)
    with pytest.raises(ValueError, match=r"^`activation_config` is required"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=config, activation_config=None)

    class DummyConfig(FakeQuantizeConfigBase):
        pass

    with pytest.raises(ValueError, match=r"^Only `HiF8QuantizeConfig` is supported"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=DummyConfig(), activation_config=config)
    with pytest.raises(ValueError, match=r"^Only `HiF8QuantizeConfig` is supported"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=config, activation_config=DummyConfig())


def test_init_rejects_non_pertensor_quant_mode():
    """HiF8QuantizeConfig also serves Lightning Indexer Q/K quantization,
    where quant_mode may be non-"pertensor" -- this wrapper's per-expert
    scale broadcast and backward-reuse-via-transpose scheme only hold for a
    single scalar scale, so a non-"pertensor" config must be rejected here
    rather than silently mishandled.
    """
    data = torch.randn(64, 128)
    pertoken_config = HiF8QuantizeConfig(quant_mode="pertoken")
    pertensor_config = _config()

    with pytest.raises(ValueError, match=r"weight_config.quant_mode.*must be 'pertensor'"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=pertoken_config, activation_config=pertoken_config)
    with pytest.raises(ValueError, match=r"activation_config.quant_mode.*must be 'pertensor'"):
        HiF8TrainingWeightWrapperTensor(data, weight_config=pertensor_config, activation_config=pertoken_config)


@pytest.mark.parametrize("device", target_devices, ids=lambda value: str(value))
def test_non_quantized_operation_preserves_plain_tensor_result(device):
    plain = torch.randn(64, 128, device=device)
    wrapped = _wrapper((64, 128), device)

    actual = torch.add(plain, wrapped)
    expected = torch.add(plain, getattr(wrapped, "_data"))  # noqa: B009

    assert isinstance(actual, torch.Tensor)
    assert not isinstance(actual, HiF8TrainingWeightWrapperTensor)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("operation", ["select", "transpose", "detach", "clone"])
def test_metadata_operations_preserve_wrapper(operation):
    weight = _wrapper((4, 64, 128))

    with torch._C.DisableTorchFunctionSubclass():
        if operation == "select":
            actual = weight[0]
            expected = getattr(weight, "_data")[0]  # noqa: B009
        elif operation == "transpose":
            actual = weight.transpose(0, 1)
            expected = getattr(weight, "_data").transpose(0, 1)  # noqa: B009
        elif operation == "detach":
            actual = weight.detach()
            expected = getattr(weight, "_data").detach()  # noqa: B009
        else:
            actual = weight.clone()
            expected = getattr(weight, "_data").clone()  # noqa: B009

    assert isinstance(actual, HiF8TrainingWeightWrapperTensor)
    assert actual.weight_config is weight.weight_config
    assert actual.activation_config is weight.activation_config
    torch.testing.assert_close(getattr(actual, "_data"), expected)  # noqa: B009


@pytest.mark.parametrize("operation", [torch.mm, F.linear])
def test_dispatch_rejects_wrapped_activation(operation):
    weight = _wrapper((64, 128))

    with pytest.raises(AssertionError, match=r"^A should not be a HiF8TrainingWeightWrapperTensor$"):
        operation(weight, weight)


@pytest.mark.parametrize("operation", [torch.addmm, F.linear])
def test_dispatch_rejects_unwrapped_weight(operation):
    activation = torch.randn(16, 64)
    weight = torch.randn(64, 128)
    wrapped_bias = _wrapper((128,))

    with pytest.raises(AssertionError, match=r"^B should be a HiF8TrainingWeightWrapperTensor$"):
        if operation is torch.addmm:
            operation(wrapped_bias, activation, weight)
        else:
            operation(activation, weight, wrapped_bias)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda activation, weight, bias: torch.mm(activation, weight.T), id="mm"),
        pytest.param(lambda activation, weight, bias: torch.matmul(activation, weight.T), id="matmul"),
        pytest.param(lambda activation, weight, bias: F.linear(activation, weight, bias), id="linear"),
        pytest.param(lambda activation, weight, bias: torch.addmm(bias, activation, weight.T), id="addmm"),
        pytest.param(
            lambda activation, weight, bias: torch.addmm(bias, activation, weight.T, beta=2.0, alpha=0.5),
            id="addmm-scaled",
        ),
    ],
)
def test_mm_dispatch_forward_backward_matches_dense_reference(operation):
    config = _config()
    activation_data = torch.randn(32, 1024, device="npu", dtype=torch.bfloat16)
    weight_data = torch.randn(2048, 1024, device="npu", dtype=torch.bfloat16)
    bias_data = torch.randn(2048, device="npu", dtype=torch.bfloat16)

    activation = torch.nn.Parameter(activation_data.clone())
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(
            weight_data.clone(),
            weight_config=config,
            activation_config=config,
        )
    )
    bias = torch.nn.Parameter(bias_data.clone())
    actual = operation(activation, weight, bias)
    actual.sum().backward()

    activation_ref = torch.nn.Parameter(activation_data.clone())
    weight_ref = torch.nn.Parameter(weight_data.clone())
    bias_ref = torch.nn.Parameter(bias_data.clone())
    expected = operation(activation_ref, weight_ref, bias_ref)
    expected.sum().backward()

    assert actual.shape == expected.shape
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD


def test_bmm_dispatch_forward_backward_matches_dense_reference():
    config = _config()
    activation_data = torch.randn(8, 32, 1024, device="npu", dtype=torch.bfloat16)
    weight_data = torch.randn(8, 1024, 2048, device="npu", dtype=torch.bfloat16)

    activation = torch.nn.Parameter(activation_data.clone())
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(
            weight_data.clone(),
            weight_config=config,
            activation_config=config,
        )
    )
    actual = torch.bmm(activation, weight)
    actual.sum().backward()

    activation_ref = torch.nn.Parameter(activation_data.clone())
    weight_ref = torch.nn.Parameter(weight_data.clone())
    expected = torch.bmm(activation_ref, weight_ref)
    expected.sum().backward()

    assert actual.shape == expected.shape
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD


def test_grouped_mm_dispatch_matches_dense_reference():
    config = _config()
    num_tokens, num_experts, K, N = 16, 4, 1024, 2048
    activation = torch.randn(num_tokens, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight_data = torch.randn(num_experts, N, K, device="npu", dtype=torch.bfloat16)
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(weight_data, weight_config=config, activation_config=config)
    )
    offsets = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device="npu")

    actual = torch._grouped_mm(activation, weight.transpose(-2, -1), offs=offsets)
    actual.sum().backward()

    activation_ref = activation.detach().clone().requires_grad_()
    weight_ref = weight_data.detach().clone().requires_grad_()
    expected = torch._grouped_mm(activation_ref, weight_ref.transpose(-2, -1), offs=offsets)
    expected.sum().backward()

    assert actual.shape == (num_tokens, N)
    assert compute_error(actual, expected) > _SQNR_THRESHOLD
    assert compute_error(activation.grad, activation_ref.grad) > _SQNR_THRESHOLD
    assert compute_error(weight.grad, weight_ref.grad) > _SQNR_THRESHOLD


@pytest.mark.parametrize("device", target_devices, ids=lambda value: str(value))
def test_deepcopy_preserves_wrapper_contract(device):
    wrapper = _wrapper((64, 128), device, dtype=torch.bfloat16)
    copied = copy.deepcopy(wrapper)

    assert isinstance(copied, HiF8TrainingWeightWrapperTensor)
    assert getattr(copied, "_data") is not getattr(wrapper, "_data")  # noqa: B009
    assert copied.weight_config == wrapper.weight_config
    assert copied.activation_config == wrapper.activation_config
    torch.testing.assert_close(getattr(copied, "_data"), getattr(wrapper, "_data"))  # noqa: B009
