# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HiF8TrainingWeightWrapperTensor.

Kept as a dedicated file (rather than folding entries into the shared
test_wrapper_tensor.py / test_wrapper_ops.py parametrize tables that cover
Base/Float8/MX/Block together) so this new backend's tests don't risk
touching those already-established, heavily-parametrized suites. Covers the
same behaviors those files check for the other backends: __init__
validation, subclass-preserving dispatch, deepcopy, FSDP hooks, and
end-to-end SQNR through an nn.Parameter + optimizer step.

SQNR thresholds are conservative placeholders -- see test_hif8_ops.py's
module docstring for why.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.float8.float8_utils import compute_error
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase

from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import HiF8QuantizeConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.wrapper_tensors import HiF8TrainingWeightWrapperTensor

from ..testing_utils import target_devices

_SQNR_THRESHOLD = 12.0


# =========================================================================
# __init__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
def test_init_stores_attrs(device):
    w = torch.randn(64, 128, device=device)
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()
    wrapper = HiF8TrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=act_config)
    assert wrapper._data is w
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
def test_init_rejects_missing_weight_config(device):
    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=r"^`weight_config` is required for HiF8TrainingWeightWrapperTensor\.$"):
        HiF8TrainingWeightWrapperTensor(w, weight_config=None, activation_config=HiF8QuantizeConfig())


@pytest.mark.parametrize("device", target_devices)
def test_init_rejects_missing_activation_config(device):
    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=r"^`activation_config` is required for HiF8TrainingWeightWrapperTensor\.$"):
        HiF8TrainingWeightWrapperTensor(w, weight_config=HiF8QuantizeConfig(), activation_config=None)


@pytest.mark.parametrize("device", target_devices)
def test_init_rejects_wrong_type_weight_config(device):
    class DummyConfig(FakeQuantizeConfigBase):
        pass

    w = torch.randn(64, 128, device=device)
    with pytest.raises(
        ValueError,
        match=r"^Only `HiF8QuantizeConfig` is supported for `weight_config` in HiF8TrainingWeightWrapperTensor\.$",
    ):
        HiF8TrainingWeightWrapperTensor(w, weight_config=DummyConfig(), activation_config=HiF8QuantizeConfig())


@pytest.mark.parametrize("device", target_devices)
def test_init_rejects_wrong_type_activation_config(device):
    class DummyConfig(FakeQuantizeConfigBase):
        pass

    w = torch.randn(64, 128, device=device)
    with pytest.raises(
        ValueError,
        match=(
            r"^Only `HiF8QuantizeConfig` is supported for `activation_config` "
            r"in HiF8TrainingWeightWrapperTensor\.$"
        ),
    ):
        HiF8TrainingWeightWrapperTensor(w, weight_config=HiF8QuantizeConfig(), activation_config=DummyConfig())


# =========================================================================
# __torch_function__ dispatch: A must not be wrapped, B must be wrapped
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("func", [torch.mm, F.linear])
def test_torch_function_arg_a_is_wrapper(func, device):
    w = torch.randn(64, 128, device=device)
    w1 = HiF8TrainingWeightWrapperTensor(w, weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig())
    w2 = HiF8TrainingWeightWrapperTensor(w, weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig())
    with pytest.raises(AssertionError, match=r"^A should not be a HiF8TrainingWeightWrapperTensor$"):
        func(w1, w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("func", [torch.addmm, F.linear])
def test_torch_function_arg_b_not_wrapper(func, device):
    bias = HiF8TrainingWeightWrapperTensor(
        torch.randn(128, device=device), weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig()
    )
    A = torch.randn(16, 64, device=device)
    B = torch.randn(64, 128, device=device)
    with pytest.raises(AssertionError, match=r"^B should be a HiF8TrainingWeightWrapperTensor$"):
        func(bias, A, B) if func is torch.addmm else func(A, B, bias)


@pytest.mark.parametrize("device", target_devices)
def test_torch_function_non_quantized_op_passes_through(device):
    """A non-matmul op (e.g. torch.add) is not intercepted for quantization."""
    w1 = torch.randn(64, 128, device=device)
    w2 = torch.randn(64, 128, device=device)
    wrapper = HiF8TrainingWeightWrapperTensor(
        w2.clone(), weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig()
    )
    result = torch.add(w1, wrapper)
    expected = torch.add(w1, w2)
    assert type(result) is torch.Tensor
    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", target_devices)
def test_bias_bypass(device):
    """Wrapped bias is unconditionally bypassed (not quantized)."""
    A = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    w_wrapped = HiF8TrainingWeightWrapperTensor(
        torch.randn(128, 64, dtype=torch.bfloat16, device=device),
        weight_config=HiF8QuantizeConfig(),
        activation_config=HiF8QuantizeConfig(),
    )
    bias = torch.randn(128, dtype=torch.bfloat16, device=device)
    bias_wrapped = HiF8TrainingWeightWrapperTensor(
        bias, weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig()
    )
    out_wrapped = F.linear(A, w_wrapped, bias_wrapped)
    out_ref = F.linear(A, w_wrapped, bias)
    assert torch.equal(out_wrapped, out_ref), "bias should not be quantized"


# =========================================================================
# Subclass-preserving dispatch (representative subset -- full op list is
# already covered generically for BaseTrainingWeightWrapperTensor in
# test_wrapper_ops.py::test_wrapper_preserves_subclass)
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "op_func",
    [
        lambda x: x[0],
        lambda x: x[0:2],
        lambda x: x.transpose(0, 1),
        lambda x: x.detach(),
        lambda x: x.clone(),
    ],
)
def test_preserves_subclass(op_func, device):
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()
    weight = torch.randn(4, 64, 128, device=device)
    wrapper = HiF8TrainingWeightWrapperTensor(weight, weight_config=weight_config, activation_config=act_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = op_func(wrapper)
        ref_result = op_func(wrapper._data)

    assert isinstance(result, HiF8TrainingWeightWrapperTensor)
    assert result.weight_config is weight_config
    assert result.activation_config is act_config
    assert torch.equal(result._data, ref_result)


# =========================================================================
# deepcopy
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
def test_deepcopy(device):
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)
    wrapper = HiF8TrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=act_config)

    wrapper_copy = copy.deepcopy(wrapper)

    assert wrapper_copy is not wrapper
    assert isinstance(wrapper_copy, HiF8TrainingWeightWrapperTensor)
    assert wrapper_copy._data is not wrapper._data
    assert torch.equal(wrapper_copy._data, wrapper._data)
    assert wrapper_copy.weight_config == wrapper.weight_config


# =========================================================================
# FSDP hooks (inherited unchanged from BaseTrainingWeightWrapperTensor --
# smoke-tested here for HiF8 specifically)
# =========================================================================


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
def test_fsdp_pre_all_gather(param_dtype):
    w = torch.randn(64, 128, dtype=torch.float32)
    wrapper = HiF8TrainingWeightWrapperTensor(
        w, weight_config=HiF8QuantizeConfig(), activation_config=HiF8QuantizeConfig()
    )
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype)

    all_gather_inputs, metadata = wrapper.fsdp_pre_all_gather(None, None, None, None, mp_policy)
    (data,) = all_gather_inputs
    assert metadata == ()
    assert data.dtype == param_dtype
    assert torch.equal(data, w.to(param_dtype))


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_post_all_gather_first_step(device):
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()
    w = torch.empty(64, 128, device="meta")
    wrapper = HiF8TrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=act_config)
    gathered = torch.empty(4, 64, 128, device=device)

    result, inner_tensors = wrapper.fsdp_post_all_gather((gathered,), None, torch.float32, out=None)
    assert type(result) is HiF8TrainingWeightWrapperTensor
    assert result._data is gathered
    assert isinstance(inner_tensors, tuple)
    assert len(inner_tensors) == 1
    assert inner_tensors[0] is gathered
    assert result.weight_config is weight_config
    assert result.activation_config is act_config


# =========================================================================
# End-to-end SQNR through nn.Parameter + optimizer step
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "call_fn, A_shape, w_shape, bias_shape, out_shape",
    [
        (lambda a, w, bias: torch.mm(a, w.T), (32, 1024), (2048, 1024), (), (32, 2048)),
        (lambda a, w, bias: F.linear(a, w), (32, 1024), (2048, 1024), (), (32, 2048)),
        (
            lambda a, w, bias: F.linear(a, w, bias),
            (32, 1024),
            (2048, 1024),
            (2048,),
            (32, 2048),
        ),
        (
            lambda a, w, bias: torch.matmul(a, w.T),
            (32, 1024),
            (2048, 1024),
            (),
            (32, 2048),
        ),
        (
            lambda a, w, bias: torch.addmm(bias, a, w.T),
            (32, 1024),
            (2048, 1024),
            (2048,),
            (32, 2048),
        ),
        (
            lambda a, w, bias: torch.addmm(bias, a, w.T, beta=2.0, alpha=0.5),
            (32, 1024),
            (2048, 1024),
            (2048,),
            (32, 2048),
        ),
    ],
)
def test_op_sqnr_and_training_step(call_fn, A_shape, w_shape, bias_shape, out_shape, device):
    """HiF8-quantized forward+backward+optimizer-step tracks a bf16 reference."""
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()

    activation_tensor = torch.randn(*A_shape, device=device, dtype=torch.bfloat16)
    weight_tensor = torch.randn(*w_shape, device=device, dtype=torch.bfloat16)

    activation = torch.nn.Parameter(activation_tensor.clone())
    weight = torch.nn.Parameter(
        HiF8TrainingWeightWrapperTensor(
            weight_tensor.clone(), activation_config=act_config, weight_config=weight_config
        )
    )
    bias = torch.nn.Parameter(torch.randn(bias_shape, device=device, dtype=torch.bfloat16))
    ref_bias = torch.nn.Parameter(bias.data.clone())

    ref_activation = torch.nn.Parameter(activation_tensor.clone())
    ref_weight = torch.nn.Parameter(weight_tensor.clone())

    learning_rate = 1
    opt_params = [weight, bias] if bias_shape else [weight]
    optimizer = torch.optim.SGD(opt_params, lr=learning_rate)
    out = call_fn(activation, weight, bias)

    ref_opt_params = [ref_weight, ref_bias] if bias_shape else [ref_weight]
    ref_optimizer = torch.optim.SGD(ref_opt_params, lr=learning_rate)
    ref_out = call_fn(ref_activation, ref_weight, ref_bias)

    assert out.shape == out_shape == ref_out.shape

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (quantization was applied)"
    assert sqnr > _SQNR_THRESHOLD, f"Forward SQNR too low ({sqnr:.1f} dB)"

    target = torch.ones_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()
    ref_loss = F.mse_loss(ref_out, target)
    ref_loss.backward()

    assert activation.grad is not None
    activation_grad_sqnr = compute_error(activation.grad, ref_activation.grad)
    assert activation_grad_sqnr > _SQNR_THRESHOLD, f"Input grad SQNR too low ({activation_grad_sqnr:.1f} dB)"

    assert weight.grad is not None
    weight_grad_sqnr = compute_error(weight.grad, ref_weight.grad)
    assert weight_grad_sqnr > _SQNR_THRESHOLD, f"Weight grad SQNR too low ({weight_grad_sqnr:.1f} dB)"

    optimizer.step()
    ref_optimizer.step()

    assert not torch.equal(weight_tensor, weight), "weight should be updated"
    new_weight_sqnr = compute_error(weight, ref_weight)
    assert new_weight_sqnr > _SQNR_THRESHOLD, f"New weight SQNR too low ({new_weight_sqnr:.1f} dB)"


@pytest.mark.parametrize("device", target_devices)
def test_op_grouped_mm_sqnr(device):
    """grouped_mm with HiF8-quantized weight produces reasonable forward+backward SQNR."""
    weight_config = HiF8QuantizeConfig()
    act_config = HiF8QuantizeConfig()

    S, E, K, N = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    A = torch.randn(S, K, dtype=torch.bfloat16, requires_grad=True, device=device)
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    wrapper = HiF8TrainingWeightWrapperTensor(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=device)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(A_ref, w_ref.transpose(-2, -1), offs=offs)
    out = torch._grouped_mm(A, param.transpose(-2, -1), offs=offs)
    assert out.shape == (S, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (quantization was applied)"
    assert sqnr > _SQNR_THRESHOLD, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.backward(torch.ones_like(ref_out))
    out.backward(torch.ones_like(out))
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > _SQNR_THRESHOLD
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > _SQNR_THRESHOLD
