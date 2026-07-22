# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F
from torchao.float8.float8_utils import compute_error
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.quant_api import quantize_

from torchtitan_npu.experiments.ao_npu.torchao_npu import ParamSwapConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.wrapper_tensors import BaseTrainingWeightWrapperTensor

from .reference_moe import MoE
from .testing_utils import (
    _expert_weight_filter,
    _moe_input,
    create_moe_model,
    target_devices,
)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "weight_config, act_config, sqnr_threshold",
    [
        (
            Float8FakeQuantizeConfig(),
            None,
            {"out": 30, "input_grad": 30, "param_grad": 22},
        ),
        (
            Float8FakeQuantizeConfig(),
            Float8FakeQuantizeConfig(),
            {"out": 27, "input_grad": 26, "param_grad": 18},
        ),
        (
            BlockQuantizeConfig(),
            MXQuantizeConfig(),
            {"out": 12, "input_grad": 12, "param_grad": 8},
        ),
    ],
)
@pytest.mark.parametrize("use_grouped_mm", [True, False])
def test_param_swap(device, weight_config, act_config, sqnr_threshold, use_grouped_mm):
    """Forward and gradient SQNR vs FP32 reference for the ParamSwap model."""
    if isinstance(weight_config, BlockQuantizeConfig) and not use_grouped_mm:
        pytest.skip("BlockFP8 requires bfloat16 (use_grouped_mm=True)")
    dtype = torch.bfloat16 if use_grouped_mm else torch.float32
    qat_model = create_moe_model(device, use_grouped_mm=use_grouped_mm, dtype=dtype)
    ref_model = copy.deepcopy(qat_model)

    quantize_(
        qat_model,
        ParamSwapConfig(
            activation_config=act_config,
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        ),
        filter_fn=lambda m, fqn: isinstance(m, MoE),
    )

    learning_rate = 0.0001
    qat_optimizer = torch.optim.SGD(qat_model.parameters(), lr=learning_rate)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=learning_rate)

    def check_finite(qat_name, qat_param, ref_name, ref_param):
        assert torch.isfinite(ref_param).all(), f"NaN values appear in the reference parameter {ref_name}"
        assert torch.isfinite(qat_param).all(), f"NaN values appear in the qat parameter {qat_name}"

    def check_all_zero(qat_name, qat_param, ref_name, ref_param):
        assert ref_param.norm() != 0, f"The reference parameter {ref_name} is all zero"
        assert qat_param.norm() != 0, f"The qat parameter {qat_name} is all zero"

    for _ in range(1):
        # Clear gradients
        qat_optimizer.zero_grad()
        ref_optimizer.zero_grad()

        qat_prev = copy.deepcopy(qat_model)

        # Generate input randomly
        qat_x = _moe_input(qat_model).requires_grad_(True)
        ref_x = qat_x.clone().detach().requires_grad_(True)

        # Propagate forward
        qat_out = qat_model(qat_x)
        ref_out = ref_model(ref_x)

        # Set up target
        target = torch.ones_like(qat_out)

        # Compute loss and propagate backward
        qat_loss = F.mse_loss(qat_out, target)
        qat_loss.backward()

        ref_loss = F.mse_loss(ref_out, target)
        ref_loss.backward()

        # Update weights
        qat_optimizer.step()
        ref_optimizer.step()

        # Check loss alignment
        loss_rel_diff = abs(qat_loss.item() - ref_loss.item()) / ref_loss.item()
        assert loss_rel_diff < 0.03, "Loss of the QAT and reference models should align."

        # Check SQNR of output
        check_finite("out", qat_out, "out", ref_out)
        check_all_zero("out", qat_out, "out", ref_out)
        out_sqnr = compute_error(qat_out, ref_out)
        assert out_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
        assert out_sqnr > sqnr_threshold["out"], (
            f"The output SQNR too low ({out_sqnr:.1f} dB), fake quant may be degrading output"
        )

        # Check SQNR of the input's gradient
        check_finite("qat_x.grad", qat_x.grad, "ref_x.grad", ref_x.grad)
        check_all_zero("qat_x.grad", qat_x.grad, "ref_x.grad", ref_x.grad)
        x_grad_sqnr = compute_error(qat_x.grad, ref_x.grad)
        assert x_grad_sqnr > sqnr_threshold["input_grad"], f"Input grad SQNR too low ({x_grad_sqnr:.1f} dB)"

        # Check SQNR of gradients of all wrapped parameters to be updated
        for (qat_name, qat_param), (ref_name, ref_param) in zip(
            qat_model.named_parameters(), ref_model.named_parameters(), strict=True
        ):
            if ref_param.requires_grad:
                is_gate = ".gate" in qat_name
                assert qat_param.requires_grad, f"{qat_name} should require gradients"
                assert qat_param.grad is not None, f"{qat_name} has no gradient"
                check_finite(
                    f"{qat_name}.grad",
                    qat_param.grad,
                    f"{ref_name}.grad",
                    ref_param.grad,
                )
                check_all_zero(
                    f"{qat_name}.grad",
                    qat_param.grad,
                    f"{ref_name}.grad",
                    ref_param.grad,
                )
                if not is_gate:
                    sqnr = compute_error(qat_param.grad, ref_param.grad)
                    assert sqnr > sqnr_threshold["param_grad"], (
                        f"Weight grad SQNR too low for {qat_name} ({sqnr:.1f} dB)"
                    )
            else:
                assert qat_param.requires_grad is False, f"{qat_name} should not require gradients"

        # Check the change of weights
        for (cur_name, cur_param), (prev_name, prev_param) in zip(
            qat_model.named_parameters(), qat_prev.named_parameters(), strict=True
        ):
            assert type(cur_param) == type(prev_param), (
                f"The type of {cur_name} changed from {type(cur_param)} to {type(prev_param)}"
            )

            assert cur_param.requires_grad == prev_param.requires_grad, (
                f"{cur_name}.requires_grad changed from {prev_param.requires_grad} to {cur_param.requires_grad}"
            )

            assert torch.isfinite(cur_param).all(), f"Elements of {cur_name} in the QAT model should all be finite."

            if cur_param.requires_grad and ".gate" not in cur_name:
                data = (
                    cur_param.data.to_tensor()
                    if isinstance(cur_param.data, BaseTrainingWeightWrapperTensor)
                    else cur_param.data
                )
                assert not torch.equal(data, prev_param.data), (
                    f"Weight {cur_name} should be updated after optimizer step."
                )
