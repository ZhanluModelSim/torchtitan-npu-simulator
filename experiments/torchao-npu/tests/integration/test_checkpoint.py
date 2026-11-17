# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DCP checkpoint save/load integration test for a QAT-wrapped MoE model.

A reference MoE model is quantized with ``ParamSwapConfig`` (routed expert
weights wrapped into ``BlockMXTrainingWeightWrapperTensor``), saved with
``torch.distributed.checkpoint``, loaded into a fresh quantized model, and the
reloaded model is verified to be identical to the pre-save model: per-tensor
type, wrapper class, quantize configs, dtype, shape, and bit-exact data, plus
bit-exact forward output.
"""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockMXTrainingWeightWrapperTensor,
)

from ..reference_moe import MoE
from ..testing_utils import _expert_weight_filter, _moe_input, create_moe_model, target_devices


def _quantize_moe(model):
    """Wrap the 3D routed expert weights into BlockMXTrainingWeightWrapperTensor."""
    quantize_(
        model,
        ParamSwapConfig(
            activation_config=MXQuantizeConfig(),
            weight_config=BlockMXQuantizeConfig(),
            params_filter_fn=_expert_weight_filter,
        ),
        filter_fn=lambda module, fqn: isinstance(module, MoE),
    )
    return model


def _inner_data(value):
    return value.to_tensor() if isinstance(value, BaseTrainingWeightWrapperTensor) else value


def _snapshot_state(state_dict):
    """Record type/config/dtype/shape/requires_grad plus a data copy per entry."""
    snapshot = {}
    for key, value in state_dict.items():
        wrapped = isinstance(value, BaseTrainingWeightWrapperTensor)
        snapshot[key] = {
            "type": type(value),
            "weight_config": value.weight_config if wrapped else None,
            "activation_config": value.activation_config if wrapped else None,
            "dtype": value.dtype,
            "shape": value.shape,
            "requires_grad": value.requires_grad,
            "data": _inner_data(value).clone(),
        }
    return snapshot


def _assert_state_equal(reference, state_dict, context):
    """Assert every entry of ``state_dict`` matches the ``reference`` snapshot."""
    assert set(reference) == set(state_dict), f"{context}: key sets differ: {set(reference) ^ set(state_dict)}"
    for key in sorted(reference):
        ref, value = reference[key], state_dict[key]
        assert isinstance(value, ref["type"]), (
            f"{context}: {key}: type {type(value).__name__} != {ref['type'].__name__}"
        )
        if isinstance(ref["type"], type) and issubclass(ref["type"], BaseTrainingWeightWrapperTensor):
            assert value.weight_config == ref["weight_config"], f"{context}: {key}: weight_config differs"
            assert value.activation_config == ref["activation_config"], f"{context}: {key}: activation_config differs"
        assert value.dtype == ref["dtype"], f"{context}: {key}: dtype differs"
        assert value.shape == ref["shape"], f"{context}: {key}: shape differs"
        assert value.requires_grad == ref["requires_grad"], f"{context}: {key}: requires_grad differs"
        assert torch.equal(_inner_data(value), ref["data"]), f"{context}: {key}: data differs"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The DCP round trip runs the grouped-mm forward, which compiles the test-only "
        "Triton kernel _fill_indices_kernel; triton_ascend builds its launcher with an "
        "outdated C++ standard that this torch rejects, so it cannot compile on this "
        "toolchain. Re-enable once Triton-Ascend is updated."
    ),
)
@pytest.mark.parametrize("device", target_devices)
def test_dcp_checkpoint_save_load_roundtrip(device, tmp_path):
    """A QAT-wrapped MoE model survives a DCP save/load round trip unchanged."""
    model = _quantize_moe(create_moe_model(device, use_grouped_mm=True, dtype=torch.bfloat16))

    wrapped_keys = {k for k, v in model.state_dict().items() if isinstance(v, BlockMXTrainingWeightWrapperTensor)}
    assert wrapped_keys == {"experts.gate_proj", "experts.down_proj", "experts.up_proj"}, (
        f"Expected the 3D routed expert weights to be wrapped, got {sorted(wrapped_keys)}"
    )

    x = _moe_input(model)
    with torch.no_grad():
        out_before = model(x)
    reference = _snapshot_state(model.state_dict())

    checkpoint_dir = tmp_path / "checkpoint"
    dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint_dir)

    reloaded = _quantize_moe(create_moe_model(device, use_grouped_mm=True, dtype=torch.bfloat16))
    dcp.load({"model": reloaded.state_dict()}, checkpoint_id=checkpoint_dir)

    _assert_state_equal(reference, reloaded.state_dict(), "reloaded model vs pre-save model")

    with torch.no_grad():
        out_after = reloaded(x)
    assert torch.equal(out_before, out_after), "Forward output differs after checkpoint load"
