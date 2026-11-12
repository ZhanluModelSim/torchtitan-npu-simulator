# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CANN registration boundary for the recipe-selected SwiGLUGroup activation.

The unit-test tier keeps only the config/override composition checks and fakes
``_ensure_cann_ops_loaded``; this smoke test loads the real ``cann_ops_nn.ops``
extension and substitutes only its device primitive on CPU to verify the built
activation's packing, clamp argument, output and gradient boundary.
"""

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import apply_overrides
from torchtitan.config.loader import ConfigLoader
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe import MoE

from torchtitan_npu.override.common.swiglu_group import AscSwiGLU
from torchtitan_npu.patches.torchtitan.models.common.moe import LimitedSwiGLU

pytest.importorskip("torch_npu")
try:
    import cann_ops_nn.ops  # noqa: F401
except Exception as error:  # the JIT build may fail on a CPU runner
    pytest.skip(f"cann_ops_nn.swiglu_group is unavailable: {error}", allow_module_level=True)

_RECIPE = ("torchtitan_npu_recipes.models.deepseek_v4", "deepseek_v4_debugmodel")
_SWIGLU = "torchtitan_npu.override.common.swiglu_group."


def _trainer_config(module, recipe):
    return ConfigLoader._load_config(module, recipe)


def _moe_layers(config):
    return [
        (index, layer) for index, layer in enumerate(config.model.layers) if isinstance(layer.feed_forward, MoE.Config)
    ]


@pytest.mark.parametrize("limit", [0.0, 0.75])
def test_recipe_selected_built_activation_reaches_registered_cann_boundary(monkeypatch, limit):
    config = _trainer_config(*_RECIPE)
    first_moe = _moe_layers(config)[0][1].feed_forward
    first_moe.routed_experts.activation_fn = SwiGLU.Config() if limit <= 0 else LimitedSwiGLU.Config(swiglu_limit=limit)
    config.override.imports = [_SWIGLU + "asc"]
    apply_overrides(config.override, config)
    activation = first_moe.routed_experts.activation_fn.build()
    assert type(activation) is AscSwiGLU
    operation = torch.ops.cann_ops_nn.swiglu_group
    assert operation.default._schema.name == "cann_ops_nn::swiglu_group"
    calls = []

    def cpu_device_primitive(h, *, weight, group_index, clamp_limit):
        calls.append((h, weight, group_index, clamp_limit))
        gate, up = h.chunk(2, dim=-1)
        if clamp_limit > 0:
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
        return F.silu(gate) * up

    monkeypatch.setattr(operation, "default", cpu_device_primitive)
    gate = torch.tensor([[-2.0, 1.0, 0.5], [0.25, 3.0, -1.0]], requires_grad=True)
    up = torch.tensor([[2.0, -3.0, 0.25], [0.5, 1.0, -2.0]], requires_grad=True)
    ref_gate, ref_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    clamped_gate = ref_gate.clamp(max=limit) if limit > 0 else ref_gate
    clamped_up = ref_up.clamp(min=-limit, max=limit) if limit > 0 else ref_up
    expected = F.silu(clamped_gate) * clamped_up

    actual = activation(torch.stack((gate, up), dim=-2), offsets=torch.tensor([2]))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(gate.grad, ref_gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(up.grad, ref_up.grad, rtol=0, atol=0)
    assert len(calls) == 1
    packed, weight, group_index, clamp_limit = calls[0]
    torch.testing.assert_close(packed, torch.cat((gate, up), dim=-1))
    assert packed.is_contiguous()
    assert weight is None and group_index is None
    assert clamp_limit == (limit if limit > 0 else -1.0)
