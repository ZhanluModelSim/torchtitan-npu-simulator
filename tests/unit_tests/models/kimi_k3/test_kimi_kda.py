# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA provider receives the legacy fusion inputs and external beta sigmoid."""

import torch

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


def test_kda_provider_preserves_input_identity_flags_and_external_sigmoid():
    config = kimi_k3_debugmodel().model_spec.model.layers[0].attention
    received = {}

    def provider(**kwargs):
        received.update(kwargs)
        return kwargs["v"] * kwargs["beta"].unsqueeze(-1), None

    config.kernel = provider
    with torch.random.fork_rng():
        torch.manual_seed(563)
        module = config.build()
        module.init_states()
        x = torch.randn(1, 8, 256, requires_grad=True)
    output = module(x)
    output.square().sum().backward()

    torch.testing.assert_close(received["beta"], torch.sigmoid(module.b_proj(x).float()))
    assert received["A_log"] is module.A_log
    assert received["dt_bias"] is module.dt_bias
    for name in ("q", "k", "v", "g"):
        assert received[name].shape == (1, 8, 8, 32)
    assert {key: value for key, value in received.items() if not isinstance(value, torch.Tensor)} == {
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "transpose_state_layout": True,
        "cu_seqlens": None,
    }
    assert x.grad is not None
    assert module.v_proj.weight.grad is not None
    assert module.b_proj.weight.grad is not None
