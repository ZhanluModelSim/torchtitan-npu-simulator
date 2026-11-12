# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh-process import order preserves DeepSeek activation and score placement."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("order", ("upstream-first", "plugin-first"))
def test_v4_import_order_preserves_clamp_and_pre_down_projection_scores(order):
    script = textwrap.dedent(
        """
        import dataclasses
        import sys

        if sys.argv[1] == "upstream-first":
            import torchtitan.models.deepseek_v4.flavors as early_flavors
            captured_factory = early_flavors.make_routed_experts_config

        import torchtitan_npu
        import torch
        import torch.nn.functional as F
        from torchtitan.config import derive
        from torchtitan.models.common.linear import GroupedLinear
        from torchtitan.models.common.moe import RoutedExperts
        from torchtitan_npu.models.common.activation import LimitedSwiGLU
        from torchtitan_npu.models.deepseek_v4 import model_registry
        from torchtitan_npu.patches.torchtitan.models.common import moe as moe_patch

        if sys.argv[1] == "upstream-first":
            assert early_flavors.make_routed_experts_config is captured_factory
        assert moe_patch.LimitedSwiGLU is LimitedSwiGLU
        config = model_registry("debugmodel", seq_len=128, num_mtp_layers=1)
        for layer in (*config.layers, *config.mtp_layers):
            experts = layer.feed_forward.routed_experts
            assert type(experts) is RoutedExperts.Config
            assert type(experts.activation_fn) is LimitedSwiGLU.Config
            assert experts.activation_fn.swiglu_limit == 10.0
            assert experts.token_dispatcher.absorb_router_scores
            shared = layer.feed_forward.shared_experts
            assert type(shared.activation_fn) is LimitedSwiGLU.Config
            assert shared.activation_fn.swiglu_limit == 10.0

        class CPUGroupedLinear(GroupedLinear):
            # CPU dense GEMMs implement the grouped backend without replacing
            # routed-expert forward, dispatcher, activation or score transport.
            @dataclasses.dataclass(kw_only=True, slots=True)
            class Config(GroupedLinear.Config):
                pass

            def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
                starts = [0, *offsets_E.tolist()]
                return torch.cat([
                    F.linear(input_RI[starts[e]:starts[e + 1]], weight_EOI[e].bfloat16())
                    for e in range(weight_EOI.shape[0])
                ])

        cfg = config.layers[2].feed_forward.routed_experts
        cfg.w13 = derive(cfg.w13, CPUGroupedLinear.Config, group_size=2, in_features=2, out_features=2)
        cfg.w2 = derive(cfg.w2, CPUGroupedLinear.Config, group_size=2, in_features=2, out_features=2)
        cfg.token_dispatcher = dataclasses.replace(cfg.token_dispatcher, num_experts=2, top_k=1)
        experts = cfg.build()
        assert type(experts) is RoutedExperts
        with torch.no_grad():
            experts.w13.weight.copy_(torch.linspace(-7.0, 8.0, 16).reshape(2, 2, 2, 2))
            experts.w2.weight.copy_(torch.tensor([[[0.8, 0.3], [-0.2, 1.3]],
                                                  [[1.1, -0.6], [0.7, 0.4]]]))
        x = torch.tensor([[2.5, 1.25], [1.5, -2.0]], dtype=torch.bfloat16, requires_grad=True)
        scores = torch.tensor([[0.123], [0.679]], requires_grad=True)
        ids = torch.tensor([[1], [0]])
        reference_x = x.detach().clone().requires_grad_()
        reference_scores = scores.detach().clone().requires_grad_()
        up_weight = experts.w13.weight.detach().clone().requires_grad_()
        down_weight = experts.w2.weight.detach().clone().requires_grad_()

        actual = experts(x, scores, ids, torch.tensor([1, 1]))
        expected_rows = []
        for row, expert in enumerate((1, 0)):
            gate, up = F.linear(reference_x[row], up_weight[expert].flatten(0, 1).bfloat16()).reshape(2, 2)
            hidden = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
            weighted = (hidden.float() * reference_scores[row]).bfloat16()
            expected_rows.append(F.linear(weighted, down_weight[expert].bfloat16()))
        expected = torch.stack(expected_rows)
        gradient = torch.tensor([[0.7, -0.4], [1.1, 0.3]], dtype=torch.bfloat16)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
        torch.testing.assert_close(scores.grad, reference_scores.grad, rtol=0, atol=0)
        torch.testing.assert_close(experts.w13.weight.grad, up_weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(experts.w2.weight.grad, down_weight.grad, rtol=0, atol=0)
        """
    )
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([str(root), *sys.path]))

    result = subprocess.run(
        [sys.executable, "-c", script, order], cwd=root, env=env, text=True, capture_output=True, timeout=120
    )

    assert result.returncode == 0, result.stdout + result.stderr
