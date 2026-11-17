# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter contracts for the V4.1 mHC fused modules.

Selection/isolation via the real debug model; arithmetic contracts run on
CPU with the fused kernels replaced by reference implementations.  The
kernel mocks prove the adapter feeds the kernels the documented operands
and keeps the autograd chain connected; NPU kernel execution is validated
on the target hardware (see the verification records).
"""

import pytest
import torch
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre

MHC_POST_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post"
MHC_SINKHORN_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn"


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


class TestMHcAdapters:
    def test_override_selects_and_preserves_config(self):
        cfg = model_registry("deepseek_v4_1_debugmodel").model
        reference = model_registry("deepseek_v4_1_debugmodel").model
        apply_overrides(
            OverrideConfig(imports=[MHC_POST_OVERRIDE, MHC_SINKHORN_OVERRIDE]),
            cfg,
        )
        from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPost, AscV41HcPre

        for layer, ref_layer in zip(cfg.layers[:6], reference.layers[:6], strict=True):
            assert isinstance(layer.hc_post, AscV41HcPost.Config)
            assert isinstance(layer.hc_attn_pre, AscV41HcPre.Config)
            # the fused configs are pure replacements: every declared field
            # keeps the reference value
            for field in ("hc_mult", "dim", "sinkhorn_iters", "hc_eps", "norm_eps"):
                assert getattr(layer.hc_attn_pre, field) == getattr(ref_layer.hc_attn_pre, field)

    def test_hc_post_matches_reference_arithmetic(self, monkeypatch):
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPost.Config().build()
        fused_module = fused.AscV41HcPost.Config().build()
        # Block-level contract: ``y`` is the collapsed single stream [B, T, D];
        # ``residual`` is the multi-stream [B, T, n, D].
        B, T, D, n = 2, 5, 8, 4
        y = torch.randn(B, T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)

        def mhc_post(residual, comb, y, post):
            out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
            )
            return out.type_as(y)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)

        # Forward and backward both flow through the mocked kernel unchanged:
        # same values, same gradients on every operand.
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        fused_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        expected = reference(*ref_inputs)
        actual = fused_module(*fused_inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        for ref_t, fused_t, name in zip(ref_inputs, fused_inputs, ("y", "residual", "post", "comb"), strict=True):
            assert ref_t.grad is not None and fused_t.grad is not None, name
            torch.testing.assert_close(fused_t.grad, ref_t.grad, rtol=1e-5, atol=1e-6)

    def test_hc_post_unfolds_the_batch_folded_stream(self, monkeypatch):
        """Sequence-parallel linears return the collapsed stream as ``[T, D]``;
        the reference broadcasts that silently, the kernel needs ``[B, T, D]``."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        fused_module = fused.AscV41HcPost.Config().build()
        B, T, D, n = 1, 5, 8, 4
        y_folded = torch.randn(T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)
        expected = HcPost.Config().build()(y_folded.unsqueeze(0), residual, post, comb)

        received = []

        def mhc_post(x, h_res, h_out, h_post):
            received.append(tuple(t.dim() for t in (x, h_res, h_out, h_post)))
            out = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + torch.sum(
                h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=2
            )
            return out.type_as(h_out)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)
        actual = fused_module(y_folded, residual, post, comb)
        assert received[0] == (4, 4, 3, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_sinkhorn_matches_reference_and_feeds_kernel(self, monkeypatch):
        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        module = fused.AscV41HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module.load_state_dict(reference.state_dict())
        n = module.hc_mult

        calls = []

        def sinkhorn(comb, *, eps, num_iters, out_flag):
            # Differentiable CPU stand-in mirroring the reference's balancing:
            # proving the adapter connects the graph, not the native kernel.
            calls.append((eps, num_iters, out_flag))
            row_max = comb.max(dim=-1, keepdim=True).values
            out = torch.exp(comb - row_max)
            out = out / out.sum(dim=-1, keepdim=True) + eps
            out = out / (out.sum(dim=-2, keepdim=True) + eps)
            for _ in range(num_iters - 1):
                out = out / (out.sum(dim=-1, keepdim=True) + eps)
                out = out / (out.sum(dim=-2, keepdim=True) + eps)
            return out, None, None

        monkeypatch.setattr(fused.torch_npu, "npu_mhc_sinkhorn", sinkhorn)

        B, T = 2, 3
        mixes = torch.randn(B, T, 2 * n + n * n, requires_grad=True)
        ref_mixes = mixes.detach().clone().requires_grad_(True)
        ref_pre, ref_post, ref_comb = reference._split_sinkhorn(ref_mixes)
        pre, post, comb = module._split_sinkhorn(mixes)

        # pre/post keep the reference arithmetic exactly; comb matches the
        # stand-in's balancing (same math as the reference), and the kernel
        # received the documented iteration knobs.
        torch.testing.assert_close(pre, ref_pre, rtol=0, atol=0)
        torch.testing.assert_close(post, ref_post, rtol=0, atol=0)
        torch.testing.assert_close(comb, ref_comb, rtol=1e-6, atol=1e-7)
        assert calls[0] == (module.hc_eps, module.sinkhorn_iters, 1)

        # the gradient connection: identical backward through both paths
        (pre.sum() + post.sum() + comb.sum()).backward()
        (ref_pre.sum() + ref_post.sum() + ref_comb.sum()).backward()
        torch.testing.assert_close(mixes.grad, ref_mixes.grad, rtol=1e-5, atol=1e-7)
