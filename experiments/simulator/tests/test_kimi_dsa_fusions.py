# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selected-key Meta outputs, actual saved storage and scoped restoration."""

import pytest
import torch
from torchtitan_simulator.compat.kimi_dsa import _native_sparse, dsa_indexer, install_kimi_dsa_native_capture
from torchtitan_simulator.compat.patches import PatchScope


def test_native_sfa_saves_actual_statistics_and_shared_latent_gradients():
    pytest.importorskip("torch_npu")
    from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture

    q = torch.empty((2, 12, 4, 512), device="meta", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.empty((2, 12, 1, 512), device="meta", dtype=torch.bfloat16, requires_grad=True)
    qr = torch.empty((2, 12, 4, 64), device="meta", dtype=torch.bfloat16, requires_grad=True)
    kr = torch.empty((2, 12, 1, 64), device="meta", dtype=torch.bfloat16, requires_grad=True)
    idx = torch.empty((2, 12, 5), device="meta", dtype=torch.int64)
    capture = OpDispatchCapture()
    with capture:
        out = _native_sparse(q, kv, qr, kr, idx, 192**-0.5, 5, "test_topk")
        out.sum().backward()
    assert out.shape == q.shape
    events = {event.raw_op_type: event for event in capture.memory_events()}
    forward = events["npu.npu_sparse_flash_attention.default"]
    backward = events["npu.npu_sparse_flash_attention_grad.default"]
    assert forward.inputs[1].storage_key == forward.inputs[2].storage_key
    saved = {tensor.storage_key for tensor in backward.inputs}
    assert all(value.storage_key in saved for value in forward.outputs)
    for value in (q, kv, qr, kr):
        assert value.grad.shape == value.shape and value.grad.dtype == value.dtype


def test_indexer_shape_and_frozen_integer_result():
    q = torch.empty((2, 12, 3, 7), dtype=torch.bfloat16, device="meta")
    k = torch.empty((2, 12, 7), dtype=torch.bfloat16, device="meta")
    weights = torch.empty((2, 12, 3), device="meta")
    out = dsa_indexer(q, k, weights, 5)
    assert out.shape == (2, 12, 5) and out.dtype == torch.int64 and not out.requires_grad
    assert dsa_indexer(q, k, weights, 20).shape == (2, 12, 12)


@pytest.mark.parametrize("fail", [False, True])
def test_dsa_patch_restores_native_selection_on_normal_and_exception_exit(fail):
    from torchtitan_npu.models.kimi_k3_1 import attention
    from torchtitan_npu.models.kimi_k3_1.dsa import KimiDSAIndexer

    original = KimiDSAIndexer._select, attention.KimiGatedDSA._project_attention
    try:
        with PatchScope() as patches:
            install_kimi_dsa_native_capture(patches)
            assert KimiDSAIndexer._select is not original[0]
            assert attention.KimiGatedDSA._project_attention is not original[1]
            if fail:
                raise RuntimeError("intentional session failure")
    except RuntimeError as error:
        assert fail and str(error) == "intentional session failure"
    assert (KimiDSAIndexer._select, attention.KimiGatedDSA._project_attention) == original


def test_indexer_fake_rejects_invalid_precision():
    from torchtitan_simulator.compat.kimi_dsa import _indexer_meta

    iq = torch.empty((1, 6, 2, 4), device="meta", dtype=torch.int64)
    ik = torch.empty((1, 6, 4), device="meta", dtype=torch.int64)
    iw = torch.empty((1, 6, 2), device="meta")
    with pytest.raises(ValueError, match="floating-point"):
        _indexer_meta(iq, ik, iw, 2)


def test_absorbing_kv_weights_preserves_output_and_gradients():
    from types import SimpleNamespace

    from torchtitan_npu.models.kimi_k3_1.attention import KimiGatedDSA
    from torchtitan_npu.models.kimi_k3_1.dsa import selected_key_attention

    batch, seq, heads, q_dim, rope_dim, k_dim, v_dim = 2, 7, 3, 4, 2, 5, 3
    generator = torch.Generator().manual_seed(71)

    def value(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()

    q, qr, kv, kr = (
        value(batch, seq, heads, q_dim),
        value(batch, seq, heads, rope_dim),
        value(batch, seq, k_dim),
        value(batch, seq, rope_dim),
    )
    weight = value(heads * (q_dim + v_dim), k_dim)
    module = SimpleNamespace(
        num_heads=heads,
        qk_nope_head_dim=q_dim,
        v_head_dim=v_dim,
        kv_lora_rank=k_dim,
        kv_b_proj=SimpleNamespace(weight=weight),
    )
    scale = (q_dim + rope_dim) ** -0.5
    position = torch.arange(seq)
    indices = (position[:, None] - torch.arange(3)).clamp_min(0).expand(batch, seq, 3)
    k, v = (
        torch.nn.functional.linear(kv, weight).reshape(batch, seq, heads, q_dim + v_dim).split((q_dim, v_dim), dim=-1)
    )
    expanded_key = torch.cat((k, kr.unsqueeze(2).expand(-1, -1, heads, -1)), dim=-1)
    expected = selected_key_attention(
        torch.cat((q, qr), dim=-1).transpose(1, 2),
        expanded_key.transpose(1, 2),
        v.transpose(1, 2),
        indices,
        scale,
    )

    def absorbed_kernel(aq, akv, aq_rope, ak_rope, idx):
        key = torch.cat((akv, ak_rope), dim=-1).expand(-1, -1, heads, -1)
        return selected_key_attention(
            torch.cat((aq, aq_rope), dim=-1).transpose(1, 2),
            key.transpose(1, 2),
            akv.expand(-1, -1, heads, -1).transpose(1, 2),
            idx,
            scale,
        ).transpose(1, 2)

    actual = KimiGatedDSA._absorbed_attention(module, q, qr, kv, kr, indices, absorbed_kernel)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    upstream = value(*actual.shape).detach()
    for gradient, reference in zip(
        torch.autograd.grad(actual, (q, qr, kv, kr, weight), upstream),
        torch.autograd.grad(expected, (q, qr, kv, kr, weight), upstream),
        strict=True,
    ):
        torch.testing.assert_close(gradient, reference, rtol=1e-11, atol=1e-12)
