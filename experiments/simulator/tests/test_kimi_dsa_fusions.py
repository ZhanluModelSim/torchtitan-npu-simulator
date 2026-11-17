# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selected-key Meta outputs, actual saved storage and scoped restoration."""

import pytest
import torch
from torchtitan_simulator.compat.kimi_dsa import dsa_attention, dsa_indexer, install_kimi_dsa_virtual_fusions
from torchtitan_simulator.compat.patches import PatchScope


def test_selected_attention_saves_forward_values_and_has_distinct_value_gradient_width():
    q = torch.empty((2, 3, 12, 7), device="meta", requires_grad=True)
    k = torch.empty_like(q, requires_grad=True)
    v = torch.empty((2, 3, 12, 4), device="meta", requires_grad=True)
    idx = torch.empty((2, 12, 5), dtype=torch.int64, device="meta")
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out, lse = dsa_attention(q, k, v, idx, 7**-0.5)
        out.sum().backward()
    assert out.shape == v.shape and lse.shape == (2, 3, 12)
    assert lse.dtype == torch.float32 and not lse.requires_grad
    assert len(saved) == 6
    # Compare storage identity, not values or coincidentally equal shapes.
    assert [t.untyped_storage()._cdata for t in saved] == [t.untyped_storage()._cdata for t in (q, k, v, idx, out, lse)]
    assert [t.grad.shape for t in (q, k, v)] == [q.shape, k.shape, v.shape]
    assert [t.grad.dtype for t in (q, k, v)] == [q.dtype, k.dtype, v.dtype]


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

    original = KimiDSAIndexer._select, attention.selected_key_attention
    try:
        with PatchScope() as patches:
            install_kimi_dsa_virtual_fusions(patches)
            assert (KimiDSAIndexer._select, attention.selected_key_attention) != original
            q = torch.empty((1, 2, 6, 4), device="meta", requires_grad=True)
            v = torch.empty((1, 2, 6, 3), device="meta", requires_grad=True)
            idx = torch.empty((1, 6, 2), device="meta", dtype=torch.int64)
            result = attention.selected_key_attention(q, q, v, idx, 0.5)
            assert result.shape == (1, 2, 6, 3)
            if fail:
                raise RuntimeError("intentional session failure")
    except RuntimeError as error:
        assert fail and str(error) == "intentional session failure"
    assert (KimiDSAIndexer._select, attention.selected_key_attention) == original


def test_dsa_fake_rejects_invalid_precision_scale_and_saved_placement():
    from torchtitan_simulator.compat.kimi_dsa import _attention_grad_meta, _indexer_meta

    iq = torch.empty((1, 6, 2, 4), device="meta", dtype=torch.int64)
    ik = torch.empty((1, 6, 4), device="meta", dtype=torch.int64)
    iw = torch.empty((1, 6, 2), device="meta")
    with pytest.raises(ValueError, match="floating-point"):
        _indexer_meta(iq, ik, iw, 2)
    q = torch.empty((1, 2, 6, 4), device="meta")
    v = torch.empty((1, 2, 6, 3), device="meta")
    idx = torch.empty((1, 6, 2), device="meta", dtype=torch.int64)
    with pytest.raises(ValueError, match="scale"):
        dsa_attention(q, q, v, idx, float("nan"))
    out = torch.empty_like(v)
    lse = torch.empty((1, 2, 6), device="meta")
    with pytest.raises(ValueError, match="placement"):
        _attention_grad_meta(q, q, v, idx, out, lse, torch.empty(v.shape), 0.5)
