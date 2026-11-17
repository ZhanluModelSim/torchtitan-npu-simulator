# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused partial-RoPE override: eager CPU parity with the split workaround."""

import pytest
import torch

from tests.unit_tests.rope_test_utils import (
    register_cpu_kernels,
    split_workaround_reference,
)
from torchtitan_npu.override.common.rope import AscPartialComplexRoPE


@pytest.mark.parametrize("inverse", [False, True])
def test_asc_partial_matches_split_workaround_forward_and_backward(inverse):
    register_cpu_kernels()
    split, rotary_dim = 12, 8
    fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=rotary_dim, max_seq_len=16, split=split))
    positions = torch.arange(5).unsqueeze(0)

    x_fused = torch.randn(2, 5, 3, split + rotary_dim, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x_fused.detach().clone().requires_grad_(True)

    out_fused = fused(x_fused, positions=positions, inverse=inverse)
    out_ref = split_workaround_reference(x_ref, split, rotary_dim, positions, inverse=inverse)

    torch.testing.assert_close(out_fused, out_ref, rtol=0, atol=0)
    assert torch.equal(out_fused, out_ref)

    grad_out = torch.randn_like(out_fused, dtype=torch.float32)
    grad_fused = torch.autograd.grad(out_fused.float(), x_fused, grad_out)[0]
    grad_ref = torch.autograd.grad(out_ref.float(), x_ref, grad_out)[0]

    torch.testing.assert_close(grad_fused, grad_ref, rtol=0, atol=0)
    assert torch.equal(grad_fused, grad_ref)


def test_asc_partial_split_does_not_fragment_cache_pool():
    split_a = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=8, max_seq_len=16, split=12))
    split_b = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=8, max_seq_len=16, split=4))

    # The cos/sin table only depends on the rotary flavor, never on split.
    assert split_a.cache is split_b.cache


# --- tail-RoPE contract guards (review: hot-path window validation) ---------


def test_asc_partial_config_rejects_negative_split_and_odd_dim():
    with pytest.raises(ValueError, match="split must be non-negative"):
        AscPartialComplexRoPE.Config(dim=4, max_seq_len=16, split=-1)
    with pytest.raises(ValueError, match="positive even width"):
        AscPartialComplexRoPE.Config(dim=3, max_seq_len=16, split=4)
    with pytest.raises(ValueError, match="positive even width"):
        AscPartialComplexRoPE.Config(dim=0, max_seq_len=16, split=4)


def test_asc_partial_rejects_width_mismatch():
    register_cpu_kernels()
    # An MTP-style misconfiguration: full-width site (512) pinned with
    # split=0 and a 64-wide rotary dim would silently rotate the wrong
    # channels; the tail-RoPE contract must fail fast instead.
    fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=64, max_seq_len=16, split=0))
    q = torch.randn(1, 5, 2, 512)
    k = torch.randn(1, 5, 2, 512)
    with pytest.raises(ValueError, match="tail-RoPE contract violated: query width 512"):
        fused(q)
    with pytest.raises(ValueError, match="tail-RoPE contract violated: query width 512"):
        fused(q, k)
    # A key wider than the query also trips the contract.
    ok = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=64, max_seq_len=16, split=448))
    with pytest.raises(ValueError, match="key width"):
        ok(torch.randn(1, 5, 2, 512), torch.randn(1, 5, 2, 256))
