# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split-aware ``CosSinRoPE`` patch regression (CPU, no CANN).

The patched default path must equal the pre-split model pattern — rotating
the trailing channels with a plain (unsplit) rope and concatenating the
untouched prefix back — for both zero and non-zero splits, with query and
key carrying different head counts.
"""

import pytest
import torch
from torchtitan.models.common.rope import CosSinRoPE

# The patch under test applies itself on import; keep this file runnable
# standalone instead of relying on another test module to load it.
import torchtitan_npu.patches.torchtitan.models.common.rope  # noqa: F401


@pytest.mark.parametrize("split", [0, 12])
def test_split_cossin_matches_manual_tail_rotation(split):
    rotary_dim = 8
    split_rope = CosSinRoPE(CosSinRoPE.Config(dim=rotary_dim, max_seq_len=16, split=split))
    plain = CosSinRoPE(CosSinRoPE.Config(dim=rotary_dim, max_seq_len=16))

    torch.manual_seed(7)
    width = split + rotary_dim
    # Different head counts: the broadcasted cache must serve both sites.
    q = torch.randn(2, 5, 3, width)
    k = torch.randn(2, 5, 7, width)
    positions = torch.arange(5).unsqueeze(0)

    out_q, out_k = split_rope(q, k, positions=positions)
    out_q_only = split_rope(q, positions=positions)

    ref_q_tail, ref_k_tail = plain(q[..., split:], k[..., split:], positions=positions)
    assert torch.equal(out_q, torch.cat((q[..., :split], ref_q_tail), dim=-1))
    assert torch.equal(out_k, torch.cat((k[..., :split], ref_k_tail), dim=-1))
    assert torch.equal(out_q_only, out_q)
    assert out_q.shape == q.shape and out_q.dtype == q.dtype
    assert out_k.shape == k.shape and out_k.dtype == k.dtype
    # The prefix channels ride along untouched.
    assert torch.equal(out_q[..., :split], q[..., :split])
    assert torch.equal(out_k[..., :split], k[..., :split])


def test_split_cossin_forward_and_grad_flow():
    split, rotary_dim = 4, 8
    rope = CosSinRoPE(CosSinRoPE.Config(dim=rotary_dim, max_seq_len=16, split=split))
    plain = CosSinRoPE(CosSinRoPE.Config(dim=rotary_dim, max_seq_len=16))

    torch.manual_seed(11)
    q = torch.randn(2, 5, 3, split + rotary_dim, requires_grad=True)
    k = torch.randn(2, 5, 3, split + rotary_dim, requires_grad=True)
    q_ref, k_ref = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    positions = torch.arange(5).unsqueeze(0)

    out_q, out_k = rope(q, k, positions=positions)
    ref_q_tail, ref_k_tail = plain(q_ref[..., split:], k_ref[..., split:], positions=positions)

    grad_q, grad_k = torch.autograd.grad((out_q.square() + out_k).sum(), (q, k))
    ref_grad_q, ref_grad_k = torch.autograd.grad(
        (ref_q_tail.square().sum() + q_ref[..., :split].square().sum() + ref_k_tail.sum() + k_ref[..., :split].sum()),
        (q_ref, k_ref),
    )
    assert torch.equal(grad_q, ref_grad_q)
    assert torch.equal(grad_k, ref_grad_k)


def test_cossin_inverse_is_explicitly_unsupported():
    rope = CosSinRoPE(CosSinRoPE.Config(dim=8, max_seq_len=16, split=4))
    cache = rope._reshape_cache(torch.randn(1, 4, 2, 12), None)
    with pytest.raises(NotImplementedError):
        CosSinRoPE.apply_rotary_emb(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8), cache, inverse=True)
