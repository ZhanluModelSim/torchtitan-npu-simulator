# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def test_pro_scores_follow_document_local_picks_across_chunks():
    from torchtitan_npu.ops.pypto_pro.li_lig_wrapper import _scores_at_picks

    gen = torch.Generator().manual_seed(17)
    q = torch.randn(130, 8, 16, generator=gen).bfloat16()
    k = torch.randn(5, 1, 16, generator=gen).bfloat16()
    w = torch.randn(130, 8, generator=gen)
    cu_q = torch.tensor([0, 8, 130], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 5], dtype=torch.int32)
    indices = torch.tensor([1, -1, 0], dtype=torch.int32).expand(130, 1, 3).clone()
    indices[8:, 0, 2] = 2

    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = _scores_at_picks(q, k, w, indices, cu_q, cu_k)

    assert scores.dtype == torch.float32
    for q_start, q_stop, k_start, k_stop in ((0, 8, 0, 2), (8, 130, 2, 5)):
        dense = (q[q_start:q_stop].float() @ k[k_start:k_stop, 0].float().T).relu()
        dense = (dense * w[q_start:q_stop, :, None]).sum(1)
        picks = indices[q_start:q_stop, 0].long()
        expected = dense.gather(1, picks.clamp_min(0)).masked_fill(picks < 0, -torch.inf)
        torch.testing.assert_close(scores[q_start:q_stop, 0], expected, atol=1e-5, rtol=1e-5)
