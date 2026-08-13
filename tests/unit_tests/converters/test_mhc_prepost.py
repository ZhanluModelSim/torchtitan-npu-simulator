# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch

import torchtitan_npu.converters.kernels.mhc_prepost as mhc_prepost
from torchtitan_npu.models.deepseek_v4.model import HcPre


def _fake_mhc_pre(x, phi, alpha, bias, **kwargs):
    batch, sequence, num_streams, dim = x.shape
    parameter_term = phi.mean() + alpha.mean() + bias.mean()
    h_in = x.float().mean(dim=2) + parameter_term
    h_post = x.float().mean(dim=-1) + parameter_term
    h_res = h_post.unsqueeze(-1).expand(batch, sequence, num_streams, num_streams)
    return h_in.to(x.dtype), h_post, h_res, h_post, h_post, h_post


def _fake_mhc_sinkhorn(h_res, **kwargs):
    normalized = h_res / (h_res.abs().mean() + 1)
    return normalized, h_res, h_res


def test_npu_hc_pre_fused_training_forward_backward():
    parent = HcPre.Config(
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        norm_eps=1e-6,
    ).build()
    module = mhc_prepost.NpuHcPreFused(parent)

    x = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    hc_fn = torch.randn(24, 32, dtype=torch.float32, requires_grad=True)
    hc_scale = torch.randn(3, dtype=torch.float32, requires_grad=True)
    hc_base = torch.randn(24, dtype=torch.float32, requires_grad=True)

    with patch.object(mhc_prepost.torch_npu, "npu_mhc_pre", side_effect=_fake_mhc_pre) as mock_pre:
        with patch.object(
            mhc_prepost.torch_npu,
            "npu_mhc_sinkhorn",
            side_effect=_fake_mhc_sinkhorn,
        ) as mock_sinkhorn:
            h_in, h_post, h_res = module(x, hc_fn, hc_scale, hc_base)

    assert mock_pre.call_args.kwargs["out_flag"] == 1
    assert mock_sinkhorn.call_args.kwargs["out_flag"] == 1
    assert (h_in.shape, h_post.shape, h_res.shape) == (
        (1, 2, 8),
        (1, 2, 4),
        (1, 2, 4, 4),
    )
    assert (h_in.dtype, h_post.dtype, h_res.dtype) == (
        torch.bfloat16,
        torch.float32,
        torch.float32,
    )

    (h_in.float().sum() + h_post.sum() + h_res.sum()).backward()

    assert x.grad is not None
    assert hc_fn.grad is not None
    assert hc_scale.grad is not None
    assert hc_base.grad is not None
