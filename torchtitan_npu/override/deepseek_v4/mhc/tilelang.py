# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 MHC overrides backed by TileLang implementations.

The HcHead override uses the in-tree TileLang fused op
``mhc_head_compute_mix_tilelang``. The HcPost override adapts the model-side
contract to the external TileKernels ``mhc_post`` operator.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from torchtitan_npu.models.deepseek_v4.mhc import HcHead, HcPost, HcPre
from torchtitan_npu.ops.tilelang import (
    mhc_head_compute_mix_tilelang,
    tilelang_mhc_head_compute_mix_a5,
    tilelang_mhc_post,
    tilelang_mhc_pre,
)
from torchtitan_npu.override import _IS_A5


def _to_local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TilelangHcHead(HcHead):
    """HcHead backed by the TileLang fused ``mhc_head_compute_mix`` kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcHead.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # ``HcHead`` only uses hc_mult to size its parameters and drops it; the
        # TileLang kernel needs it as ``num_stream``.
        self.hc_mult = config.hc_mult
        self._use_a5_backend = _IS_A5

    def _forward_tilelang(self, x: Tensor) -> Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "TilelangHcHead expects local tensor input; apply HcHeadParallelStyle with local TP input."
            )
        hc_fn = _to_local_tensor(self.hc_fn)
        hc_base = _to_local_tensor(self.hc_base)
        hc_scale = _to_local_tensor(self.hc_scale)

        is_tnd = x.dim() == 3

        if is_tnd:
            x = x.flatten(1).unsqueeze(1)  # [T, N, D] -> [T, 1, N*D]
        elif x.dim() == 4:
            x = x.flatten(2)  # [B, S, N, D] -> [B, S, N*D]
        else:
            raise ValueError(
                f"TilelangHcHead expects 3D [T, N, D] or 4D [B, S, N, D] tensor, but got input with shape {x.shape}"
            )

        y = mhc_head_compute_mix_tilelang(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            None,
            False,
            self.norm_eps,
            self.eps,
            self.hc_mult,
        )

        if is_tnd:
            y = y.squeeze(1)  # [T, 1, out_features] -> [T, out_features]

        return y

    def _forward_a5(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, DTensor) or any(
            isinstance(parameter, DTensor) for parameter in (self.hc_fn, self.hc_base, self.hc_scale)
        ):
            raise ValueError("A5 TileLang HcHead does not support DTensor; use a local-tensor path")
        if x.device.type != "npu":
            raise ValueError(f"A5 TileLang HcHead requires NPU input, got {x.device}")
        if x.ndim != 4:
            raise ValueError(f"A5 TileLang HcHead expects [B,S,N,H], got {tuple(x.shape)}")
        if self.hc_mult != 4 or x.shape[-2] != self.hc_mult:
            raise ValueError("A5 TileLang HcHead currently supports hc_mult=4 and N=4")
        if x.shape[-1] % 64 != 0:
            raise ValueError("A5 TileLang HcHead requires hidden size divisible by 64")
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("A5 TileLang HcHead requires contiguous BF16 input")

        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_fn.float()) * rsqrt
        mixes = tilelang_mhc_head_compute_mix_a5(
            mixes.contiguous(), self.hc_scale.float(), self.hc_base.float(), self.eps
        )
        y = torch.sum(mixes.unsqueeze(-1) * x.reshape(shape), dim=2)
        return y.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_a5(x) if self._use_a5_backend else self._forward_tilelang(x)


class TilelangHcPost(HcPost):
    """Run TileKernels ``mhc_post`` while preserving the model-side contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError(f"TileLang HcPost requires NPU inputs, got {x.device}")
        if x.ndim != 3:
            raise ValueError(f"TileLang HcPost expects x with shape [B,S,H], got {tuple(x.shape)}")
        if residual.ndim != 4:
            raise ValueError(f"TileLang HcPost expects residual with shape [B,S,N,H], got {tuple(residual.shape)}")
        batch, seq, hidden = x.shape
        if residual.shape[:2] != (batch, seq) or residual.shape[-1] != hidden:
            raise ValueError(
                "TileLang HcPost requires residual shape [B,S,N,H] to match x [B,S,H], "
                f"got x={tuple(x.shape)}, residual={tuple(residual.shape)}"
            )
        num_heads = residual.shape[2]
        if post.shape != (batch, seq, num_heads):
            raise ValueError(
                "TileLang HcPost requires post shape [B,S,N], "
                f"got post={tuple(post.shape)} for x={tuple(x.shape)} and N={num_heads}"
            )
        if comb.shape != (batch, seq, num_heads, num_heads):
            raise ValueError(
                f"TileLang HcPost requires comb shape [B,S,N,N], got comb={tuple(comb.shape)} for N={num_heads}"
            )
        if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise ValueError(
                "TileLang HcPost requires x and residual to use torch.bfloat16, "
                f"got x={x.dtype}, residual={residual.dtype}"
            )
        if post.dtype != torch.float32 or comb.dtype != torch.float32:
            raise ValueError(
                f"TileLang HcPost requires post and comb to use torch.float32, got post={post.dtype}, comb={comb.dtype}"
            )
        if any(not tensor.is_contiguous() for tensor in (x, residual, post, comb)):
            raise ValueError("TileLang HcPost requires contiguous x, residual, post and comb inputs")

        return tilelang_mhc_post(x, residual, post.unsqueeze(-1), comb)


class TilelangHcPre(HcPre):
    """Run TileKernels HcPre without HcPost-coupled grad accumulation.

    Norm stays in PyTorch because the external kernel's validated hidden
    sizes start above the mini-model's 256-wide shape. The existing
    ``torch_npu`` Sinkhorn also keeps this adapter compatible with
    TileLang releases whose PTO backend cannot lower ``tl.simd.vdup``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, DTensor) or any(
            isinstance(parameter, DTensor) for parameter in (self.hc_fn, self.hc_base, self.hc_scale)
        ):
            raise ValueError(
                "TileLang HcPre does not support DTensor/full_dtensor; use the validated spmd_types backend"
            )
        if x.device.type != "npu":
            raise ValueError(f"TileLang HcPre requires NPU input, got {x.device}")
        if x.ndim != 4:
            raise ValueError(f"TileLang HcPre expects x with shape [B,S,N,H], got {tuple(x.shape)}")
        if self.hc_mult != 4:
            raise ValueError(f"TileLang HcPre currently supports only hc_mult=4, got {self.hc_mult}")
        if x.shape[-2] != self.hc_mult:
            raise ValueError(f"TileLang HcPre expects N={self.hc_mult}, got x shape {tuple(x.shape)}")
        if x.shape[-1] % 64 != 0:
            raise ValueError(f"TileLang HcPre requires a hidden size divisible by 64, got x shape {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"TileLang HcPre requires torch.bfloat16 input, got {x.dtype}")
        if not x.is_contiguous():
            raise ValueError("TileLang HcPre requires contiguous input")

        return tilelang_mhc_pre(
            x,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
            self.norm_eps,
        )
