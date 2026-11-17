# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 MHC overrides backed by TileLang implementations.

V4.1 keeps a cross-layer pre-mix graph, so ``TilelangV41HcPre`` adapts the
model-side ``forward(x, pre_mix) -> (y, pre, post, comb)`` contract to the
TileKernels split/apply primitives (the freshly computed ``pre`` is returned
for the next sub-layer and the branches collapse with the incoming
``pre_mix``).  ``TilelangV41HcPost`` adapts ``forward(y, residual, post, comb)``
to the external TileKernels ``mhc_post`` operator.
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre
from torchtitan_npu.ops.tilelang import tilelang_mhc_post, tilelang_mhc_pre_v41


class TilelangV41HcPre(HcPre):
    """V4.1 HcPre backed by TileKernels Split/Apply and torch_npu Sinkhorn."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def _to_local(self, tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def forward(self, x_BLHcD: torch.Tensor, pre_mix_BLHc: torch.Tensor):
        if isinstance(x_BLHcD, DTensor):
            raise ValueError(
                "TileLang V4.1 HcPre does not support DTensor activations; use the validated spmd_types backend"
            )
        if x_BLHcD.device.type != "npu":
            raise ValueError(f"TileLang V4.1 HcPre requires NPU input, got {x_BLHcD.device}")
        if x_BLHcD.ndim != 4:
            raise ValueError(f"TileLang V4.1 HcPre expects x with shape [B,S,N,D], got {tuple(x_BLHcD.shape)}")
        if self.hc_mult != 4:
            raise ValueError(f"TileLang V4.1 HcPre currently supports only hc_mult=4, got {self.hc_mult}")
        if x_BLHcD.shape[-2] != self.hc_mult:
            raise ValueError(f"TileLang V4.1 HcPre expects N={self.hc_mult}, got x shape {tuple(x_BLHcD.shape)}")
        if x_BLHcD.shape[-1] % 64 != 0:
            raise ValueError(
                f"TileLang V4.1 HcPre requires a hidden size divisible by 64, got x shape {tuple(x_BLHcD.shape)}"
            )
        if x_BLHcD.dtype != torch.bfloat16:
            raise ValueError(f"TileLang V4.1 HcPre requires torch.bfloat16 input, got {x_BLHcD.dtype}")
        if not x_BLHcD.is_contiguous():
            raise ValueError("TileLang V4.1 HcPre requires contiguous input")
        batch, seq = x_BLHcD.shape[:2]
        if pre_mix_BLHc.ndim != 3:
            raise ValueError(f"TileLang V4.1 HcPre expects pre_mix with shape [B,S,N], got {tuple(pre_mix_BLHc.shape)}")
        if pre_mix_BLHc.shape != (batch, seq, self.hc_mult):
            raise ValueError(
                "TileLang V4.1 HcPre requires pre_mix shape [B,S,N] to match x [B,S,N,D], "
                f"got pre_mix={tuple(pre_mix_BLHc.shape)} for x={tuple(x_BLHcD.shape)}"
            )
        if pre_mix_BLHc.dtype != torch.float32:
            raise ValueError(f"TileLang V4.1 HcPre requires float32 pre_mix, got {pre_mix_BLHc.dtype}")
        if not pre_mix_BLHc.is_contiguous():
            raise ValueError("TileLang V4.1 HcPre requires contiguous pre_mix")

        return tilelang_mhc_pre_v41(
            x_BLHcD,
            pre_mix_BLHc,
            self._to_local(self.hc_fn).float(),
            self._to_local(self.hc_scale).float(),
            self._to_local(self.hc_base).float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )


class TilelangV41HcPost(HcPost):
    """Run TileKernels ``mhc_post`` while preserving the V4.1 model contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(
        self,
        y_BLD: torch.Tensor,
        residual_BLHcD: torch.Tensor,
        post_BLHc: torch.Tensor,
        comb_BLHcHc: torch.Tensor,
    ) -> torch.Tensor:
        # Sequence-parallel linears hand the collapsed stream back batch-folded
        # (``[T, D]`` with B=1); normalize every operand to the explicit
        # batch-first form the TileKernels kernel requires.
        y_was_2d = y_BLD.ndim == 2
        residual_was_3d = residual_BLHcD.ndim == 3
        post_was_2d = post_BLHc.ndim == 2
        comb_was_3d = comb_BLHcHc.ndim == 3
        if y_was_2d:
            y_BLD = y_BLD.unsqueeze(0)
        if residual_was_3d:
            residual_BLHcD = residual_BLHcD.unsqueeze(0)
        if post_was_2d:
            post_BLHc = post_BLHc.unsqueeze(0)
        if comb_was_3d:
            comb_BLHcHc = comb_BLHcHc.unsqueeze(0)

        if y_BLD.device.type != "npu":
            raise ValueError(f"TileLang V4.1 HcPost requires NPU inputs, got {y_BLD.device}")
        if y_BLD.ndim != 3:
            raise ValueError(f"TileLang V4.1 HcPost expects y with shape [B,S,D], got {tuple(y_BLD.shape)}")
        if residual_BLHcD.ndim != 4:
            raise ValueError(
                f"TileLang V4.1 HcPost expects residual with shape [B,S,N,D], got {tuple(residual_BLHcD.shape)}"
            )
        batch, seq, hidden = y_BLD.shape
        if residual_BLHcD.shape[:2] != (batch, seq) or residual_BLHcD.shape[-1] != hidden:
            raise ValueError(
                "TileLang V4.1 HcPost requires residual shape [B,S,N,D] to match y [B,S,D], "
                f"got y={tuple(y_BLD.shape)}, residual={tuple(residual_BLHcD.shape)}"
            )
        num_streams = residual_BLHcD.shape[2]
        if post_BLHc.shape != (batch, seq, num_streams):
            raise ValueError(
                "TileLang V4.1 HcPost requires post shape [B,S,N], "
                f"got post={tuple(post_BLHc.shape)} for y={tuple(y_BLD.shape)} and N={num_streams}"
            )
        if comb_BLHcHc.shape != (batch, seq, num_streams, num_streams):
            raise ValueError(
                "TileLang V4.1 HcPost requires comb shape [B,S,N,N], "
                f"got comb={tuple(comb_BLHcHc.shape)} for N={num_streams}"
            )
        if y_BLD.dtype != torch.bfloat16 or residual_BLHcD.dtype != torch.bfloat16:
            raise ValueError(
                "TileLang V4.1 HcPost requires y and residual to use torch.bfloat16, "
                f"got y={y_BLD.dtype}, residual={residual_BLHcD.dtype}"
            )
        if post_BLHc.dtype != torch.float32 or comb_BLHcHc.dtype != torch.float32:
            raise ValueError(
                "TileLang V4.1 HcPost requires post and comb to use torch.float32, "
                f"got post={post_BLHc.dtype}, comb={comb_BLHcHc.dtype}"
            )
        if any(not tensor.is_contiguous() for tensor in (y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc)):
            raise ValueError("TileLang V4.1 HcPost requires contiguous y, residual, post and comb inputs")

        out = tilelang_mhc_post(y_BLD, residual_BLHcD, post_BLHc.unsqueeze(-1), comb_BLHcHc)
        if residual_was_3d:
            out = out.squeeze(0)
        return out


@override(
    target=HcPre.Config,
    exact=True,
    description="V4.1 single-pass mHC pre backed by TileKernels Split/Apply and torch_npu Sinkhorn",
)
def tilelang_hc_pre(cfg: HcPre.Config) -> TilelangV41HcPre.Config:
    return derive(cfg, TilelangV41HcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description="V4.1 mHC post backed by the external TileKernels mhc_post operator",
)
def tilelang_hc_post(cfg: HcPost.Config) -> TilelangV41HcPost.Config:
    return derive(cfg, TilelangV41HcPost.Config)
