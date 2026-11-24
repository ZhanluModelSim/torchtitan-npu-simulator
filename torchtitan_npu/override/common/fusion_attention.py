# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: modality-encoder attention on the CANN fused dense kernel.

The encoder attention is maskless non-causal dense MHA with an fp32 softmax;
``npu_fusion_attention`` computes the same math in one kernel (BSND layout,
``sparse_mode=0``).  The module-level seam keeps the vendor call in one place
so the simulator can bridge its autograd on Meta devices.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

from torchtitan_npu.models.system3.model import EncoderAttention


def _encoder_fusion_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_num: int, scale: float) -> torch.Tensor:
    """Single seam for the fused op; the simulator replaces it on Meta devices."""
    result = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
    )
    return result[0]


class AscEncoderAttention(EncoderAttention):
    """Encoder MHA on the fused CANN dense kernel.

    BSND takes the reshaped projections directly, so the reference transposes
    disappear; the kernel's internal fp32 softmax matches the reference fp32
    normalize.  The fused op is causal-mask free under ``sparse_mode=0``,
    matching the bidirectional encoder semantics.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(EncoderAttention.Config):
        pass

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim)
        k = k.view(b, s, self.heads, self.head_dim)
        v = v.view(b, s, self.heads, self.head_dim)
        out = _encoder_fusion_attention(q, k, v, self.heads, self.scale)
        return self.o(out.reshape(b, s, -1))


@override(
    target=EncoderAttention.Config,
    description="CANN fused dense attention via npu_fusion_attention",
)
def asc(cfg: EncoderAttention.Config) -> "AscEncoderAttention.Config":
    return derive(cfg, AscEncoderAttention.Config)
