# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SLA2 attention for mm_gc: full-width MHA projections + per-head QK RMSNorm +
RoPE + the Sparse Linear Attention core (``core.SparseLinearAttention``).

RoPE frequencies are computed inside ``forward`` from the runtime sequence
length so that CP (full-sequence all-gather) and TP layouts self-adapt. All
SLA2 internals are per-head computations; TP shards the head dimension, and
the per-head-dim router/alpha weights stay replicated.
"""

import contextlib
from dataclasses import dataclass

import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module

from .core import SparseLinearAttention


def _local_weight(w: torch.Tensor) -> torch.Tensor:
    return w.to_local() if isinstance(w, DTensor) else w


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, Dh]; freqs_cis: complex64 [S, Dh/2]."""
    ctx = (
        torch.amp.autocast(x.device.type, enabled=False)
        if x.device.type != "meta"
        else contextlib.nullcontext()
    )
    with ctx:
        x_f = x.float()
        x_c = torch.view_as_complex(x_f.reshape(*x_f.shape[:-1], -1, 2))
        freqs = freqs_cis.view((1, 1, *freqs_cis.shape))
        out = torch.view_as_real(x_c * freqs).flatten(3)
        return out.type_as(x)


class SLA2Attention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        head_dim: int
        seq_len: int
        rope_theta: float = 10000.0
        sla2_topk_1m: float = 0.005
        sla2_topk_5m: float = 0.001
        sla2_blkq: int = 64
        sla2_blkk: int = 64
        sla2_feature_map: str = "softmax"
        sla2_tie_feature_map_qk: bool = True
        sla2_stage: int = 1
        sla2_mode: str = "train"
        sla2_router_data_path: str | None = None
        use_bf16: bool = True
        norm_eps: float = 1e-6
        layer_idx: int = 0

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(
                f"dim {config.dim} must be divisible by n_heads {config.n_heads}"
            )
        if config.dim // config.n_heads != config.head_dim:
            raise ValueError(
                f"dim/n_heads {config.dim // config.n_heads} must equal "
                f"head_dim {config.head_dim}"
            )
        if config.sla2_blkq <= 0 or config.sla2_blkk <= 0:
            raise ValueError("SLA2 block sizes must be positive")

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.seq_len = config.seq_len
        self.rope_theta = config.rope_theta

        self.to_q = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=False,
        ).build()
        self.to_k = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=False,
        ).build()
        self.to_v = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=False,
        ).build()
        self.to_out = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=False,
        ).build()
        self.norm_q = RMSNorm.Config(
            normalized_shape=config.head_dim,
            eps=config.norm_eps,
        ).build()
        self.norm_k = RMSNorm.Config(
            normalized_shape=config.head_dim,
            eps=config.norm_eps,
        ).build()

        num_blocks = (config.seq_len + config.sla2_blkq - 1) // config.sla2_blkq
        num_key_blocks = (
            config.seq_len + config.sla2_blkk - 1
        ) // config.sla2_blkk
        ratio = (
            config.sla2_topk_1m
            if config.seq_len <= 1_000_000
            else config.sla2_topk_5m
        )
        topk_blocks = max(1, int(num_key_blocks * ratio))
        sla2_topk = topk_blocks / num_key_blocks

        self.core = SparseLinearAttention(
            head_dim=config.head_dim,
            topk=sla2_topk,
            L=config.seq_len,
            feature_map=config.sla2_feature_map,
            BLKQ=config.sla2_blkq,
            BLKK=config.sla2_blkk,
            use_bf16=config.use_bf16,
            tie_feature_map_qk=config.sla2_tie_feature_map_qk,
            layer_idx=config.layer_idx,
            mode=config.sla2_mode,
            stage=config.sla2_stage,
            router_data_path=config.sla2_router_data_path,
        )
        self.sla2_topk_blocks = topk_blocks
        self.num_query_blocks = num_blocks
        self.tp_group = None
        self.local_n_heads = config.n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        w_q = _local_weight(self.to_q.weight)
        w_k = _local_weight(self.to_k.weight)
        w_v = _local_weight(self.to_v.weight)
        q = F.linear(x, w_q).view(B, S, self.local_n_heads, self.head_dim)
        k = F.linear(x, w_k).view(B, S, self.local_n_heads, self.head_dim)
        v = F.linear(x, w_v).view(B, S, self.local_n_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        freqs_cis = precompute_freqs_cis(
            self.head_dim, S, self.rope_theta
        ).to(x.device)
        q = apply_rotary_emb(q.transpose(1, 2), freqs_cis)
        k = apply_rotary_emb(k.transpose(1, 2), freqs_cis)

        v = v.transpose(1, 2)

        out = self.core(q, k, v)
        out = out.to(x.dtype).transpose(1, 2).reshape(
            B, S, self.local_n_heads * self.head_dim
        )
        w_out = _local_weight(self.to_out.weight)
        out = F.linear(out, w_out)
        if self.tp_group is not None:
            out = dist_nn.all_reduce(out, group=self.tp_group)
        return out
