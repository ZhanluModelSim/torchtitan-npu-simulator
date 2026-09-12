# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sparse Linear Attention (SLA) core for mm_gc.

Interface follows the reference SLA implementation: block-granular router with
soft/hard top-k block selection (stage 1 / stage 2), a sparse softmax branch on
the selected key blocks, a global linear-attention branch with softmax feature
map, and a per-query-block learnable ``alpha`` blend.

The fused Triton/ACLNN sparse-attention kernel is optional: when
``mm_gc/kernel.py`` exposing ``_attention`` is absent, stage 2 falls back to an
eager masked-attention path with identical input/output shapes so that the
single-card skeleton stays runnable on CPU and meta devices.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kernel import _attention
except ImportError:
    _attention = None


def mean_pool(x: torch.Tensor, blk: int) -> torch.Tensor:
    B, H, L, D = x.shape
    nb = (L + blk - 1) // blk
    pad = nb * blk - L
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    x_sum = x.view(B, H, nb, blk, D).sum(dim=-2)
    counts = torch.full(
        (nb,), float(blk), dtype=x_sum.dtype, device=x_sum.device
    )
    if pad > 0:
        counts[-1] = float(L - (nb - 1) * blk)
    return x_sum / counts.view(1, 1, nb, 1)


def soft_top_k(
    scores: torch.Tensor,
    k: int,
    temperature: float = 0.1,
    dim: int = -1,
    max_iter: int = 50,
    tol: float = 1e-2,
) -> torch.Tensor:
    if dim != -1 and dim != scores.ndim - 1:
        scores = scores.transpose(dim, -1)

    x_min = scores.min(dim=-1, keepdim=True)[0]
    x_max = scores.max(dim=-1, keepdim=True)[0]

    t = torch.zeros_like(x_min)

    low = -10.0 - x_max / temperature
    high = 10.0 - x_min / temperature

    early_stop = not (scores.is_meta or torch.compiler.is_compiling())
    for _ in range(max_iter):
        t_mid = (low + high) * 0.5
        logits = scores / temperature + t_mid
        probs = torch.sigmoid(logits)
        current_sum = probs.sum(dim=-1, keepdim=True)
        diff = current_sum - k
        if early_stop and bool(torch.all(torch.abs(diff) < tol)):
            t = t_mid
            break
        mask = diff > 0
        low = torch.where(mask, low, t_mid)
        high = torch.where(~mask, high, t_mid)
        t = t_mid

    soft_logits = scores / temperature + t
    soft_mask = torch.sigmoid(soft_logits)

    return soft_mask


def get_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    BLKQ: int = 64,
    BLKK: int = 64,
    proj_q: nn.Linear | None = None,
    proj_k: nn.Linear | None = None,
    dtype: torch.dtype = torch.bfloat16,
    stage: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)

    q_proj = proj_q(pooled_qblocks).to(dtype)
    k_proj = proj_k(pooled_kblocks).to(dtype)
    pooled_score = q_proj @ k_proj.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))

    if stage == 1:
        sparse_map = soft_top_k(pooled_score, topk)
        lut = None
    else:
        lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
        sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
        sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


class SparseLinearAttention(nn.Module):
    """Reference-interface SLA module.

    Args:
        head_dim: per-head dimension of q/k/v.
        topk: ratio of key blocks selected for sparse attention.
        L: sequence length; fixes the number of alpha blocks at build time.
        feature_map: feature map for the linear branch, one of
            ['hedgehog', 'elu', 'relu', 'softmax'] (hedgehog raises here).
        BLKQ: query block size.
        BLKK: key block size.
        use_bf16: compute branches in bfloat16 (default) or float16.
        tie_feature_map_qk: use the same feature map for query and key.
        layer_idx: layer index; required for stage-2 router loading.
        mode: "train" or "infer".
        stage: 1 (trainable router, soft top-k) or 2 (frozen router loaded
            from ``router_data_path``, hard top-k).
        router_data_path: directory holding ``block<i>/block_<i>_2.pt``
            router checkpoints; required for stage 2.
    """

    def __init__(
        self,
        head_dim: int,
        topk: float,
        L: int,
        feature_map: str = "softmax",
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        tie_feature_map_qk: bool = True,
        layer_idx: int | None = None,
        mode: str = "infer",
        stage: int = 1,
        router_data_path: str | None = None,
    ):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        if mode not in ("train", "infer"):
            raise ValueError(f"mode must be 'train' or 'infer', got {mode}")
        if stage == 2 and router_data_path is None:
            raise ValueError("stage == 2 requires router_data_path")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.L = L
        self.layer_idx = layer_idx
        self.router_data_path = router_data_path
        self.stage = stage
        self.mode = mode

        self.proj_q = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        self.proj_k = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        num_blocks = (L + self.BLKQ - 1) // self.BLKQ
        self.alpha = nn.Parameter(torch.empty(num_blocks, 1))
        with torch.no_grad():
            self.alpha.fill_(0.8)

        if feature_map == "elu":

            def elu_feature_map(x):
                return F.elu(x) + 1

            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == "relu":
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == "softmax":

            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)

            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f"Not supported feature map {feature_map}.")

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        if mode == "train":
            if self.stage == 2:
                self.init_weights_2_()
                self.proj_q.weight.requires_grad = False
                self.proj_q.bias.requires_grad = False
                self.proj_k.weight.requires_grad = False
                self.proj_k.bias.requires_grad = False
            else:
                self.init_weights_1_()

    def init_weights_1_(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.proj_q.weight)
            nn.init.zeros_(self.proj_q.bias)
            nn.init.eye_(self.proj_k.weight)
            nn.init.zeros_(self.proj_k.bias)
            self.alpha.fill_(0.8)

    def init_weights_2_(self) -> None:
        data = torch.load(
            self.router_data_path + f"/block{self.layer_idx}/block_{self.layer_idx}_2.pt"
        )
        with torch.no_grad():
            self.proj_q.weight.data = data["proj_q.weight"]
            self.proj_q.bias.data = data["proj_q.bias"]
            self.proj_k.weight.data = data["proj_k.weight"]
            self.proj_k.bias.data = data["proj_k.bias"]
            self.alpha.data = data["alpha"]

    def _sparse_attention_eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparse_map: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        key_mask = torch.repeat_interleave(
            torch.repeat_interleave(sparse_map > 0, self.BLKQ, dim=2),
            self.BLKK,
            dim=3,
        )[:, :, :L, :L]
        attention_weights = F.softmax(
            scores.masked_fill(~key_mask, float("-inf")), dim=-1
        )
        return torch.matmul(attention_weights, v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_sparsity: bool = False,
    ) -> torch.Tensor:
        B, num_heads, L, head_dim = q.size()
        if L != self.L:
            raise ValueError(
                f"runtime sequence length {L} differs from the length {self.L} "
                "the SLA core was built with"
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sparse_map, lut, real_topk = get_block_map(
            q,
            k,
            topk_ratio=self.topk,
            BLKQ=self.BLKQ,
            BLKK=self.BLKK,
            proj_q=self.proj_q,
            proj_k=self.proj_k,
            dtype=self.dtype,
            stage=self.stage,
        )

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        if self.stage == 1:
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            soft_mask_ = torch.repeat_interleave(
                torch.repeat_interleave(sparse_map, self.BLKQ, dim=2),
                self.BLKK,
                dim=3,
            )
            attention_weights = F.softmax(
                scores * soft_mask_[:, :, :L, :L], dim=-1
            )
            o_s = torch.matmul(attention_weights, v)
        elif _attention is not None:
            o_s = _attention.apply(
                q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK
            )
        else:
            o_s = self._sparse_attention_eager(q, k, v, sparse_map, L)

        q = self.feature_map_q(q).contiguous().to(self.dtype)
        k = self.feature_map_k(k).contiguous().to(self.dtype)

        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))

        o_l = calc_linear(q, k, v)

        block_indices = torch.arange(L, device=q.device) // self.BLKQ
        alpha_per_position = self.alpha[block_indices].view(1, 1, L, 1)

        weighted_o_s = alpha_per_position * o_s
        weighted_o_l = (1 - alpha_per_position) * o_l

        o = (weighted_o_s + weighted_o_l).to(self.dtype)
        return o
