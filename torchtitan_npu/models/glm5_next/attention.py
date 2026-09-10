# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""glm5_next text attention: KDA linear attention + MLA-DSA sparse attention.

Contract: see MODEL_CONTRACT.md (section 4).

- ``GlmDeltaAttention`` (KDA): fused qkv short conv, low-rank forget gate with
  safe lower bound, sigmoid beta, ``chunk_kda`` kernel seam (same family as
  kimi_k3), low-rank output gate + gated RMSNorm.
- ``GlmDsaAttention``: NoPE MLA (``qk_rope_head_dim=0``) + k-pool compressed
  DSA indexer. The indexer is frozen (``requires_grad=False``) matching the raw
  reference's ``@torch.no_grad()`` inference path. Training-state sparse
  attention is gather-based over the static top-k union, which keeps every
  shape value-independent and meta-clean.

All modules avoid ``.item()``/``.tolist()``/data-dependent control flow so they
run on meta tensors.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class ShortConv1d(nn.Module):
    """Fused depthwise causal short conv over concatenated qkv (kernel=4).

    Under TP the qkv channels split per head, so the local channel layout is
    three contiguous slices (q/k/v) of the global channels. ``set_local_slice``
    records the global start of each slice so forward can rebuild the local
    weight view; with no TP the full weight applies directly.
    """

    def __init__(self, channels: int, kernel_size: int = 4, activation: str = "silu"):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=0,
            bias=False,
        )
        self._channel_starts: tuple[int, int, int] | None = None
        self._local_channels: int | None = None

    def set_local_slice(self, channel_starts: tuple[int, int, int], local_channels: int) -> None:
        self._channel_starts = channel_starts
        self._local_channels = local_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, channels]
        x_t = x.transpose(1, 2)
        weight = _local(self.conv.weight)
        if self._local_channels is not None:
            lc = self._local_channels
            weight = torch.cat([weight[s : s + lc] for s in self._channel_starts], dim=0)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        x_t = F.conv1d(x_t, weight, bias=None, stride=1, groups=x_t.shape[1])[..., : x.shape[1]]
        x = x_t.transpose(1, 2)
        if self.activation == "silu":
            x = F.silu(x)
        return x


class RMSNormGated(nn.Module):
    """RMSNorm with sigmoid gating: norm(x) * sigmoid(gate), fp32 math."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight.float() * x
        return (x * F.sigmoid(gate.float())).to(input_dtype)


class GlmForgetGate(nn.Module):
    """Low-rank forget gate with per-head decay and safe lower bound.

    ``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``.
    ``A_log``/``dt_bias`` stay fp32 (raw ``_keep_in_fp32_modules_strict``).
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, lower_bound: float | None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lower_bound = lower_bound
        self.f_a_proj = Linear.Config(in_features=hidden_size, out_features=head_dim, bias=False).build()
        self.f_b_proj = Linear.Config(in_features=head_dim, out_features=num_heads * head_dim, bias=False).build()
        self.dt_bias = nn.Parameter(torch.empty(num_heads * head_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        gate = self.f_b_proj(self.f_a_proj(hidden_states))
        dt_bias = _local(self.dt_bias).view(1, 1, -1)
        g = (gate.float() + dt_bias).view(b, s, self.num_heads, self.head_dim)
        a_log = _local(self.A_log).view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(a_log)
        if self.lower_bound is not None:
            return self.lower_bound * torch.sigmoid(decay_rate * g)
        g = torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))
        return -decay_rate * g


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x / inv_norm


class GlmDeltaAttention(nn.Module):
    """KDA (Kimi Delta Attention) with GLM fused-qkv short conv.

    ``_chunk_kda`` is the production-kernel dispatch point (kimi_k3 family
    ``triton_ascend_kernels.chunk_kda``); the sequential reference below is the
    exact-math fallback for debug-scale real execution and unit tests.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        gate_lower_bound: float | None = -5.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        qkv_dim = num_heads * head_dim
        self.qkv_dim = qkv_dim

        self.q_proj = Linear.Config(in_features=hidden_size, out_features=qkv_dim, bias=False).build()
        self.k_proj = Linear.Config(in_features=hidden_size, out_features=qkv_dim, bias=False).build()
        self.v_proj = Linear.Config(in_features=hidden_size, out_features=qkv_dim, bias=False).build()
        self.conv1d = ShortConv1d(3 * qkv_dim, kernel_size=conv_kernel_size, activation="silu")
        self.forget_gate = GlmForgetGate(hidden_size, num_heads, head_dim, gate_lower_bound)
        self.b_proj = Linear.Config(in_features=hidden_size, out_features=num_heads, bias=False).build()
        self.g_a_proj = Linear.Config(in_features=hidden_size, out_features=head_dim, bias=False).build()
        self.g_b_proj = Linear.Config(in_features=head_dim, out_features=qkv_dim, bias=False).build()
        self.o_norm = RMSNormGated(head_dim, eps=norm_eps)
        self.o_proj = Linear.Config(in_features=qkv_dim, out_features=hidden_size, bias=False).build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_masks, positions
        b, s, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)],
            dim=-1,
        )
        mixed_qkv = self.conv1d(mixed_qkv)
        query, key, value = torch.split(mixed_qkv, [self.qkv_dim] * 3, dim=-1)
        query = query.view(b, s, h, d)
        key = key.view(b, s, h, d)
        value = value.view(b, s, h, d)

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        core_attn_out = self._chunk_kda(query, key, value, g, beta)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(b, s, h, d)
        output = self.o_norm(core_attn_out, gate).reshape(b, s, self.qkv_dim)
        return self.o_proj(output)

    def _chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Production-kernel dispatch (kimi_k3 ``chunk_kda`` family).

        The fused kernel receives the pre-computed gate ``g`` (decay/A_log/
        lower-bound already applied in ``GlmForgetGate``), so the kernel runs
        with ``use_gate_in_kernel=False`` and no safe-gate re-scaling. When
        ``triton_ascend_kernels`` is unavailable (CPU/debug environments) the
        exact-math sequential reference below is used and a one-time warning
        is logged; the simulator always replaces this seam with a shape-only
        shim recording the fused op name (``apply_glm5_next_shims``).
        """
        try:
            from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda
        except ImportError:
            if not self._warned_no_fused_kernel:
                logger.warning(
                    "triton_ascend_kernels is not installed; glm5_next KDA falls "
                    "back to the sequential reference implementation (debug-scale "
                    "only). The simulator records the fused op via its shape-only "
                    "shim instead."
                )
                self._warned_no_fused_kernel = True
            return self._sequential_kda(q, k, v, g, beta)

        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=None,
            dt_bias=None,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=False,
            lower_bound=None,
            transpose_state_layout=True,
            cu_seqlens=None,
        )
        return output

    @staticmethod
    def _sequential_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Token-recurrent KDA reference (raw ``recurrent_kimi_delta_attention``).

        fp32 state math, qk l2norm in-kernel semantics, no initial state
        (training path without cache).
        """
        initial_dtype = q.dtype
        q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
        q = l2norm(q)
        k = l2norm(k)
        scale = 1.0 / (q.shape[-1] ** 0.5)
        q = q * scale

        b, s, h, k_dim = k.shape
        v_dim = v.shape[-1]
        state = torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32, device=q.device)
        outputs = []
        for t in range(s):
            g_t = g[:, t].exp().unsqueeze(-1)  # [B, H, K, 1]
            k_t = k[:, t]  # [B, H, K]
            v_t = v[:, t]  # [B, H, V]
            b_t = beta[:, t].unsqueeze(-1)  # [B, H, 1]

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]
            delta = (v_t - kv_mem) * b_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            q_t = q[:, t]  # [B, H, K]
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))  # [B, H, V]
        return torch.stack(outputs, dim=1).to(initial_dtype)


class GlmDsaIndexer(nn.Module):
    """DeepSeek Sparse Attention indexer with k-pool compression.

    Scores compressed k-pool candidates, expands selected pools into raw token
    indices and appends the current incomplete tail pool. Output width is
    static: ``select_pools * kpool + (kpool - 1)``. The indexer is frozen
    (no gradients) matching the raw ``@torch.no_grad()`` reference.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        n_heads: int = 64,
        head_dim: int = 128,
        topk: int = 8192,
        kpool: int = 8,
        always_select_tail: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        self.kpool = kpool
        self.always_select_tail = always_select_tail
        self.softmax_scale = head_dim**-0.5

        self.wq_b = Linear.Config(in_features=q_lora_rank, out_features=n_heads * head_dim, bias=False).build()
        self.wk = Linear.Config(in_features=hidden_size, out_features=head_dim, bias=False).build()
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = Linear.Config(in_features=hidden_size, out_features=n_heads, bias=False).build()
        self.kpool_ape = nn.Parameter(torch.zeros(kpool, head_dim))
        self.kpool_gate = nn.Parameter(torch.zeros(head_dim, hidden_size))
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor, q_resid: torch.Tensor) -> torch.Tensor:
        """Return int64 top-k token indices ``[B, S, K]``; -1 marks invalid."""
        b, s, _ = hidden_states.shape
        hd, nh = self.head_dim, self.n_heads
        if s % self.kpool != 0:
            raise ValueError(
                f"DSA indexer requires seq_len={s} divisible by index_kpool={self.kpool}"
            )
        pools = s // self.kpool

        q = self.wq_b(q_resid).view(b, s, nh, hd)
        k = self.k_norm(self.wk(hidden_states))  # [B, S, hd]
        gate_scores = F.linear(hidden_states, _local(self.kpool_gate))  # [B, S, hd]

        # Pool compression: learn a weighted average over each complete pool.
        keys_r = k.view(b, pools, self.kpool, hd)
        logits = (gate_scores.view(b, pools, self.kpool, hd).float() + _local(self.kpool_ape).float()[None, None]).to(
            keys_r.dtype
        )
        probabilities = logits.softmax(dim=2)
        pool_keys = (probabilities * keys_r).sum(dim=2)  # [B, P, hd]

        scores = torch.matmul(q.float(), pool_keys.float().transpose(-1, -2).unsqueeze(1))  # [B, nh, S, P]
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(hidden_states).float() * (nh**-0.5)  # [B, S, nh]
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)  # [B, S, P]

        # Causality: a pool is selectable only if its last token <= query pos.
        positions = torch.arange(s, device=hidden_states.device)
        pool_end = positions.view(pools, self.kpool)[:, -1]  # [P]
        causal = pool_end[None, None, :] <= positions[None, :, None]  # [1, S, P]
        index_scores = index_scores.masked_fill(~causal, torch.finfo(index_scores.dtype).min)

        select_pools = min(self.topk // self.kpool, pools)
        selected = index_scores.topk(select_pools, dim=-1).indices  # [B, S, select_pools]

        pool_offsets = torch.arange(self.kpool, device=hidden_states.device)
        token_indices = selected.unsqueeze(-1) * self.kpool + pool_offsets  # [B, S, select_pools, kpool]
        token_valid = token_indices <= positions[None, :, None, None]
        token_indices = token_indices.masked_fill(~token_valid, -1)
        topk_indices = token_indices.flatten(-2)  # [B, S, select_pools * kpool]

        if self.always_select_tail:
            tail_width = self.kpool - 1
            tail_start = (positions + 1) - (positions + 1) % self.kpool  # [S]
            tail_indices = tail_start[:, None] + torch.arange(tail_width, device=hidden_states.device)[None, :]
            tail_indices = tail_indices[None].expand(b, s, tail_width)
            tail_valid = (tail_indices <= positions[None, :, None]) & (tail_indices < s)
            tail_indices = tail_indices.masked_fill(~tail_valid, -1)
            topk_indices = torch.cat([topk_indices, tail_indices], dim=-1)

        return topk_indices


class GlmDsaAttention(nn.Module):
    """NoPE MLA + k-pool DSA indexer with gather-based sparse attention.

    Training path has no KV cache: the sparse union over top-k indices is
    gathered from the same-sequence K/V, so every shape is static. This is
    mathematically equivalent to SDPA over an additive top-k mask; the fused
    production kernel replaces it at the same seam (MODEL_CONTRACT.md §11).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        indexer_heads: int = 64,
        indexer_head_dim: int = 128,
        index_topk: int = 8192,
        index_kpool: int = 8,
        index_kpool_always_select_tail: bool = True,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if qk_nope_head_dim != v_head_dim:
            raise ValueError(
                "glm5_next DSA requires qk_nope_head_dim == v_head_dim (kv_b_proj "
                "produces a fused k/v latent)"
            )
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.q_a_proj = Linear.Config(in_features=hidden_size, out_features=q_lora_rank, bias=False).build()
        self.q_a_norm = RMSNorm.Config(normalized_shape=q_lora_rank, eps=rms_norm_eps).build()
        self.q_b_proj = Linear.Config(in_features=q_lora_rank, out_features=num_heads * qk_nope_head_dim, bias=False).build()
        self.kv_a_proj_with_mqa = Linear.Config(in_features=hidden_size, out_features=kv_lora_rank, bias=False).build()
        self.kv_a_norm = RMSNorm.Config(normalized_shape=kv_lora_rank, eps=rms_norm_eps).build()
        self.kv_b_proj = Linear.Config(
            in_features=kv_lora_rank, out_features=num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        ).build()
        self.o_proj = Linear.Config(in_features=num_heads * v_head_dim, out_features=hidden_size, bias=False).build()

        self.indexer = GlmDsaIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            n_heads=indexer_heads,
            head_dim=indexer_head_dim,
            topk=index_topk,
            kpool=index_kpool,
            always_select_tail=index_kpool_always_select_tail,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_masks, positions
        b, s, _ = hidden_states.shape

        q_resid = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_resid).view(b, s, self.num_heads, self.qk_nope_head_dim)

        kv_pass = self.kv_a_norm(self.kv_a_proj_with_mqa(hidden_states))  # [B, S, kv_lora]
        kv = self.kv_b_proj(kv_pass).view(b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        topk_indices = self.indexer(hidden_states, q_resid)  # [B, S, K] int64
        valid = topk_indices.ge(0)
        safe_indices = topk_indices.clamp(0, s - 1)
        batch_idx = torch.arange(b, device=hidden_states.device)[:, None, None]

        key_sel = key[batch_idx, safe_indices]  # [B, S, K, H, D]
        value_sel = value[batch_idx, safe_indices]  # [B, S, K, H, D]

        attn_scores = torch.einsum("bshd,bskhd->bhsk", query, key_sel) * self.scaling
        attn_scores = attn_scores.masked_fill(
            ~valid[:, None], torch.finfo(attn_scores.dtype).min
        )
        attn_probs = attn_scores.float().softmax(dim=-1).to(query.dtype)
        attn_output = torch.einsum("bhsk,bskhd->bshd", attn_probs, value_sel)  # [B, S, H, D]

        return self.o_proj(attn_output.reshape(b, s, self.num_heads * self.v_head_dim))
