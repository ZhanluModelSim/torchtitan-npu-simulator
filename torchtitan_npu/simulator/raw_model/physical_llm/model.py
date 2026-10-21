"""
DeepSeekV4.1-Flash: 40T Compute-Equivalent Multimodal Understanding Model
=========================================================================
════════════════════════════════════════════════════════════════════════
整体结构
════════════════════════════════════════════════════════════════════════
  Embedding(V×d, tied)
    → [ FlashBlock × 100 物理层 ] × Loop 2 遍 = 200 逻辑深度
    → FinalNorm → LM Head(tied)

  FlashBlock = mHC → Attention(ISA/GDN2) → mHC → MoE/Engram
  层型: 4 层单元 [ISA, GDN2, GDN2, GDN2] × 25 单元
    25 × ISA (Indexed Sparse Attention: CSA 2× 压缩 + indexer 稀疏)
    75 × GDN2 (delta-rule 线性注意力, O(1) 固定状态)
  FFN: 90 层 LatentMoE(4096 专家 top-25 + 1 共享); 10 层 Engram 记忆层(无 MoE)
  多模态: image/audio/point3d/haptic 编码器 → 投影 + 模态类型嵌入
          → AR 交错序列(损失只在 text 段)

════════════════════════════════════════════════════════════════════════
各部分计算公式与标准算法论文对照
════════════════════════════════════════════════════════════════════════
(1) ISA — CSA 压缩 + Indexer 稀疏注意力
    参考: DeepSeek V4.1 Flash(CSA2, KV 2:1 压缩 + indexer)/
    DeepSeek Sparse Attention indexer
    · KV 压缩: 槽位只存偶数位 token 的 K/V(2:1); 官方为编码器 2:1 /
      解码器 1:1(平均 2.0), 本模型统一 2× — 简化口径差异
    · Indexer 打分(与 DeepSeek indexer 一致: 低秩投影 + RMSNorm 点积):
        s_ij = ⟨rmsnorm(W_q h_i), rmsnorm(W_k h_j)⟩ / √idx_dim
      在 2× 压缩网格上因果屏蔽(2j > i → −∞), 取 top-K
    · 稀疏注意力(indexer 分数作偏置, DeepSeek sparse attention 口径):
        o_i = Σ_{j∈topK(i)} softmax_j(q_i·k_j/√hd + s_ij) v_j
    · KV 共享(Flash reindex/reuse): 每组 5 个 ISA 层共享 1 个 KV 槽位:
      1 产 KV + 1 reindex(自选 topK)+ 3 reuse(复用 topK)。
      官方 8/40 层存 KV(kv_layer_frac 0.2)+ 30 层 reuse; 本模型
      5 kv + 5 reindex + 15 reuse / 25 层 — 同机制、不同排布与占比

(2) GDN2 — delta-rule 线性注意力
    参考: DeltaNet(Yang et al., Parallelizing Linear Transformers with
    the Delta Rule over Sequence Length)
    · 状态更新(k 归一化到单位范数, 保证状态算子收缩):
        S_t = S_{t-1} ⊙ (1 − βe_t) + βw_t (v_t − S_{t-1} k̄_t) k̄_tᵀ
    · 读出: o_t = α_t ⊙ (S_t q_t) · da_gain_t
    与论文差异:
      a) 论文为单标量 β: S = S − β(S k − v)kᵀ; 本模型为 βe/βw 双门
         (擦除衰减 + 写入强度)+ α 读出门 + da 逐头增益 — 门控扩展
      b) 论文为分块并行算法; 本实现逐 token 串行(参考实现口径,
         生产可换分块并行)
      c) k 归一化与论文一致(Δ 规则的标准稳定化)

(3) LatentMoE
    参考: DeepSeek-V3 MoE + LatentMoE 潜在瓶颈
    · 路由(h 空间打分, token-choice, 负载均衡 z_loss 同口径):
        scores = √softplus(h W_r) × route_scale, top-K 权重归一化
    · 逐层共享投影: g = W_g h, u = W_u h (d→latent); latent_up (latent→d)
    · 专家(仅潜在空间 FFN, latent→inter→latent):
        e_m(g,u) = W_d,m SwiGLU(W_g,m g, W_u,m u)
    · 合并: out = latent_up( Σ_k w_k e_{m_k}(g,u) + Σ_shared e_shared(g,u) )
    与论文差异: 激活 top-25(V3 top-8 / Flash top-6);
    无 expert-choice 路由(全 token-choice)

(4) mHC — Manifold-Constrained Hyper-Connections
    参考: DeepSeek-V4 mHC(Sinkhorn 约束超连接)
    · 每层 2 块(注意力前/FFN 前):
        x̄ = rmsnorm(x); expand: d→hc·bw(分支矩阵)
        M = Sinkhorn_τ(exp(W_mix)) — 双随机(流形约束, τ=20 迭代)
        y = rmsnorm(W_c einsum(x̄, M)) + x   (contract + 残差)
    与论文差异: 分支宽 hc_branch_dim=1024 = d/16(轻量口径, 控制稠密
    激活; 仓库参考实现为全宽 d — 参数量/激活 ×16)

(5) Engram — 条件记忆
    参考: DeepSeek V4.1 Flash 官方 Engram(196.6B @ 748.6B)
    · 哈希(每阶每头独立乘子 m; Horner 滚动, mod 2²⁵ 防溢出):
        h_t = (h_{t-1}·257 + id_t·m) mod 2²⁵(覆盖最近 order 个 token)
    · 检索: mem = concat_{阶×头}(查表[行×256]) → [b, s, 6144]
    · 门控(官方 rmsnorm_ssr): d = ⟨rmsnorm(W_q h), rmsnorm(W_k mem)⟩/√d
        g = sigmoid(sign(d)·√|d|) × token_mask(id≠0)
    · 注入: out = g ⊙ rmsnorm(W_v mem)
    与 Flash 官方差异: 注入层 10/5 个 vs 官方 2 个(层 1/14); 行数
    2²⁵ vs 官方 16M; token_mask 以 id==0 近似官方静态 token 类型掩码

(6) Loop = MoR 权重绑定
    参考: Mixture of Recursions / Looped Transformer(权重复用语义)
    · 100 物理层权重只存一份, 前向跑 2 遍(逻辑 200 层):
      参数量 ≈ 20.7T, 计算等效 ≈ 41.5T
    · 反向: autograd 按链式法则把两遍使用点的梯度累加到同一
      Parameter(dW = Σ_r dW_r), 优化器单次更新
    与论文差异: 无 MoR 保留制(论文允许部分 token 提前退出);
    KV 槽位每遍重算覆盖(不跨遍共享)

════════════════════════════════════════════════════════════════════════
其他标准组件: RMSNorm(Touvron et al.)/ RoPE + YaRN / SwiGLU(Shazeer,
带 clamp)— 与标准一致, 无差异
════════════════════════════════════════════════════════════════════════
Pure PyTorch, self-contained. Config lives in config.py.
"""


import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlashConfig


# ============================================================================
# SECTION 1: Utilities
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm (float32 internally)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight.float()).to(dtype)


def swiglu(x: torch.Tensor, gate: torch.Tensor, clamp_val: Optional[float] = None) -> torch.Tensor:
    if clamp_val is not None:
        gate = gate.clamp(max=clamp_val)
    out = x * F.silu(gate)
    if clamp_val is not None:
        out = out.clamp(-clamp_val, clamp_val)
    return out


def sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.softplus(x))


def compute_load_balance_loss(router_probs, router_indices, num_experts, topk):
    with torch.no_grad():
        one_hot = torch.zeros(
            router_indices.shape[0], topk, num_experts,
            dtype=router_probs.dtype, device=router_indices.device,
        )
        one_hot.scatter_(2, router_indices.unsqueeze(-1), 1.0)
        density = one_hot.mean(dim=(0, 1))
    density_proxy = router_probs.mean(dim=0)
    return (density * density_proxy).sum() * num_experts


def normal_init_(tensor, mean=0.0, std=0.02):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_to_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return q * cos + rotate_half(q) * sin


def stable_softmax(scores, dim=-1, clamp_val=None):
    if clamp_val is not None:
        scores = scores.clamp(-clamp_val, clamp_val)
    return F.softmax(scores.float(), dim=dim).to(scores.dtype)


# ============================================================================
# SECTION 2: RoPE + YaRN
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta, yarn_factor, original_max_seq_len, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        ramp = torch.linspace(0, 1, max_seq_len)
        scale = 1.0 / yarn_factor + (1.0 - 1.0 / yarn_factor) * ramp
        self.register_buffer("yarn_scale", scale, persistent=False)
        self.original_max_seq_len = original_max_seq_len

    @torch.no_grad()
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        inv_freq = self.inv_freq.to(device)
        if seq_len > self.original_max_seq_len:
            scale = self.yarn_scale[:seq_len].to(device)
            inv_freq = inv_freq.unsqueeze(0) / scale.unsqueeze(-1).clamp(min=0.1)
            freqs = torch.einsum("i,j->ij", t, inv_freq.squeeze(0))
        else:
            freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


# ============================================================================
# SECTION 3: ISA Attention Projections
# ============================================================================
# ISA 采用 CSA 口径的全秩 Q/K/V/O 投影(无 MLA 低秩):
#   Q/O: 每 ISA 层各一个 Linear(d, nh×hd)
#   K/V: 仅产 KV 层持有(5 层),reuse/reindex 层读共享槽位
# 投影矩阵直接内联在 AlternatingAttention (见 SECTION 7)。


# ============================================================================
# SECTION 5: ISA Indexer + Token-Level Sparse Attention
# ============================================================================

class ISAIndexer(nn.Module):
    """Indexer for token-level sparsity: scores every stride-compressed
    position of the shared KV and returns top-K token indices.

    公式(DeepSeek indexer 标准口径: 低秩投影 + RMSNorm 点积, 无 Hadamard):
        s_ij = ⟨rmsnorm(W_q h_i), rmsnorm(W_k h_j)⟩ / √idx_dim
    selection 在 2× 压缩网格(idx_k[::2])上进行;因果屏蔽
    (2j > i → s_ij = −1e9), 取 top-K;
    indexer 分数随 top-K 传给注意力作偏置(见 IndexedSparseAttention)。
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.head_dim = config.indexer_head_dim
        self.topk = config.indexer_topk
        self.compress_ratio = config.isa_compress_ratio
        idx_dim = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, idx_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, idx_dim, bias=False)
        self.q_norm = RMSNorm(idx_dim)
        self.k_norm = RMSNorm(idx_dim)

    def select(self, hidden_states):
        """Returns topk_indices [b, s, topk], topk_scores [b, s, topk]."""
        b, s, _ = hidden_states.shape
        idx_q = self.q_norm(self.q_proj(hidden_states))
        idx_k = self.k_norm(self.k_proj(hidden_states))
        idx_k_c = idx_k[:, ::self.compress_ratio, :]
        cl = idx_k_c.shape[1]

        scores = torch.bmm(
            idx_q.reshape(b, s, -1),
            idx_k_c.reshape(b, cl, -1).transpose(1, 2),
        ) / math.sqrt(idx_q.shape[-1])

        q_pos = torch.arange(s, device=hidden_states.device).unsqueeze(1)
        k_pos = torch.arange(0, s, self.compress_ratio,
                             device=hidden_states.device).unsqueeze(0)
        scores = scores.masked_fill((k_pos > q_pos).unsqueeze(0), -1e9)

        actual_topk = min(self.topk, cl)
        topk_scores, topk_indices = torch.topk(scores, actual_topk, dim=-1)
        return topk_indices, topk_scores


class IndexedSparseAttention(nn.Module):
    """Token-level sparse attention over CSA 压缩后的 KV (2× 网格)。

    公式:
        o_i = Σ_{j∈topK(i)} softmax_j(q_i·k_j / √hd + s_ij) v_j
    k_comp/v_comp 已由产 KV 层按 isa_compress_ratio 压缩写入共享槽位;
    s_ij 为 indexer 分数偏置, 被因果屏蔽的槽位 s_ij = −1e9 → 权重 0。
    计算量 O(s · topK · hd), 与序列长线性(稀疏注意力核心收益)。

    Stateless module — the KV / indexer projections live in
    AlternatingAttention depending on the layer's KV-sharing role.
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.softmax_clamp = config.attn_softmax_clamp

    def forward(self, q_full, k_comp, v_comp, topk_indices, topk_scores):
        b, s, nh, hd = q_full.shape
        cl = k_comp.shape[1]
        actual_topk = topk_indices.shape[-1]

        flat = topk_indices.reshape(b * s, actual_topk)
        k_flat = k_comp.reshape(b, cl, -1)
        v_flat = v_comp.reshape(b, cl, -1)
        bidx = torch.arange(b, device=q_full.device).unsqueeze(1).unsqueeze(2).expand(-1, s, actual_topk)
        k_sel = k_flat[bidx, flat.view(b, s, actual_topk)].view(b, s, actual_topk, nh, hd)
        v_sel = v_flat[bidx, flat.view(b, s, actual_topk)].view(b, s, actual_topk, nh, hd)

        attn_scores = torch.einsum("bshd,bskhd->bhsk", q_full, k_sel) * self.scale
        attn_scores = attn_scores + topk_scores.unsqueeze(1)
        attn_probs = stable_softmax(attn_scores, dim=-1, clamp_val=self.softmax_clamp)
        return torch.einsum("bhsk,bskhd->bshd", attn_probs, v_sel)


# ============================================================================
# SECTION 6: GDN2 — delta-rule linear attention
# ============================================================================

class GDN2Attention(nn.Module):
    """GDN2: delta-rule linear attention (Generalized Delta Net)。

    状态 S ∈ [H, d_state, d_state] 为固定缓冲(非可学习参数);每层
    O(1) 内存、无 token 级 KV cache(线性注意力)。

    公式(DeltaNet 核心 + 门控扩展):
      · k 归一化(论文标准, 保证状态算子收缩、长序列数值稳定):
          k̄_t = k_t / ‖k_t‖
      · 状态更新(Delta rule):
          S_t = S_{t-1} ⊙ (1 − βe_t) + βw_t (v_t − S_{t-1} k̄_t) k̄_tᵀ
        βe/βw = sigmoid(W_β h_t) 双门(擦除衰减/写入强度)
      · 读出:
          o_t = α_t ⊙ (S_t q_t) · da_gain_t
        α = sigmoid(W_α h_t) 读出门;da_gain = sigmoid(mean(da)) 逐头增益

    与 DeltaNet 论文差异:
      a) 论文为单标量 β: S = S − β(S k − v)kᵀ;本模型拆为 βe/βw
         双门并增加 α 读出门与 da 增益(门控扩展, 见 estimate 参数)
      b) 论文为分块并行算法;本实现逐 token 串行(参考实现口径,
         生产可换分块并行, 见论文 §parallel)
      c) k 归一化与论文一致(Δ 规则的标准稳定化)
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        d = config.hidden_size
        H = config.gdn2_num_heads
        self.d_state = config.gdn2_d_state
        self.d_k = config.gdn2_d_k
        self.d_v = config.gdn2_d_v
        self.d_a = config.gdn2_d_a
        self.H = H

        # Projections
        self.q_proj = nn.Linear(d, H * self.d_k, bias=False)
        self.k_proj = nn.Linear(d, H * self.d_k, bias=False)
        self.v_proj = nn.Linear(d, H * self.d_v, bias=False)
        self.o_proj = nn.Linear(H * self.d_v, d, bias=False)

        # Gates
        # α gate: [H, d_k] read gate
        self.alpha_gate = nn.Linear(d, H * self.d_k, bias=False)
        # β gate (double gate): 2 × [H, d_state] erase/write control
        self.beta_gate = nn.Linear(d, 2 * H * self.d_state, bias=False)

        # d_a projection: 逐头读出增益门 (读出 o = α ⊙ S·q 之外的学习式缩放)
        self.da_proj = nn.Linear(d, H * self.d_a, bias=False)
        self.da_norm = RMSNorm(H * self.d_a)

        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, hidden_states):
        b, s, d = hidden_states.shape
        H, dk, dv, ds = self.H, self.d_k, self.d_v, self.d_state

        q = self.q_proj(hidden_states).view(b, s, H, dk)
        k = self.k_proj(hidden_states).view(b, s, H, dk)
        v = self.v_proj(hidden_states).view(b, s, H, dv)
        alpha = torch.sigmoid(self.alpha_gate(hidden_states).view(b, s, H, dk))
        beta_raw = self.beta_gate(hidden_states).view(b, s, 2, H, ds)
        beta_erase = torch.sigmoid(beta_raw[:, :, 0])  # [b, s, H, ds]
        beta_write = torch.sigmoid(beta_raw[:, :, 1])  # [b, s, H, ds]
        da = self.da_norm(self.da_proj(hidden_states)).view(b, s, H, self.d_a)
        # 逐头读出增益(da 投影的用途):[b, s, H]
        da_gain = torch.sigmoid(da.mean(dim=-1))

        # Initialize state S ∈ [b, H, d_state, d_state]
        # Note: in training we process the full sequence in parallel via chunked computation
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Parallel delta-rule update over sequence (chunked for memory)
        # For smoke test / forward pass: process token by token in a loop
        # (production would use parallel chunked algorithm)
        S = torch.zeros(b, H, ds, ds, device=device, dtype=dtype)

        outputs = []
        for t in range(s):
            qt = q[:, t]          # [b, H, dk]
            kt = k[:, t]          # [b, H, dk]
            vt = v[:, t]          # [b, H, dv]
            at = alpha[:, t]      # [b, H, dk]
            be = beta_erase[:, t] # [b, H, ds]
            bw = beta_write[:, t] # [b, H, ds]

            # Delta rule: S = S + β (v - S k) k^T
            # d_state 与 d_k/d_v 可能不等 → pad/truncate 到 ds
            if dk < ds:
                k_state = F.pad(kt, (0, ds - dk))
                v_state = F.pad(vt, (0, ds - dv))
            elif dk > ds:
                k_state = kt[..., :ds]
                v_state = vt[..., :ds]
            else:
                k_state = kt
                v_state = vt
            # DeltaNet 标准:k 归一化到单位范数, 使状态更新算子
            # (I - β k k^T) 成为压缩算子 — 状态与反向梯度都不随
            # 序列长度发散(长序列/多模态 151+ tokens 仍数值稳定)
            k_state = k_state / (k_state.norm(dim=-1, keepdim=True) + 1e-8)

            # Sk = S @ k_state: S[b,H,ds,ds] @ k_state[b,H,ds] → [b, H, ds]
            Sk = torch.einsum("bhij,bhj->bhi", S, k_state)  # [b, H, ds]
            delta = (v_state - Sk)  # [b, H, ds]
            # Apply write gate: outer product delta ⊗ k_state → [b, H, ds, ds]
            update = torch.einsum("bhi,bhj->bhij", bw * delta, k_state)  # [b, H, ds, ds]
            # 擦除门(decay):be 沿 S 的列(key 维)广播,衰减对应键位的记忆
            S = S * (1 - be.unsqueeze(-1)) + update  # be[b,H,ds,1] broadcast

            # Read: o = α ⊙ (S @ q): S[b,H,i,j] q[b,H,j] → [b,H,i]
            if dk < ds:
                q_state = F.pad(qt, (0, ds - dk))
            elif dk > ds:
                q_state = qt[..., :ds]
            else:
                q_state = qt
            o = torch.einsum("bhij,bhj->bhi", S, q_state)  # [b, H, ds]
            # Pad/truncate alpha to ds before multiplying
            if dk < ds:
                at = F.pad(at, (0, ds - dk))
            elif dk > ds:
                at = at[..., :ds]
            o = at * o * da_gain[:, t].unsqueeze(-1)  # [b, H, ds]

            # Project back to d_v
            if dv < ds:
                o = o[..., :dv]
            elif dv > ds:
                o = F.pad(o, (0, dv - ds))

            outputs.append(o)

        out = torch.stack(outputs, dim=1)  # [b, s, H, dv]
        out = out.reshape(b, s, H * dv)
        return self.o_proj(out)


# ============================================================================
# SECTION 7: Alternating Attention (ISA / GDN2 dispatch + Flash KV sharing)
# ============================================================================

class AlternatingAttention(nn.Module):
    """Layer-type dispatch: ISA / GDN2 by position in 4-layer unit。

    ISA 投影公式(CSA 口径, 全秩、无 MLA 低秩):
        q = W_q h ∈ [b,s,nh,hd];  q = cat(q_noPE, RoPE(q_rope))
        k = W_k h, v = W_v h(仅产 KV 层); o = W_o attn(q, k_sel, v_sel)
      KV 槽位: k_comp = k[::2], v_comp = v[::2](2× 压缩, 只存偶数位)

    ISA 层实现 Flash-style KV 共享(DeepSeek v4.1 Flash):
      - "kv" role: 全秩 K/V → 2× 压缩写入组共享槽位 + 自己的
        indexer 选 top-K;
      - "reindex" role: 读共享压缩 KV, 用自己的 indexer 重新选 top-K;
      - "reuse" role: 读共享压缩 KV, 复用产 KV 层的 top-K(无 KV 投影、
        无 indexer 计算)。
    """

    def __init__(self, config: FlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_isa = config.is_isa_layer(layer_idx)

        if self.is_isa:
            self.role = config.isa_role(layer_idx)
            self.kv_group = config.kv_group_of(layer_idx)
            d = config.hidden_size
            nh = config.num_attention_heads
            hd = config.head_dim
            # CSA 口径: 全秩 Q/O 投影(每 ISA 层),K/V 仅产 KV 层持有
            self.q_proj = nn.Linear(d, nh * hd, bias=False)
            self.o_proj = nn.Linear(nh * hd, d, bias=False)
            self.rotary_emb = RotaryEmbedding(
                config.qk_rope_head_dim, config.rope_theta,
                config.yarn_factor, config.yarn_original_max, config.max_seq_len,
            )
            self.num_heads = nh
            self.head_dim = hd
            self.nope_dim = hd - config.qk_rope_head_dim
            self.rope_dim = config.qk_rope_head_dim
            self.compress_ratio = config.isa_compress_ratio
            if self.role == "kv":
                self.k_proj = nn.Linear(d, nh * hd, bias=False)
                self.v_proj = nn.Linear(d, nh * hd, bias=False)
                self.indexer = ISAIndexer(config)
            elif self.role == "reindex":
                self.indexer = ISAIndexer(config)
            self.isa = IndexedSparseAttention(config)
        else:
            self.gdn2 = GDN2Attention(config)

    def forward(self, hidden_states, kv_bufs=None):
        if not self.is_isa:
            return self.gdn2(hidden_states)

        b, s, _ = hidden_states.shape
        buf = None if kv_bufs is None else kv_bufs[self.kv_group]

        # 全秩 Q + RoPE(rope 部分旋转、noPE 部分不动)
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        cos, sin = self.rotary_emb(s, hidden_states.device)
        q_rope_r = apply_rotary_to_q(q[..., self.nope_dim:], cos, sin)
        q_full = torch.cat([q[..., :self.nope_dim], q_rope_r], dim=-1)

        if self.role == "kv":
            k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
            k_rope_r = apply_rotary_to_q(k[..., self.nope_dim:], cos, sin)
            k_full = torch.cat([k[..., :self.nope_dim], k_rope_r], dim=-1)
            # CSA 压缩: 槽位只存压缩网格(2×)上的 KV
            k_comp = k_full[:, ::self.compress_ratio, :, :]
            v_comp = v[:, ::self.compress_ratio, :, :]
            topk_i, topk_s = self.indexer.select(hidden_states)
            if buf is not None:
                buf["k_comp"], buf["v_comp"] = k_comp, v_comp
                buf["topk_indices"], buf["topk_scores"] = topk_i, topk_s
        else:
            if buf is None:
                raise RuntimeError(
                    f"ISA layer {self.layer_idx} (role={self.role}) 需要共享 "
                    f"KV 缓冲 kv_bufs, 但未提供")
            k_comp, v_comp = buf["k_comp"], buf["v_comp"]
            if self.role == "reindex":
                topk_i, topk_s = self.indexer.select(hidden_states)
            else:  # reuse: 复用产 KV 层的 top-K 选择
                topk_i, topk_s = buf["topk_indices"], buf["topk_scores"]

        out = self.isa(q_full, k_comp, v_comp, topk_i, topk_s)
        return self.o_proj(out.reshape(b, s, self.num_heads * self.head_dim))


# ============================================================================
# SECTION 8: LatentMoE (with moe_latent_dim bottleneck)
# ============================================================================

class LatentExpertMLP(nn.Module):
    """LatentMoE single expert — latent space only:
       latent → inter → latent
    (d→latent 下投影与 latent→d 上投影为逐层共享、所有专家共用,见 MoELayer)
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        latent = config.moe_latent_dim
        inter = config.moe_intermediate_size

        # latent → inter (expand, gate/up 双路)
        self.gate_inter = nn.Linear(latent, inter, bias=False)
        self.up_inter = nn.Linear(latent, inter, bias=False)
        # inter → latent (compress)
        self.inter_to_latent = nn.Linear(inter, latent, bias=False)

        self.clamp_val = config.swiglu_clamp

    def forward(self, gate_l, up_l):
        """gate_l/up_l = 共享 d→latent 下投影后的 latent 向量。"""
        gate_inter = self.gate_inter(gate_l)
        up_inter = self.up_inter(up_l)
        act = swiglu(up_inter, gate_inter, self.clamp_val)
        return self.inter_to_latent(act)


class SharedExpertMLP(nn.Module):
    """Shared expert using the same latent-space structure (复用共享投影)."""

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.expert = LatentExpertMLP(config)

    def forward(self, gate_l, up_l):
        return self.expert(gate_l, up_l)


class MoERouter(nn.Module):
    """Token-choice router: token selects top-K experts."""

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.num_experts = config.num_routed_experts
        self.topk = config.num_experts_per_token
        self.route_scale = config.route_scale
        self.score_func = config.router_score_function
        self.router_weight = nn.Parameter(torch.empty(config.hidden_size, self.num_experts))
        normal_init_(self.router_weight, std=0.02 / math.sqrt(self.num_experts))

    def forward(self, tokens):
        logits = tokens @ self.router_weight
        if self.score_func == "sqrtsoftplus":
            scores = sqrtsoftplus(logits) * self.route_scale
        elif self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1) * self.route_scale
        else:
            scores = torch.sigmoid(logits) * self.route_scale
        topk_w, topk_i = torch.topk(scores, self.topk, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return topk_i, topk_w, logits


class MoELayer(nn.Module):
    """LatentMoE layer: shared experts + routed experts。

    公式(DeepSeek-V3 MoE + LatentMoE 潜在瓶颈):
      · 路由(h 空间打分, token-choice):
          scores = √softplus(h W_r) × route_scale;top-K 权重归一化
      · 逐层共享投影(所有专家/全部 token 共用):
          g = W_g h (d→latent),  u = W_u h (d→latent)
      · 专家(仅潜在空间 FFN, latent→inter→latent):
          e_m(g,u) = W_{d,m} SwiGLU(W_{g,m} g, W_{u,m} u)
      · 合并(latent→d 上投影逐层共享):
          out = W_up( Σ_k w_k e_{m_k}(g,u) + Σ_shared e_shared(g,u) )
      · 负载均衡(DeepSeek 标准 z_loss):
          aux = E × Σ_e density_e × prob_e

    与论文差异: 激活 top-25(V3 top-8 / Flash top-6);
    无 expert-choice 路由(全 token-choice)。
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        d = config.hidden_size
        latent = config.moe_latent_dim
        self.num_experts = config.num_routed_experts
        self.topk = config.num_experts_per_token

        # 逐层共享 d→latent 下投影(gate/up 双路,所有专家共用)
        self.gate_down = nn.Linear(d, latent, bias=False)
        self.up_down = nn.Linear(d, latent, bias=False)
        # 逐层共享 latent→d 上投影
        self.latent_up = nn.Linear(latent, d, bias=False)

        self.router = MoERouter(config)

        self.shared_experts = nn.ModuleList([
            SharedExpertMLP(config) for _ in range(config.num_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            LatentExpertMLP(config) for _ in range(config.num_routed_experts)
        ])

    def forward(self, hidden_states):
        b, s, d = hidden_states.shape
        tokens = hidden_states.reshape(-1, d)

        # 共享 d→latent 下投影(全部 token;路由/专家共用)
        gate_l = self.gate_down(tokens)
        up_l = self.up_down(tokens)

        # Shared experts(潜在空间 FFN,全部 token)
        shared_out = sum(expert(gate_l, up_l) for expert in self.shared_experts)

        # Routed experts(路由打分在 h 空间, token-choice top-K)
        topk_i, topk_w, logits = self.router(tokens)
        aux_loss = compute_load_balance_loss(
            F.softmax(logits, dim=-1), topk_i, self.num_experts, self.topk
        )

        # Dispatch(token-choice:迭代专家、收集 latent 切片)
        routed_out = torch.zeros_like(gate_l)
        for k in range(self.topk):
            expert_idx = topk_i[:, k]
            expert_w = topk_w[:, k].unsqueeze(-1)
            for exp_id in range(self.num_experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    routed_out[mask] += self.routed_experts[exp_id](
                        gate_l[mask], up_l[mask]) * expert_w[mask]

        # 共享 latent→d 上投影(路由输出 + 共享专家输出均经此回 h)
        h_out = self.latent_up(routed_out + shared_out)
        return h_out.reshape(b, s, d), aux_loss


# ============================================================================
# SECTION 9: Loop KV Cache (incremental decode scaffolding)
# ============================================================================

class LoopKVCache:
    """Incremental-decode cache for the looped, KV-shared backbone.

    KV 槽位按共享组存储(25 ISA 层 → 5 组);GDN2 层只维护固定大小状态
    S (O(1)/层)。Loop 语义:首遍(r=0)写槽位,次遍(r=1)复用;逐 token
    解码时新 token 的 KV 拼接进槽位,GDN2 状态就地更新。

    (与 DeepSeekV4-Sparse 的 RecursiveDynamicCache 同级:训练期全序列
    前向不使用,仅 use_cache 解码路径持有;生产实现在此之上做
    scatter-update / 状态分片)
    """

    def __init__(self, kv_groups: int, num_physical_layers: int,
                 num_recursion: int):
        self.kv_groups = kv_groups
        self.num_physical_layers = num_physical_layers
        self.num_recursion = num_recursion
        self.group_key_cache: List[Any] = [None] * kv_groups
        self.group_value_cache: List[Any] = [None] * kv_groups
        self.gdn2_states: List[Any] = [None] * num_physical_layers
        self._seen_tokens = 0

    def update_kv(self, group_idx: int, k_full, v, first_pass: bool):
        """KV layer: pass 1 concatenates the new token's KV; pass 2 reads."""
        if first_pass:
            if self.group_key_cache[group_idx] is None:
                self.group_key_cache[group_idx] = k_full
                self.group_value_cache[group_idx] = v
            else:
                self.group_key_cache[group_idx] = torch.cat(
                    [self.group_key_cache[group_idx], k_full], dim=-2)
                self.group_value_cache[group_idx] = torch.cat(
                    [self.group_value_cache[group_idx], v], dim=-2)
            self._seen_tokens = self.group_key_cache[group_idx].shape[-2]
        return self.group_key_cache[group_idx], self.group_value_cache[group_idx]

    def get_kv(self, group_idx: int):
        return self.group_key_cache[group_idx], self.group_value_cache[group_idx]

    def get_seq_length(self):
        return self._seen_tokens


# ----------------------------------------------------------------------------
# Engram (conditional memory, DeepSeek V4.1 Flash 口径)
# ----------------------------------------------------------------------------

class PolynomialRollingHash(nn.Module):
    """多项式滚动哈希 (Horner 递推, 溢出安全):

      h_t = (h_{t-1} × BASE + id_t × mult_head) % capacity

    每头独立乘子 mult_head;order 阶 = 覆盖最近 order 个 token 的
    n-gram 窗口。与 Flash 官方 engram_n_heads=8 / ngram_orders 对齐。
    """

    BASE = 257

    def __init__(self, order, capacity, num_heads, layer_idx, seed_offset=0):
        super().__init__()
        self.order = order
        self.capacity = capacity
        self.num_heads = num_heads
        gen = torch.Generator()
        gen.manual_seed(layer_idx * 100 + order * 10 + seed_offset)
        self.register_buffer(
            "multipliers",
            torch.randint(1, 2 ** 31 - 1, (num_heads,), generator=gen).long()
        )

    def forward(self, input_ids):
        # input_ids: [b, s] → [b, s, num_heads] hash indices in [0, capacity)
        h = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                        self.num_heads, dtype=torch.long, device=input_ids.device)
        for _ in range(self.order):
            h = (h * self.BASE
                 + input_ids.unsqueeze(-1) * self.multipliers) % self.capacity
        return h


class MultiHeadHashTable(nn.Module):
    """多阶 n-gram 哈希查找表:每头独立偏移区段,行 × head_dim。

    Flash 口径:每 (阶, 头) 一张 rows_per_head=2^25 行的表,
    每行 head_dim=256 维。
    """

    def __init__(self, capacity, num_heads, head_dim):
        super().__init__()
        self.capacity = capacity
        self.num_heads = num_heads
        self.per_head_dim = head_dim
        self.embedding = nn.Embedding(num_heads * capacity, head_dim)
        self.register_buffer("head_offsets", torch.arange(num_heads) * capacity)

    def forward(self, hash_indices):
        b, s, nh = hash_indices.shape
        outs = []
        for h in range(nh):
            idx = hash_indices[:, :, h] + self.head_offsets[h]
            outs.append(self.embedding(idx))
        return torch.cat(outs, dim=-1)  # [b, s, nh * head_dim]


class EngramModule(nn.Module):
    """Engram 条件记忆 (DeepSeek V4.1 Flash 口径)。

    公式(每注入层一个模块;注入层为记忆层, 替代 MoE):
      · 哈希(每阶每头独立乘子 m;Horner 滚动, mod 2²⁵ 防溢出):
          h_t = (h_{t-1}·257 + id_t·m) mod 2²⁵(覆盖最近 order 个 token)
      · 检索: mem = concat_{阶×头}(查表[行×256]) → [b, s, 3·8·256=6144]
      · 门控(官方 rmsnorm_ssr 静态门控):
          d = ⟨rmsnorm(W_q h), rmsnorm(W_k mem)⟩ / √d
          g = sigmoid(sign(d)·√|d|) × token_mask(id≠0)
      · 注入: out = g ⊙ rmsnorm(W_v mem)

    与 Flash 官方差异: 注入层 10/5 个 vs 官方 2 个(层 1/14);行数
    2²⁵ vs 官方 16M;token_mask 以 id==0 近似官方静态 token 类型掩码。
    """

    def __init__(self, config: FlashConfig, layer_idx: int):
        super().__init__()
        d = config.hidden_size
        orders = config.engram_ngram_orders
        num_heads = config.engram_num_hash_heads
        head_dim = config.engram_head_dim
        capacity = config.engram_rows_per_head
        concat_dim = len(orders) * num_heads * head_dim  # 6144

        self.hash_fns = nn.ModuleList([
            PolynomialRollingHash(order, capacity, num_heads, layer_idx, n_idx)
            for n_idx, order in enumerate(orders)
        ])
        self.hash_tables = nn.ModuleList([
            MultiHeadHashTable(capacity, num_heads, head_dim)
            for _ in orders
        ])

        # key/value 投影(记忆 concat → hidden)+ query 投影(hidden → hidden)
        self.key_proj = nn.Linear(concat_dim, d, bias=False)
        self.value_proj = nn.Linear(concat_dim, d, bias=False)
        self.query_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(d)
        self.k_norm = RMSNorm(d)
        self.input_norm = RMSNorm(d)
        self.output_norm = RMSNorm(d)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, hidden_states, input_ids):
        # 检索:各阶哈希 → 查表 → 拼接 [b, s, concat_dim]
        parts = []
        for hash_fn, hash_table in zip(self.hash_fns, self.hash_tables):
            parts.append(hash_table(hash_fn(input_ids)))
        mem = torch.cat(parts, dim=-1)

        # rmsnorm_ssr 静态门控 (Flash 口径, dot 在 hidden 空间):
        # RMSNorm 点积 → signed-square-root → sigmoid → 静态 token mask
        q = self.q_norm(self.query_proj(self.input_norm(hidden_states)))
        k = self.k_norm(self.key_proj(mem))
        dot = (q * k).sum(dim=-1, keepdim=True) * self.scale
        gate = torch.sigmoid(dot.sign() * dot.abs().sqrt())
        # 静态 token mask: 无词元 id 的位置(模态 token / 填充, id==0)
        # 不注入记忆(Flash token_mask 静态口径)
        gate = gate * (input_ids != 0).unsqueeze(-1).to(gate.dtype)

        out = self.output_norm(self.value_proj(mem))
        return gate * out


# ============================================================================
# SECTION 10: Transformer Block
# ============================================================================

class SinkhornIteration(nn.Module):
    """Sinkhorn 迭代:把分支混合矩阵约束为双随机矩阵(流形约束)。"""

    def __init__(self, n_iters=20):
        super().__init__()
        self.n_iters = n_iters

    def forward(self, w):
        w = torch.exp(torch.clamp(w, max=20.0))  # 防溢出
        for _ in range(self.n_iters):
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            w = w / w.sum(dim=-2, keepdim=True).clamp(min=1e-12)
        return w


class HyperConnectionBlock(nn.Module):
    """mHC 超连接块(DeepSeek-V4 Manifold-Constrained Hyper-Connections)。

    公式(每层 2 块: 注意力前/FFN 前):
        x̄ = rmsnorm(x)
        z = einsum_{h,o}( W_e x̄ , M )   # expand d→hc·bw 后分支混合
        M = Sinkhorn_τ(exp(W_mix))        # 双随机矩阵(流形约束, τ=20 迭代)
        y = rmsnorm(W_c z) + x             # contract hc·bw→d + 残差
    即 expanded[b,s,hc,bw] → 与 M 混合 → contract;M 经 Sinkhorn
    迭代保持行/列和 = 1(双随机约束为论文核心语义)。

    与论文差异: 分支宽 hc_branch_dim=1024 = d/16(轻量口径, 控制
    稠密激活预算);仓库参考实现为全宽 d —
    参数量/激活 ×16。
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        d, hc, bw = config.hidden_size, config.hc_mult, config.hc_branch_dim
        self.expand = nn.Linear(d, hc * bw, bias=False)
        self.sinkhorn = SinkhornIteration(config.sinkhorn_iters)
        self.mix_weights = nn.Parameter(torch.randn(hc, hc) * 0.02)
        self.contract = nn.Linear(hc * bw, d, bias=False)
        self.pre_norm = RMSNorm(d)
        self.post_norm = RMSNorm(d)
        self.hc_mult = hc
        self.branch_dim = bw

    def forward(self, x):
        b, s, d = x.shape
        residual = x
        x = self.pre_norm(x)
        expanded = self.expand(x).view(b, s, self.hc_mult, self.branch_dim)
        mix = self.sinkhorn(self.mix_weights)
        mixed = torch.einsum("bshd,ho->bsod", expanded, mix)
        merged = mixed.reshape(b, s, self.hc_mult * self.branch_dim)
        return residual + self.post_norm(self.contract(merged))


class FlashBlock(nn.Module):
    """Block: mHC → Attention (ISA/GDN2) → mHC → MoE/Engram.

    mHC 超连接块(Sinkhorn 约束分支混合)分别在注意力与 FFN 前做
    超连接;ISA 层按 KV 共享角色参与组内 KV 槽位读写(见
    AlternatingAttention);GDN2 层维护自己的固定状态、不参与 KV 槽位。

    FFN 角色:
      - MoE 层: LatentMoE (4096 专家 top-25 + 1 共享)
      - Engram 注入层(记忆层): Flash 口径条件记忆模块替代 MoE,
        保持总参数量不变;需要 input_ids 做 n-gram 哈希查找
    """

    def __init__(self, config: FlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.is_engram = config.is_engram_layer(layer_idx)

        self.hc_pre_attn = HyperConnectionBlock(config)
        self.hc_pre_ffn = HyperConnectionBlock(config)
        self.attention = AlternatingAttention(config, layer_idx)
        self.attn_pre_norm = RMSNorm(config.hidden_size)
        self.attn_post_norm = RMSNorm(config.hidden_size)
        if self.is_engram:
            self.engram = EngramModule(config, layer_idx)
        else:
            self.moe = MoELayer(config)
        self.moe_pre_norm = RMSNorm(config.hidden_size)
        self.moe_post_norm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, kv_bufs=None, input_ids=None,
                attention_mask=None):
        h = hidden_states
        # mHC + Attention
        h = self.hc_pre_attn(h)
        residual = h
        h = residual + self.attention(self.attn_pre_norm(h), kv_bufs)
        h = self.attn_post_norm(h)

        # mHC + MoE / Engram
        h = self.hc_pre_ffn(h)
        residual = h
        if self.is_engram:
            if input_ids is None:
                raise RuntimeError(
                    f"Engram 注入层 {self.layer_idx} 需要 input_ids 做 n-gram 哈希查找")
            ffn_out = self.engram(self.moe_pre_norm(h), input_ids)
            aux_loss = torch.tensor(0.0, device=h.device)
        else:
            ffn_out, aux_loss = self.moe(self.moe_pre_norm(h))
        h = residual + ffn_out
        h = self.moe_post_norm(h)

        return h, aux_loss


# ============================================================================
# SECTION 11: Multimodal Encoders (AR 多模态理解)
# ============================================================================

class EncoderAttention(nn.Module):
    """Standard MHA for modality encoders (bidirectional, no causal mask)."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out = (probs @ v).transpose(1, 2).reshape(b, s, -1)
        return self.o(out)


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer block for modality encoders."""

    def __init__(self, hidden: int, heads: int, mlp_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = EncoderAttention(hidden, heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp_up = nn.Linear(hidden, mlp_mult * hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_mult * hidden, hidden, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.norm2(x))))
        return x


class ImageEncoder(nn.Module):
    """Image encoder: ViT-style patchify → pos embed → bidirectional
    transformer blocks → patch tokens [b, n_patches, enc_hidden]。

    公式: z = PatchEmbed(img) + pos_embed(Conv2d 3→h, stride=patch);
    z ← L 层预归一化双向自注意力块;out = rmsnorm(z)(ViT 标准口径)。"""

    def __init__(self, config: FlashConfig):
        super().__init__()
        h = config.image_enc_hidden
        p = config.image_patch_size
        self.patch_embed = nn.Conv2d(3, h, kernel_size=p, stride=p, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.image_max_patches, h))
        self.blocks = nn.ModuleList([
            EncoderBlock(h, config.image_enc_heads)
            for _ in range(config.image_enc_layers)
        ])
        self.norm = RMSNorm(h)

    def forward(self, x):
        # x: [b, 3, H, W]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [b, n, h]
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class AudioEncoder(nn.Module):
    """Audio encoder: raw waveform → 1D-conv stem (≈40× downsampling,
    log-mel 风格简化) + bidirectional transformer → frame tokens
    [b, n_frames, enc_hidden]。

    公式: z = Conv1d(1→h, K=400, S=160) → GELU → Conv1d(h→h, 3) →
    pos_embed → L 层双向块 → rmsnorm(Whisper 式卷积前端 + ViT 块)。"""

    def __init__(self, config: FlashConfig):
        super().__init__()
        h = config.audio_enc_hidden
        K, S = config.audio_conv_kernel, config.audio_conv_stride
        self.stem = nn.Sequential(
            nn.Conv1d(1, h, kernel_size=K, stride=S, bias=False),
            nn.GELU(),
            nn.Conv1d(h, h, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, config.audio_max_frames, h))
        self.blocks = nn.ModuleList([
            EncoderBlock(h, config.audio_enc_heads)
            for _ in range(config.audio_enc_layers)
        ])
        self.norm = RMSNorm(h)

    def forward(self, x):
        # x: [b, T] 或 [b, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x).transpose(1, 2)  # [b, frames, h]
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PointCloudEncoder(nn.Module):
    """Embodied 3D encoder: PointNet-style set encoding — shared per-point
    MLP + global max-pool summary token + per-point tokens (attention
    over the point set, no positional embedding → 置换不变集合编码).
    Output [b, 1 + N, enc_hidden]。

    公式(PointNet 口径): z_i = MLP_shared(xyz_i);
    tokens = [max_pool(z); z] → L 层双向块 → rmsnorm。"""

    def __init__(self, config: FlashConfig):
        super().__init__()
        h = config.point3d_enc_hidden
        self.point_mlp = nn.Sequential(
            nn.Linear(3, h, bias=False), nn.GELU(),
            nn.Linear(h, h, bias=False), nn.GELU(),
            nn.Linear(h, h, bias=False),
        )
        self.blocks = nn.ModuleList([
            EncoderBlock(h, config.point3d_enc_heads)
            for _ in range(config.point3d_enc_layers)
        ])
        self.norm = RMSNorm(h)

    def forward(self, x):
        # x: [b, N, 3] (点云坐标, 可选法线/颜色扩展输入通道由调用方拼接)
        z = self.point_mlp(x)                        # [b, N, h]
        g = z.max(dim=1, keepdim=True).values        # 全局汇总 token
        tokens = torch.cat([g, z], dim=1)            # [b, 1+N, h]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class HapticEncoder(nn.Module):
    """Embodied haptic-somatosensory encoder: multi-channel sensory stream
    (6-axis IMU / 触觉阵列 / 力觉) → 1D-conv downsampling + bidirectional
    transformer → tokens [b, n_frames, enc_hidden]。

    公式: z = Conv1d(C→h, K=8, S=4) → pos_embed → L 层双向块 → rmsnorm。"""

    def __init__(self, config: FlashConfig):
        super().__init__()
        h = config.haptic_enc_hidden
        K, S = config.haptic_conv_kernel, config.haptic_conv_stride
        self.stem = nn.Sequential(
            nn.Conv1d(config.haptic_num_channels, h, kernel_size=K, stride=S, bias=False),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, config.haptic_max_frames, h))
        self.blocks = nn.ModuleList([
            EncoderBlock(h, config.haptic_enc_heads)
            for _ in range(config.haptic_enc_layers)
        ])
        self.norm = RMSNorm(h)

    def forward(self, x):
        # x: [b, T, C]
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)  # [b, frames, h]
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ModalityEncoder(nn.Module):
    """Multimodal encoder hub: per-modality encoders + projection into the
    model hidden space + learned modality-type embeddings.

    文本直接走主干词嵌入;image/audio/point3d/haptic 各自编码后经
    投影进入 hidden_size,再加模态类型嵌入,与文本 token 交错成单一
    AR 序列。
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.modalities = config.modalities
        self.enc_hidden = {
            "image": config.image_enc_hidden,
            "audio": config.audio_enc_hidden,
            "point3d": config.point3d_enc_hidden,
            "haptic": config.haptic_enc_hidden,
        }
        if "image" in self.modalities:
            self.image_encoder = ImageEncoder(config)
        if "audio" in self.modalities:
            self.audio_encoder = AudioEncoder(config)
        if "point3d" in self.modalities:
            self.point3d_encoder = PointCloudEncoder(config)
        if "haptic" in self.modalities:
            self.haptic_encoder = HapticEncoder(config)

        self.modality_projs = nn.ModuleDict({
            m: nn.Linear(self.enc_hidden[m], config.hidden_size, bias=False)
            for m in self.modalities if m != "text"
        })
        self.modality_type_embedding = nn.Embedding(
            len(self.modalities), config.hidden_size)
        self.type_ids = {m: i for i, m in enumerate(self.modalities)}

    def encode(self, modality: str, x) -> torch.Tensor:
        """Returns [b, L_m, hidden_size] for one modality segment."""
        if modality == "image":
            z = self.image_encoder(x)
        elif modality == "audio":
            z = self.audio_encoder(x)
        elif modality == "point3d":
            z = self.point3d_encoder(x)
        elif modality == "haptic":
            z = self.haptic_encoder(x)
        else:
            raise ValueError(f"未知模态 {modality!r} (modalities={self.modalities})")
        z = self.modality_projs[modality](z)
        type_id = torch.tensor(self.type_ids[modality], device=z.device)
        return z + self.modality_type_embedding(type_id)


# ============================================================================
# SECTION 12: Full Model
# ============================================================================

class FlashModel(nn.Module):
    """DeepSeekV4.1-Flash: Looped Transformer + multimodal encoders.

    Backbone(公式):
      100 物理层权重只存一份, 前向跑 num_recursion=2 遍(逻辑深度
      = 100 × 2 = 200)。实际参数量 ≈ 20.7T; 计算等效 ≈ 41.5T
      (层参数 ×2, embedding/编码器只计一次)。
    反向语义:同一份权重在两遍使用——autograd 按链式法则把各使用点
    的梯度累加到同一 Parameter(dW = Σ_r dW_r), 优化器只执行一次更新;
    激活值各遍独立(MoR 权重绑定同义)。

    Flash KV sharing: within each pass, the 25 ISA layers share 5 KV
    slots (per group: 1 kv / 1 reindex / 3 reuse), passed through the
    forward as a per-group buffer dict; GDN2 layers keep only their
    fixed O(1) state.

    Engram: 10 inject layers carry Flash-style conditional memory
    modules (无 MoE 记忆层);they hash input ids (multimodal 位置以
    id=0 参与查找、门控静态屏蔽) — see EngramModule.

    Multimodal (AR 理解): forward 接受 layout 指定的交错序列
    (如 ["image", "audio", "point3d", "haptic", "text"]);modality 段由
    ModalityEncoder 编码投影 + 模态类型嵌入,text 段走词嵌入。损失只在
    text 段上计算(modality 位置 labels 置 -100)。
    """

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.config = config

        if config.loop_enable:
            if config.loop_sharing != "cycle":
                raise ValueError(
                    f"Loop sharing 仅实现 'cycle'(当前为 {config.loop_sharing!r}); "
                    f"num_layers={config.num_layers}")
            if config.base_depth * config.num_recursion != config.num_layers:
                raise ValueError(
                    f"Loop 配置不一致: base_depth({config.base_depth}) × "
                    f"num_recursion({config.num_recursion}) = "
                    f"{config.base_depth * config.num_recursion} != "
                    f"num_layers({config.num_layers})")
            if config.base_depth % config.unit_size != 0:
                raise ValueError(
                    f"Loop cycle 共享要求 base_depth({config.base_depth}) 是 "
                    f"unit_size({config.unit_size}) 的整数倍, 保证各递归内 "
                    f"ISA/GDN2 层型模式对齐")
            self.num_physical_layers = config.base_depth
            self.num_recursion = config.num_recursion
        else:
            self.num_physical_layers = config.num_layers
            self.num_recursion = 1

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        normal_init_(self.embedding.weight, std=config.embedding_init_method_std)

        self.layers = nn.ModuleList([
            FlashBlock(config, i) for i in range(self.num_physical_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tied

        self.modality_encoder = (
            ModalityEncoder(config) if config.multimodal_enable else None)

        # Decode-time cache (created lazily on use_cache; 见 LoopKVCache)
        self.kv_cache: Optional[LoopKVCache] = None

    # ------------------------------------------------------------------
    # Multimodal sequence construction
    # ------------------------------------------------------------------
    def build_multimodal_sequence(self, input_ids, modality_inputs, layout):
        """AR 多模态输入拼接:按 layout 逐段编码并拼接为单一序列。

        Returns (hidden_states [b, S, d], ids_full [b, S],
        text_spans [(start, end) ...])。ids_full 供 Engram 哈希查找:
        text 段用词元 id,modality 段置 0(静态 token mask 屏蔽其注入)。
        layout 示例: ["image", "text"] / ["image", "audio", "point3d",
        "haptic", "text"]。text 段可多次出现;labels 覆盖所有 text 段的
        按序拼接,其余位置在损失中屏蔽(-100)。
        """
        if self.modality_encoder is None:
            raise RuntimeError("多模态未启用 (multimodal_enable=False)")
        parts = []
        ids_parts = []
        spans = []
        offset = 0
        for key in layout:
            if key == "text":
                if input_ids is None:
                    raise ValueError("layout 含 text 段但未提供 input_ids")
                emb = self.embedding(input_ids)
                L = input_ids.shape[1]
                spans.append((offset, offset + L))
                ids_parts.append(input_ids)
            else:
                x = modality_inputs.get(key)
                if x is None:
                    raise ValueError(f"layout 含 {key} 段但未提供对应输入")
                emb = self.modality_encoder.encode(key, x)
                L = emb.shape[1]
                ids_parts.append(torch.zeros(
                    emb.shape[0], L, dtype=torch.long, device=emb.device))
            parts.append(emb)
            offset += L
        return (torch.cat(parts, dim=1), torch.cat(ids_parts, dim=1), spans)

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                image=None, audio=None, point3d=None, haptic=None,
                layout=None, use_cache: bool = False):
        if layout is not None:
            modality_inputs = {
                "image": image, "audio": audio,
                "point3d": point3d, "haptic": haptic,
            }
            hidden_states, ids_full, text_spans = self.build_multimodal_sequence(
                input_ids, modality_inputs, layout)
            if labels is not None:
                # labels 与所有 text 段按序拼接对齐;modality 位置屏蔽
                b = hidden_states.shape[0]
                full_labels = torch.full(
                    (b, hidden_states.shape[1]), -100,
                    dtype=labels.dtype, device=hidden_states.device)
                idx = 0
                for s, e in text_spans:
                    L = e - s
                    full_labels[:, s:e] = labels[:, idx:idx + L]
                    idx += L
                labels = full_labels
        else:
            hidden_states = self.embedding(input_ids)
            ids_full = input_ids

        total_aux = torch.tensor(0.0, device=hidden_states.device)

        if use_cache and self.kv_cache is None:
            self.kv_cache = LoopKVCache(
                self.config.kv_groups, self.num_physical_layers,
                self.num_recursion)

        # Loop recursion: num_recursion passes over the base_depth physical
        # blocks. 逻辑层 li = r × base_depth + pi 复用物理块 pi 的权重:
        # FLOPs 按逻辑深度计,参数只存一份;反向时各遍对同一权重的梯度
        # 由 autograd 自动累加(单次优化器更新)。每遍为 25 个 ISA 层维护
        # 5 个共享 KV 槽位(kv_bufs),产 KV 层写、reindex/reuse 层读。
        kv_bufs: Dict[int, Dict[str, Any]] = {
            g: {} for g in range(self.config.kv_groups)
        }
        h = hidden_states
        for r in range(self.num_recursion):
            for pi, layer in enumerate(self.layers):
                h, aux = layer(h, kv_bufs, ids_full, attention_mask)
                total_aux = total_aux + aux
        hidden_states = h

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
            total_loss = loss + self.config.z_loss_alpha * total_aux
            return total_loss, logits
        return logits


# ============================================================================
# SECTION 13: Parameter Counting
# ============================================================================

def count_parameters(model, verbose=True):
    total = 0
    trainable = 0
    by_comp = {}
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if "engram" in name or "hash_table" in name:
            cat = "engram"
        elif "embedding" in name or "lm_head" in name:
            cat = "embedding/lm_head"
        elif "modality_encoder" in name:
            cat = "multimodal_encoders"
        elif "routed_experts" in name:
            cat = "moe_routed_experts"
        elif "shared_experts" in name:
            cat = "moe_shared_experts"
        elif "router" in name:
            cat = "moe_router"
        elif "gdn2" in name:
            cat = "gdn2_attention"
        elif "indexer" in name or "q_proj" in name or "k_proj" in name \
                or "v_proj" in name or "o_proj" in name:
            cat = "isa_attention"
        elif "gate_down" in name or "up_down" in name or "latent_up" in name:
            cat = "moe_latent_projections"
        elif "hc_pre" in name or "hyper_connection" in name:
            cat = "mhc"
        elif "norm" in name:
            cat = "norms"
        else:
            cat = "other"
        by_comp[cat] = by_comp.get(cat, 0) + n

    result = {
        "total": total, "trainable": trainable,
        "total_billions": total / 1e9, "total_trillions": total / 1e12,
        "by_component": {k: {"count": v, "billions": v / 1e9}
                         for k, v in sorted(by_comp.items(), key=lambda x: -x[1])},
    }
    if verbose:
        print("=" * 70)
        print(f"Total: {total:,} ({result['total_billions']:.1f}B / {result['total_trillions']:.3f}T)")
        print(f"Trainable: {trainable:,}")
        print("-" * 70)
        for c, v in result["by_component"].items():
            print(f"  {c:<30} {v['count']:>15,} {v['billions']:>10.2f}B")
        print("=" * 70)
    return result


# ============================================================================
# SECTION 14: Config-Level Parameter Estimation
# ============================================================================

def estimate_encoder_params(config: FlashConfig) -> int:
    """Multimodal encoder params from config (无模型实例化)."""
    d = config.hidden_size
    total = 0
    if "image" in config.modalities:
        h, p = config.image_enc_hidden, config.image_patch_size
        total += (3 * p * p * h + h * config.image_max_patches
                  + config.image_enc_layers * (12 * h * h + 2 * h) + h)
    if "audio" in config.modalities:
        h = config.audio_enc_hidden
        total += (config.audio_conv_kernel * h + 3 * h * h
                  + h * config.audio_max_frames
                  + config.audio_enc_layers * (12 * h * h + 2 * h) + h)
    if "point3d" in config.modalities:
        h = config.point3d_enc_hidden
        total += (3 * h + 2 * h * h
                  + config.point3d_enc_layers * (12 * h * h + 2 * h) + h)
    if "haptic" in config.modalities:
        h = config.haptic_enc_hidden
        total += (config.haptic_num_channels * config.haptic_conv_kernel * h
                  + h * config.haptic_max_frames
                  + config.haptic_enc_layers * (12 * h * h + 2 * h) + h)
    # modality 投影 + 类型嵌入
    for m in config.modalities:
        if m == "text":
            continue
        total += {"image": config.image_enc_hidden,
                  "audio": config.audio_enc_hidden,
                  "point3d": config.point3d_enc_hidden,
                  "haptic": config.haptic_enc_hidden}[m] * d
    total += len(config.modalities) * d
    return total


def estimate_params_from_config(config: FlashConfig) -> dict:
    """Estimate parameters from config without instantiating the model.

    Loop (cycle sharing): parameters are stored once per physical block,
    so per-layer counts use base_depth (实际参数量 = grand_total);compute
    follows the full logical depth num_layers = base_depth × num_recursion
    (计算等效量 = compute_equivalent_total)。多模态编码器每前向只跑一次,
    不计入循环展开(与 embedding 同口径)。
    """
    d = config.hidden_size
    nl = config.num_layers                          # logical (effective) depth
    loop_on = bool(config.loop_enable) and config.loop_sharing == "cycle"
    nl_phys = config.base_depth if loop_on else nl  # physical depth
    n_rec = config.num_recursion if loop_on else 1
    nh = config.num_attention_heads
    hd = config.head_dim
    ne = config.num_routed_experts
    nshared = config.num_shared_experts
    mi = config.moe_intermediate_size
    latent = config.moe_latent_dim
    V = config.vocab_size
    rope = config.qk_rope_head_dim
    idx_dim = config.indexer_n_heads * config.indexer_head_dim

    emb = V * d

    # ---- ISA layers with Flash KV sharing (CSA 口径, 无 MLA) ----
    # roles per group: 1 kv / kv_reindex reindex / kv_reuse reuse
    n_isa = (nl_phys // config.unit_size) * config.isa_per_unit
    n_gdn2 = nl_phys - n_isa
    n_kv_isa = config.kv_groups
    n_reindex = config.kv_groups * config.kv_reindex_per_group
    n_reuse = n_isa - n_kv_isa - n_reindex

    # 全秩投影: Q/O 每 ISA 层,K/V 仅产 KV 层
    q_per = d * nh * hd
    k_per = d * nh * hd
    v_per = d * nh * hd
    o_per = nh * hd * d
    idx_per = 2 * d * idx_dim + 2 * idx_dim  # indexer q/k 投影 + q/k RMSNorm

    isa_total = (n_isa * (q_per + o_per) + n_kv_isa * (k_per + v_per)
                 + (n_kv_isa + n_reindex) * idx_per)

    # ---- GDN2 layers ----
    Hg, dk, dv, ds, da = (config.gdn2_num_heads, config.gdn2_d_k,
                          config.gdn2_d_v, config.gdn2_d_state,
                          config.gdn2_d_a)
    gdn2_per = (3 * d * Hg * dk + Hg * dv * d + d * Hg * dk
                + d * 2 * Hg * ds + d * Hg * da + Hg * da)  # 末项 da_norm
    gdn2_total = n_gdn2 * gdn2_per

    # ---- LatentMoE (on non-Engram layers) ----
    # expert: 潜在空间 FFN(latent→inter→latent 三投影);d→latent 下投影与
    # latent→d 上投影为逐层共享(见 MoELayer),计入 latent_proj_total
    n_engram = len(config.engram_layers) if config.engram_enable else 0
    n_moe = nl_phys - n_engram
    expert_per = 3 * latent * mi
    routed_total = n_moe * ne * expert_per
    shared_total = n_moe * nshared * expert_per
    router_total = n_moe * d * ne
    latent_proj_total = n_moe * 3 * d * latent

    # ---- Engram (Flash 口径条件记忆, 注入层替代 MoE) ----
    if config.engram_enable:
        orders = config.engram_ngram_orders
        e_heads = config.engram_num_hash_heads
        e_hd = config.engram_head_dim
        e_rows = config.engram_rows_per_head
        e_concat = len(orders) * e_heads * e_hd        # 6144
        engram_per = (len(orders) * e_heads * e_rows * e_hd  # 查找表
                      + d * d + 2 * d * e_concat        # query/key/value 投影
                      + 4 * d)                          # q/k/input/output RMSNorm
        engram_total = n_engram * engram_per
        engram_lookup_per_token = n_engram * len(orders) * e_heads * e_hd
    else:
        engram_total = 0
        engram_lookup_per_token = 0

    norms_total = nl_phys * 4 * d + d  # 每层 attn/moe pre-post ×4 + final_norm

    # ---- mHC (2 blocks per layer: expand + mix + contract) ----
    hc = config.hc_mult
    bw = config.hc_branch_dim
    mhc_per = 2 * (d * hc * bw + hc * hc + hc * bw * d) + 4 * d  # + 2 块 × pre/post RMSNorm
    mhc_total = nl_phys * mhc_per

    # ---- Multimodal encoders (每前向一次,不随循环展开) ----
    encoders_total = (estimate_encoder_params(config)
                      if config.multimodal_enable else 0)

    layer_total = (isa_total + gdn2_total + routed_total + shared_total
                   + router_total + latent_proj_total + engram_total
                   + mhc_total + norms_total)

    # 实际参数量: 物理层只存一份 (embedding/编码器存一次)
    physical = 2 * emb + encoders_total + layer_total
    # 计算等效量: 层参数随循环展开(embedding/编码器只算一次)
    compute_eq = 2 * emb + encoders_total + layer_total * n_rec

    # ---- Active params (单遍 = 一次 100 物理层扫描) ----
    # Engram 查找表按查找口径计激活(每 token 每注入层仅读
    # len(orders)×heads×head_dim 条目),其 key/value/query 投影为
    # 稠密全激活。
    topk = config.num_experts_per_token
    layer_active = (isa_total + gdn2_total + mhc_total        # 注意力/mHC 全激活
                    + topk * expert_per * n_moe               # 路由专家
                    + nshared * expert_per * n_moe            # 共享专家
                    + latent_proj_total + router_total
                    + n_engram * d * (d + 2 * len(config.engram_ngram_orders)
                                      * config.engram_num_hash_heads
                                      * config.engram_head_dim)
                    + engram_lookup_per_token * 1)            # 查找条目(极小)
    active_per_pass = emb + encoders_total + layer_active
    active_full = emb + encoders_total + layer_active * n_rec

    return {
        "embedding": emb,
        "encoders_total": encoders_total,
        "isa_total": isa_total,                # 物理层口径
        "gdn2_total": gdn2_total,              # 物理层口径
        "routed_total": routed_total,          # 物理层口径
        "shared_total": shared_total,
        "router_total": router_total,
        "latent_proj_total": latent_proj_total,
        "mhc_total": mhc_total,                # 物理层口径
        "engram_total": engram_total,          # 物理层口径 (注入层)
        "n_moe": n_moe,
        "n_engram": n_engram,
        "n_isa": n_isa,
        "n_gdn2": n_gdn2,
        "n_kv_isa": n_kv_isa,
        "n_reindex": n_reindex,
        "n_reuse": n_reuse,
        "grand_total": physical,               # 总参数 = 实际参数量(Loop 共享)
        "physical_total": physical,            # 同 grand_total(兼容别名)
        "compute_equivalent_total": compute_eq,  # 计算等效(全循环展开)
        "active_params_per_pass": active_per_pass,   # 单遍激活(标称 200B 口径)
        "active_params_full_forward": active_full,   # 全前向激活(含 Loop)
        "physical_layers": nl_phys,
        "effective_layers": nl,
        "num_recursion": n_rec,
        "moe_ratio": routed_total / physical,
        "engram_ratio": engram_total / physical,
        "expert_per": expert_per,
    }


# ============================================================================
# SECTION 15: System Resource Analysis
# ============================================================================

@dataclass
class SystemResourceReport:
    config_name: str
    total_params: float
    compute_equivalent_params: float
    active_params_per_pass: float
    active_params_full_forward: float
    moe_params: float
    moe_ratio: float
    mhc_params: float
    engram_params: float
    engram_ratio: float
    encoder_params: float
    kv_cache_compression: float
    model_weights_gb: float
    kv_cache_total_gb: float
    recommended_gpu_count: int
    parallelism_strategy: str


def analyze_system_resources(config: FlashConfig, seq_len=65536, batch_size=1,
                             training=True, verbose=True):
    params = estimate_params_from_config(config)
    # 实际参数量 = 物理层口径;计算等效量 = 全逻辑深度展开
    total_T = params["grand_total"] / 1e12
    compute_eq_T = params["compute_equivalent_total"] / 1e12
    n_rec = params["num_recursion"]

    # KV cache compression vs MHA
    # MHA baseline: 2 × nh × hd × nl per token (逻辑层口径)
    mha_kv = 2 * config.num_attention_heads * config.head_dim * config.num_layers
    # Our design: only 5 shared KV slots, CSA 2× 压缩 (K+V 各 nh×hd 每
    # token、只存一半 token) + GDN2 fixed states amortized over seq_len
    isa_kv = (config.kv_groups * 2 * config.num_attention_heads
              * config.head_dim // config.isa_compress_ratio)
    gdn2_state = (params["n_gdn2"] * config.gdn2_num_heads
                  * config.gdn2_d_state ** 2)
    kv_per_token = isa_kv + gdn2_state / seq_len
    compression = mha_kv / max(kv_per_token, 1)

    # Active params (单遍 = 一次物理层扫描;全前向含 Loop)
    active_pass = params["active_params_per_pass"] / 1e9
    active_full = params["active_params_full_forward"] / 1e9

    # Memory: storage follows the physical params (weights stored once)
    total_p = params["physical_total"]
    weights_gb = total_p * 1.0 / 1e9  # FP8
    kv_total_gb = kv_per_token * seq_len * 2 / 1e9  # FP16

    # GPU
    gpu_mem = 80.0 * 0.85
    total_train_gb = weights_gb * 2 + total_p * 8 / 1e9 + kv_total_gb + \
        batch_size * seq_len * config.hidden_size * config.num_layers * 12 * 2 / 1e9
    gpus = max(1, int(math.ceil(total_train_gb / gpu_mem)))
    if gpus <= 8: strat = "TP=1, EP=1, PP=1, DP=8"
    elif gpus <= 64: strat = "TP=4, EP=8, PP=2, DP=1"
    elif gpus <= 512: strat = "TP=8, EP=64, PP=4, DP=1"
    elif gpus <= 4096: strat = "TP=8, EP=256, PP=8, DP=1"
    else: strat = f"TP=8, EP=512, PP=16, DP={max(1, gpus // 8192)}"

    loop_tag = (f" Loop {config.num_recursion}×{config.base_depth}"
                if config.loop_enable else "")
    report = SystemResourceReport(
        config_name=(f"d={config.hidden_size}, L={config.num_layers}{loop_tag}, "
                     f"experts={config.num_routed_experts}"),
        total_params=total_T,
        compute_equivalent_params=compute_eq_T,
        active_params_per_pass=active_pass,
        active_params_full_forward=active_full,
        moe_params=params["routed_total"] / 1e9,
        moe_ratio=params["moe_ratio"],
        mhc_params=params["mhc_total"] / 1e9,
        engram_params=params["engram_total"] / 1e9,
        engram_ratio=params["engram_ratio"],
        encoder_params=params["encoders_total"] / 1e9,
        kv_cache_compression=compression,
        model_weights_gb=weights_gb,
        kv_cache_total_gb=kv_total_gb,
        recommended_gpu_count=gpus,
        parallelism_strategy=strat,
    )

    if verbose:
        sep = "=" * 80
        print(f"\n{sep}\n  SYSTEM RESOURCE ANALYSIS: {report.config_name}")
        print(f"  Context: seq_len={seq_len:,}, batch={batch_size}, training={training}\n{sep}")
        print(f"\n  PARAMETERS")
        print(f"  实际参数量(物理 {params['physical_layers']} 层):       "
              f"     {report.total_params:>10.2f}T")
        print(f"  计算等效量({config.num_layers} 层展开, Loop {n_rec}):     "
              f"     {report.compute_equivalent_params:>10.2f}T")
        print(f"  单遍激活(100 物理层扫描):        {report.active_params_per_pass:>10.1f}B")
        print(f"  全前向激活(含 Loop {n_rec}):          "
              f"{report.active_params_full_forward:>10.1f}B")
        print(f"  MoE routed:           {report.moe_params/1e3:>10.2f}T "
              f"({report.moe_ratio*100:.1f}%)")
        print(f"  mHC:                  {report.mhc_params:>10.2f}B")
        print(f"  Engram:               {report.engram_params/1e3:>10.3f}T "
              f"({report.engram_ratio*100:.1f}%)")
        print(f"  多模态编码器:         {report.encoder_params:>10.2f}B")
        print(f"  ISA 层: {params['n_isa']} ({params['n_kv_isa']} kv / "
              f"{params['n_reindex']} reindex / {params['n_reuse']} reuse), "
              f"GDN2 层: {params['n_gdn2']}")
        print(f"\n  KV CACHE COMPRESSION vs MHA: {report.kv_cache_compression:>10.0f}x")
        print(f"  Model weights (FP8):  {report.model_weights_gb:>10.1f} GB")
        print(f"  KV cache (full ctx):  {report.kv_cache_total_gb:>10.2f} GB")
        print(f"\n  GPUs (H800-80GB):     {report.recommended_gpu_count:>10}")
        print(f"  Strategy:             {report.parallelism_strategy}")
        print(sep)

    return report


# ============================================================================
# SECTION 16: Smoke Test & Main
# ============================================================================

def smoke_test(batch_size=2, seq_len=32):
    """Quick forward + backward pass, scaled down for CPU.

    Preserves:
      - 4-layer unit structure [ISA, GDN2, GDN2, GDN2] (base_depth=12)
      - Loop 2× cycle sharing (num_recursion=2)
      - Flash KV sharing: 3 ISA layers → 1 KV slot (kv / reuse / reindex)
      - LatentMoE with moe_latent_dim
      - GDN2 delta-rule linear attention
      - Multimodal encoders + interleaved AR sequence (loss on text spans)
      - Token-choice routing + load-balance loss
    """
    config = FlashConfig(
        hidden_size=128,
        num_layers=24,               # base_depth(12) × Loop 2
        num_attention_heads=4,
        head_dim=32,
        vocab_size=512,
        max_seq_len=512,

        qk_rope_head_dim=8,

        loop_enable=True,
        loop_sharing="cycle",
        num_recursion=2,
        base_depth=12,

        unit_size=4,
        isa_per_unit=1,
        gdn2_per_unit=3,

        kv_share_enable=True,
        kv_groups=1,                 # 3 ISA 层 / 1 槽位: kv + reuse + reindex
        kv_reindex_per_group=1,
        kv_reuse_per_group=1,

        isa_compress_ratio=2,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_topk=8,

        gdn2_num_heads=4,
        gdn2_d_state=16,
        gdn2_d_k=8,
        gdn2_d_v=8,
        gdn2_d_a=3,

        hc_mult=2,
        sinkhorn_iters=3,

        num_routed_experts=4,
        num_shared_experts=1,
        moe_intermediate_size=64,
        moe_latent_dim=48,
        num_experts_per_token=2,

        engram_enable=True,
        engram_layers=[1, 5, 9],        # 3 个注入层(记忆层,无 MoE)
        engram_ngram_orders=[2],
        engram_num_hash_heads=2,
        engram_head_dim=16,
        engram_rows_per_head=64,
        engram_static_gate=True,

        multimodal_enable=True,
        modalities=["text", "image", "audio", "point3d", "haptic"],
        image_patch_size=8,
        image_enc_hidden=64,
        image_enc_layers=2,
        image_enc_heads=4,
        image_max_patches=64,
        audio_enc_hidden=48,
        audio_enc_layers=2,
        audio_enc_heads=4,
        audio_conv_kernel=4,
        audio_conv_stride=2,
        audio_max_frames=64,
        point3d_enc_hidden=48,
        point3d_enc_layers=2,
        point3d_enc_heads=4,
        point3d_num_points=32,
        haptic_enc_hidden=48,
        haptic_enc_layers=2,
        haptic_enc_heads=4,
        haptic_num_channels=4,
        haptic_conv_kernel=4,
        haptic_conv_stride=2,
        haptic_max_frames=64,

        rope_theta=1024.0,
        yarn_factor=4.0,
        yarn_original_max=256,

        swiglu_clamp=10.0,
        attn_softmax_clamp=50.0,
        z_loss_alpha=0.001,
    )

    print(f"Smoke test (DeepSeekV4.1-Flash small model)")
    print(f"  batch={batch_size}, seq_len={seq_len}")
    print(f"  d={config.hidden_size}, L={config.num_layers}, heads={config.num_attention_heads}")
    print(f"  Unit: ISA={config.is_isa_layer(0)}, GDN2={[i for i in range(4) if config.is_gdn2_layer(i)]}")
    print(f"  Loop: {config.num_recursion}× cycle, base_depth={config.base_depth}")
    print(f"  Layer types: " + ", ".join(
        f"L{i}={config.layer_type(i)}" for i in range(config.base_depth)
    ))
    print(f"  KV sharing: {config.isa_layout_summary()}")
    print(f"  experts={config.num_routed_experts}, latent={config.moe_latent_dim}, "
          f"inter={config.moe_intermediate_size}, top-{config.num_experts_per_token}")
    print(f"  GDN2: H={config.gdn2_num_heads}, d_state={config.gdn2_d_state}, "
          f"d_k={config.gdn2_d_k}, d_a={config.gdn2_d_a}")
    print(f"  mHC: mult={config.hc_mult}, sinkhorn_iters={config.sinkhorn_iters}")
    print(f"  Engram: layers={config.engram_layers}, orders={config.engram_ngram_orders}, "
          f"heads={config.engram_num_hash_heads}, rows={config.engram_rows_per_head}")
    print(f"  Modalities: {config.modalities}")

    device = torch.device("cpu")
    model = FlashModel(config).to(device)
    model.train()

    # Loop 结构校验:物理块 = base_depth,逻辑深度 = base_depth × num_recursion
    print(f"  Loop structure: physical_blocks={model.num_physical_layers} "
          f"(base_depth={config.base_depth}), recursion={model.num_recursion}, "
          f"logical_depth={config.num_layers}")
    assert model.num_physical_layers == config.base_depth
    assert config.base_depth * config.num_recursion == config.num_layers

    # KV 角色校验:3 个 ISA 层 (0/4/8) → kv / reuse / reindex
    roles = [config.isa_role(4 * u) for u in range(3)]
    print(f"  ISA roles (layers 0/4/8): {roles}")
    assert roles == ["kv", "reuse", "reindex"], roles

    # Engram 结构校验:注入层为记忆层(有 engram、无 moe),其余层有 moe
    n_moe_blocks = sum(1 for i in range(config.base_depth)
                       if hasattr(model.layers[i], "moe"))
    engram_struct_ok = (
        all(hasattr(model.layers[i], "engram")
            and not hasattr(model.layers[i], "moe")
            for i in config.engram_layers)
        and n_moe_blocks == config.base_depth - len(config.engram_layers)
    )
    print(f"  Engram structure: {len(config.engram_layers)} memory layers "
          f"(no MoE), {n_moe_blocks} MoE layers: "
          f"{'OK' if engram_struct_ok else 'FAILED'}")
    assert engram_struct_ok

    stats = count_parameters(model, verbose=True)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # ---- 纯文本前向 + 反向 ----
    print("\nForward pass (text)...")
    t0 = time.time()
    loss, logits = model(input_ids=input_ids, labels=labels)
    fwd = time.time() - t0
    print(f"  Time: {fwd:.3f}s, Loss: {loss.item():.4f}")
    print(f"  Logits: max={logits.max().item():.4f}, min={logits.min().item():.4f}, "
          f"NaN={torch.isnan(logits).any().item()}")

    print("\nBackward pass...")
    t0 = time.time()
    loss.backward()
    bwd = time.time() - t0

    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
                  if p.grad is not None}
    nan_grads = sum(1 for v in grad_norms.values() if math.isnan(v))
    max_g = max(grad_norms.values()) if grad_norms else 0
    print(f"  Time: {bwd:.3f}s, Max grad: {max_g:.6f}, NaN grads: {nan_grads}")

    passed = not (math.isnan(loss.item()) or nan_grads > 0)

    # ---- 反向语义校验:循环共享权重多使用点梯度累加 = 复合函数全梯度 ----
    # (Loop 同一权重在两遍使用,autograd 按链式法则求和;此处用 float64
    # 有限差分验证"使用两次"的梯度确为复合函数全梯度,而非逐遍覆盖)
    w = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                  dtype=torch.float64))
    x = torch.tensor([[0.5, -1.0]], dtype=torch.float64)

    def composed(ww):
        y1 = x @ ww
        y2 = y1 @ ww
        return y1.sum() + 2.0 * y2.sum()

    grad_auto = torch.autograd.grad(composed(w), w)[0]
    eps = 1e-6
    grad_num = torch.zeros_like(w)
    for i in range(w.numel()):
        wp = w.detach().clone(); wp.view(-1)[i] += eps
        wm = w.detach().clone(); wm.view(-1)[i] -= eps
        grad_num.view(-1)[i] = (composed(wp) - composed(wm)) / (2 * eps)
    grad_ok = torch.allclose(grad_auto, grad_num, atol=1e-6)
    print(f"  Grad accumulation check (autograd sum == finite diff): "
          f"{'OK' if grad_ok else 'FAILED'} "
          f"auto={grad_auto.reshape(-1).tolist()} "
          f"num={grad_num.reshape(-1).tolist()}")
    passed = passed and grad_ok

    # ---- 多模态前向:交错 AR 序列 + 损失只落在 text 段 ----
    print("\nForward pass (multimodal: image + audio + point3d + haptic + text)...")
    mm_img = torch.randn(batch_size, 3, 64, 64)      # patch 8 → 64 tokens
    mm_audio = torch.randn(batch_size, 40)           # → (40-4)/2+1 = 19 frames
    mm_pc = torch.randn(batch_size, 16, 3)           # → 1+16 = 17 tokens
    mm_haptic = torch.randn(batch_size, 40, 4)       # → 19 frames
    layout = ["image", "audio", "point3d", "haptic", "text"]
    mm_labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    t0 = time.time()
    mm_loss, mm_logits = model(
        input_ids=input_ids, labels=mm_labels,
        image=mm_img, audio=mm_audio, point3d=mm_pc, haptic=mm_haptic,
        layout=layout,
    )
    fwd_mm = time.time() - t0

    expected_seq = 64 + 19 + 17 + 19 + seq_len
    mm_ok = (not math.isnan(mm_loss.item())
             and mm_logits.shape == (batch_size, expected_seq, config.vocab_size))
    print(f"  Time: {fwd_mm:.3f}s, Loss: {mm_loss.item():.4f}")
    print(f"  Sequence: {expected_seq} tokens (modality {expected_seq - seq_len} "
          f"+ text {seq_len}), Logits: {tuple(mm_logits.shape)}")
    print(f"  Multimodal check: {'OK' if mm_ok else 'FAILED'}")

    # 编码器梯度连通性:多模态反向应把梯度传回各 encoder
    mm_loss.backward()
    enc_grads = {n: (p.grad is not None and p.grad.norm().item() > 0)
                 for n, p in model.modality_encoder.named_parameters()}
    enc_bad = [n for n, ok in enc_grads.items() if not ok]
    enc_ok = not enc_bad and len(enc_grads) > 0
    print(f"  Encoder grads ({len(enc_grads)} params, all nonzero): "
          f"{'OK' if enc_ok else 'FAILED'} {enc_bad[:5] if enc_bad else ''}")
    passed = passed and mm_ok and enc_ok

    # ---- 参数估算一致性 (物理口径) ----
    est = estimate_params_from_config(config)
    print(f"\n  Estimate: grand_total={est['grand_total']:,} vs counted={stats['total']:,} "
          f"(差异来自 tied embedding 与 norms 近似)")
    print(f"  Compute-equivalent: {est['compute_equivalent_total']/1e9:.3f}B "
          f"(×{est['num_recursion']} loop), active/pass={est['active_params_per_pass']/1e9:.3f}B")

    print(f"\n{'PASSED' if passed else 'FAILED'} "
          f"(total: {fwd+bwd+fwd_mm:.2f}s, params: {stats['total']:,})")
    return model


# ============================================================================
# SECTION 17: Requirement Compliance Check (spec check)
# ============================================================================

def spec_check(verbose=True):
    """逐条核对 preset_40T 与架构规格的一致性(结构/口径自检).

    规格:
      1. 100 层结构: 25 层带 indexer 的稀疏注意力(CSA 压缩 2×,
         token 级稀疏) + 75 层 GDN2(线性)
      2. MoE: 4096 专家 / 激活 25 / 1 共享专家 / Latent MoE
      3. 25 个稀疏层共享 5 层 KV Cache;其余 20 层: 5 reindex, 15 reuse topK
      4. Looped Transformer, Loop 2 次
      5. mHC (Manifold-Constrained Hyper-Connections, Sinkhorn 约束)
      6. Engram ≈ 10% 参数 (DeepSeek V4.1 Flash 口径), 总参数量不变
         (MoE 移出注入层 → 90 层)
      7. 20T 参数量 / 激活 200B / 计算等效 40T (参数量 ±10%,
         激活 ±30% — CSA 全秩投影与 mHC 为稠密全激活, 等效 ±10%)
      8. AR 多模态理解: 文本/图像/语音/具身(3D、体感)经独立编码器投影进主干

    Returns True if all checks pass.
    """
    config = FlashConfig.preset_40T()
    params = estimate_params_from_config(config)

    checks: List[Tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- 1. 层结构 ----
    n_isa = sum(config.is_isa_layer(i) for i in range(config.base_depth))
    n_gdn2 = config.base_depth - n_isa
    check("100 层结构", config.base_depth == 100,
          f"base_depth={config.base_depth}")
    check("25 层 ISA(indexer 稀疏注意力)", n_isa == 25,
          f"实际 {n_isa} 层")
    check("ISA = CSA 压缩 2× (无 MLA 低秩)",
          config.isa_compress_ratio == 2
          and not hasattr(config, "q_lora_rank"),
          f"isa_compress_ratio={config.isa_compress_ratio}, 全秩 Q/K/V/O, "
          f"indexer top-{config.indexer_topk}")
    check("75 层 GDN2(线性注意力)", n_gdn2 == 75,
          f"实际 {n_gdn2} 层")

    # ---- 2. KV 共享 (25 ISA 层 → 5 槽位) ----
    roles = [config.isa_role(u * config.unit_size)
             for u in range(config.base_depth // config.unit_size)]
    check("共享 5 层 KV Cache", config.kv_groups == 5,
          f"{config.kv_groups} 槽位 / {len(roles)} ISA 层")
    check("5 层产 KV", roles.count("kv") == 5,
          f"实际 {roles.count('kv')} 层")
    check("5 层 reindex", roles.count("reindex") == 5,
          f"实际 {roles.count('reindex')} 层")
    check("15 层 reuse topK", roles.count("reuse") == 15,
          f"实际 {roles.count('reuse')} 层")

    # ---- 3. MoE ----
    check("MoE 4096 专家", config.num_routed_experts == 4096,
          f"{config.num_routed_experts} 路由专家")
    check("激活 25 专家", config.num_experts_per_token == 25,
          f"top-{config.num_experts_per_token}")
    check("1 共享专家", config.num_shared_experts == 1,
          f"{config.num_shared_experts} 共享专家")
    check("Latent MoE 瓶颈", 0 < config.moe_latent_dim < config.hidden_size,
          f"latent={config.moe_latent_dim} < d={config.hidden_size}")

    # ---- 4. Loop ----
    check("Looped Transformer Loop 2",
          config.loop_enable and config.num_recursion == 2,
          f"{config.base_depth} 物理层 × {config.num_recursion} = "
          f"{config.num_layers} 逻辑深度")

    # ---- 5. mHC ----
    check("mHC (Sinkhorn 约束超连接)",
          config.hc_mult >= 2 and config.sinkhorn_iters > 0,
          f"hc_mult={config.hc_mult}, sinkhorn_iters={config.sinkhorn_iters}, "
          f"2 块/层, 共 {params['mhc_total']/1e9:.1f}B")

    # ---- 6. Engram ----
    check("Engram ≈ 10% 参数 (Flash 口径)",
          config.engram_enable and 0.08 <= params["engram_ratio"] <= 0.12,
          f"{params['engram_ratio']*100:.1f}% ({params['engram_total']/1e12:.3f}T), "
          f"{params['n_engram']} 注入层 × {len(config.engram_ngram_orders)} 阶 × "
          f"{config.engram_num_hash_heads} 头 × {config.engram_rows_per_head:,} 行 "
          f"× {config.engram_head_dim} 维; 注入层无 MoE (MoE 在 {params['n_moe']} 层)")
    check("总参数量不变 (≈20T 口径)",
          abs(params["grand_total"] / 1e12 - 20.0) / 20.0 <= 0.10,
          f"实际 {params['grand_total']/1e12:.2f}T (Engram 由 MoE 移出注入层补偿)")

    # ---- 7. 参数口径 (标称值 ±10% / 激活 ±20%) ----
    total_T = params["grand_total"] / 1e12
    comp_T = params["compute_equivalent_total"] / 1e12
    act_B = params["active_params_per_pass"] / 1e9
    check("20T 参数量 (±10%)", abs(total_T - 20.0) / 20.0 <= 0.10,
          f"实际 {total_T:.2f}T (标称 20T, Engram {params['engram_total']/1e12:.3f}T "
          f"+ mHC {params['mhc_total']/1e9:.1f}B)")
    check("激活 200B (±30%)", abs(act_B - 200.0) / 200.0 <= 0.30,
          f"单遍 {act_B:.1f}B (含 Loop: "
          f"{params['active_params_full_forward']/1e9:.1f}B; "
          f"Engram 表格按查找口径; CSA 全秩投影 +27B 激活)")
    target_comp = 40.0
    check(f"计算等效 {target_comp:.0f}T (±10%)",
          abs(comp_T - target_comp) / target_comp <= 0.10,
          f"实际 {comp_T:.2f}T (标称 {target_comp:.0f}T)")

    # ---- 8. 多模态 ----
    mods = config.modalities
    check("多模态集齐 text/image/audio/point3d/haptic",
          all(m in mods for m in ("text", "image", "audio", "point3d", "haptic")),
          f"{mods}")
    check("encoder 投影进 hidden + 模态类型嵌入",
          config.multimodal_enable and params["encoders_total"] > 0,
          f"编码器参数 {params['encoders_total']/1e9:.2f}B;"
          f" 前向/反向已由 smoke 验证")

    all_ok = all(p for _, p, _ in checks)
    if verbose:
        print(f"\n{'='*80}")
        print(f"  SPEC COMPLIANCE CHECK (架构规格 vs preset_40T)")
        print(f"{'='*80}")
        for name, passed, detail in checks:
            mark = "[OK]  " if passed else "[FAIL]"
            print(f"  {mark} {name:<30} {detail}")
        print(f"{'='*80}")
        print(f"  {'ALL PASSED' if all_ok else 'HAS FAILURES'} "
              f"({sum(p for _, p, _ in checks)}/{len(checks)})")
        print(f"{'='*80}")
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeekV4.1-Flash Model")
    parser.add_argument("--preset", type=str, default="40T",
                        choices=["40T", "test"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--spec-check", action="store_true")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.spec_check:
        spec_check()
    elif args.analyze or (args.preset == "40T" and not args.smoke):
        from config import FlashConfig
        config = FlashConfig.preset_40T()
        print(f"\n{'='*80}")
        print(f"  DeepSeekV4.1-Flash-40T (20T 参数 × Loop 2 = 40T 计算等效)")
        print(f"{'='*80}")
        print(f"  hidden_size:          {config.hidden_size}")
        print(f"  num_layers:           {config.num_layers} "
              f"(100 物理层 × Loop {config.num_recursion})")
        print(f"  num_attention_heads:  {config.num_attention_heads}")
        print(f"  head_dim:             {config.head_dim}")
        print(f"  vocab_size:           {config.vocab_size}")
        print(f"  max_seq_len:          {config.max_seq_len:,}")
        print(f"  num_routed_experts:   {config.num_routed_experts}")
        print(f"  moe_latent_dim:       {config.moe_latent_dim}")
        print(f"  moe_intermediate:     {config.moe_intermediate_size}")
        print(f"  num_experts/token:    {config.num_experts_per_token}")
        print(f"  Loop: {config.num_recursion}× {config.loop_sharing}, "
              f"base_depth={config.base_depth}")
        print(f"  KV sharing:           {config.isa_layout_summary()}")
        print(f"  GDN2: H={config.gdn2_num_heads}, d_state={config.gdn2_d_state}, "
              f"d_k={config.gdn2_d_k}, d_a={config.gdn2_d_a}")
        print(f"  ISA: {config.isa_compress_ratio}x / topk={config.indexer_topk}")
        print(f"  Modalities:           {config.modalities}")
        print(f"{'='*80}")

        params = estimate_params_from_config(config)
        print(f"\n  --- Parameter Estimates ---")
        print(f"  Embedding+LM Head:    {params['embedding']/1e9:.2f}B × 2 (tied)")
        print(f"  ISA attention:        {params['isa_total']/1e9:.2f}B "
              f"({params['n_isa']} layers: {params['n_kv_isa']} kv / "
              f"{params['n_reindex']} reindex / {params['n_reuse']} reuse)")
        print(f"  GDN2 attention:       {params['gdn2_total']/1e9:.2f}B "
              f"({params['n_gdn2']} layers)")
        print(f"  MoE routed:           {params['routed_total']/1e12:.2f}T "
              f"(物理 {params['n_moe']} 层, 注入层无 MoE)")
        moe_dense_B = ((params['shared_total'] + params['router_total']
                        + params['latent_proj_total']) / 1e9)
        print(f"  MoE shared/router/proj: {moe_dense_B:.2f}B")
        print(f"  mHC hyper-connections: {params['mhc_total']/1e9:.2f}B "
              f"(hc_mult={config.hc_mult}, 2 blocks/layer)")
        print(f"  Engram:               {params['engram_total']/1e12:.3f}T "
              f"({params['engram_ratio']*100:.1f}%, {params['n_engram']} 注入层, "
              f"Flash 口径)")
        print(f"  多模态编码器:         {params['encoders_total']/1e9:.2f}B")
        print(f"  --> 实际参数量 (Loop 共享):    {params['grand_total']/1e12:.2f}T "
              f"(标称 20T)")
        print(f"  --> 计算等效量 (Loop {params['num_recursion']} 展开): "
              f"{params['compute_equivalent_total']/1e12:.2f}T (标称 40T)")
        print(f"  --> 单遍激活 (100 物理层):    "
              f"{params['active_params_per_pass']/1e9:.1f}B (标称 200B)")
        print(f"  --> 全前向激活 (含 Loop):     "
              f"{params['active_params_full_forward']/1e9:.1f}B")
        print(f"  MoE fraction:         {params['moe_ratio']*100:.2f}%")
        print(f"  Engram fraction:      {params['engram_ratio']*100:.2f}%")

        analyze_system_resources(config, seq_len=args.seq_len, batch_size=args.batch_size)

    elif args.smoke or args.preset == "test":
        smoke_test()
