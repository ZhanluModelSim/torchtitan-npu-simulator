"""
model.py
纯PyTorch实现，无custom‑ops，无分布式并行
SLA2Attention + Multi‑Head‑MoE（MAGI‑2 preview风格朴素实现）
前2层、后2层使用Dense FFN，中间层使用Multi‑Head‑MoE
ref: https://github.com/SandAI-org/MAGI-2-preview/blob/main/inference/model/magi2_preview.py#L2941
"""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    max_seq_len: int = 5 * 10**6
    vocab_size: int = 102400
    dim: int = 12288
    inter_dim: int = 16384          # Dense FFN intermediate dim
    n_layers: int = 96
    n_heads: int = 96
    head_dim: int = 128
    # ========= Multi‑Head‑MoE 参数 =========
    moe_num_heads: int = 32                 # S=32 MoE heads
    moe_head_hidden_size: int = 384         # hs=384 per moe‑head
    experts_per_head: int = 1024            # Es=1024 experts per head
    k_per_head: int = 6                     # Ks=6 activated per head
    moe_shared_inter_dim: int = 6144        # shared FFN:12288‑6144‑12288
    moe_expert_inter_mult: int = 4          # 384*4=1536 per routed expert
    # RoPE
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    # SLA2 Attention
    sla2_topk_1m: float = 0.005   # keep 0.5% →99.5% sparse @1M
    sla2_topk_5m: float = 0.001   # keep 0.1% →99.9% sparse @5M
    sla2_blkq: int = 64
    sla2_blkk: int = 64
    sla2_feature_map: Literal["softmax"] = "softmax"
    sla2_tie_feature_map_qk: bool = True
    sla2_stage: int = 2
    norm_eps: float = 1e-6


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(x.device.type, enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ========== 内联块SLA2核心实现（带自动padding） ==========
def block_mean_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    B, H, T, Dh = x.shape
    Nb = T // block_size
    x = x.view(B, H, Nb, block_size, Dh)
    return x.mean(dim=-2)


class BlockRouter(nn.Module):
    def __init__(self, d_head: int, proj_dim: int = 64):
        super().__init__()
        self.proj_q = nn.Linear(d_head, proj_dim)
        self.proj_k = nn.Linear(d_head, proj_dim)

    def forward(self, q_pool: torch.Tensor, k_pool: torch.Tensor):
        qp = self.proj_q(q_pool)
        kp = self.proj_k(k_pool)
        score_mat = torch.matmul(qp, kp.transpose(-1, -2))
        return score_mat


def soft_topk_mask(score_mat: torch.Tensor, topk_blocks: int, temperature: float, training: bool):
    B, H, Nb_q, Nb_k = score_mat.shape
    if training:
        logits = score_mat / temperature
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits = logits + gumbel_noise
        soft_mask = F.softmax(logits, dim=-1)
        vals, _ = torch.topk(soft_mask, k=topk_blocks, dim=-1)
        thresh = vals[..., [-1]]
        mask_c = torch.sigmoid((soft_mask - thresh) / temperature)
    else:
        _, top_idx = torch.topk(score_mat, k=topk_blocks, dim=-1)
        mask_c = torch.zeros_like(score_mat)
        mask_c.scatter_(-1, top_idx, 1.0)
    return mask_c


class InnerBlockSLA2(nn.Module):
    """输入输出均 [B,H,T,Dh]，内部自动padding，无因果mask"""
    def __init__(
        self,
        d_head: int,
        block_size: int,
        topk_blocks: int,
        router_proj_dim: int = 64,
        temp: float = 0.1,
        n_heads: int = 96
    ):
        super().__init__()
        self.d_head = d_head
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.temp = temp
        self.router = BlockRouter(d_head=d_head, proj_dim=router_proj_dim)
        self.alpha = nn.Parameter(torch.full((n_heads, 1), 0.5))

    def block_reshape(self, x: torch.Tensor):
        B, H, T, Dh = x.shape
        Nb = T // self.block_size
        return x.view(B, H, Nb, self.block_size, self.d_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, H, T_orig, Dh = q.shape
        bs = self.block_size
        pad_len = (bs - (T_orig % bs)) % bs
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len), value=0.0)
            k = F.pad(k, (0, 0, 0, pad_len), value=0.0)
            v = F.pad(v, (0, 0, 0, pad_len), value=0.0)
        T_pad = q.size(2)
        Nb = T_pad // bs

        q_pool = block_mean_pool(q, bs)
        k_pool = block_mean_pool(k, bs)
        score_mat = self.router(q_pool, k_pool)
        M_c = soft_topk_mask(score_mat, self.topk_blocks, self.temp, self.training)
        M_not = 1.0 - M_c

        q_blk = self.block_reshape(q)
        k_blk = self.block_reshape(k)
        v_blk = self.block_reshape(v)

        # sparse branch O_s
        M_c_exp = M_c.unsqueeze(3).unsqueeze(-1)
        M_c_exp = M_c_exp.expand(-1, -1, -1, bs, -1, bs)
        M_c_exp = M_c_exp.flatten(2, 3).flatten(-2, -1)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (Dh ** 0.5)
        attn_score = attn_score.masked_fill(M_c_exp < 1e-6, -1e9)
        attn_weight_s = F.softmax(attn_score, dim=-1)
        O_s = torch.matmul(attn_weight_s, v)

        # linear branch O_l phi=softmax
        phi_q = F.softmax(q, dim=-1)
        phi_k = F.softmax(k, dim=-1)

        phi_k_blk = phi_k.view(B, H, Nb, bs, Dh)
        phi_v_blk = v.view(B, H, Nb, bs, Dh)
        phi_k_masked = torch.einsum("bhqk,bhkd->bhqd", M_not, phi_k_blk)
        phi_v_masked = torch.einsum("bhqk,bhkd->bhqd", M_not, phi_v_blk)

        kv_l = torch.matmul(phi_k_masked.transpose(-1, -2), phi_v_masked)
        O_l_raw = torch.matmul(phi_q.unsqueeze(-2), kv_l).squeeze(-2)
        phi_k_sum = torch.sum(phi_k_masked, dim=-2)
        norm = torch.matmul(phi_q.unsqueeze(-1), phi_k_sum.unsqueeze(-1)).squeeze(-1)
        O_l = O_l_raw / (norm + 1e-8)

        # fusion
        alpha = torch.sigmoid(self.alpha)
        O = alpha.unsqueeze(0).unsqueeze(-1) * O_s + (1.0 - alpha.unsqueeze(0).unsqueeze(-1)) * O_l

        if pad_len > 0:
            O = O[:, :, :T_orig, :]
        return O

# ===================== SLA2Attention 【仅本类修改，其余全部原样保留】 =====================
class SLA2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.dim
        self.num_heads = args.n_heads
        self.head_dim = args.head_dim
        self.norm_eps = args.norm_eps

        self.to_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_dim, self.norm_eps)
        self.norm_k = RMSNorm(self.head_dim, self.norm_eps)
        self.to_out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.block_size_k = args.sla2_blkk
        self._sla2_inner: Optional[InnerBlockSLA2] = None

    def _get_topk_blocks(self, seq_len: int) -> int:
        ratio = self.args.sla2_topk_1m if seq_len <= 1_000_000 else self.args.sla2_topk_5m
        num_blk = seq_len // self.block_size_k
        return max(1, int(num_blk * ratio))

    def forward(self, x, rotary_emb):
        B, S, _ = x.shape
        q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))

        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)

        topk_blk = self._get_topk_blocks(S)
        if self._sla2_inner is None:
            self._sla2_inner = InnerBlockSLA2(
                d_head=self.head_dim,
                block_size=self.block_size_k,
                topk_blocks=topk_blk,
                router_proj_dim=64,
                temp=0.1,
                n_heads=self.num_heads
            )

        out = self._sla2_inner(q, k, v)
        return self.to_out(out.flatten(2, 3).type_as(x))

# ===================== Dense FFN =====================
class DenseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, inter_dim = args.dim, args.inter_dim
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ===================== Multi‑Head‑MoE 朴素实现 =====================
class MoEHeadExpert(nn.Module):
    """单个MoE‑head内的路由专家: hs → hs*mult → hs (SwiGLU)"""
    def __init__(self, hs: int, inter_mult: int):
        super().__init__()
        inter = hs * inter_mult
        self.w1 = nn.Linear(hs, inter, bias=False)
        self.w2 = nn.Linear(inter, hs, bias=False)
        self.w3 = nn.Linear(hs, inter, bias=False)

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return out * weight


class MultiHeadGate(nn.Module):
    """逐MoE‑head独立Gate；输入 [N, S, hs]，输出每个head的top‑k权重与专家索引"""
    def __init__(self, moe_num_heads: int, moe_head_hidden_size: int, experts_per_head: int, k_per_head: int):
        super().__init__()
        self.moe_num_heads = moe_num_heads
        self.hs = moe_head_hidden_size
        self.experts_per_head = experts_per_head
        self.k_per_head = k_per_head
        # 每个head独立router权重: [S, hs, Es]
        self.router = nn.Parameter(torch.empty(moe_num_heads, moe_head_hidden_size, experts_per_head))

    def forward(self, x: torch.Tensor):
        """
        x: [N, S, hs]
        return weights:[N, S, Ks], indices:[N, S, Ks]
        """
        # matmul per head: [N,S,hs] @ [S,hs,Es] -> [N,S,Es]
        logits = torch.einsum("nsh,she->nse", x, self.router)
        scores = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(scores, k=self.k_per_head, dim=-1)
        return weights, indices


class MultiHeadMoE(nn.Module):
    """
    MAGI‑2 preview风格 Multi‑Head‑MoE 朴素单卡实现
    1.输入投影 dim → S × hs
    2.每个MoE‑head独立Gate路由
    3.每个head内部执行专家计算
    4.聚合S个head输出回dim；叠加独立共享专家输出
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.dim = args.dim
        self.moe_s = args.moe_num_heads
        self.hs = args.moe_head_hidden_size
        self.experts_per_head = args.experts_per_head
        self.k_per_head = args.k_per_head
        # 输入投影 dim → S*hs
        self.proj_in = nn.Linear(self.dim, self.moe_s * self.hs, bias=False)
        # 输出聚合 S*hs → dim
        self.proj_out = nn.Linear(self.moe_s * self.hs, self.dim, bias=False)
        # multi‑head gate
        self.gate = MultiHeadGate(
            moe_num_heads=self.moe_s,
            moe_head_hidden_size=self.hs,
            experts_per_head=self.experts_per_head,
            k_per_head=self.k_per_head
        )
        # 每个head持有一组专家: ModuleList[S]，每个head内部是ModuleList[Es]
        self.per_head_experts = nn.ModuleList()
        for _ in range(self.moe_s):
            head_experts = nn.ModuleList([
                MoEHeadExpert(self.hs, args.moe_expert_inter_mult)
                for _ in range(self.experts_per_head)
            ])
            self.per_head_experts.append(head_experts)
        # Shared Expert SwiGLU dim‑shared_inter‑dim
        self.shared_w1 = nn.Linear(self.dim, args.moe_shared_inter_dim, bias=False)
        self.shared_w3 = nn.Linear(self.dim, args.moe_shared_inter_dim, bias=False)
        self.shared_w2 = nn.Linear(args.moe_shared_inter_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        n_token = B * T
        x_flat = x.reshape(n_token, D)
        # step1: project to multi‑head subspace [N, S*hs] -> [N, S, hs]
        x_sub = self.proj_in(x_flat).reshape(n_token, self.moe_s, self.hs)
        # step2: per‑head routing
        weights, indices = self.gate(x_sub)  # weights/indices: [N, S, Ks]
        # step3: per‑head expert compute
        moe_out_sub = torch.zeros_like(x_sub)  # [N, S, hs]
        for head_idx in range(self.moe_s):
            head_x = x_sub[:, head_idx, :]          # [N, hs]
            head_w = weights[:, head_idx, :]        # [N, Ks]
            head_idxes = indices[:, head_idx, :]    # [N, Ks]
            head_experts = self.per_head_experts[head_idx]
            cnt = torch.bincount(head_idxes.flatten(), minlength=self.experts_per_head).tolist()
            for eid in range(self.experts_per_head):
                if cnt[eid] == 0:
                    continue
                exp = head_experts[eid]
                row_pos, top_pos = torch.where(head_idxes == eid)
                moe_out_sub[row_pos, head_idx, :] += exp(head_x[row_pos], head_w[row_pos, top_pos, None])
        # step4: aggregate moe heads back to dim
        moe_out = self.proj_out(moe_out_sub.reshape(n_token, self.moe_s * self.hs))
        # step5: shared expert branch
        shared_out = self.shared_w2(F.silu(self.shared_w1(x_flat)) * self.shared_w3(x_flat))
        final = moe_out + shared_out
        return final.reshape(B, T, D)

# ===================== TransformerLayer =====================
class TransformerLayer(nn.Module):
    def __init__(self, layer_idx: int, args: ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = SLA2Attention(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        is_dense_layer = (layer_idx < 2) or (layer_idx >= args.n_layers - 2)
        if is_dense_layer:
            self.feed_forward = DenseFeedForward(args)
        else:
            self.feed_forward = MultiHeadMoE(layer_idx, args)

    def forward(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> Tuple[torch.Tensor, None]:
        h = x + self.attention(self.attention_norm(x), rotary_emb=rotary_emb)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, None

# ===================== 顶层模型 =====================
class HybridMultiHeadMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerLayer(i, args) for i in range(args.n_layers)])
        self.norm_final = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    @torch.no_grad()
    def precompute_freqs_cis(self, seq_len: int, device):
        dim = self.args.head_dim
        theta = self.args.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        x = self.embeddings(tokens)
        freqs_cis = self.precompute_freqs_cis(S, device=x.device)
        for layer in self.layers:
            x, _ = layer(x, rotary_emb=freqs_cis)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits


def build_model():
    args = ModelArgs()
    model = HybridMultiHeadMoEModel(args)
    return model, args


if __name__ == "__main__":
    model, args = build_model()
    print(f"n_layers={args.n_layers}, dim={args.dim}")
    print(f"moe_num_heads={args.moe_num_heads}, experts_per_head={args.experts_per_head}")
    print(f"k_per_head={args.k_per_head}")
    # 简短测试
    B = 1
    S = 197
    dummy_tokens = torch.randint(0, args.vocab_size, (B, S))
    logits = model(dummy_tokens)
    print(f"dummy forward pass ok, logits shape {logits.shape}")
