"""Generic block-diffusion LLM (model.py): parameter schema, network, decode
loop, runtime config, registration, and closed-form memory accounting in one
self-contained file.

One implementation, two configurations:
- dense SwiGLU FFN when `num_experts == 0`;
- top-k routed experts + 1 shared expert when `num_experts > 0`.
All sizes come from model_config/*.json; nothing is hard-wired to a checkpoint.

Two entry points on the model, kept strictly separated:
- forward(input_ids, cache_kv=None, is_causal=False) -> [bsz, seq_len, vocab]:
  ONE pass. is_causal=False (denoising): bidirectional attention over the
  KV-cached context plus the given tokens (the current canvas);
  is_causal=True (block boundary / AR): context fully visible, causal within
  the given tokens; single-token input needs no mask. Logits are produced at
  every input position.
- generate(input_ids, cache_kv=None) -> BlockDiffusionOutput: the decode loop.
  dlm mode runs D = time_step_per_block denoising passes — each pass commits
  the most confident predictions among the still-unresolved canvas positions,
  rewriting the canvas that conditions the next pass — then one causal
  boundary pass re-encodes the completed block (K/V for later blocks,
  last-position logits for the first token of the next block); nfe = D + 1.
  ar mode is the autoregressive baseline: one causal pass.

Pure PyTorch throughout; grouped-expert compute is expressed as gather + bmm
so the traced graph carries the same op shapes as a fused grouped GEMM.

Decode modes (via runtime_config.decode_mode): dlm (default) | ar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Iterator, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from workloads.models.base.base_config import BaseModelArgs, BaseRuntimeConfig
from analyzer.analyzers.utils import get_dtype_size
from zhanlu.entry.workload.model_register import ClassType, build_in_model


# ----------------------------------------------------------------- parameters


@build_in_model.register(name="MindIBlockDiffusion", class_type=ClassType.MODEL_CONFIG_CLASS)
@dataclass
class BlockDiffusionArgs(BaseModelArgs):
    """Block-diffusion LLM hyperparameters. Field names match the keys of
    model_config/*.json; defaults mirror the flagship 10.05T MoE member
    (model_config/block_diffusion_moe_10t.json).

    10.05T total parameters (99.5% in routed experts), 129B activated (1.28%),
    97 layers. Deployment requires TP x EP clusters (tp=8 / ep=512 -> 48.7 GiB
    per card; see README). Dense vs MoE is decided by `num_experts`.
    """

    model_type: str = "block_diffusion"
    model_name: str = "block_diffusion"
    archtype: str = "moe"            # informational; dense/MoE is decided by num_experts

    vocab_size: int = 262144
    hidden_size: int = 8192
    num_hidden_layers: int = 97
    num_attention_heads: int = 64
    num_key_value_heads: int = 8     # GQA (also the upper bound for tp_size)
    head_dim: int = 128
    intermediate_size: int = 14336   # dense FFN / MoE shared expert

    # MoE FFN: num_experts > 0 -> routed experts + 1 shared expert; 0 -> dense SwiGLU
    num_experts: int = 1024
    moe_intermediate_size: int = 4096
    num_experts_per_tok: int = 8     # top-k

    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144
    torch_dtype: str = "bfloat16"

    # diffusion fields
    block_size: int = 256            # tokens per block (the canvas)
    mask_token_id: int = 100         # placeholder token for unresolved canvas positions
    max_denoise_steps: int = 48      # upper bound on denoising passes per block

    def is_moe(self) -> bool:
        return self.num_experts > 0

    def h_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path):
        """Build from a config json, ignoring keys the dataclass does not define
        (HF-style config.json carries extra keys such as architectures /
        bos_token_id that would break a plain **kwargs constructor)."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {fl.name for fl in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@build_in_model.register(name="MindIBlockDiffusion", class_type=ClassType.RUNTIME_COMFIG_CLASS)
class RuntimeConfig(BaseRuntimeConfig):
    """Per-run deployment knobs. Parallelism splits:

    - tp_size shards attention heads and the vocab dimension of embedding /
      output head (upper bound: num_key_value_heads);
    - ep_size shards routed experts across cards (num_experts // ep_size
      experts per card).

    Large configs need both: the flagship weighs 18.3 TiB in bf16.
    """

    def __init__(self, config: BlockDiffusionArgs):
        super().__init__()
        self.config = config

        self.input_dtype = torch.bfloat16
        self.weight_dtype = torch.bfloat16
        self.vector_dtype = torch.bfloat16
        self.softmax_dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16

        self.decode_mode = "dlm"  # dlm | ar
        self.batch_size = 1
        self.global_batch_size = 8
        self.num_ranks = 1
        self.tp_size = 1
        self.ep_size = 1

        self.seq_len = 1
        self.max_new_tokens = 256
        self.cache_len = 4608
        self.block_length = config.block_size
        self.time_step_per_block = 2
        # kept for analyzer/test compatibility (unused by the dlm/ar paths)
        self.threshold = 0.0
        self.acceptance_length = 8.0
        self.is_causal = True
        self.is_prefill = False
        self.is_in_server = False

    # dp_size must track num_ranks unless explicitly overridden after
    # construction (the analyzer may write it back). A plain attribute would
    # go stale when num_ranks changes, so it is a property with a setter.
    @property
    def dp_size(self):
        return self.__dict__.get("_dp_size", self.num_ranks)

    @dp_size.setter
    def dp_size(self, value):
        self.__dict__["_dp_size"] = value


# ------------------------------------------------------------- position/masks


def precompute_rope_cache(head_dim: int, max_positions: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for rotary position embeddings.

    RoPE encodes absolute position as rotation: the dimension pair
    (x_{2i}, x_{2i+1}) of every query/key vector is rotated by m * theta_i at
    position m, with theta_i = theta^(-2i/d). Inner products then depend only
    on the position difference m - n, which is exactly the position dependence
    attention needs. Returns (cos, sin), each [max_positions, head_dim // 2].
    """
    assert head_dim % 2 == 0, "RoPE pairs dimensions; head_dim must be even"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each dimension pair of x by its per-position angle, in real
    arithmetic: for the pair (x_i, x_j) rotated by angle a,
    (x_i', x_j') = (x_i·cos a − x_j·sin a, x_i·sin a + x_j·cos a).

    x: [b, h, s, head_dim]; cos/sin: [s, head_dim // 2]. Dimensions are paired
    half-against-half (i with i + d/2). Computed in fp32 for numerical
    stability, cast back to the input dtype on exit.
    """
    s = x.size(2)
    cos = torch.cat([cos[:s], cos[:s]], dim=-1)[None, None]   # [1, 1, s, d]
    sin = torch.cat([sin[:s], sin[:s]], dim=-1)[None, None]
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotate_half = torch.cat((-x2, x1), dim=-1)
    return (xf * cos + rotate_half * sin).to(x.dtype)


def block_prefix_attn_mask(
    prefix_len: int, block_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive attention mask for the block-boundary causal pass: every block
    row sees the whole KV-cache prefix and attends causally *within* the block.

    Single-token decode (block_len == 1) needs no mask at all.
    Returns [block_len, prefix_len + block_len], 0 where visible / -inf where masked.
    """
    mask = torch.zeros(block_len, prefix_len + block_len, dtype=torch.float32, device=device)
    tri = torch.full((block_len, block_len), float("-inf"))
    mask[:, prefix_len:] = torch.triu(tri, diagonal=1)
    return mask.to(dtype)


# ------------------------------------------------------------------- modules


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Statistics are computed in fp32 and the
    output is cast back to the input dtype: mean-of-squares over bf16 loses
    precision for long hidden vectors."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class Attention(nn.Module):
    """Grouped-query attention with an explicit head_dim (hidden_size need not
    equal heads x head_dim).

    Pass semantics:
    - denoising (is_causal=False): bidirectional over (cache prefix + block);
    - block boundary (is_causal=True, seq_len > 1): prefix fully visible,
      causal within the block (explicit additive mask);
    - single-token AR decode: attends everything, no mask.
    """

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads // max(1, runtime_config.tp_size)
        self.n_kv_heads = max(1, config.num_key_value_heads // max(1, runtime_config.tp_size))
        assert self.n_heads % self.n_kv_heads == 0, "GQA requires n_heads % n_kv_heads == 0"
        self.head_dim = config.h_head_dim()
        self.softmax_dtype = runtime_config.softmax_dtype

        qkv_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, qkv_dim, bias=config.attention_bias,
            dtype=runtime_config.weight_dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, kv_dim, bias=config.attention_bias,
            dtype=runtime_config.weight_dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, kv_dim, bias=config.attention_bias,
            dtype=runtime_config.weight_dtype,
        )
        self.o_proj = nn.Linear(
            qkv_dim, config.hidden_size, bias=config.attention_bias,
            dtype=runtime_config.weight_dtype,
        )
        cos, sin = precompute_rope_cache(
            self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, is_causal: bool, cache_kv=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, self.rope_cos[:seq_len], self.rope_sin[:seq_len])
        k = apply_rotary_emb(k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

        # Denoising and boundary passes attend the finished context (KV cache)
        # plus the current block; the cache is concatenated on the sequence axis.
        prefix_len = 0
        if cache_kv is not None:
            k = torch.cat([cache_kv[0], k], dim=2)
            v = torch.cat([cache_kv[1], v], dim=2)
            prefix_len = cache_kv[0].size(2)

        # GQA: query heads are processed in n_kv_heads groups that share one
        # key/value head each; the group axis broadcasts without materializing
        # repeated copies of K/V.
        group = self.n_heads // self.n_kv_heads
        q4 = q.view(bsz, self.n_kv_heads, group, seq_len, self.head_dim)
        k4 = k.unsqueeze(2)                                          # [b, kvh, 1, sk, d]
        scores = torch.matmul(q4, k4.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if is_causal and seq_len > 1:
            scores = scores + block_prefix_attn_mask(
                prefix_len, seq_len, scores.dtype, scores.device).view(1, 1, 1, seq_len, -1)
        probs = F.softmax(scores, dim=-1, dtype=self.softmax_dtype).to(v.dtype)
        attn = torch.matmul(probs, v.unsqueeze(2))                   # [b, kvh, group, s, d]
        attn = attn.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn)


class MLP(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x)). Doubles as the MoE shared expert."""

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig,
                 intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, inter, bias=config.mlp_bias,
            dtype=runtime_config.weight_dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size, inter, bias=config.mlp_bias,
            dtype=runtime_config.weight_dtype,
        )
        self.down_proj = nn.Linear(
            inter, config.hidden_size, bias=config.mlp_bias,
            dtype=runtime_config.weight_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """MoE router: top-k expert selection with renormalized routing weights.
    Scoring runs in fp32 (logit magnitudes across 1000+ experts vary widely)."""

    def __init__(self, config: BlockDiffusionArgs):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices


class RoutedExperts(nn.Module):
    """Routed-expert weights, stacked per local expert: w13 [E_l, D, 2I] and
    w2 [E_l, I, D], with E_l = num_experts // ep_size experts resident per card.

    forward treats routing as (token, expert) pairs and evaluates each pair as
    an independent SwiGLU GEMM against its expert's weights (gather + bmm).
    Per-pair weight reads match the traffic of a grouped GEMM with one group
    per pair, so the traced graph is cost-equivalent to a fused implementation.
    """

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        e_local = max(1, config.num_experts // max(1, runtime_config.ep_size))
        self.n_local_experts = e_local
        self.w13 = nn.Parameter(torch.empty(
            e_local, config.hidden_size, 2 * config.moe_intermediate_size,
            dtype=runtime_config.weight_dtype,
        ))
        self.w2 = nn.Parameter(torch.empty(
            e_local, config.moe_intermediate_size, config.hidden_size,
            dtype=runtime_config.weight_dtype,
        ))

    def forward(self, x_pairs: torch.Tensor, pair_weights: torch.Tensor) -> torch.Tensor:
        """x_pairs: [P, D] (token, expert) pairs; pair_weights: [P] routing weights."""
        p = x_pairs.size(0)
        # Deterministic round-robin expert assignment. Real routing is
        # data-dependent; the round-robin keeps the pair count exact (P full
        # expert GEMMs on this card) while every op stays shape-static, which
        # meta-device tracing requires.
        e_idx = torch.arange(p, device=x_pairs.device) % self.n_local_experts
        h = torch.bmm(x_pairs.unsqueeze(1), self.w13[e_idx]).squeeze(1)  # [P, 2I]
        gate, up = h.chunk(2, dim=-1)
        act = F.silu(gate) * up
        y = torch.bmm(act.unsqueeze(1), self.w2[e_idx]).squeeze(1)       # [P, D]
        # Routing weights are computed in fp32; scale back in the activation
        # dtype so fp32 does not leak into subsequent layers.
        return y * pair_weights.to(y.dtype).unsqueeze(-1)


class MoE(nn.Module):
    """Top-k routed experts plus one always-active shared expert.

    EP accounting: a card evaluates only the (token, expert) pairs routed to
    its resident experts, approximately N * top_k / ep_size pairs for N tokens
    (dispatch / all-to-all communication is not modeled; zero-padding the
    partial routed output back to N rows is a zero-cost shape op). With
    ep_size == 1 the card holds every expert and the reduction is exact:
    each token contributes exactly top_k pairs.
    """

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.ep_size = max(1, runtime_config.ep_size)
        self.gate = Gate(config)
        self.experts = RoutedExperts(config, runtime_config)
        # The shared expert is replicated on every card (accounted as /tp).
        self.shared_expert = MLP(config, runtime_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        tokens = x.reshape(-1, x.shape[-1])
        n = tokens.size(0)

        weights, _indices = self.gate(tokens)
        # Materialize every (token, expert) pair: each token spawns exactly
        # top_k pairs, then keep this card's share of them.
        pairs_x = tokens.unsqueeze(1).expand(n, self.top_k, tokens.size(-1)).reshape(n * self.top_k, -1)
        pairs_w = weights.reshape(n * self.top_k)

        p_local = max(1, n * self.top_k // self.ep_size)
        routed = self.experts(pairs_x[:p_local], pairs_w[:p_local])
        if p_local == n * self.top_k:
            routed = routed.view(n, self.top_k, -1).sum(dim=1)  # exact per-token reduction
        else:
            if routed.shape[0] < n:  # zero-pad the partial result back to N rows
                routed = torch.cat(
                    [routed, routed.new_zeros(n - routed.shape[0], routed.shape[1])], 0)
            routed = routed[:n]

        out = routed + self.shared_expert(tokens)
        return out.reshape(bsz, seq_len, -1)


class Block(nn.Module):
    """Pre-norm residual block: norm -> attention -> residual, norm -> FFN -> residual."""

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, runtime_config)
        if config.is_moe():
            self.mlp = MoE(config, runtime_config)
        else:
            self.mlp = MLP(config, runtime_config)

    def forward(self, x: torch.Tensor, is_causal: bool, cache_kv=None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, is_causal=is_causal, cache_kv=cache_kv)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Backbone(nn.Module):
    """Token embedding, transformer blocks, final norm."""

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        tp = max(1, runtime_config.tp_size)
        self.embed_tokens = nn.Embedding(
            config.vocab_size // tp, config.hidden_size, dtype=runtime_config.input_dtype,
        )
        self.layers = nn.ModuleList(
            [Block(config, runtime_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, is_causal: bool, cache_kv=None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, is_causal=is_causal, cache_kv=cache_kv)
        return self.norm(x)


@dataclass
class BlockDiffusionOutput:
    """Result of decoding one block: last-position logits and the number of
    forward passes (network function evaluations) spent."""

    logits: torch.Tensor
    nfe: int


def commit_schedule(block_len: int, steps: int) -> Iterator[Tuple[int, int]]:
    """Proportional commit schedule for one block: yields, per denoising
    pass, (m, c) = (positions still unresolved at pass entry, positions to
    commit in this pass), with ceil(L*k/steps) cumulative commits after
    pass k. Guarantees every position is committed by the final pass.
    Depends only on Python ints, so every tensor shape in the loop is static,
    which meta-device tracing requires."""
    prev = 0
    for k in range(1, steps + 1):
        cur = math.ceil(block_len * k / steps)
        yield block_len - prev, cur - prev
        prev = cur


# --------------------------------------------------------------------- model


@build_in_model.register(name="MindIBlockDiffusion", class_type=ClassType.MODEL_CLASS)
class BlockDiffusionLM(nn.Module):
    """Block-diffusion LM: backbone + output head + decode loop.

    forward is one pass (see module docstring); generate drives the decode
    loop on top of it.
    """

    def __init__(self, config: BlockDiffusionArgs, runtime_config: RuntimeConfig):
        super().__init__()
        self.config = config
        self.runtime_config = runtime_config
        self.backbone = Backbone(config, runtime_config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size // max(1, runtime_config.tp_size),
            bias=False, dtype=runtime_config.input_dtype,
        )

    # ------------------------------------------------------------------ pass

    def forward(self, input_ids: torch.Tensor, cache_kv=None, is_causal: bool = False) -> torch.Tensor:
        """One forward pass; returns logits at every input position:
        [bsz, seq_len, vocab_size // tp_size].

        is_causal=False: bidirectional over (cache prefix + input) — the
        denoising-pass semantics. is_causal=True: prefix fully visible, causal
        within the input — the block-boundary / AR semantics. The canvas is
        re-embedded on every call: bidirectional attention means hidden states
        cannot be cached across canvas revisions.
        """
        hidden = self.backbone(input_ids, is_causal=is_causal, cache_kv=cache_kv)
        return self.lm_head(hidden)

    # ----------------------------------------------------------------- decode

    def generate(self, input_ids: torch.Tensor, cache_kv=None) -> BlockDiffusionOutput:
        """Decode one block according to runtime_config.decode_mode."""
        mode = str(self.runtime_config.decode_mode).lower()
        if mode == "ar":
            return self._generate_ar(input_ids, cache_kv)
        if mode == "dlm":
            return self._generate_dlm(input_ids, cache_kv)
        raise ValueError(f"Unsupported decode_mode '{self.runtime_config.decode_mode}'")

    def _generate_dlm(self, input_ids: torch.Tensor, cache_kv=None) -> BlockDiffusionOutput:
        """Denoise the current block, then re-encode it at the block boundary.

        State carried across denoising passes: the canvas itself (committed
        tokens replace unresolved placeholders via scatter, so each pass
        conditions on a more-resolved block) and the unresolved-position flags
        (which positions still need predictions). The boundary pass consumes
        the completed block — every position committed — re-encoding it
        causally for the KV cache and emitting last-position logits for the
        first token of the next block.
        """
        bsz, block_len = input_ids.shape
        steps = max(1, int(self.runtime_config.time_step_per_block))
        steps = min(steps, int(self.config.max_denoise_steps))

        unresolved = torch.ones(bsz, block_len, dtype=torch.int8, device=input_ids.device)
        for m_before, n_commit in commit_schedule(block_len, steps):
            # Denoising pass: one bidirectional forward over (cache + canvas);
            # re-embedding and re-encoding the whole canvas is required because
            # bidirectional attention invalidates all hidden states whenever
            # any canvas position changes.
            logits = self.forward(input_ids, is_causal=False, cache_kv=cache_kv)  # [bsz, L, V]
            # Consider only the still-unresolved positions (sorted to the front).
            order = unresolved.argsort(dim=1, descending=True, stable=True)
            masked_pos = order[:, :m_before]
            sel = masked_pos.unsqueeze(-1).expand(*masked_pos.shape, logits.size(-1))
            sel_logits = logits.gather(1, sel)                         # [bsz, m, V]
            # Commit the n_commit most confident predictions into the canvas.
            conf, tok = sel_logits.max(dim=-1)
            top = conf.topk(n_commit, dim=-1).indices                  # [bsz, c]
            pos = masked_pos.gather(1, top)
            input_ids = input_ids.scatter(1, pos, tok.gather(1, top))
            unresolved = unresolved.scatter(
                1, pos, torch.zeros_like(unresolved.gather(1, pos)))

        # Block boundary: causal re-encode of the completed block; only the
        # last position's logits are needed (first token of the next block).
        logits = self.forward(input_ids, is_causal=True, cache_kv=cache_kv)
        return BlockDiffusionOutput(logits=logits[:, -1:, :], nfe=steps + 1)

    def _generate_ar(self, input_ids: torch.Tensor, cache_kv=None) -> BlockDiffusionOutput:
        """Autoregressive baseline: one causal pass, last-position logits."""
        logits = self.forward(input_ids, is_causal=True, cache_kv=cache_kv)
        return BlockDiffusionOutput(logits=logits[:, -1:, :], nfe=1)

    # ------------------------------------------------------------------ memory

    @staticmethod
    def kv_size(args: BlockDiffusionArgs, runtime_config: RuntimeConfig) -> float:
        """KV-cache bytes (GiB): 2 (K and V) x batch x cache_len x kv_heads x
        head_dim x layers x itemsize."""
        itemsize = get_dtype_size(runtime_config.kv_cache_dtype)
        kv_cache = (
            2
            * runtime_config.batch_size
            * runtime_config.cache_len
            * args.num_key_value_heads
            * args.h_head_dim()
            * args.num_hidden_layers
            * itemsize
        )
        return kv_cache / 1024 ** 3

    @staticmethod
    def weight_size(args: BlockDiffusionArgs, runtime_config: RuntimeConfig) -> float:
        """Per-card weights (GiB). Sharding: attention + embedding + output
        head follow tp; routed experts follow ep; the replicated shared expert
        is approximated as /tp. With untied embeddings the output head is a
        second vocab x hidden matrix next to the embedding."""
        w = get_dtype_size(runtime_config.weight_dtype)
        hidden = args.hidden_size
        head_dim = args.h_head_dim()
        tp = max(1, runtime_config.tp_size)
        ep = max(1, runtime_config.ep_size)

        embedding = args.vocab_size * hidden * w / tp
        head = 0 if args.tie_word_embeddings else args.vocab_size * hidden * w / tp
        qo = args.num_attention_heads * head_dim * hidden * 2
        kv = args.num_key_value_heads * head_dim * hidden * 2
        attn = (qo + kv) * args.num_hidden_layers * w / tp

        if args.is_moe():
            routed = args.num_experts * 3 * hidden * args.moe_intermediate_size \
                * args.num_hidden_layers * w / ep
            shared = 3 * hidden * args.intermediate_size * args.num_hidden_layers * w / tp
            ffn = routed + shared
        else:
            ffn = 3 * hidden * args.intermediate_size * args.num_hidden_layers * w / tp

        return (embedding + head + attn + ffn) / 1024 ** 3

    def memory_size(self):
        args, rc = self.config, self.runtime_config
        return self.weight_size(args, rc) + self.kv_size(args, rc)


if __name__ == "__main__":
    # CPU smoke test: single-pass forward, then both decode modes through the
    # generation mixin; checks output shapes and the per-block pass count.
    torch.set_default_dtype(torch.bfloat16)
    args = BlockDiffusionArgs(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, num_experts=0,
        block_size=8, mask_token_id=100, max_position_embeddings=512,
    )
    runtime = RuntimeConfig(args)
    model = BlockDiffusionLM(args, runtime)

    canvas = torch.full((2, args.block_size), args.mask_token_id, dtype=torch.long)
    cache = torch.rand(2, 1, 2, 16, 16)  # [2(K/V), bsz, kv_heads, cache_len, head_dim]

    logits = model(canvas, cache_kv=cache, is_causal=False)
    print("single-pass logits:", logits.shape)

    runtime.decode_mode = "dlm"
    out = model.generate(canvas, cache_kv=cache)
    print("dlm logits:", out.logits.shape, "nfe:", out.nfe)

    runtime.decode_mode = "ar"
    out = model.generate(torch.randint(0, args.vocab_size, (2, 8)))
    print("ar  logits:", out.logits.shape, "nfe:", out.nfe)
