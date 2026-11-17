# Copyright (c) 2026 Huawei Technologies Co., Ltd. all rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this tree.

"""SLA2Attention + Multi-Head-MoE decoder (MAGI-2 preview style).

Faithful port of the external reference ``multi-head_moe.py`` (pure-PyTorch
naive implementation). The SLA2 attention path executes the reference's own
small eager operators (``apply_rotary_emb``, local ``RMSNorm``, block pooling,
block router, soft top-k mask, the two attention branches and the alpha mix);
nothing on that path is replaced by fused or virtual operators.

Approved deviations from the reference (all documented, none substitutes an
operator kind):

- Three latent shape bugs make the reference unrunnable; the intended math is
  implemented with the layouts the reference docstrings state:
  RoPE broadcasts ``freqs_cis`` over ``unsqueeze(1)`` (reference had
  ``unsqueeze(2)``), the linear-branch einsum is ``"bhqk,bhkld->bhqld"``
  (reference subscript was illegal), and ``SLA2Attention`` transposes at the
  ``[B,S,H,D]/[B,H,T,D]`` boundary the reference crossed silently.
- ``InnerBlockSLA2`` receives ``topk_blocks`` per call instead of caching it
  in a lazily-constructed module (the reference caches the first call's
  value, which goes stale when the sequence length changes).
- Router/gate weights get a standard small-std normal init (the reference
  leaves ``torch.empty`` buffers uninitialized); ``alpha`` stays at the
  reference's 0.5.
- The naive per-head/per-expert Python loop of the reference's Multi-Head-MoE
  is expressed through the shared token dispatcher + grouped-GEMM experts in
  BOTH modes (owner decision: the reference's ``bincount``/``torch.where``
  cannot run under Meta capture — their output shapes depend on tensor
  values — and grouped GEMM is the production expert execution).

Fusion boundary: the two module-level seam functions
``run_sla2_sparse_attention`` / ``run_sla2_linear_attention`` (the sparse and
linear branch cores). Everything around them stays native dispatch in both
modes: block pooling, block router, soft top-k mask, ``M_not = 1 - M_c``,
padding/crop, the alpha branch mix, RoPE and the norms. The simulator's
``SimulationConfig.enable_sla2_fusion`` switch (default OFF) replaces exactly
these two functions with virtual fused ops for Meta capture
(``torchtitan_simulator.compat.multi_head_moe``).

Shape suffixes: B batch, S sequence, H attention heads, D model dim,
Dh head dim, T tokens, Hm MoE heads, hs MoE head hidden, Es experts per
head, Ks activated experts per head, E total routed experts.
"""

import dataclasses
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "BlockRouter",
    "InnerBlockSLA2",
    "MultiHeadGate",
    "MultiHeadMoE",
    "MultiHeadMoEModel",
    "RMSNorm",
    "SLA2Attention",
    "TransformerBlock",
    "apply_rotary_emb",
    "run_sla2_linear_attention",
    "run_sla2_sparse_attention",
]


@dataclass(kw_only=True, slots=True)
class RopeGeometry:
    """RoPE geometry metadata the framework Decoder contract reads.

    Pure configuration (``max_seq_len`` validation/cache bookkeeping); the
    SLA2 path computes RoPE frequencies via the reference's
    ``precompute_freqs_cis`` and builds no RoPE module.
    """

    max_seq_len: int = 4096


def _replicated_sharding(parameter: str) -> ShardingConfig:
    """Replicated state placement for model-owned plain parameters."""
    return ShardingConfig(state_shardings={parameter: dense_param_placement(tp=spmd.R)})


def _apply_rotary_emb_math(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs_cis.unsqueeze(1)).flatten(3)
    return out.type_as(x)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Reference ``apply_rotary_emb``; ``unsqueeze(1)`` fixes its broadcast bug.

    The reference multiplies by ``freqs_cis.unsqueeze(2)`` while ``freqs_cis``
    is ``[S, Dh/2]`` and ``x`` is ``[B, S, H, Dh]``; dim 1 is the sequence
    axis, so ``unsqueeze(1)`` is the broadcast the docstrings intend. The
    reference disables autocast around the math; Meta tensors carry no
    autocast state, so Meta capture skips that context (behavior-identical:
    there is nothing to disable) while real devices keep it.
    """
    if x.device.type == "meta":
        return _apply_rotary_emb_math(x, freqs_cis)
    with torch.amp.autocast(x.device.type, enabled=False):
        return _apply_rotary_emb_math(x, freqs_cis)


class RMSNorm(Module):
    """Reference RMSNorm: ``x * rsqrt(mean(x^2) + eps) * weight`` in fp32."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("weight")

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = torch.nn.Parameter(torch.ones(config.dim))

    def reset_parameters(self) -> None:
        # Reference init: ``torch.ones(dim)``.
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ========== 内联块SLA2核心实现（带自动padding，源码小算子） ==========


def run_sla2_sparse_attention(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    block_mask_BHqq: torch.Tensor,
    selected_blocks: int,
) -> torch.Tensor:
    """Fusion boundary of the SLA2 sparse (block-selected softmax) branch.

    Default eager path: the reference's own small operators below (full score
    matrix, 1/sqrt(Dh) scaling after the score matmul, -1e9 fill, softmax).
    ``selected_blocks`` is the per-row selected key-block count consumed by
    the opt-in fused boundary (``SimulationConfig.enable_sla2_fusion``); the
    eager math materializes the full matrix and ignores the count.
    """
    bs = q_BHTD.size(-2) // block_mask_BHqq.size(-2)
    mask_c_exp = block_mask_BHqq.unsqueeze(3).unsqueeze(-1)
    mask_c_exp = mask_c_exp.expand(-1, -1, -1, bs, -1, bs)
    mask_c_exp = mask_c_exp.flatten(2, 3).flatten(-2, -1)
    attn_score = torch.matmul(q_BHTD, k_BHTD.transpose(-1, -2)) / (q_BHTD.size(-1) ** 0.5)
    attn_score = attn_score.masked_fill(mask_c_exp < 1e-6, -1e9)
    attn_weight_s = F.softmax(attn_score, dim=-1)
    return torch.matmul(attn_weight_s, v_BHTD)


def run_sla2_linear_attention(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    complement_mask_BHqq: torch.Tensor,
    selected_blocks: int,
) -> torch.Tensor:
    """Fusion boundary of the SLA2 linear (softmax-feature) branch.

    Default eager path: per query block, a KV state is accumulated over the
    complement-masked key blocks (``M_not = 1 - M_c``), then every token reads
    its block state and normalizes by the matching feature mass (einsum fixed
    per the documented bug). ``selected_blocks`` is the per-row complement
    key-block count consumed by the opt-in fused boundary; the eager math
    materializes the per-block states and ignores the count.
    """
    B, H, T, Dh = q_BHTD.shape
    bs = T // complement_mask_BHqq.size(-2)
    nb = complement_mask_BHqq.size(-2)
    phi_q = F.softmax(q_BHTD, dim=-1)
    phi_k = F.softmax(k_BHTD, dim=-1)
    phi_k_blk = phi_k.view(B, H, nb, bs, Dh)
    phi_v_blk = v_BHTD.view(B, H, nb, bs, Dh)
    phi_k_masked = torch.einsum("bhqk,bhkld->bhqld", complement_mask_BHqq, phi_k_blk)
    phi_v_masked = torch.einsum("bhqk,bhkld->bhqld", complement_mask_BHqq, phi_v_blk)
    kv_state = torch.matmul(phi_k_masked.transpose(-1, -2), phi_v_masked)
    kv_state = kv_state.repeat_interleave(bs, dim=2)
    out_raw = torch.matmul(phi_q.unsqueeze(-2), kv_state).squeeze(-2)
    phi_k_sum = phi_k_masked.sum(dim=-2).repeat_interleave(bs, dim=2)
    norm = torch.matmul(phi_q.unsqueeze(-2), phi_k_sum.unsqueeze(-1)).squeeze(-1)
    return out_raw / (norm + 1e-8)


class BlockRouter(Module):
    """Block-pair scorer for SLA2 block selection (proj_q/proj_k + dot)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        proj_q: Linear.Config
        proj_k: Linear.Config

        def __post_init__(self) -> None:
            if self.proj_q.in_features != self.proj_k.in_features:
                raise ValueError("BlockRouter projections must share the head width")
            if self.proj_q.out_features != self.proj_k.out_features:
                raise ValueError("BlockRouter projections must share the score width")

    def __init__(self, config: Config):
        super().__init__()
        self.proj_q = config.proj_q.build()
        self.proj_k = config.proj_k.build()

    def forward(self, q_pool_BHbD: torch.Tensor, k_pool_BHbD: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.proj_q(q_pool_BHbD), self.proj_k(k_pool_BHbD).transpose(-1, -2))


class InnerBlockSLA2(Module):
    """The SLA2 core: block-sparse softmax branch + masked linear branch.

    Inputs and outputs are ``[B, H, T, Dh]``; padding to a block multiple and
    the final crop happen here exactly like the reference. There is no causal
    mask (reference docstring: "无因果mask"). The alpha branch mix stays
    inside the core on the ``[B, H, T, D]`` head axis, like the reference.
    The two branch computations live in the module-level seam functions
    (the fusion boundary); this forward owns pooling, routing, masking and
    the mix.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        d_head: int
        n_heads: int
        block_size: int
        router: BlockRouter.Config
        temp: float = 0.1

        def __post_init__(self) -> None:
            if self.block_size <= 0:
                raise ValueError(f"SLA2 block_size must be positive, got {self.block_size}")
            if self.router.proj_q.in_features != self.d_head:
                raise ValueError("the block router must score the head width")
            if self.sharding_config is None:
                # alpha is a plain model-owned parameter; declare it replicated
                # so the spmd_types path distributes it as a DTensor before
                # FSDP wraps it.
                self.sharding_config = _replicated_sharding("alpha")

    def __init__(self, config: Config):
        super().__init__()
        self.block_size = config.block_size
        self.temp = config.temp
        self.router = config.router.build()
        self.alpha = torch.nn.Parameter(torch.full((config.n_heads, 1), 0.5))

    def reset_parameters(self) -> None:
        # The reference leaves alpha at 0.5 (sigmoid ~= 0.62 branch mix) and
        # never initializes the router weights (``torch.empty``); this port
        # gives the router a standard small-std normal init so training starts
        # from a defined state.
        torch.nn.init.normal_(self.router.proj_q.weight, std=0.02)
        torch.nn.init.normal_(self.router.proj_k.weight, std=0.02)

    def _soft_topk_mask(self, score_BHqq: torch.Tensor, topk_blocks: int) -> torch.Tensor:
        """Training: gumbel-softmax soft mask; eval: hard top-k scatter."""
        if self.training:
            logits = score_BHqq / self.temp
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            logits = logits + gumbel
            soft_mask = F.softmax(logits, dim=-1)
            vals, _ = torch.topk(soft_mask, k=topk_blocks, dim=-1)
            thresh = vals[..., [-1]]
            return torch.sigmoid((soft_mask - thresh) / self.temp)
        _, top_idx = torch.topk(score_BHqq, k=topk_blocks, dim=-1)
        mask = torch.zeros_like(score_BHqq)
        mask.scatter_(-1, top_idx, 1.0)
        return mask

    def forward(
        self,
        q_BHTD: torch.Tensor,
        k_BHTD: torch.Tensor,
        v_BHTD: torch.Tensor,
        *,
        topk_blocks: int,
    ) -> torch.Tensor:
        B, H, T_orig, Dh = q_BHTD.shape
        bs = self.block_size
        pad_len = (bs - (T_orig % bs)) % bs
        if pad_len > 0:
            q_BHTD = F.pad(q_BHTD, (0, 0, 0, pad_len), value=0.0)
            k_BHTD = F.pad(k_BHTD, (0, 0, 0, pad_len), value=0.0)
            v_BHTD = F.pad(v_BHTD, (0, 0, 0, pad_len), value=0.0)
        T_pad = q_BHTD.size(2)
        nb = T_pad // bs
        if not 1 <= topk_blocks <= nb:
            raise ValueError(f"topk_blocks must be in [1, {nb}], got {topk_blocks}")

        q_pool = q_BHTD.view(B, H, nb, bs, Dh).mean(dim=-2)
        k_pool = k_BHTD.view(B, H, nb, bs, Dh).mean(dim=-2)
        score_BHqq = self.router(q_pool, k_pool)
        mask_c = self._soft_topk_mask(score_BHqq, topk_blocks)
        mask_not = 1.0 - mask_c

        # The two branch seams are the fusion boundary: eager small operators
        # by default; the simulator's opt-in ``enable_sla2_fusion`` replaces
        # exactly these two functions with virtual fused ops for capture.
        out_sparse = run_sla2_sparse_attention(q_BHTD, k_BHTD, v_BHTD, mask_c, topk_blocks)
        out_linear = run_sla2_linear_attention(q_BHTD, k_BHTD, v_BHTD, mask_not, nb - topk_blocks)

        # fusion: per-head branch mix on the [B, H, T, D] head axis
        alpha = torch.sigmoid(self.alpha)
        out = alpha.unsqueeze(0).unsqueeze(-1) * out_sparse + (1.0 - alpha.unsqueeze(0).unsqueeze(-1)) * out_linear

        if pad_len > 0:
            out = out[:, :, :T_orig, :]
        return out


class SLA2Attention(Module):
    """SLA2 attention: QKV projections, per-head QK-norm, RoPE, SLA2 core.

    The reference feeds ``InnerBlockSLA2`` a ``[B, S, H, Dh]`` tensor while
    the core (and its own docstring) expects ``[B, H, T, Dh]``; this port
    transposes at the boundary (documented fix). RoPE is the reference's own
    complex-multiply ``apply_rotary_emb``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        head_dim: int
        block_size: int
        rope_theta: float = 10000.0
        # Framework-contract metadata only (see RopeGeometry); never built.
        rope: RopeGeometry = dataclasses.field(default_factory=RopeGeometry)
        sla2_topk_ratio_1m: float = 0.005
        sla2_topk_ratio_5m: float = 0.001
        to_q: Linear.Config
        to_k: Linear.Config
        to_v: Linear.Config
        to_out: Linear.Config
        norm_q: RMSNorm.Config
        norm_k: RMSNorm.Config
        inner: InnerBlockSLA2.Config

        def __post_init__(self) -> None:
            if self.to_q.out_features != self.n_heads * self.head_dim:
                raise ValueError("to_q must project to n_heads * head_dim")
            if self.inner.d_head != self.head_dim or self.inner.n_heads != self.n_heads:
                raise ValueError("the inner SLA2 core geometry must match the attention geometry")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.rope_theta = config.rope_theta
        self.sla2_topk_ratio_1m = config.sla2_topk_ratio_1m
        self.sla2_topk_ratio_5m = config.sla2_topk_ratio_5m

        self.to_q = config.to_q.build()
        self.to_k = config.to_k.build()
        self.to_v = config.to_v.build()
        self.to_out = config.to_out.build()
        self.norm_q = config.norm_q.build()
        self.norm_k = config.norm_k.build()
        self.inner = config.inner.build()

    def _get_topk_blocks(self, seq_len: int) -> int:
        ratio = self.sla2_topk_ratio_1m if seq_len <= 1_000_000 else self.sla2_topk_ratio_5m
        num_blk = seq_len // self.block_size
        return max(1, int(num_blk * ratio))

    def forward(
        self,
        x_BSD: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        _, S, _ = x_BSD.shape
        q = self.to_q(x_BSD).unflatten(2, (self.n_heads, self.head_dim))
        k = self.to_k(x_BSD).unflatten(2, (self.n_heads, self.head_dim))
        v = self.to_v(x_BSD).unflatten(2, (self.n_heads, self.head_dim))

        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)

        # [B, H, T, Dh] for the SLA2 core (documented layout fix)
        q_BHTD = q.transpose(1, 2)
        k_BHTD = k.transpose(1, 2)
        v_BHTD = v.transpose(1, 2)

        out_BHTD = self.inner(q_BHTD, k_BHTD, v_BHTD, topk_blocks=self._get_topk_blocks(S))
        return self.to_out(out_BHTD.transpose(1, 2).flatten(2, 3).type_as(x_BSD))


class MultiHeadGate(Module):
    """Per-MoE-head independent gate over the head's hidden subspace.

    Inputs [T, Hm, hs]; each head scores its Es experts with its own router
    matrix and keeps the top Ks. The router parameter is replicated (expert
    sharding applies to the grouped experts, not the gate), matching the
    reference geometry where routing runs on the full per-head expert set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        moe_heads: int
        head_hidden: int
        experts_per_head: int
        k_per_head: int
        _debug_force_load_balance: bool = False

        def __post_init__(self) -> None:
            if not 0 < self.k_per_head <= self.experts_per_head:
                raise ValueError(
                    f"k_per_head ({self.k_per_head}) must be in (0, experts_per_head] ({self.experts_per_head})"
                )
            if self.sharding_config is None:
                # The gate router is replicated: expert sharding applies to the
                # grouped experts; the gate scores the full per-head expert set.
                self.sharding_config = _replicated_sharding("router")

    def __init__(self, config: Config):
        super().__init__()
        self.moe_heads = config.moe_heads
        self.head_hidden = config.head_hidden
        self.experts_per_head = config.experts_per_head
        self.k_per_head = config.k_per_head
        self._debug_force_load_balance = config._debug_force_load_balance
        self.router = torch.nn.Parameter(torch.empty(config.moe_heads, config.head_hidden, config.experts_per_head))

    def reset_parameters(self) -> None:
        # The reference creates the gate weights with ``torch.empty`` (never
        # initialized); this port uses a standard small-std normal init.
        torch.nn.init.normal_(self.router, std=0.02)

    def forward(self, x_TSH: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = self.router.to_local() if isinstance(self.router, DTensor) else self.router
        logits = torch.einsum("tsh,she->tse", x_TSH, router)
        scores = F.softmax(logits, dim=-1)
        if self._debug_force_load_balance:
            # Balanced round-robin per head (mirrors the common router's debug
            # mode) so capture geometry does not depend on data-dependent
            # routing; the scores still come from the real gate output.
            tokens, heads, _ = scores.shape
            ids = (
                torch.arange(tokens * heads * self.k_per_head, device=scores.device, dtype=torch.int64).reshape(
                    tokens, heads, self.k_per_head
                )
                % self.experts_per_head
            )
            return scores.gather(dim=-1, index=ids), ids
        weights, indices = torch.topk(scores, k=self.k_per_head, dim=-1)
        return weights, indices


class MultiHeadMoE(Module):
    """Multi-Head MoE: per-head routing over one shared grouped-GEMM expert pool.

    The reference loops heads and experts in Python with per-expert gathers;
    production executes a token dispatch + grouped GEMM per layer. This port
    flattens (head, expert) into one expert id (head * Es + expert) and reuses
    the shared RoutedExperts (token dispatcher + grouped experts): the head
    axis of the projected subspace becomes the dispatcher's sequence axis.
    Owner decision: grouped GEMM in BOTH capture modes (the reference's
    ``bincount``/``torch.where`` loop cannot run under Meta capture).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        moe_heads: int
        head_hidden: int
        proj_in: Linear.Config
        proj_out: Linear.Config
        router: MultiHeadGate.Config
        routed_experts: RoutedExperts.Config
        shared_experts: FeedForward.Config

        def __post_init__(self) -> None:
            if self.proj_in.out_features != self.moe_heads * self.head_hidden:
                raise ValueError("proj_in must project to moe_heads * head_hidden")
            if self.proj_out.in_features != self.moe_heads * self.head_hidden:
                raise ValueError("proj_out must aggregate moe_heads * head_hidden")
            if self.router.moe_heads != self.moe_heads or self.router.head_hidden != self.head_hidden:
                raise ValueError("the gate geometry must match the head geometry")
            if self.routed_experts.inner_experts.dim != self.head_hidden:
                raise ValueError("the grouped experts must consume the head hidden width")
            if self.routed_experts.inner_experts.num_experts != self.router.experts_per_head * self.moe_heads:
                raise ValueError("the expert pool must hold experts_per_head * moe_heads experts")

    def __init__(self, config: Config):
        super().__init__()
        self.moe_heads = config.moe_heads
        self.head_hidden = config.head_hidden
        self.proj_in = config.proj_in.build()
        self.router = config.router.build()
        self.routed_experts = config.routed_experts.build()
        self.proj_out = config.proj_out.build()
        self.shared_experts = config.shared_experts.build()

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        B, L, D = x_BLD.shape
        tokens = B * L
        x_TD = x_BLD.reshape(tokens, D)
        x_TSH = self.proj_in(x_TD).view(tokens, self.moe_heads, self.head_hidden)

        weights_TSK, indices_TSK = self.router(x_TSH)
        # flatten (head, expert) into one expert id space
        head_offset = (
            torch.arange(self.moe_heads, device=x_TSH.device, dtype=torch.int64) * self.router.experts_per_head
        )
        expert_ids_TSK = indices_TSK + head_offset.view(1, -1, 1)
        routing_map_TSE = torch.zeros(
            tokens,
            self.moe_heads,
            self.router.experts_per_head * self.moe_heads,
            dtype=torch.bool,
            device=x_TSH.device,
        ).scatter_(-1, expert_ids_TSK, True)
        num_tokens_per_expert_E = routing_map_TSE.sum(dim=(0, 1))

        moe_out_TSH = self.routed_experts(
            x_TSH,
            weights_TSK,
            expert_ids_TSK,
            num_tokens_per_expert_E,
        )
        moe_out_TD = self.proj_out(moe_out_TSH.reshape(tokens, -1))
        out_TD = moe_out_TD + self.shared_experts(x_TD)
        return out_TD.view(B, L, D)


class TransformerBlock(Module):
    """Decoder block: SLA2 attention + dense FFN or Multi-Head MoE.

    Residual layout matches the reference: ``h = x + attn(norm(x))`` then
    ``out = h + ffn(norm(h))``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: SLA2Attention.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        feed_forward: FeedForward.Config | None = None
        moe: MultiHeadMoE.Config | None = None

        def __post_init__(self) -> None:
            if (self.feed_forward is None) == (self.moe is None):
                raise ValueError("exactly one of feed_forward / moe must be set per block")

    def __init__(self, config: Config):
        super().__init__()
        self.moe_enabled = config.moe is not None
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        if self.moe_enabled:
            self.moe = config.moe.build()  # pyrefly: ignore [missing-attribute]
        else:
            self.feed_forward = config.feed_forward.build()  # pyrefly: ignore [missing-attribute]

    def forward(self, x_BLD: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        h = x_BLD + self.attention(self.attention_norm(x_BLD), rotary_emb)
        h_normed = self.ffn_norm(h)
        if self.moe_enabled:
            return h + self.moe(h_normed)  # pyrefly: ignore [missing-attribute]
        return h + self.feed_forward(h_normed)  # pyrefly: ignore [missing-attribute]


class MultiHeadMoEModel(Decoder):
    """SLA2Attention + Multi-Head-MoE autoregressive decoder.

    ``forward`` follows the reference ``HybridMultiHeadMoEModel.forward``
    literally: embeddings -> ``precompute_freqs_cis`` once -> layers ->
    ``norm_final`` -> ``lm_head``. The runner's ``positions`` seam is accepted
    and ignored (the reference derives RoPE frequencies from the sequence
    length); this family is DP/FSDP/EP only (TP is rejected at config level,
    CP in ``build_attention_masks``), so the framework's PP/metadata plumbing
    has no consumer here. Framework attribute names are kept
    (``tok_embeddings``/``norm``/``lm_head`` = reference
    ``embeddings``/``norm_final``/``lm_head``); the head is executed by the
    loss when ``_skip_lm_head`` is set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "multi_head_moe does not implement tensor parallelism: the SLA2 head axis "
                    "and the per-head gate would both need head-sharded layouts; capture "
                    "covers DP/FSDP/EP only"
                )
            from .sharding import set_multi_head_moe_sharding_config

            set_multi_head_moe_sharding_config(self, enable_ep=parallelism.expert_parallel_degree > 1)
            if parallelism.expert_parallel_degree > 1:
                from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

                for layer_cfg in self.layers:
                    if layer_cfg.moe is not None:
                        layer_cfg.moe.routed_experts.token_dispatcher = AllToAllTokenDispatcher.Config(
                            num_experts=layer_cfg.moe.routed_experts.inner_experts.num_experts,
                            top_k=layer_cfg.moe.router.k_per_head,
                        )

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            # Debug-grade MFU estimate: total parameters plus the per-token
            # active-parameter count from the config geometry (routed experts
            # contribute their top-k share only). Matches the reference's
            # 2-dense-front / 2-dense-back layer split.
            nparams = sum(parameter.numel() for parameter in model.parameters())
            active = 0
            for layer in self.layers:
                attn = layer.attention
                d = attn.to_q.in_features
                active += 4 * attn.to_q.out_features * d
                proj_dim = attn.inner.router.proj_q.out_features
                active += 2 * attn.head_dim * proj_dim
                if layer.moe is not None:
                    moe = layer.moe
                    gate = moe.router
                    active += moe.proj_in.out_features * moe.proj_in.in_features
                    active += moe.proj_out.in_features * moe.proj_out.out_features
                    active += gate.moe_heads * gate.head_hidden * gate.experts_per_head
                    experts = moe.routed_experts.inner_experts
                    per_expert = 3 * experts.hidden_dim * experts.dim
                    active += gate.moe_heads * gate.k_per_head * per_expert
                    shared = moe.shared_experts
                    active += 3 * shared.w1.out_features * shared.w1.in_features
                else:
                    ffn = layer.feed_forward
                    active += 3 * ffn.w1.out_features * ffn.w1.in_features
            return nparams, 6 * active * seq_len

    @torch.no_grad()
    def precompute_freqs_cis(self, seq_len: int, device):
        # Reference ``precompute_freqs_cis``; geometry comes from the (shared)
        # SLA2 attention config.
        attn = next(iter(self.layers.values())).attention
        dim = attn.head_dim
        theta = attn.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
        *,
        attention_metadata=None,
    ) -> torch.Tensor:
        del positions, attention_masks, attention_metadata  # maskless family; see docstring
        x = self.tok_embeddings(tokens)
        freqs_cis = self.precompute_freqs_cis(x.size(1), device=x.device)
        for layer in self.layers.values():
            x = layer(x, rotary_emb=freqs_cis)
        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x

    def build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
        """SLA2 has no causal mask; positions pass through unchanged.

        The runner's transitional input seam. CP is rejected: the SLA2 block
        pooling and per-block linear state span the full sequence, so a
        sequence split needs a real CP recipe that this family does not
        implement.
        """
        if cp_mesh is not None:
            if load_balancer_type is not None:
                raise NotImplementedError("multi_head_moe capture does not use a CP load balancer")
            raise NotImplementedError("multi_head_moe does not support context parallel")
        return tokens, labels, dict(extra_inputs)
