# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""System3 Flash decoder (DeepSeekV4.1-Flash reference port).

Faithful port of the external reference ``external_models/system3`` (pure
PyTorch, ``model.py`` + ``config.py``). The backbone is a looped transformer
(``base_depth`` physical blocks replayed ``num_recursion`` times over one
shared ``kv_bufs`` dict), with per-4-layer-unit ``[ISA, GDN2, GDN2, GDN2]``
layer types, Flash KV sharing across ISA layers (kv / reindex / reuse roles),
LatentMoE, mHC Sinkhorn hyper-connections, Engram n-gram conditional memory
and interleaved multimodal encoders. Every eager operator on the forward path
is the reference's own small operators; nothing is replaced by fused or
virtual operators.

Approved deviations from the reference (none substitutes an operator kind on
the eager path; all documented per the fidelity rules):

- Routed-expert execution: the reference loops experts with data-dependent
  boolean-mask indexing (``gate_l[mask]``); the masked gather shapes depend on
  tensor values and cannot run under Meta capture. The port expresses the
  same token-choice top-k math through the shared token dispatcher + grouped
  GEMM in BOTH modes (owner-approved deviation class, same ruling as the
  multi-head-moe port): per-expert gate/up/down grouped GEMMs with the
  reference's exact SwiGLU clamp placement (gate clamp ``max`` before silu,
  product clamp ``±`` after), weights multiplied outside the expert (combine).
  The ``(gate, up)`` latent pair rides the dispatcher as one concatenated
  feature dim; combine reuses the shared dispatcher and scatters into the
  latent width.
- z-loss wiring: the reference adds ``z_loss_alpha * aux`` to the loss its
  forward returns. The training stack computes the loss outside the model, so
  the same gradient (``z_loss_alpha * d(aux)/dtheta``, no per-token
  normalization) is injected with an identity-forward autograd function on the
  MoE output carrier. Numerically identical to the reference total loss.
- Router / shared expert / Engram / mHC / encoders keep the reference's own
  modules and math verbatim (including the ``rmsnorm_ssr`` static gate and
  fp32 softmax normalize).
- Encoder stems declared as ``nn.Sequential`` in the reference are expressed
  as leaf modules holding the same Conv/GELU/Linear sequence, so every
  parameter carries an sharding annotation for the ``spmd_types`` backend.
  The op stream is identical.
- ``torch.jit.script`` decorators on the RoPE helpers are dropped: TorchScript
  compilation is an execution detail and does not dispatch cleanly under Meta
  capture; the math is the reference's verbatim.
- ``LoopKVCache`` (the reference's incremental-decode scaffold) is not ported:
  it is decode-time only and never executes in training capture.
- Weight tying: the reference ties the LM head weight to the token embedding
  (one matrix). The capture flavor keeps two independent matrices — the
  capture stack's parameter ledger indexes both names separately (same
  deviation class as the other simulator families). Forward math is
  unchanged; the only semantic difference is whether their gradients are
  shared, which only affects training dynamics.
- Multimodal inputs enter through the runner's input-prep seam
  (``build_attention_masks``), which synthesizes deterministic modality
  tensors for the multimodal debug flavor; the text flavor keeps the encoders
  out of the model entirely (``multimodal_enable=False``), matching the
  text-only synthetic capture contract.

Shape suffixes: B batch, S sequence, D model dim, H heads, Dh head dim,
L latent width, F expert intermediate width, E experts, K experts per token,
T tokens.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, Conv1d, Conv2d
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher, LocalTokenDispatcher
from torchtitan.protocols.module import Module, ModuleList
from torchtitan.protocols.sharding import ShardingConfig

_register_autograd_function = getattr(spmd, "register_autograd_function", None)

__all__ = [
    "AlternatingAttention",
    "AudioEncoder",
    "EncoderAttention",
    "EncoderBlock",
    "EngramModule",
    "FlashBlock",
    "GDN2Attention",
    "HapticEncoder",
    "HyperConnectionBlock",
    "ISAIndexer",
    "ImageEncoder",
    "IndexedSparseAttention",
    "LatentGroupedExperts",
    "LatentMoELayer",
    "LatentMoERouter",
    "LatentRoutedExperts",
    "ModalityEncoder",
    "MultiHeadHashTable",
    "PointCloudEncoder",
    "PolynomialRollingHash",
    "RopeGeometry",
    "SharedLatentExpertMLP",
    "SinkhornIteration",
    "System3HostHashTable",
    "System3Model",
    "System3RMSNorm",
]


# ============================================================================
# Section 1: utilities (reference verbatim)
# ============================================================================


def _replicated_sharding(*parameters: str) -> ShardingConfig:
    """Replicated state placement for model-owned plain parameters."""
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in parameters})


def swiglu(x: torch.Tensor, gate: torch.Tensor, clamp_val: float | None = None) -> torch.Tensor:
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
            router_indices.shape[0],
            topk,
            num_experts,
            dtype=router_probs.dtype,
            device=router_indices.device,
        )
        one_hot.scatter_(2, router_indices.unsqueeze(-1), 1.0)
        density = one_hot.mean(dim=(0, 1))
    density_proxy = router_probs.mean(dim=0)
    return (density * density_proxy).sum() * num_experts


def normal_init_(tensor, mean=0.0, std=0.02):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_to_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # cos/sin are computed by RotaryEmbedding in float32; cast to q's dtype so
    # fp16/bf16 attention is not silently promoted to float32.
    cos = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    return q * cos + rotate_half(q) * sin


def stable_softmax(scores, dim=-1, clamp_val=None):
    if clamp_val is not None:
        scores = scores.clamp(-clamp_val, clamp_val)
    return F.softmax(scores.float(), dim=dim).to(scores.dtype)


class System3RMSNorm(Module):
    """RMSNorm (float32 internally), reference verbatim."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("weight")

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.dim))
        self.eps = config.eps

    def reset_parameters(self) -> None:
        # Reference init: ``torch.ones(hidden_size)``.
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight.float()).to(dtype)


# ============================================================================
# Section 2: RoPE + YaRN (reference verbatim)
# ============================================================================


@dataclass(kw_only=True, slots=True)
class RopeGeometry:
    """Framework-contract rope metadata the Decoder config path reads.

    The ISA path builds its real ``RotaryEmbedding`` from the same geometry at
    build time; this object only carries ``max_seq_len`` for the framework's
    sequence-length resize (``Decoder.Config.update_from_config``).
    """

    max_seq_len: int = 512


class RotaryEmbedding(Module):
    # yarn_scale cache is shared across all ISA layers (identical config);
    # the 8M-context entry would be ~33MB, so the reference caches one copy.
    _yarn_cache: ClassVar[dict[tuple[float, int, int], torch.Tensor]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        theta: float
        yarn_factor: float
        original_max_seq_len: int
        max_seq_len: int

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.max_seq_len = config.max_seq_len
        inv_freq = 1.0 / (config.theta ** (torch.arange(0, config.dim, 2).float() / config.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Standard YaRN interpolation: positions <= original_max unscaled;
        # beyond that a 0->1 ramp stretches remote wavelengths back into the
        # trained range (inv_freq / yarn_factor**ramp).
        key = (config.yarn_factor, config.original_max_seq_len, config.max_seq_len)
        if key not in RotaryEmbedding._yarn_cache:
            ramp = torch.zeros(config.max_seq_len)
            if config.max_seq_len > config.original_max_seq_len:
                ramp[config.original_max_seq_len :] = torch.linspace(
                    0, 1, config.max_seq_len - config.original_max_seq_len
                )
            RotaryEmbedding._yarn_cache[key] = torch.exp(ramp * math.log(config.yarn_factor))
        self.yarn_scale = RotaryEmbedding._yarn_cache[key]  # shared tensor (not a buffer)
        self.original_max_seq_len = config.original_max_seq_len

    @torch.no_grad()
    def forward(self, seq_len, device):
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len({seq_len}) exceeds max_seq_len({self.max_seq_len})")
        t = torch.arange(seq_len, device=device).float()
        inv_freq = self.inv_freq.to(device)
        if seq_len > self.original_max_seq_len:
            scale = self.yarn_scale[:seq_len].to(device)
            inv_freq = inv_freq.unsqueeze(0) / scale.unsqueeze(-1)  # [seq_len, dim/2]
            freqs = t.unsqueeze(-1) * inv_freq
        else:
            freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


# ============================================================================
# Section 3: ISA indexer + token-level sparse attention (reference verbatim)
# ============================================================================


class ISAIndexer(Module):
    """Indexer for token-level sparsity over the stride-compressed KV grid.

    Reference scoring: RMSNorm'd low-rank projections, then one flattened
    dot product over the index width scaled by ``1/sqrt(width)`` on the
    compressed grid — no ReLU, no per-head gate, no head-sum. The indexer
    scores are passed to the attention as bias.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        n_heads: int
        head_dim: int
        topk: int
        compress_ratio: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        q_norm: System3RMSNorm.Config
        k_norm: System3RMSNorm.Config

        def __post_init__(self) -> None:
            if self.q_proj.out_features != self.n_heads * self.head_dim:
                raise ValueError("indexer q_proj must project to n_heads * head_dim")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.topk = config.topk
        self.compress_ratio = config.compress_ratio
        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

    def select(self, hidden_states, attention_mask=None):
        """Returns topk_indices [b, s, topk] int64, topk_scores [b, s, topk]."""
        idx_q = self.q_norm(self.q_proj(hidden_states))
        idx_k = self.k_norm(self.k_proj(hidden_states))
        idx_k_c = idx_k[:, :: self.compress_ratio, :]
        return self._select_topk(idx_q, idx_k_c, attention_mask)

    def _select_topk(self, idx_q, idx_k_c, attention_mask=None):
        """Causal top-K selection over the compressed indexer grid (seam).

        Reference scoring: one flattened dot product over the index width,
        scaled by ``1/sqrt(width)``; no ReLU, no gate, no head-sum.
        """
        b, s, width = idx_q.shape
        cl = idx_k_c.shape[1]

        scores = torch.bmm(
            idx_q.reshape(b, s, width),
            idx_k_c.reshape(b, cl, width).transpose(1, 2),
        ) / math.sqrt(width)

        q_pos = torch.arange(s, device=idx_q.device).unsqueeze(1)
        k_pos = torch.arange(0, s, self.compress_ratio, device=idx_q.device).unsqueeze(0)
        # Mask value uses the dtype minimum (-1e9 is not representable in fp16).
        mask_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill((k_pos > q_pos).unsqueeze(0), mask_val)
        if attention_mask is not None:
            k_valid = attention_mask.bool()[:, :: self.compress_ratio]  # [b, cl]
            scores = scores.masked_fill(~k_valid.unsqueeze(1), mask_val)

        actual_topk = min(self.topk, cl)
        topk_scores, topk_indices = torch.topk(scores, actual_topk, dim=-1)
        return topk_indices, topk_scores


class IndexedSparseAttention(Module):
    """Token-level sparse attention over the CSA-compressed KV (2x grid).

    o_i = sum_{j in topK(i)} softmax_j(q_i.k_j / sqrt(hd) + s_ij) v_j; the
    gathered k/v come from the shared compressed slot; masked slots carry a
    -inf indexer bias so their softmax weight is zero.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        softmax_clamp: float
        compress_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.softmax_clamp = config.softmax_clamp
        self.compress_ratio = config.compress_ratio

    def forward(self, q_full, k_comp, v_comp, topk_indices, topk_scores, attention_mask=None):
        return self._sparse_attention(q_full, k_comp, v_comp, topk_indices, topk_scores, attention_mask)

    def _sparse_attention(self, q_full, k_comp, v_comp, topk_indices, topk_scores, attention_mask=None):
        """Selected-key attention over the compressed KV grid (seam)."""
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
        # Deviation from the reference (owner-approved, PR #24 review): the
        # reference clamps after the padding fill (``stable_softmax`` receives
        # dtype-minimum scores and the clamp raises masked keys back to the
        # clamp value), so masked keys keep exp(-clamp - LSE) weight and fully
        # padded rows attend uniformly. Clamp first, then mask with -inf:
        # masked keys contribute exactly zero and a row without a single valid
        # key yields zero attention weights instead of a uniform average.
        attn_scores = attn_scores.clamp(-self.softmax_clamp, self.softmax_clamp)
        if attention_mask is not None:
            k_valid = attention_mask.bool()[:, :: self.compress_ratio]  # [b, cl]
            k_pad = ~k_valid[bidx, flat.view(b, s, actual_topk)]  # [b, s, k]
            attn_scores = attn_scores.masked_fill(k_pad.unsqueeze(1), float("-inf"))
            attn_probs = stable_softmax(attn_scores, dim=-1)
            # Softmax over an all-masked row is NaN; zero it explicitly.
            row_has_valid = (~k_pad).any(dim=-1)  # [b, s]
            attn_probs = torch.where(row_has_valid[:, None, :, None], attn_probs, torch.zeros_like(attn_probs))
        else:
            attn_probs = stable_softmax(attn_scores, dim=-1)
        return torch.einsum("bhsk,bskhd->bshd", attn_probs, v_sel)


# ============================================================================
# Section 4: GDN2 — delta-rule linear attention (reference verbatim)
# ============================================================================


class GDN2Attention(Module):
    """GDN2: delta-rule linear attention (Generalized Delta Net).

    State S in [b, H, d_state, d_state] is a fresh zero buffer per forward
    (fixed O(1) memory, no token-level KV cache); the reference's per-token
    serial update is kept verbatim (k normalized to unit norm first, beta
    erase/write double gate, alpha read gate, per-head da gain). No explicit
    positional encoding (DeltaNet: position is implicit in the update order).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        d_state: int
        d_k: int
        d_v: int
        d_a: int
        # Framework-contract rope metadata (Decoder config path resizes it
        # like every attention); GDN2 itself uses no positional encoding.
        rope: RopeGeometry
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        o_proj: Linear.Config
        alpha_gate: Linear.Config
        beta_gate: Linear.Config
        da_proj: Linear.Config
        da_norm: System3RMSNorm.Config

        def __post_init__(self) -> None:
            if self.q_proj.out_features != self.num_heads * self.d_k:
                raise ValueError("GDN2 q_proj must project to H * d_k")
            if self.o_proj.in_features != self.num_heads * self.d_v:
                raise ValueError("GDN2 o_proj must consume H * d_v")
            if self.beta_gate.out_features != 2 * self.num_heads * self.d_state:
                raise ValueError("GDN2 beta_gate must project to 2 * H * d_state")

    def __init__(self, config: Config):
        super().__init__()
        d = config.hidden_size
        H = config.num_heads
        self.d_state = config.d_state
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.d_a = config.d_a
        self.H = H

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.o_proj = config.o_proj.build()
        self.alpha_gate = config.alpha_gate.build()
        self.beta_gate = config.beta_gate.build()
        self.da_proj = config.da_proj.build()
        self.da_norm = config.da_norm.build()

    def forward(self, hidden_states, kv_bufs=None, attention_mask=None):
        # kv_bufs is accepted for the uniform block call signature (GDN2 keeps
        # only its fixed per-forward state); the mask gates state writes.
        del kv_bufs
        b, s, _d = hidden_states.shape
        H, dk, dv, ds = self.H, self.d_k, self.d_v, self.d_state
        # Padding mask (True/1 = valid): padding positions do not write the
        # state S, so later tokens read unpolluted state.
        mask = attention_mask.bool() if attention_mask is not None else None

        q = self.q_proj(hidden_states).view(b, s, H, dk)
        k = self.k_proj(hidden_states).view(b, s, H, dk)
        v = self.v_proj(hidden_states).view(b, s, H, dv)
        alpha = torch.sigmoid(self.alpha_gate(hidden_states).view(b, s, H, dk))
        beta_raw = self.beta_gate(hidden_states).view(b, s, 2, H, ds)
        beta_erase = torch.sigmoid(beta_raw[:, :, 0])  # [b, s, H, ds]
        beta_write = torch.sigmoid(beta_raw[:, :, 1])  # [b, s, H, ds]
        da = self.da_norm(self.da_proj(hidden_states)).view(b, s, H, self.d_a)
        # Per-head read gain: [b, s, H]
        da_gain = torch.sigmoid(da.mean(dim=-1))

        # The delta-rule update runs token by token in the reference
        # (production may swap in a chunked-parallel algorithm; the reference
        # is serial). The seam keeps that swap a one-method change.
        out = self._delta_rule_scan(q, k, v, alpha, beta_erase, beta_write, da_gain, mask)
        out = out.reshape(b, s, H * dv)
        return self.o_proj(out)

    def _delta_rule_scan(self, q, k, v, alpha, beta_erase, beta_write, da_gain, mask):
        """Serial reference delta-rule scan; the simulator swaps a chunked boundary in."""
        b, s, H = q.shape[0], q.shape[1], q.shape[2]
        ds, dv = beta_erase.shape[-1], v.shape[-1]
        dk = q.shape[-1]
        device = q.device
        dtype = q.dtype

        S = torch.zeros(b, H, ds, ds, device=device, dtype=dtype)

        outputs = []
        for t in range(s):
            qt = q[:, t]  # [b, H, dk]
            kt = k[:, t]  # [b, H, dk]
            vt = v[:, t]  # [b, H, dv]
            at = alpha[:, t]  # [b, H, dk]
            be = beta_erase[:, t]  # [b, H, ds]
            bw = beta_write[:, t]  # [b, H, ds]

            # d_state may differ from d_k/d_v -> pad/truncate to ds. The key
            # and value conversions are independent (d_k, d_v, d_state are
            # freely configurable): deriving the value conversion from the dk
            # relation left v_state mis-sized whenever dk == ds but dv != ds,
            # or dk > ds but dv < ds (reference carries the same latent defect
            # and crashes on those geometries; fixed here by owner-approved
            # deviation, bitwise-identical on every reference-runnable config).
            if dk < ds:
                k_state = F.pad(kt, (0, ds - dk))
            elif dk > ds:
                k_state = kt[..., :ds]
            else:
                k_state = kt
            if dv < ds:
                v_state = F.pad(vt, (0, ds - dv))
            elif dv > ds:
                v_state = vt[..., :ds]
            else:
                v_state = vt
            # DeltaNet standard: normalize k to unit norm so the state update
            # operator (I - beta k k^T) is a contraction (float32 norm for
            # numerical stability).
            k_state = (k_state.float() / (k_state.float().norm(dim=-1, keepdim=True) + 1e-8)).to(k_state.dtype)

            Sk = torch.einsum("bhij,bhj->bhi", S, k_state)  # [b, H, ds]
            delta = v_state - Sk  # [b, H, ds]
            update = torch.einsum("bhi,bhj->bhij", bw * delta, k_state)  # [b, H, ds, ds]
            # Erase gate decays the key-side columns of S.
            if mask is None:
                S = S * (1 - be.unsqueeze(-1)) + update  # be[b,H,ds,1] broadcast
            else:
                mt = mask[:, t].view(b, 1, 1, 1)  # right-aligned broadcast
                S = S * (1 - be.unsqueeze(-1) * mt) + update * mt

            # Read: o = alpha * (S @ q) * da_gain
            if dk < ds:
                q_state = F.pad(qt, (0, ds - dk))
            elif dk > ds:
                q_state = qt[..., :ds]
            else:
                q_state = qt
            o = torch.einsum("bhij,bhj->bhi", S, q_state)  # [b, H, ds]
            if dk < ds:
                at = F.pad(at, (0, ds - dk))
            elif dk > ds:
                at = at[..., :ds]
            o = at * o * da_gain[:, t].unsqueeze(-1)  # [b, H, ds]

            # Project back to d_v.
            if dv < ds:
                o = o[..., :dv]
            elif dv > ds:
                o = F.pad(o, (0, dv - ds))

            outputs.append(o)

        return torch.stack(outputs, dim=1)  # [b, s, H, dv]


# ============================================================================
# Section 5: Alternating attention (ISA / GDN2 dispatch + Flash KV sharing)
# ============================================================================


class AlternatingAttention(Module):
    """Layer-type dispatch: ISA / GDN2 by position in the 4-layer unit.

    ISA keeps full-rank Q/O projections on every ISA layer; K/V projections
    and the indexer only exist on the layer's KV-sharing role requires them
    ("kv" role owns K/V + indexer, "reindex" owns an indexer, "reuse" reads
    the shared slot's top-K). GDN2 layers hold only their own delta-rule
    module.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        is_isa: bool
        hidden_size: int
        num_heads: int
        head_dim: int
        qk_rope_head_dim: int
        compress_ratio: int
        softmax_clamp: float
        # Framework-contract rope metadata (Decoder config path); built into
        # the ISA RotaryEmbedding below.
        rope: RopeGeometry
        role: str = "kv"
        kv_group: int = 0
        q_proj: Linear.Config | None = None
        o_proj: Linear.Config | None = None
        k_proj: Linear.Config | None = None
        v_proj: Linear.Config | None = None
        indexer: ISAIndexer.Config | None = None
        isa: IndexedSparseAttention.Config | None = None
        rotary: RotaryEmbedding.Config | None = None
        gdn2: GDN2Attention.Config | None = None

        def __post_init__(self) -> None:
            if self.is_isa:
                if self.q_proj is None or self.o_proj is None or self.isa is None:
                    raise ValueError("ISA layers require q/o projections and the sparse core")
                if self.role in ("kv", "reindex") and self.indexer is None:
                    raise ValueError(f"ISA role {self.role!r} requires an indexer")
                if self.role == "kv" and (self.k_proj is None or self.v_proj is None):
                    raise ValueError("ISA 'kv' role requires K/V projections")
                if self.rotary is None:
                    raise ValueError("ISA layers require the rotary embedding config")
            elif self.gdn2 is None:
                raise ValueError("non-ISA layers require the GDN2 config")

    def __init__(self, config: Config):
        super().__init__()
        self.layer_idx = config.layer_idx
        self.is_isa = config.is_isa

        if self.is_isa:
            self.role = config.role
            self.kv_group = config.kv_group
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.nope_dim = config.head_dim - config.qk_rope_head_dim
            self.rope_dim = config.qk_rope_head_dim
            self.compress_ratio = config.compress_ratio
            self.q_proj = config.q_proj.build()  # pyrefly: ignore [missing-attribute]
            self.o_proj = config.o_proj.build()  # pyrefly: ignore [missing-attribute]
            self.rotary_emb = config.rotary.build()  # pyrefly: ignore [missing-attribute]
            if config.role == "kv":
                self.k_proj = config.k_proj.build()  # pyrefly: ignore [missing-attribute]
                self.v_proj = config.v_proj.build()  # pyrefly: ignore [missing-attribute]
            if config.role in ("kv", "reindex"):
                self.indexer = config.indexer.build()  # pyrefly: ignore [missing-attribute]
            self.isa = config.isa.build()  # pyrefly: ignore [missing-attribute]
        else:
            self.gdn2 = config.gdn2.build()  # pyrefly: ignore [missing-attribute]

    def forward(self, hidden_states, kv_bufs=None, attention_mask=None):
        if not self.is_isa:
            return self.gdn2(hidden_states, attention_mask=attention_mask)  # pyrefly: ignore [missing-attribute]

        b, s, _ = hidden_states.shape
        buf = None if kv_bufs is None else kv_bufs[self.kv_group]

        # Full-rank Q + RoPE (rope dims rotate, noPE dims pass through).
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        cos, sin = self.rotary_emb(s, hidden_states.device)
        q_rope_r = apply_rotary_to_q(q[..., self.nope_dim :], cos, sin)
        q_full = torch.cat([q[..., : self.nope_dim], q_rope_r], dim=-1)

        if self.role == "kv":
            k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
            k_rope_r = apply_rotary_to_q(k[..., self.nope_dim :], cos, sin)
            k_full = torch.cat([k[..., : self.nope_dim], k_rope_r], dim=-1)
            # CSA compression: the slot stores the compressed-grid KV only.
            k_comp = k_full[:, :: self.compress_ratio, :, :]
            v_comp = v[:, :: self.compress_ratio, :, :]
            topk_i, topk_s = self.indexer.select(hidden_states, attention_mask)
            if buf is not None:
                buf["k_comp"], buf["v_comp"] = k_comp, v_comp
                buf["topk_indices"], buf["topk_scores"] = topk_i, topk_s
        else:
            if buf is None:
                raise RuntimeError(
                    f"ISA layer {self.layer_idx} (role={self.role}) needs the shared "
                    f"KV buffer kv_bufs, but none was provided"
                )
            k_comp, v_comp = buf["k_comp"], buf["v_comp"]
            if self.role == "reindex":
                topk_i, topk_s = self.indexer.select(hidden_states, attention_mask)
            else:  # reuse: the KV layer's top-K selection
                topk_i, topk_s = buf["topk_indices"], buf["topk_scores"]

        out = self.isa(q_full, k_comp, v_comp, topk_i, topk_s, attention_mask)
        return self.o_proj(out.reshape(b, s, self.num_heads * self.head_dim))


# ============================================================================
# Section 6: LatentMoE (shared bottleneck projections + grouped experts)
# ============================================================================


if _register_autograd_function is not None:  # pragma: no branch

    @_register_autograd_function
    class _ZLossInjection(torch.autograd.Function):
        """Identity-forward autograd injecting the z-loss gradient on backward.

        The forward result carries the carrier's type, which ``typecheck_forward``
        restates: the body is opaque to the checker. The reference computes
        ``total_loss = ce + z_loss_alpha * sum(aux)`` inside the model; the
        training stack computes the loss outside, so the identical gradient (no
        per-token normalization) rides on the MoE output carrier. Saving the
        scaled aux tensor keeps ``ones_like`` exact.
        """

        @staticmethod
        def forward(ctx, carrier: torch.Tensor, scaled_aux: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore [bad-override]
            ctx.save_for_backward(scaled_aux)
            return carrier

        @staticmethod
        def typecheck_forward(carrier: torch.Tensor, scaled_aux: torch.Tensor) -> torch.Tensor:
            result = _ZLossInjection.apply(carrier, scaled_aux)
            spmd.assert_type(result, spmd.get_local_type(carrier))
            return result

        @staticmethod
        def backward(  # pyrefly: ignore [bad-override]
            ctx, grad_carrier: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            (scaled_aux,) = ctx.saved_tensors
            return grad_carrier, torch.ones_like(scaled_aux)

else:  # pragma: no cover

    class _ZLossInjection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, carrier: torch.Tensor, scaled_aux: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore [bad-override]
            ctx.save_for_backward(scaled_aux)
            return carrier

        @staticmethod
        def typecheck_forward(carrier: torch.Tensor, scaled_aux: torch.Tensor) -> torch.Tensor:
            return _ZLossInjection.apply(carrier, scaled_aux)

        @staticmethod
        def backward(  # pyrefly: ignore [bad-override]
            ctx, grad_carrier: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            (scaled_aux,) = ctx.saved_tensors
            return grad_carrier, torch.ones_like(scaled_aux)


class LatentGroupedExperts(Module):
    """Grouped-GEMM execution of the reference LatentExpertMLP math.

    Per expert (weights ``(E, F, D)``/``(E, D, F)`` like the shared grouped
    experts): ``out = W_down silu(clamp_max(W_gate g)) * (W_up u)`` with the
    product clamped ``±swiglu_limit`` — the reference's exact clamp placement
    (gate clamped max before silu, value path unclamped, product clamped).
    The grouped GEMM runs in bf16 like the shared ``GroupedExperts`` (under
    the bf16 mixed-precision capture the expert math is dtype-identical to
    the reference; under fp32 capture the bf16 grouped GEMM is the shared
    production deviation, documented in the header).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int  # latent width
        hidden_dim: int  # expert intermediate width
        num_experts: int
        swiglu_limit: float

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit
        self.w_gate_EFD = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.w_up_EFD = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.w_down_EDF = nn.Parameter(torch.empty(config.num_experts, config.dim, config.hidden_dim))

    def reset_parameters(self) -> None:
        # The reference smoke leaves the expert weights at torch.empty (zero
        # pages); production declares a trunc-normal(0.02) init like the
        # shared grouped experts.
        for weight in (self.w_gate_EFD, self.w_up_EFD, self.w_down_EDF):
            nn.init.trunc_normal_(weight, std=0.02, a=-0.04, b=0.04)

    def forward(
        self,
        g_RD: torch.Tensor,
        u_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w_gate_EFD, DTensor):
            w_gate_EFD = self.w_gate_EFD.to_local()
            assert isinstance(self.w_up_EFD, DTensor)
            w_up_EFD = self.w_up_EFD.to_local()
            assert isinstance(self.w_down_EDF, DTensor)
            w_down_EDF = self.w_down_EDF.to_local()
        else:
            w_gate_EFD = self.w_gate_EFD
            w_up_EFD = self.w_up_EFD
            w_down_EDF = self.w_down_EDF

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_RF = self._grouped_mm(A=g_RD.bfloat16(), B_t=w_gate_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        up_RF = self._grouped_mm(A=u_RD.bfloat16(), B_t=w_up_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        # Reference swiglu(up_inter, gate_inter, clamp): gate clamped max
        # before silu, product clamped both sides after.
        act_RF = swiglu(up_RF, gate_RF, self.swiglu_limit)
        return self._grouped_mm(A=act_RF, B_t=w_down_EDF.bfloat16().transpose(-2, -1), offs=offsets_E).type_as(g_RD)

    def _grouped_mm(self, *, A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """Grouped matmul ``A @ B_t`` with per-expert token offsets."""
        return torch._grouped_mm(A, B_t, offs=offs)


class LatentRoutedExperts(Module):
    """Token dispatcher + latent-pair grouped experts.

    The dispatcher carries the concatenated ``(gate, up)`` latent pair as one
    feature dim so EP exchanges both halves once; the grouped experts consume
    the split halves and emit the latent output, combined through the shared
    dispatcher (score multiply + scatter-add, output width = latent). The
    ``gate`` tensor only provides the combine's output shape (latent width).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: LatentGroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config | AllToAllTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()  # pyrefly: ignore [missing-attribute]
        self.token_dispatcher = config.token_dispatcher.build()  # pyrefly: ignore [missing-attribute]

    def parallelize(self, parallel_dims) -> None:
        """Wire the EP mesh on the dispatcher (shared RoutedExperts recipe)."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(ep_mesh=parallel_dims.get_optional_mesh("ep"))

    def forward(
        self,
        gate_l_TL: torch.Tensor,
        up_l_TL: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        z_TD = torch.cat([gate_l_TL, up_l_TL], dim=-1)
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            z_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            g_RD, u_RD = routed_input_RD.chunk(2, dim=-1)
            routed_output_RL = self.inner_experts(g_RD, u_RD, num_global_tokens_per_local_expert_e)
        # The shared combine multiplies by the expert-sorted scores and
        # scatter-adds into a zeros_like(shape-provider) buffer; the gate
        # tensor provides the latent output shape.
        return self.token_dispatcher.combine(routed_output_RL, metadata, gate_l_TL)


class SharedLatentExpertMLP(Module):
    """Shared expert with the reference's latent-space structure (dense math).

    latent -> inter (gate/up pair) -> latent, SwiGLU with the reference clamp;
    consumes the shared ``(gate, up)`` down-projections like the reference.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate_inter: Linear.Config
        up_inter: Linear.Config
        inter_to_latent: Linear.Config
        clamp_val: float

        def __post_init__(self) -> None:
            if self.gate_inter.in_features != self.up_inter.in_features:
                raise ValueError("shared expert gate/up projections must share the latent width")

    def __init__(self, config: Config):
        super().__init__()
        self.gate_inter = config.gate_inter.build()
        self.up_inter = config.up_inter.build()
        self.inter_to_latent = config.inter_to_latent.build()
        self.clamp_val = config.clamp_val

    def forward(self, gate_l, up_l):
        gate_inter = self.gate_inter(gate_l)
        up_inter = self.up_inter(up_l)
        act = swiglu(up_inter, gate_inter, self.clamp_val)
        return self.inter_to_latent(act)


class LatentMoERouter(Module):
    """Token-choice router with the reference's exact scoring math.

    scores = score_fn(logits) * route_scale; top-K; weights normalized in
    float32 (``fp16/bf16 clamp floor 1e-8`` would round to 0 and 0/0 -> NaN);
    ``_debug_force_load_balance`` replaces the top-K ids with a round-robin
    selection while keeping the real gathered scores (shared debug contract).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_experts: int
        topk: int
        route_scale: float
        score_function: Literal["sqrtsoftplus", "softmax", "sigmoid"] = "sqrtsoftplus"
        _debug_force_load_balance: bool = False

        def __post_init__(self) -> None:
            if not 0 < self.topk <= self.num_experts:
                raise ValueError(f"topk ({self.topk}) must be in (0, num_experts] ({self.num_experts})")
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("router_weight")

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.route_scale = config.route_scale
        self.score_function = config.score_function
        self._debug_force_load_balance = config._debug_force_load_balance
        self.router_weight = nn.Parameter(torch.empty(config.hidden_size, config.num_experts))

    def reset_parameters(self) -> None:
        # Reference init: normal_(std=0.02 / sqrt(num_experts)).
        normal_init_(self.router_weight, std=0.02 / math.sqrt(self.num_experts))

    def forward(self, tokens):
        logits = tokens @ self.router_weight
        if self.score_function == "sqrtsoftplus":
            scores = sqrtsoftplus(logits) * self.route_scale
        elif self.score_function == "softmax":
            scores = F.softmax(logits, dim=-1) * self.route_scale
        else:
            scores = torch.sigmoid(logits) * self.route_scale
        topk_w, topk_i = torch.topk(scores, self.topk, dim=-1)
        if self._debug_force_load_balance:
            # Balanced round-robin ids; the gathered scores stay real so the
            # normalized weights below keep the reference's data dependency.
            n_tokens = topk_i.shape[0]
            ids = (
                torch.arange(n_tokens * self.topk, device=scores.device, dtype=torch.int64).reshape(n_tokens, self.topk)
                % self.num_experts
            )
            topk_w = scores.gather(dim=-1, index=ids)
            topk_i = ids
        # Normalize in float32 (fp16/bf16 would round the clamp floor to 0).
        topk_w = topk_w.float()
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        topk_w = topk_w.to(scores.dtype)
        return topk_i, topk_w, logits


class LatentMoELayer(Module):
    """LatentMoE layer: shared bottleneck projections + routed/shared experts.

    gate_down/up_down project d -> latent (shared by all tokens/experts);
    routed experts run in latent space via the token dispatcher; the shared
    expert runs the same latent math densely on every token; latent_up maps
    latent -> d. The reference's load-balance z-loss gradient is injected on
    the output carrier with coefficient ``z_loss_alpha``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        z_loss_alpha: float
        gate_down: Linear.Config
        up_down: Linear.Config
        latent_up: Linear.Config
        router: LatentMoERouter.Config
        routed_experts: LatentRoutedExperts.Config
        shared_experts: list[SharedLatentExpertMLP.Config]

        def __post_init__(self) -> None:
            if self.gate_down.out_features != self.routed_experts.inner_experts.dim:
                raise ValueError("gate_down must project to the latent width")
            if self.latent_up.in_features != self.routed_experts.inner_experts.dim:
                raise ValueError("latent_up must consume the latent width")
            if self.router.num_experts != self.routed_experts.inner_experts.num_experts:
                raise ValueError("router expert count must match the grouped experts")

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.router.num_experts
        self.topk = config.router.topk
        self.z_loss_alpha = config.z_loss_alpha
        self.gate_down = config.gate_down.build()
        self.up_down = config.up_down.build()
        self.latent_up = config.latent_up.build()
        self.router = config.router.build()
        self.routed_experts = config.routed_experts.build()
        self.shared_experts = ModuleList(
            [shared.build() for shared in config.shared_experts]  # pyrefly: ignore [missing-attribute]
        )

    def forward(self, hidden_states):
        b, s, d = hidden_states.shape
        tokens = hidden_states.reshape(-1, d)

        # Shared d->latent down projections (router and experts share them).
        gate_l = self.gate_down(tokens)
        up_l = self.up_down(tokens)

        # Shared experts (latent-space FFN, every token).
        shared_out = sum(expert(gate_l, up_l) for expert in self.shared_experts)

        # Routed experts (h-space scoring, token-choice top-K).
        topk_i, topk_w, logits = self.router(tokens)

        # num_local_tokens_per_expert: faithful equivalent of the reference's
        # per-expert mask (count of tokens selecting each expert).
        routing_map_TSE = torch.zeros(
            tokens.shape[0],
            self.num_experts,
            dtype=torch.bool,
            device=tokens.device,
        ).scatter_(-1, topk_i, True)
        num_tokens_per_expert_E = routing_map_TSE.sum(dim=0)

        routed_out = self.routed_experts(gate_l, up_l, topk_w, topk_i, num_tokens_per_expert_E)

        # Shared latent->d up projection (routed + shared both return via it).
        h_out = self.latent_up(routed_out + shared_out)

        # Reference z-loss (DeepSeek standard) injected on the output carrier.
        aux_loss = compute_load_balance_loss(F.softmax(logits, dim=-1), topk_i, self.num_experts, self.topk)
        scaled = aux_loss * self.z_loss_alpha
        return _ZLossInjection.typecheck_forward(h_out.reshape(b, s, d), scaled)


# ============================================================================
# Section 7: Engram conditional memory (reference verbatim)
# ============================================================================


class PolynomialRollingHash(Module):
    """Polynomial rolling hash (Horner recurrence, overflow-safe).

    h_t = (h_{t-1} * BASE + id_t * mult_head) % capacity; per-head independent
    multipliers; order-n window covers the last n tokens.
    """

    BASE = 257

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        order: int
        capacity: int
        num_heads: int
        layer_idx: int
        seed_offset: int = 0

    def __init__(self, config: Config):
        super().__init__()
        self.order = config.order
        self.capacity = config.capacity
        self.num_heads = config.num_heads
        gen = torch.Generator()
        gen.manual_seed(config.layer_idx * 100 + config.order * 10 + config.seed_offset)
        self.register_buffer(
            "multipliers",
            torch.randint(1, 2**31 - 1, (config.num_heads,), generator=gen).long(),
        )

    def forward(self, input_ids):
        # input_ids: [b, s] -> [b, s, num_heads] hash indices in [0, capacity)
        # position t's hash is decided by the last `order` tokens; positions
        # before the window are zero-filled.
        b, s = input_ids.shape
        h = torch.zeros(b, s, self.num_heads, dtype=torch.long, device=input_ids.device)
        for j in range(self.order):
            shifted = torch.zeros_like(input_ids)
            if j < s:
                shifted[:, j:] = input_ids[:, : s - j]
            h = (h * self.BASE + shifted.unsqueeze(-1) * self.multipliers) % self.capacity
        return h


class MultiHeadHashTable(Module):
    """Multi-order n-gram hash lookup table: per-head offset segments."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        capacity: int
        num_heads: int
        head_dim: int

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("weight", "head_offsets")

    def __init__(self, config: Config):
        super().__init__()
        self.capacity = config.capacity
        self.num_heads = config.num_heads
        self.per_head_dim = config.head_dim
        # The reference holds an nn.Embedding; the weight lives directly on
        # this module (same aten.embedding op) so the sharding annotation is
        # a plain parameter placement.
        self.weight = nn.Parameter(torch.empty(config.num_heads * config.capacity, config.head_dim))
        self.register_buffer("head_offsets", torch.arange(config.num_heads) * config.capacity)

    def reset_parameters(self) -> None:
        # nn.Embedding default init (the reference stores a plain Embedding).
        nn.init.normal_(self.weight)

    def forward(self, hash_indices):
        _b, _s, nh = hash_indices.shape
        outs = []
        for h in range(nh):
            idx = hash_indices[:, :, h] + self.head_offsets[h]
            outs.append(F.embedding(idx, self.weight))
        return torch.cat(outs, dim=-1)  # [b, s, nh * head_dim]


class _System3HostLookup(torch.autograd.Function):
    """Eager lookup for the host hash table with owner-local sparse gradients.

    EP=1 gathers rows from the local CPU table; EP>1 routes deduplicated row
    requests to their EP owners through the production all-to-all exchange
    (``torchtitan_npu.models.deepseek_v4_1.engram.lookup``) and reverses it in
    backward. Either way the backward accumulates only the owner-local rows
    into the table's pending sparse gradient; the dense table gradient is
    never materialized.
    """

    @staticmethod
    def forward(ctx, weight, flat_ids, table, lookup_rows, lookup_grad_rows):
        if table._ep_size > 1:
            rows = lookup_rows(ctx, weight, flat_ids, table.lookup_mesh.get_group())
            ctx.distributed = True
        else:
            local_ids = flat_ids.reshape(-1).to(device=weight.device, copy=True)
            rows = weight.index_select(0, local_ids).to(device=flat_ids.device)
            rows = rows.view(*flat_ids.shape, weight.shape[1])
            ctx.save_for_backward(flat_ids)
            ctx.distributed = False
        ctx.table = table
        ctx.lookup_grad_rows = lookup_grad_rows
        return rows

    @staticmethod
    def backward(ctx, grad_rows):  # pyrefly: ignore [bad-override]
        table = ctx.table
        if ctx.distributed:
            grad_local_rows, local_ids = ctx.lookup_grad_rows(ctx, grad_rows)
        else:
            (flat_ids,) = ctx.saved_tensors
            local_ids = flat_ids.reshape(-1).to(device=table.weight.device, dtype=torch.int64, copy=True)
            grad_local_rows = grad_rows.reshape(-1, grad_rows.shape[-1])
        table.accumulate_sparse_gradient(local_ids, grad_local_rows)
        return None, None, None, None, None


class System3HostHashTable(MultiHeadHashTable):
    """Official-paradigm execution of the reference hash table.

    DeepSeek's public Engram deployment stores the n-gram tables host-side,
    shards the rows over the EP mesh, routes lookups through an all-to-all
    exchange, and trains the tables with owner-local sparse gradients fed to
    SparseAdam; dense table gradients are never materialized. This subclass
    keeps the reference hash math verbatim (the rolling-hash ids, per-head
    offsets and the head-major ``[B, S, orders*heads*Dh]`` memory layout) and
    swaps only that storage/lookup/training execution, following the
    production DeepSeek-V4.1 ``HostEngramTable`` recipe (owner-approved
    deviation from the reference preview's dense device table).

    The weight is CPU-resident in real builds; the meta capture build keeps
    it on the ambient meta device (the host-offload flag carries the
    residency fact, ``_apply`` keeps real builds off device transforms), FSDP
    excludes it via ``ignored_params`` (the table shards itself over EP
    instead) and the sparse-gradient lifecycle follows the shared
    ``uses_host_offload`` protocol that the HostSparse optimizer container
    and the gradient clipper drive. Meta capture replaces ``_lookup_rows``
    with a shape-visible decomposition (see
    ``torchtitan_simulator.compat.system3_engram``).
    """

    uses_host_offload = True

    @dataclass(kw_only=True, slots=True)
    class Config(MultiHeadHashTable.Config):
        layer_idx: int = 0
        pin_memory: bool = False

    def __init__(self, config: Config):
        self.pin_memory = config.pin_memory
        self.layer_idx = config.layer_idx
        self._table_mesh = None
        self._ep_mesh = None
        self._ep_rank = 0
        self._ep_size = 1
        self._pending_sparse_grad: torch.Tensor | None = None
        self._replica_group = None
        self._replica_size = 1
        super().__init__(config)
        # The reference allocates on the ambient default device. Real builds
        # keep the table on CPU (production Engram residency); the meta
        # capture build follows the ambient meta device so the estimator's
        # counted-device snapshot sees the weight and reports its host
        # budget through the ``_engram_host_offload`` flag (the
        # DeepSeek-V4.1 table follows the same recipe).
        ambient = torch.get_default_device()
        host_device = ambient if ambient.type == "meta" else torch.device("cpu")
        weight = self._parameters.pop("weight")
        assert isinstance(weight, torch.nn.Parameter)
        weight = nn.Parameter(
            torch.empty(
                weight.shape,
                dtype=weight.dtype,
                device=host_device,
                pin_memory=self.pin_memory if host_device.type == "cpu" else False,
            ),
            requires_grad=weight.requires_grad,
        )
        self.register_parameter("weight", weight)
        self._mark_host_weight()

    def _mark_host_weight(self) -> None:
        # Optimizer construction follows parallelize/to_empty, so preserve the
        # Host marker and checkpoint shard suffix on replacement parameters.
        self.weight._engram_host_offload = True  # type: ignore[attr-defined]
        self.weight._engram_checkpoint_suffix = self._checkpoint_suffix()  # type: ignore[attr-defined]

    def _checkpoint_suffix(self) -> str:
        if self._ep_size <= 1:
            return ""
        return f".ep_shard_{self._ep_rank:05d}_of_{self._ep_size:05d}"

    def _apply(self, fn, recurse=True):
        """Apply device transforms to metadata while keeping weight on CPU."""
        weight = self._parameters.pop("weight")
        assert isinstance(weight, torch.nn.Parameter)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            # The table weight was created outside this transform's scope;
            # real builds must keep it CPU-resident, so re-register the
            # original parameter untouched. (Meta capture builds never route
            # through ``_apply``: the weight is born on the ambient meta
            # device in ``__init__``.)
            self.register_parameter("weight", weight)
            self._mark_host_weight()
        return result

    def _init_self_parameters(self) -> None:
        # Rank-dependent streams avoid repeating the same local shard on
        # every EP owner; mirrors the production HostEngramTable seeding.
        shard_rank = self._table_mesh.get_local_rank() if self._table_mesh is not None else self._ep_rank
        seed = (torch.initial_seed() + 1000003 * self.layer_idx + 9176 * shard_rank) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.reset_parameters()

    # -- lookup ----------------------------------------------------------

    @property
    def lookup_mesh(self):
        return self._ep_mesh

    def _lookup_rows(self, flat_ids: torch.Tensor) -> torch.Tensor:
        """Official host lookup: local at EP=1, all-to-all routed at EP>1."""
        from torchtitan_npu.models.deepseek_v4_1.engram.lookup import _lookup_grad_rows, _lookup_rows

        return _System3HostLookup.apply(self.weight, flat_ids, self, _lookup_rows, _lookup_grad_rows)

    def forward(self, hash_indices):
        _b, _s, nh = hash_indices.shape
        # Head-major flat row ids: identical rows and concat order to the
        # reference per-head ``F.embedding`` loop.
        flat_ids = hash_indices + self.head_offsets.view(1, 1, nh)
        rows = self._lookup_rows(flat_ids)  # [b, s, nh, head_dim]
        return rows.reshape(*hash_indices.shape[:-1], nh * self.per_head_dim)

    # -- EP sharding -------------------------------------------------------

    def parallelize(self, parallel_dims) -> None:
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        ep_size = ep_mesh.size() if ep_mesh is not None else 1
        self._ep_mesh = ep_mesh
        self._table_mesh = None
        if ep_size <= 1:
            self._ep_rank = 0
            self._ep_size = 1
            return
        rows = self.weight.shape[0]
        if rows % ep_size != 0:
            raise ValueError(f"Engram hash table has {rows} physical rows, not divisible by its EP degree {ep_size}.")
        # Keep weight out of generic SpmdLayout distribution. Hash metadata is
        # still parallelized by the inherited sharding config.
        global_weight = self._parameters.pop("weight")
        assert isinstance(global_weight, torch.nn.Parameter)
        try:
            super().parallelize(parallel_dims)
        finally:
            local_rows = global_weight.shape[0] // ep_size
            local_weight = nn.Parameter(
                torch.empty(
                    local_rows,
                    global_weight.shape[1],
                    dtype=global_weight.dtype,
                    device=global_weight.device,
                ),
                requires_grad=global_weight.requires_grad,
            )
            self.register_parameter("weight", local_weight)

        self._ep_rank = ep_mesh.get_local_rank() if ep_mesh is not None else 0
        self._ep_size = ep_size
        self._mark_host_weight()

    def wire_sparse_grad_replicas(self, *, edp_mesh, edp_mesh_dims=None) -> None:
        """Record the ranks that hold a copy of this EP shard.

        FSDP would reduce the replicas of the parameters it manages; this
        table is deliberately unmanaged, so it sums its owner-local sparse
        gradients itself along the sparse data-parallel axes. The EP axis
        holds disjoint row ranges and is excluded from the reduction.
        """
        self._replica_group = None
        self._replica_size = 1
        if edp_mesh is None:
            return
        if edp_mesh_dims is not None:
            replica_axes = (edp_mesh_dims.replicate, edp_mesh_dims.shard)
            # Dense FSDP may fold (dp_shard, cp) into a tuple of shard axes.
            axes = tuple(name for axis in replica_axes if axis for name in ((axis,) if isinstance(axis, str) else axis))
            if not axes:
                return
            replica_mesh = edp_mesh[axes]
        else:
            replica_mesh = edp_mesh
        if replica_mesh.size() <= 1:
            return
        # Replica groups for different EP shards are disjoint. Only members
        # participate in creating their group; no private mesh flattening API.
        self._replica_group = cast(
            "dist.ProcessGroup",
            (
                replica_mesh.get_group()
                if replica_mesh.ndim == 1
                else dist.new_group(ranks=replica_mesh.mesh.flatten().tolist(), use_local_synchronization=True)
            ),
        )
        self._replica_size = replica_mesh.size()

    @torch.no_grad()
    def reduce_sparse_gradient_across_replicas(self) -> None:
        """Sum this step's sparse gradient over the replicas of this shard.

        Called from the gradient clipper, which TorchTitan runs unconditionally
        before every optimizer step, so the norm is taken on the reduced
        gradient. Each rank pads its owner-local rows to the group maximum,
        all-gathers them, and coalesces: every replica ends up with the same
        summed gradient, which keeps their weights and SparseAdam state in step
        without ever materializing a dense shard gradient.
        """
        group = self._replica_group
        if group is None:
            return

        device = self.head_offsets.device
        pending = self._pending_sparse_grad
        if pending is None:
            local_ids = torch.empty(0, dtype=torch.int64)
            local_values = torch.empty(0, self.weight.shape[1], dtype=torch.float32)
        else:
            local_ids = pending.indices()[0]
            local_values = pending.values()

        # Row counts differ per replica, so the gather has to be padded to the
        # largest one; collecting the counts first also says where to cut.
        counts = torch.zeros(self._replica_size, dtype=torch.int64, device=device)
        counts[dist.get_rank(group)] = local_ids.numel()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        max_rows = int(counts.max())
        if max_rows == 0:
            return

        padded_ids = torch.zeros(max_rows, dtype=torch.int64, device=device)
        padded_ids[: local_ids.numel()] = local_ids.to(device=device)
        padded_values = torch.zeros(max_rows, self.weight.shape[1], dtype=torch.float32, device=device)
        padded_values[: local_values.shape[0]] = local_values.to(device=device)

        gathered_ids = [torch.empty_like(padded_ids) for _ in range(self._replica_size)]
        gathered_values = [torch.empty_like(padded_values) for _ in range(self._replica_size)]
        dist.all_gather(gathered_ids, padded_ids, group=group)
        dist.all_gather(gathered_values, padded_values, group=group)

        ids_parts = []
        values_parts = []
        for replica in range(self._replica_size):
            rows = int(counts[replica])
            if rows:
                ids_parts.append(gathered_ids[replica][:rows].to(device="cpu", copy=True))
                values_parts.append(gathered_values[replica][:rows].to(device="cpu", copy=True))

        self._pending_sparse_grad = torch.sparse_coo_tensor(
            torch.cat(ids_parts).unsqueeze(0),
            torch.cat(values_parts),
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()

    # -- checkpoint shards --------------------------------------------------

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        suffix = self._checkpoint_suffix()
        if suffix:
            destination[prefix + "weight" + suffix] = destination.pop(prefix + "weight")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        suffix = self._checkpoint_suffix()
        shard_key = prefix + "weight" + suffix
        weight_key = prefix + "weight"
        if suffix and shard_key in state_dict:
            state_dict[weight_key] = state_dict.pop(shard_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # -- sparse-gradient lifecycle -------------------------------------------

    def accumulate_sparse_gradient(
        self,
        local_row_ids: torch.Tensor,
        grad_rows: torch.Tensor,
    ) -> None:
        """Accumulate one microbatch's owner-local rows in FP32 on CPU."""
        local_row_ids = local_row_ids.reshape(-1)
        grad_rows = grad_rows.reshape(local_row_ids.numel(), self.weight.shape[1])
        ids = local_row_ids.to(device="cpu", dtype=torch.int64)
        values = grad_rows.to(device="cpu", dtype=torch.float32)
        if ids.numel():
            min_id = int(ids.min())
            max_id = int(ids.max())
            if min_id < 0 or max_id >= self.weight.shape[0]:
                raise IndexError(
                    "Engram hash lookup returned local row outside "
                    f"[0, {self.weight.shape[0]}): got range [{min_id}, {max_id}] "
                    f"across {ids.numel()} sparse entries."
                )
        sparse_grad = torch.sparse_coo_tensor(
            ids.unsqueeze(0),
            values,
            size=self.weight.shape,
            dtype=torch.float32,
            device="cpu",
            check_invariants=False,
        ).coalesce()
        if self._pending_sparse_grad is None:
            self._pending_sparse_grad = sparse_grad
        else:
            self._pending_sparse_grad = (self._pending_sparse_grad + sparse_grad).coalesce()

    def pending_sparse_grad(self) -> torch.Tensor | None:
        """The accumulated CPU sparse gradient, before it reaches SparseAdam."""
        return self._pending_sparse_grad

    def scale_pending_sparse_grad(self, scale: float) -> None:
        """Apply the global gradient-clipping coefficient to the sparse gradient."""
        if self._pending_sparse_grad is None:
            return
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            self._pending_sparse_grad.indices(),
            self._pending_sparse_grad.values() * scale,
            self._pending_sparse_grad.shape,
        ).coalesce()

    def prepare_sparse_optimizer_step(self) -> None:
        if self._pending_sparse_grad is None:
            self.weight.grad = None
            return
        self.weight.grad = self._pending_sparse_grad.to(dtype=self.weight.dtype)

    def clear_sparse_gradient(self) -> None:
        self._pending_sparse_grad = None
        self.weight.grad = None

    def mark_sparse_step_complete(self) -> None:
        self._pending_sparse_grad = None

    def refresh_lookup_storage(self, row_ids: torch.Tensor) -> None:
        """Refresh derived lookup rows after an optimizer update, if any.

        FP32 lookup uses the master weight directly and needs no refresh.
        """


class EngramModule(Module):
    """Engram conditional memory (DeepSeek V4.1 Flash convention).

    Per inject layer: 3-order rolling-hash lookup -> concat memory
    [b, s, orders*heads*head_dim] -> rmsnorm_ssr static gate
    (RMSNorm dot -> signed-square-root -> sigmoid -> static id!=0 token mask)
    -> gated value projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        key_proj: Linear.Config
        value_proj: Linear.Config
        query_proj: Linear.Config
        q_norm: System3RMSNorm.Config
        k_norm: System3RMSNorm.Config
        input_norm: System3RMSNorm.Config
        output_norm: System3RMSNorm.Config
        hash_fns: list[PolynomialRollingHash.Config]
        hash_tables: list[MultiHeadHashTable.Config]

        def __post_init__(self) -> None:
            if len(self.hash_fns) != len(self.hash_tables):
                raise ValueError("each n-gram order needs exactly one hash table")
            concat_dim = self.hash_fns[0].num_heads * self.hash_tables[0].head_dim * len(self.hash_fns)
            if self.key_proj.in_features != concat_dim or self.value_proj.in_features != concat_dim:
                raise ValueError("engram key/value projections must consume the concatenated memory")

    def __init__(self, config: Config):
        super().__init__()
        d = config.hidden_size
        self.hash_fns = ModuleList([hash_fn.build() for hash_fn in config.hash_fns])
        self.hash_tables = ModuleList([table.build() for table in config.hash_tables])
        self.key_proj = config.key_proj.build()
        self.value_proj = config.value_proj.build()
        self.query_proj = config.query_proj.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()
        self.input_norm = config.input_norm.build()
        self.output_norm = config.output_norm.build()
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, hidden_states, input_ids):
        # Retrieve: per-order hash -> table lookup -> concat [b, s, concat_dim]
        parts = []
        for hash_fn, hash_table in zip(self.hash_fns, self.hash_tables, strict=True):
            parts.append(hash_table(hash_fn(input_ids)))
        mem = torch.cat(parts, dim=-1)

        # rmsnorm_ssr static gate: RMSNorm dot -> signed sqrt -> sigmoid ->
        # static token mask (positions without a token id, id == 0, inject
        # no memory).
        x_proj = self.query_proj(self.input_norm(hidden_states))
        return self._gate(
            x_proj, self.key_proj(mem), self.value_proj(mem), input_ids
        ).to(hidden_states.dtype)

    def _gate(self, x_proj, table_proj, value_proj, input_ids):
        """rmsnorm_ssr static gate over the projected engram streams.

        ``x_proj``/``table_proj``/``value_proj`` are the pre-norm projected
        query/key/value streams; returns the unrounded gated value stream in
        the projected-stream dtype. The fused capture boundary replaces this
        method wholesale with the same math (fp32-internal gate math, result
        rounded back to the stream dtype by the caller).
        """
        q = self.q_norm(x_proj)
        k = self.k_norm(table_proj)
        dot = (q * k).sum(dim=-1, keepdim=True) * self.scale
        gate = torch.sigmoid(dot.sign() * dot.abs().sqrt())
        gate = gate * (input_ids != 0).unsqueeze(-1).to(gate.dtype)

        out = self.output_norm(value_proj)
        # V4.1 engram convention: the fp32 host-resident table and the gate
        # math stay fp32-internal; the caller rounds the product back to the
        # stream dtype so the lookup does not promote the residual chain.
        return gate * out


# ============================================================================
# Section 8: mHC hyper-connections (reference verbatim)
# ============================================================================


class SinkhornIteration(Module):
    """Sinkhorn iteration constraining the branch mix matrix to be doubly
    stochastic (float32 throughout: exp(20) overflows fp16)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_iters: int = 20

    def __init__(self, config: Config):
        super().__init__()
        self.n_iters = config.n_iters

    def forward(self, w):
        orig_dtype = w.dtype
        w = torch.exp(torch.clamp(w.float(), max=20.0))
        for _ in range(self.n_iters):
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            w = w / w.sum(dim=-2, keepdim=True).clamp(min=1e-12)
        return w.to(orig_dtype)  # normalized values <= 1, safe in the original dtype


class HyperConnectionBlock(Module):
    """mHC block: expand -> Sinkhorn-constrained branch mix -> contract + residual."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        hc_mult: int
        branch_dim: int
        expand: Linear.Config
        contract: Linear.Config
        pre_norm: System3RMSNorm.Config
        post_norm: System3RMSNorm.Config
        sinkhorn: SinkhornIteration.Config

        def __post_init__(self) -> None:
            if self.expand.out_features != self.hc_mult * self.branch_dim:
                raise ValueError("mHC expand must project to hc_mult * branch_dim")
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("mix_weights")

    def __init__(self, config: Config):
        super().__init__()
        self.expand = config.expand.build()
        self.sinkhorn = config.sinkhorn.build()
        self.mix_weights = nn.Parameter(torch.randn(config.hc_mult, config.hc_mult) * 0.02)
        self.contract = config.contract.build()
        self.pre_norm = config.pre_norm.build()
        self.post_norm = config.post_norm.build()
        self.hc_mult = config.hc_mult
        self.branch_dim = config.branch_dim

    def reset_parameters(self) -> None:
        # Reference init: ``torch.randn(hc_mult, hc_mult) * 0.02``. In-place so
        # the tensor follows the parameter's device (meta build, then NPU).
        with torch.no_grad():
            self.mix_weights.normal_().mul_(0.02)

    def forward(self, x):
        b, s, _d = x.shape
        residual = x
        x = self.pre_norm(x)
        expanded = self.expand(x).view(b, s, self.hc_mult, self.branch_dim)
        mix = self.sinkhorn(self.mix_weights)
        # Eager einsum would promote the bf16 activations with the fp32 mix
        # matrix to fp32 math; state the promotion explicitly so the Meta
        # capture sees one dtype. Like the V4.1 fused HcPre (fp32-internal
        # math, ``.to(x.dtype)`` return), the fp32 accumulation is kept but
        # the result rounds back to the stream dtype so the residual chain
        # stays bf16 under the fsdp2 param_dtype view.
        mixed = torch.einsum("bshd,ho->bsod", expanded.float(), mix.float()).to(x.dtype)
        merged = mixed.reshape(b, s, self.hc_mult * self.branch_dim)
        return residual + self.post_norm(self.contract(merged))


# ============================================================================
# Section 9: transformer block (mHC -> attention -> mHC -> MoE/Engram)
# ============================================================================


class FlashBlock(Module):
    """Block: mHC -> Attention (ISA/GDN2) -> mHC -> MoE / Engram.

    ISA layers participate in the per-group KV slot dict (kv role writes,
    reindex re-selects, reuse reads); GDN2 layers keep only their fixed state.
    Engram inject layers are memory layers (no MoE) and need input_ids for
    the n-gram hash.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        layer_type: str
        hc_pre_attn: HyperConnectionBlock.Config
        hc_pre_ffn: HyperConnectionBlock.Config
        attention: AlternatingAttention.Config | GDN2Attention.Config  # pyrefly: ignore [bad-override]
        attn_pre_norm: System3RMSNorm.Config
        attn_post_norm: System3RMSNorm.Config
        moe_pre_norm: System3RMSNorm.Config
        moe_post_norm: System3RMSNorm.Config
        moe: LatentMoELayer.Config | None = None
        engram: EngramModule.Config | None = None

        def __post_init__(self) -> None:
            if (self.moe is None) == (self.engram is None):
                raise ValueError("exactly one of moe / engram must be set per block")

    def __init__(self, config: Config):
        super().__init__()
        self.layer_idx = config.layer_idx
        self.layer_type = config.layer_type
        self.moe_enabled = config.moe is not None
        self.hc_pre_attn = config.hc_pre_attn.build()
        self.hc_pre_ffn = config.hc_pre_ffn.build()
        self.attention = config.attention.build()
        self.attn_pre_norm = config.attn_pre_norm.build()
        self.attn_post_norm = config.attn_post_norm.build()
        if self.moe_enabled:
            self.moe = config.moe.build()  # pyrefly: ignore [missing-attribute]
        else:
            self.engram = config.engram.build()  # pyrefly: ignore [missing-attribute]
        self.moe_pre_norm = config.moe_pre_norm.build()
        self.moe_post_norm = config.moe_post_norm.build()

    def forward(self, hidden_states, kv_bufs=None, input_ids=None, attention_mask=None):
        h = hidden_states
        # mHC + Attention
        h = self.hc_pre_attn(h)
        residual = h
        h = residual + self.attention(self.attn_pre_norm(h), kv_bufs, attention_mask)
        h = self.attn_post_norm(h)

        # mHC + MoE / Engram
        h = self.hc_pre_ffn(h)
        residual = h
        if self.moe_enabled:
            ffn_out = self.moe(self.moe_pre_norm(h))  # pyrefly: ignore [missing-attribute]
        else:
            if input_ids is None:
                raise RuntimeError(f"Engram inject layer {self.layer_idx} needs input_ids for the n-gram hash")
            ffn_out = self.engram(self.moe_pre_norm(h), input_ids)  # pyrefly: ignore [missing-attribute]
        h = residual + ffn_out
        h = self.moe_post_norm(h)

        return h


# ============================================================================
# Section 10: multimodal encoders (reference verbatim, sharding-annotated)
# ============================================================================


class EncoderAttention(Module):
    """Standard MHA for modality encoders (bidirectional, no causal mask)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden: int
        heads: int
        qkv: Linear.Config
        o: Linear.Config

        def __post_init__(self) -> None:
            if self.hidden % self.heads != 0:
                raise ValueError(f"encoder hidden ({self.hidden}) must divide heads ({self.heads})")
            if self.qkv.out_features != 3 * self.hidden:
                raise ValueError("encoder qkv must project to 3 * hidden")

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden // config.heads
        self.qkv = config.qkv.build()
        self.o = config.o.build()
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


class EncoderBlock(Module):
    """Pre-norm bidirectional transformer block for modality encoders."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: System3RMSNorm.Config
        attn: EncoderAttention.Config
        norm2: System3RMSNorm.Config
        mlp_up: Linear.Config
        mlp_down: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.attn = config.attn.build()
        self.norm2 = config.norm2.build()
        self.mlp_up = config.mlp_up.build()
        self.mlp_down = config.mlp_down.build()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.norm2(x))))
        return x


class ImageEncoder(Module):
    """Image encoder: ViT-style patchify -> pos embed -> bidirectional blocks
    -> rmsnorm. The reference's ``nn.Conv2d`` patch stem is a configurable
    ``Conv2d`` wrapper (identical op)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        max_patches: int
        patch_embed: Conv2d.Config
        pos_embed_sharding: ShardingConfig | None = None
        blocks: list[EncoderBlock.Config]
        norm: System3RMSNorm.Config

        def __post_init__(self) -> None:
            if self.pos_embed_sharding is None:
                self.pos_embed_sharding = _replicated_sharding("pos_embed")
            if self.patch_embed.bias:
                raise ValueError("the reference patch stem has no bias")

    def __init__(self, config: Config):
        super().__init__()
        self.patch_embed = config.patch_embed.build()
        self.max_patches = config.max_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_patches, config.patch_embed.out_channels))
        self._sharding_config = config.pos_embed_sharding
        self.blocks = nn.ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()

    def reset_parameters(self) -> None:
        # Reference init: the position embedding stays at zeros.
        nn.init.zeros_(self.pos_embed)

    def forward(self, x):
        # x: [b, 3, H, W]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [b, n, h]
        if x.shape[1] > self.max_patches:
            raise ValueError(f"image patch count {x.shape[1]} exceeds image_max_patches={self.max_patches}")
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class AudioEncoder(Module):
    """Audio encoder: raw waveform -> 1D-conv stem (~40x downsample) +
    bidirectional transformer -> frame tokens. The reference's conv/GELU
    ``nn.Sequential`` stem is expressed as explicit leaf modules (identical
    op stream) so the conv weights carry sharding annotations."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        max_frames: int
        stem_conv1: Conv1d.Config
        stem_conv2: Conv1d.Config
        pos_embed_sharding: ShardingConfig | None = None
        blocks: list[EncoderBlock.Config]
        norm: System3RMSNorm.Config

        def __post_init__(self) -> None:
            if self.pos_embed_sharding is None:
                self.pos_embed_sharding = _replicated_sharding("pos_embed")

    def __init__(self, config: Config):
        super().__init__()
        self.stem_conv1 = config.stem_conv1.build()
        self.stem_act1 = GELU(GELU.Config())
        self.stem_conv2 = config.stem_conv2.build()
        self.stem_act2 = GELU(GELU.Config())
        self.max_frames = config.max_frames
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_frames, config.stem_conv1.out_channels))
        self._sharding_config = config.pos_embed_sharding
        self.blocks = nn.ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.pos_embed)

    def forward(self, x):
        # x: [b, T] or [b, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem_conv1(x)
        x = self.stem_act1(x)
        x = self.stem_conv2(x)
        x = self.stem_act2(x)
        x = x.transpose(1, 2)  # [b, frames, h]
        if x.shape[1] > self.max_frames:
            raise ValueError(f"audio frame count {x.shape[1]} exceeds audio_max_frames={self.max_frames}")
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PointCloudEncoder(Module):
    """Embodied 3D encoder (PointNet-style set encoding): shared per-point MLP
    + global max-pool summary token + per-point tokens (permutation-invariant)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mlp_in: Linear.Config
        mlp_hidden: Linear.Config
        mlp_out: Linear.Config
        blocks: list[EncoderBlock.Config]
        norm: System3RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.mlp_in = config.mlp_in.build()
        self.mlp_act1 = GELU(GELU.Config())
        self.mlp_hidden = config.mlp_hidden.build()
        self.mlp_act2 = GELU(GELU.Config())
        self.mlp_out = config.mlp_out.build()
        self.blocks = nn.ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()

    def forward(self, x):
        # x: [b, N, 3] point coordinates
        z = self.mlp_in(x)
        z = self.mlp_act1(z)
        z = self.mlp_hidden(z)
        z = self.mlp_act2(z)
        z = self.mlp_out(z)  # [b, N, h]
        g = z.max(dim=1, keepdim=True).values  # global summary token
        tokens = torch.cat([g, z], dim=1)  # [b, 1+N, h]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class HapticEncoder(Module):
    """Embodied haptic-somatosensory encoder: multi-channel stream ->
    1D-conv downsample + bidirectional transformer -> tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        max_frames: int
        stem_conv: Conv1d.Config
        pos_embed_sharding: ShardingConfig | None = None
        blocks: list[EncoderBlock.Config]
        norm: System3RMSNorm.Config

        def __post_init__(self) -> None:
            if self.pos_embed_sharding is None:
                self.pos_embed_sharding = _replicated_sharding("pos_embed")

    def __init__(self, config: Config):
        super().__init__()
        self.stem_conv = config.stem_conv.build()
        self.stem_act = GELU(GELU.Config())
        self.max_frames = config.max_frames
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_frames, config.stem_conv.out_channels))
        self._sharding_config = config.pos_embed_sharding
        self.blocks = nn.ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.pos_embed)

    def forward(self, x):
        # x: [b, T, C]
        x = self.stem_conv(x.transpose(1, 2))
        x = self.stem_act(x)
        x = x.transpose(1, 2)  # [b, frames, h]
        if x.shape[1] > self.max_frames:
            raise ValueError(f"haptic frame count {x.shape[1]} exceeds haptic_max_frames={self.max_frames}")
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ModalityEncoder(Module):
    """Multimodal encoder hub: per-modality encoders + projection into the
    model hidden space + learned modality-type embeddings. Text goes through
    the backbone word embeddings directly."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        modalities: list[str]
        image_encoder: ImageEncoder.Config | None = None
        audio_encoder: AudioEncoder.Config | None = None
        point3d_encoder: PointCloudEncoder.Config | None = None
        haptic_encoder: HapticEncoder.Config | None = None
        modality_projs: dict[str, Linear.Config]
        modality_type_embedding: Embedding.Config

        def __post_init__(self) -> None:
            expected = set(self.modalities) - {"text"}
            present = set(self.modality_projs)
            if expected != present:
                raise ValueError(f"modality projections {sorted(present)} must cover {sorted(expected)}")

    def __init__(self, config: Config):
        super().__init__()
        self.modalities = config.modalities
        if config.image_encoder is not None:
            self.image_encoder = config.image_encoder.build()
        if config.audio_encoder is not None:
            self.audio_encoder = config.audio_encoder.build()
        if config.point3d_encoder is not None:
            self.point3d_encoder = config.point3d_encoder.build()
        if config.haptic_encoder is not None:
            self.haptic_encoder = config.haptic_encoder.build()
        self.modality_projs = nn.ModuleDict({name: proj.build() for name, proj in config.modality_projs.items()})
        self.modality_type_embedding = config.modality_type_embedding.build()
        self.type_ids = {m: i for i, m in enumerate(config.modalities)}

    def encode(self, modality: str, x) -> torch.Tensor:
        """Returns [b, L_m, hidden_size] for one modality segment."""
        # Raw modality payloads arrive as fp32 data; without a cast they would
        # promote the whole encoder chain (and then the segment concat) to
        # fp32, defeating the fsdp2 bf16 param view. Cast at the entry to the
        # projection weight dtype (bf16 under the param_dtype view, fp32 in
        # pure-fp32 runs); encoders keep fp32-internal/cast-back math.
        x = x.to(self.modality_projs[modality].weight.dtype)
        if modality == "image":
            z = self.image_encoder(x)  # pyrefly: ignore [missing-attribute]
        elif modality == "audio":
            z = self.audio_encoder(x)  # pyrefly: ignore [missing-attribute]
        elif modality == "point3d":
            z = self.point3d_encoder(x)  # pyrefly: ignore [missing-attribute]
        elif modality == "haptic":
            z = self.haptic_encoder(x)  # pyrefly: ignore [missing-attribute]
        else:
            raise ValueError(f"unknown modality {modality!r} (modalities={self.modalities})")
        z = self.modality_projs[modality](z)
        type_id = torch.tensor(self.type_ids[modality], device=z.device)
        return z + self.modality_type_embedding(type_id)


# ============================================================================
# Section 11: full model (looped backbone + multimodal AR sequence)
# ============================================================================


class System3Model(Decoder):
    """System3 Flash: looped transformer + multimodal encoders.

    The backbone runs the ``base_depth`` physical blocks ``num_recursion``
    times (logical depth = physical x recursion; parameters stored once, the
    optimizer sees each Parameter once with accumulated gradients). Each pass
    rebuilds the per-group KV slots (kv role writes, reindex/reuse read); the
    slots are NOT shared across passes (reference semantics). The head is
    executed by the loss when ``_skip_lm_head`` is set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # The decoder-wide norm is this family's fp32 RMSNorm (upstream field
        # type is the common RMSNorm.Config).
        norm: System3RMSNorm.Config  # pyrefly: ignore [bad-override]
        # Loop geometry (validated by the config builder).
        num_recursion: int = 1
        # Flash KV sharing.
        kv_share_enable: bool = True
        kv_groups: int = 1
        unit_size: int = 4
        # Multimodal.
        multimodal_enable: bool = False
        modality_encoder: ModalityEncoder.Config | None = None

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "system3 does not implement tensor parallelism: the ISA head axis, the "
                    "GDN2 head axis and the mHC branch mixing would all need sharded layouts; "
                    "capture covers DP/FSDP/EP only"
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            from .sharding import set_system3_sharding_config

            set_system3_sharding_config(self, enable_ep=parallelism.expert_parallel_degree > 1)
            if parallelism.expert_parallel_degree > 1:
                from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

                for layer_cfg in self.layers:
                    if layer_cfg.moe is not None:
                        layer_cfg.moe.routed_experts.token_dispatcher = AllToAllTokenDispatcher.Config(
                            num_experts=layer_cfg.moe.routed_experts.inner_experts.num_experts,
                            top_k=layer_cfg.moe.router.topk,
                        )

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            # Debug-grade estimate: stored parameters (physical blocks once)
            # plus the compute-equivalent active-parameter FLOPs over the full
            # loop (routed experts contribute their top-k share only; Engram
            # tables count as lookup).
            nparams = sum(parameter.numel() for parameter in model.parameters())
            active = 0
            for layer in self.layers:
                attn = layer.attention
                d = attn.hidden_size
                dk = getattr(attn, "d_k", None)
                if dk is None:
                    # ISA layer (AlternatingAttention.Config): Q/O projections
                    # on every ISA layer; K/V on the kv-role layer.
                    active += 2 * attn.num_heads * attn.head_dim * d
                    if attn.role == "kv":
                        active += 2 * attn.num_heads * attn.head_dim * d
                    if attn.indexer is not None:
                        idx_dim = attn.indexer.n_heads * attn.indexer.head_dim
                        active += 2 * d * idx_dim
                else:
                    # GDN2 layer (GDN2Attention.Config).
                    active += 3 * d * attn.num_heads * dk
                    active += attn.num_heads * attn.d_v * d
                    active += d * attn.num_heads * dk  # alpha gate
                    active += d * 2 * attn.num_heads * attn.d_state  # beta gate
                    active += d * attn.num_heads * attn.d_a  # da proj
                if layer.moe is not None:
                    moe = layer.moe
                    latent = moe.gate_down.out_features
                    active += 3 * d * latent  # gate_down/up_down/latent_up
                    active += d * moe.router.num_experts
                    per_expert = 3 * moe.routed_experts.inner_experts.hidden_dim * latent
                    active += moe.router.topk * per_expert
                    for shared in moe.shared_experts:
                        active += 3 * shared.gate_inter.out_features * shared.gate_inter.in_features
            for layer in self.layers:
                for hc in (layer.hc_pre_attn, layer.hc_pre_ffn):
                    active += 2 * hc.expand.in_features * hc.expand.out_features
            active *= self.num_recursion
            active += self.dim * self.vocab_size  # embedding lookup + head GEMM
            return nparams, 6 * active * seq_len

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_recursion = config.num_recursion
        self.kv_share_enable = config.kv_share_enable
        self.kv_groups = config.kv_groups
        self.unit_size = config.unit_size
        self.multimodal_enable = config.multimodal_enable
        self.modality_encoder = config.modality_encoder.build() if config.modality_encoder is not None else None

    def _n_kv_slots(self) -> int:
        return self.kv_groups if self.kv_share_enable else len(self.layers) // self.unit_size

    # ------------------------------------------------------------------
    # Multimodal sequence construction (reference verbatim)
    # ------------------------------------------------------------------
    def build_multimodal_sequence(self, input_ids, modality_inputs, layout):
        """AR multimodal input assembly: encode per layout segment and concat
        into one sequence. Returns (hidden_states [b, S, d], ids_full [b, S]);
        ids_full feeds the Engram hash (modality segments are id 0, which the
        static gate masks). Each segment type appears at most once."""

        if self.modality_encoder is None:
            raise RuntimeError("multimodal is not enabled (multimodal_enable=False)")
        parts = []
        ids_parts = []
        seen = set()
        for key in layout:
            if key in seen:
                raise ValueError(
                    f"layout contains a duplicate segment {key!r}: each segment type "
                    f"(including text) may appear at most once"
                )
            seen.add(key)
            if key == "text":
                emb = self.tok_embeddings(input_ids)
                ids_parts.append(input_ids)
            else:
                x = modality_inputs.get(key)
                if x is None:
                    raise ValueError(f"layout contains {key} but no matching input was provided")
                emb = self.modality_encoder.encode(key, x)
                ids_parts.append(torch.zeros(emb.shape[0], emb.shape[1], dtype=torch.long, device=emb.device))
            parts.append(emb)
        return torch.cat(parts, dim=1), torch.cat(ids_parts, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
        *,
        attention_metadata=None,
        image: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        point3d: torch.Tensor | None = None,
        haptic: torch.Tensor | None = None,
        layout: list[str] | None = None,
    ) -> torch.Tensor:
        del positions, attention_masks, attention_metadata  # maskless capture family
        if layout is not None:
            modality_inputs = {"image": image, "audio": audio, "point3d": point3d, "haptic": haptic}
            hidden_states, ids_full = self.build_multimodal_sequence(tokens, modality_inputs, layout)
        else:
            hidden_states = self.tok_embeddings(tokens)
            ids_full = tokens

        # The attention_mask stays None in capture: the synthetic text rows are
        # full length (no padding), which is exactly the reference's
        # mask=None branch (all positions valid).
        attention_mask = None

        # Loop recursion: num_recursion passes over the physical blocks; the
        # KV slots are rebuilt each pass (per-group dict, overwritten by the
        # kv-role layer).
        n_slots = self._n_kv_slots()
        kv_bufs: dict[int, dict[str, torch.Tensor]] = {g: {} for g in range(n_slots)}
        h = hidden_states
        for _ in range(self.num_recursion):
            for layer in self.layers.values():
                h = layer(h, kv_bufs, ids_full, attention_mask)

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    def build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
        """Runner input-prep seam.

        Text flavor: no causal/padding mask (full-length synthetic rows;
        the reference treats mask=None as all-valid). CP is rejected: the ISA
        compressed grid, GDN2 state and Engram hash all span the full
        sequence.

        Multimodal flavor: synthesizes the deterministic modality segments in
        front of the text (reference smoke geometry: image 64x64/patch 8,
        audio 40 samples, point3d 16 points, haptic 40x4) and returns the
        full-sequence labels with -100 on the modality positions so the loss
        lands on the text span only.
        """
        if cp_mesh is not None:
            if load_balancer_type is not None:
                raise NotImplementedError("system3 capture does not use a CP load balancer")
            raise NotImplementedError("system3 does not support context parallel")
        if not self.multimodal_enable:
            return tokens, labels, dict(extra_inputs)

        # Deterministic synthetic modality segments in front of the text span
        # (debug-multimodal flavor only). The shapes are derived from the
        # config tree so every encoder contract holds; values are irrelevant
        # for capture (real multimodal data arrives through a dataloader,
        # which the text-only runner stack does not provide).
        b, s_text = tokens.shape
        device = tokens.device
        modality_cfg = self.config.modality_encoder  # pyrefly: ignore [missing-attribute]
        assert modality_cfg is not None

        layout: list[str] = []
        modality_kwargs: dict[str, torch.Tensor] = {}
        seg_lengths: dict[str, int] = {}
        image_cfg = modality_cfg.image_encoder
        if image_cfg is not None:
            patch = image_cfg.patch_embed.kernel_size
            side = math.isqrt(image_cfg.max_patches) * patch
            image = torch.zeros(b, image_cfg.patch_embed.in_channels, side, side, device=device)
            layout.append("image")
            modality_kwargs["image"] = image
            seg_lengths["image"] = image_cfg.max_patches
        audio_cfg = modality_cfg.audio_encoder
        if audio_cfg is not None:
            kernel = audio_cfg.stem_conv1.kernel_size
            stride = audio_cfg.stem_conv1.stride
            frames = min(19, audio_cfg.max_frames)
            samples = (frames - 1) * stride + kernel
            audio = torch.zeros(b, samples, device=device)
            layout.append("audio")
            modality_kwargs["audio"] = audio
            seg_lengths["audio"] = (samples - kernel) // stride + 1
        point3d = torch.zeros(b, 16, 3, device=device)
        layout.append("point3d")
        modality_kwargs["point3d"] = point3d
        seg_lengths["point3d"] = 17  # 1 global summary token + 16 points
        haptic_cfg = modality_cfg.haptic_encoder
        if haptic_cfg is not None:
            kernel = haptic_cfg.stem_conv.kernel_size
            stride = haptic_cfg.stem_conv.stride
            frames = min(19, haptic_cfg.max_frames)
            samples = (frames - 1) * stride + kernel
            haptic = torch.zeros(b, samples, haptic_cfg.stem_conv.in_channels, device=device)
            layout.append("haptic")
            modality_kwargs["haptic"] = haptic
            seg_lengths["haptic"] = (samples - kernel) // stride + 1

        layout.append("text")
        s_total = sum(seg_lengths.values()) + s_text
        full_labels = torch.full((b, s_total), -100, dtype=labels.dtype, device=device)
        full_labels[:, s_total - s_text :] = labels
        positions = torch.arange(s_total, device=device).unsqueeze(0).expand(b, -1)
        kwargs = dict(extra_inputs)
        kwargs.update(modality_kwargs)
        kwargs.update({"layout": layout, "positions": positions})
        return tokens, full_labels, kwargs
