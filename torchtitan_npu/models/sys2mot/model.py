# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sys2MoT VLA: DeepSeek layer-paired Mixture-of-Transformers (reference port).

Faithful port of the external reference
``external_models/vla_physical_ai/VLA/sys2_mot`` (version ``1.0.0-mot``, pure
PyTorch single file, DeepSeek-V4.1-Flash derived student). The model is a
DeepSeek-style student VLM (ViT with 2D RoPE, CSA2 sparse attention over local
raw KV + shared compressed global KV, sqrt(softplus) routed MoE with one shared
expert, mHC Sinkhorn hyper-connections, Engram multihash n-gram memory) paired
layer-by-layer with action experts: every VLM layer exposes its native
attention-update KV to the corresponding action expert, and action queries read
prefix + state/action KV in ONE joint softmax. The training path is the
reference's uncached joint forward (``VLA.forward``); the cached ten-step
sampling scaffold (``prefill``/``sample_actions``/``PrefixCache``), distiller,
checkpoint IO and CLI are decode/distillation tooling that never executes in
training capture and are not ported.

Approved deviations from the reference (none substitutes an operator kind on
the eager path; all documented per the fidelity rules):

- Routed-expert execution: the reference loops experts with data-dependent
  boolean-mask indexing (``torch.where((ids == i) & valid...)`` +
  ``index_add``); the masked row shapes depend on tensor values and cannot run
  under Meta capture. The port expresses the same token-choice top-k math
  through the shared token dispatcher + grouped GEMM in BOTH modes
  (owner-approved deviation class, same ruling as the multi-head-moe and
  system3 ports): per-expert gate/up/down grouped GEMMs with the reference's
  exact SwiGLU clamp placement (gate clamp ``max`` before silu, up clamp ``±``),
  weights normalized over the gathered raw scores and multiplied at combine
  (``LocalTokenDispatcher.combine``), the shared expert running densely on
  every token. The grouped experts stack the per-expert weights into
  ``(E, F, D)``/``(E, D, F)`` tensors — identical numel, fewer parameter
  tensors — and run ``torch._grouped_mm`` in bf16 like every other simulator
  family (under fp32 capture the bf16 grouped GEMM is the shared production
  deviation). Invalid tokens are dispatched alongside valid ones (the
  reference skips them); the final ``* valid`` mask zeroes their contribution.
- Routing-bias auxiliary update: the reference mutates the per-modality
  routing-bias buffers each training forward under ``no_grad`` using boolean
  row selection + ``torch.bincount`` (``aten::bincount`` has no Meta kernel —
  probe evidence in the capture test). Under the simulator's shared debug
  contract (``_debug_force_load_balance`` set on every MoE family) the update
  is skipped and the top-k ids become a balanced round-robin selection with
  the real gathered scores; real-device training keeps the reference update.
- Framework-fit reorganization (no op-stream change): the backbone merges into
  the ``Decoder`` layout (``Backbone.embedding`` -> ``tok_embeddings``, block
  ``ffn`` -> ``moe`` with ``routed_experts.inner_experts`` so the shared FSDP
  helper and the ``moe_force_load_balance`` propagation apply), and the
  ``nn.Sequential`` time MLP is expressed as two leaf ``Linear`` modules with
  the same SiLU between them. Every forward op is the reference's own.
- Multimodal capture inputs: images/state/noisy-actions/timestep enter through
  the runner's input-prep seam (``build_attention_masks``), which synthesizes
  deterministic tensors (3 views at the tiny 84px geometry, minimum text
  length = ``training.seq_len``) and derives the flow-matching timestep batch
  from a fixed arithmetic recipe shared with the loss (the reference seeds
  torch RNG; capture is structure-only and Meta has no generator).
- Loss wiring: the reference trains with
  ``flow + 0.5*KD + 0.1*hidden + 0.1*CE + 0.01*indexer``. The KD/hidden terms
  need a frozen teacher forward, which the single-model capture stack does not
  run; the captured joint loss (``Sys2MotJointLoss``) keeps the three
  student-only terms — flow MSE against ``noise - actions``, text CE executed
  through ``set_lm_head`` over the text span only (labels' final column is
  nulled to the reference's shifted-CE target set), and the indexer
  distillation aux carried out of the model forward as its own tensor.
- Weight tying: the reference already keeps an untied LM head; nothing to
  change. RoPE freqs are recomputed per call (reference verbatim); the
  ``rope`` field on the attention config only carries the framework's
  sequence-length contract (``Decoder.Config.update_from_config`` resize).

Shape suffixes: B batch, V views, S sequence, D width, H heads, Dh head dim,
R routed rows, E experts, K experts per token, F expert intermediate width.
"""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv2d
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.module import Module, ModuleList
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "ActionExpert",
    "ActionMoT",
    "AttentionRuntime",
    "Block",
    "Compressor",
    "DeepSeekViT",
    "Engram",
    "GroupedExperts",
    "HyperMix",
    "Indexer",
    "MoE",
    "RMSNorm",
    "RoutedExperts",
    "Router",
    "SparseAttention",
    "SwiGLU",
    "Sys2MotModel",
    "VisionBlock",
    "synthetic_action_batch",
]


def _replicated_sharding(*parameters: str) -> ShardingConfig:
    """Replicated state placement for model-owned plain parameters."""
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in parameters})


@dataclass(kw_only=True, slots=True)
class RopeGeometry:
    """Framework-contract rope metadata the Decoder config path reads.

    The reference recomputes RoPE freqs inline on every call and has no YaRN
    cache; this object only carries ``max_seq_len`` for the framework's
    sequence-length resize (``Decoder.Config.update_from_config``).
    """

    max_seq_len: int = 512


def rotary(x: torch.Tensor, pos: torch.Tensor, dim: int, theta: float = 10000.0, inverse: bool = False):
    """Rotate the last dim channels; x [B,L,H,D], pos [L] or [B,L]. (reference verbatim)"""
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = pos.float()[..., None] * freq
    angles = angles[None, :, None] if pos.ndim == 1 else angles[:, :, None]
    if inverse:
        angles = -angles
    a, b = x[..., -dim:].float().chunk(2, -1)
    tail = torch.cat((a * angles.cos() - b * angles.sin(), b * angles.cos() + a * angles.sin()), -1)
    return torch.cat((x[..., :-dim], tail.to(x.dtype)), -1)


class RMSNorm(Module):
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
        # Reference init: ``torch.ones(dim)``.
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(Module):
    """Clamped SwiGLU MLP, reference verbatim (``limit=0`` disables clamping)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate: Linear.Config
        up: Linear.Config
        down: Linear.Config
        limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()  # pyrefly: ignore [missing-attribute]
        self.up = config.up.build()  # pyrefly: ignore [missing-attribute]
        self.down = config.down.build()  # pyrefly: ignore [missing-attribute]
        self.limit = config.limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate(x), self.up(x)
        if self.limit:
            g, u = g.clamp(max=self.limit), u.clamp(-self.limit, self.limit)
        return self.down(F.silu(g) * u)


# ============================================================================
# Vision encoder (reference verbatim)
# ============================================================================


class VisionBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vision_width: int
        vision_heads: int
        vision_mlp: int
        n1: RMSNorm.Config
        n2: RMSNorm.Config
        qkv: Linear.Config
        out: Linear.Config
        ffn: SwiGLU.Config

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.vision_heads
        self.dim = config.vision_width // config.vision_heads
        self.n1 = config.n1.build()  # pyrefly: ignore [missing-attribute]
        self.n2 = config.n2.build()  # pyrefly: ignore [missing-attribute]
        self.qkv = config.qkv.build()  # pyrefly: ignore [missing-attribute]
        self.out = config.out.build()  # pyrefly: ignore [missing-attribute]
        self.ffn = config.ffn.build()  # pyrefly: ignore [missing-attribute]

    def forward(self, x: torch.Tensor, ypos: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, self.dim).permute(2, 0, 1, 3, 4)

        # Half the channels use vertical position; half horizontal position.
        def rotate_2d(t: torch.Tensor) -> torch.Tensor:
            a, z = t.chunk(2, -1)
            return torch.cat((rotary(a, ypos, self.dim // 2), rotary(z, xpos, self.dim // 2)), -1)

        q, k = rotate_2d(q), rotate_2d(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.out(y.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.n2(x))


class DeepSeekViT(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vision_width: int
        vision_layers: int
        vision_heads: int
        vision_mlp: int
        patch_size: int
        downsample: int
        image_size: int
        patch: Conv2d.Config
        blocks: list[VisionBlock.Config]
        norm: RMSNorm.Config
        align1: Linear.Config
        align2: Linear.Config
        # LLM width after alignment; the delimiters live in this space.
        width: int

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                # Only the delimiter parameters are direct states; the patch
                # conv and align projections are annotated by sharding.py.
                self.sharding_config = _replicated_sharding("start", "end", "newline")

    def __init__(self, config: Config):
        super().__init__()
        self.vision_width = config.vision_width
        self.width = config.width
        self.downsample = config.downsample
        self.image_size = config.image_size
        self.patch = config.patch.build()  # pyrefly: ignore [missing-attribute]
        self.blocks = ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()  # pyrefly: ignore [missing-attribute]
        self.align1 = config.align1.build()  # pyrefly: ignore [missing-attribute]
        self.align2 = config.align2.build()  # pyrefly: ignore [missing-attribute]
        self.start, self.end, self.newline = [nn.Parameter(torch.empty(config.width)) for _ in range(3)]

    def reset_parameters(self) -> None:
        # Reference init: ``torch.randn(width) * 0.02`` for the delimiters.
        for parameter in (self.start, self.end, self.newline):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, views, channels, height, width = images.shape
        x = self.patch(images.reshape(b * views, channels, height, width))
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        ypos = torch.arange(gh, device=x.device).repeat_interleave(gw)
        xpos = torch.arange(gw, device=x.device).repeat(gh)
        for block in self.blocks:
            x = block(x, ypos, xpos)
        x = self.norm(x).transpose(1, 2).reshape(b * views, -1, gh, gw)
        r = self.downsample
        x = F.pad(x, (0, -gw % r, 0, -gh % r))
        out_h, out_w = x.shape[-2] // r, x.shape[-1] // r
        # Same channel-first 3x3 pixel unshuffle ordering as official unfold.
        x = F.unfold(x, r, stride=r).transpose(1, 2)
        x = self.align2(F.gelu(self.align1(x))).view(b * views, out_h, out_w, -1)
        nl = self.newline.to(x.dtype).view(1, 1, 1, -1).expand(b * views, out_h, 1, -1)
        x = torch.cat((x, nl), 2).flatten(1, 2)
        start = self.start.to(x.dtype).view(1, 1, -1).expand(b * views, 1, -1)
        end = self.end.to(x.dtype).view(1, 1, -1).expand(b * views, 1, -1)
        return torch.cat((start, x, end), 1).reshape(b, -1, self.width)


# ============================================================================
# Routed MoE: sqrt(softplus) router + grouped-GEMM experts + shared expert
# ============================================================================


class Router(Module):
    """Token-choice router with the reference's exact scoring math.

    ``raw = sqrt(softplus(W x))`` in float32; the per-modality balance bias is
    selected by the image mask; top-k on ``raw + bias``; weights normalized
    over the gathered raw scores (``clamp_min(1e-20)``) and scaled by
    ``route_scale``. ``_debug_force_load_balance`` replaces the top-k ids with
    a balanced round-robin selection while keeping the real gathered scores
    (shared debug contract, set by the simulator for every MoE family).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        width: int
        num_experts: int
        topk: int
        route_scale: float
        _debug_force_load_balance: bool = False

        def __post_init__(self) -> None:
            if not 0 < self.topk <= self.num_experts:
                raise ValueError(f"topk ({self.topk}) must be in (0, num_experts] ({self.num_experts})")
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("weight", "bias_text", "bias_image")

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.width))
        self.register_buffer("bias_text", torch.zeros(config.num_experts))
        self.register_buffer("bias_image", torch.zeros(config.num_experts))

    def reset_parameters(self) -> None:
        # Reference init: ``nn.Linear(width, experts, bias=False)`` default.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, flat: torch.Tensor, image_mask_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = F.softplus(F.linear(flat.float(), self.weight.float())).sqrt()
        bias = torch.where(image_mask_flat[:, None], self.bias_image.float(), self.bias_text.float())
        ids = (raw + bias).topk(self.topk, -1).indices
        if self._debug_force_load_balance:
            # Balanced round-robin ids; the gathered scores stay real so the
            # normalized weights below keep the reference's data dependency.
            n_tokens = ids.shape[0]
            ids = (
                torch.arange(n_tokens * self.topk, device=raw.device, dtype=torch.int64).reshape(n_tokens, self.topk)
                % self.num_experts
            )
        weights = raw.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20) * self.route_scale
        return ids, weights

    def update_routing_bias(self, ids: torch.Tensor, image_mask_flat: torch.Tensor, valid_flat: torch.Tensor) -> None:
        """Reference per-forward balance update (real-device training only).

        Skipped in capture: the selection uses boolean row masking and
        ``torch.bincount``, neither of which runs under Meta (no fake/Meta
        kernel), and the debug balanced-routing contract supersedes the bias
        dynamics anyway.
        """
        with torch.no_grad():
            for is_image, buffer in ((True, self.bias_image), (False, self.bias_text)):
                take = valid_flat & (image_mask_flat == is_image)
                counts = torch.bincount(ids[take].flatten(), minlength=self.num_experts).float()
                if take.any():
                    buffer.add_(0.001 * (counts.mean() - counts).sign().to(buffer.dtype))


class GroupedExperts(Module):
    """Grouped-GEMM execution of the reference per-expert SwiGLU math.

    Per expert: ``out = W_down silu(clamp_max(W_gate x)) * clamp_±(W_up x)`` —
    the reference's exact clamp placement (gate clamped ``max`` before silu,
    up clamped both sides, product unclamped). Weights are stacked
    ``(E, F, D)``/``(E, D, F)`` (identical numel to the reference's per-expert
    ``nn.Linear`` modules, fewer tensors). The grouped GEMM runs in bf16 like
    every shared grouped-experts implementation (under fp32 capture the bf16
    grouped GEMM is the shared production deviation, documented in the header).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        width: int
        expert_dim: int
        num_experts: int
        swiglu_limit: float

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                self.sharding_config = _replicated_sharding("w_gate_EFD", "w_up_EFD", "w_down_EDF")

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit
        self.w_gate_EFD = nn.Parameter(torch.empty(config.num_experts, config.expert_dim, config.width))
        self.w_up_EFD = nn.Parameter(torch.empty(config.num_experts, config.expert_dim, config.width))
        self.w_down_EDF = nn.Parameter(torch.empty(config.num_experts, config.width, config.expert_dim))

    def reset_parameters(self) -> None:
        # Reference init: per-expert ``nn.Linear`` defaults. Applying
        # kaiming_uniform to the stacked tensor would use fan_in = F*D; loop
        # the expert slices so each expert matches its reference ``nn.Linear``
        # exactly (fan_in = width for gate/up, expert_dim for down).
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w_gate_EFD[expert_idx], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_up_EFD[expert_idx], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_down_EDF[expert_idx], a=math.sqrt(5))

    def forward(
        self,
        x_RD: torch.Tensor,
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
        gate_RF = self._grouped_mm(A=x_RD.bfloat16(), B_t=w_gate_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        up_RF = self._grouped_mm(A=x_RD.bfloat16(), B_t=w_up_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        # Reference SwiGLU clamp placement: gate clamped max before silu, up
        # clamped both sides, product unclamped.
        act_RF = F.silu(gate_RF.clamp(max=self.swiglu_limit)) * up_RF.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self._grouped_mm(
            A=act_RF.bfloat16(), B_t=w_down_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    def _grouped_mm(self, *, A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """Grouped matmul ``A @ B_t`` with per-expert token offsets."""
        return torch._grouped_mm(A, B_t, offs=offs)


class RoutedExperts(Module):
    """Token dispatcher + grouped experts.

    Each token rides the dispatcher once per selected expert (the reference's
    ``expert(flat[rows])`` gather); the shared combine multiplies by the
    expert-sorted normalized scores and scatter-adds into the output width —
    the reference's ``weights[rows, ranks, None]`` multiply.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: GroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config

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
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(routed_input_RD, num_global_tokens_per_local_expert_e)
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)


class MoE(Module):
    """Routed MoE FFN: sqrt(softplus) router + grouped experts + shared expert.

    The reference sums the shared expert and the routed contributions per
    token, then multiplies by the token-validity mask; the grouped-GEMM port
    keeps that exact order. The routing-bias buffer update runs on real
    device training only (deviation documented in the header).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        router: Router.Config
        routed_experts: RoutedExperts.Config
        shared: SwiGLU.Config

        def __post_init__(self) -> None:
            if self.router.num_experts != self.routed_experts.inner_experts.num_experts:
                raise ValueError("router expert count must match the grouped experts")

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.router.num_experts
        self.topk = config.router.topk
        self.router = config.router.build()  # pyrefly: ignore [missing-attribute]
        self.routed_experts = config.routed_experts.build()  # pyrefly: ignore [missing-attribute]
        self.shared = config.shared.build()  # pyrefly: ignore [missing-attribute]

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        ids, weights = self.router(flat, image_mask.flatten())
        routing_map_TSE = torch.zeros(
            flat.shape[0],
            self.num_experts,
            dtype=torch.bool,
            device=flat.device,
        ).scatter_(-1, ids, True)
        num_tokens_per_expert_E = routing_map_TSE.sum(dim=0)
        y = self.shared(flat) + self.routed_experts(
            flat,
            weights.to(flat.dtype),
            ids,
            num_tokens_per_expert_E,
        )
        if self.training and not self.router._debug_force_load_balance:
            # Auxiliary-loss-free load balancing: update only selection bias.
            self.router.update_routing_bias(ids, image_mask.flatten(), valid.flatten())
        return y.view(shape) * valid[:, :, None]


# ============================================================================
# Engram multihash n-gram memory (reference verbatim)
# ============================================================================


class Engram(Module):
    """Multihash 2/3/4-gram memory. Hashes exclude images and padding.

    The released tokenizer's compressed-ID map is deliberately not assumed.
    These deterministic hashes define a student memory requiring distillation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        buckets: int
        heads: int
        dim: int
        max_ngram: int
        width: int
        hc_mult: int
        eps: float
        table: Embedding.Config
        project: Linear.Config

        @property
        def columns(self) -> int:
            return self.heads * (self.max_ngram - 1)

        def __post_init__(self) -> None:
            if self.table.num_embeddings != self.columns * self.buckets:
                raise ValueError("Engram table must hold columns * buckets rows")
            if self.project.in_features != self.columns * self.dim:
                raise ValueError("Engram projection must consume the concatenated columns")
            if self.sharding_config is None:
                # The table/projection are child module states (sharding.py);
                # only the gate weights are direct parameters.
                self.sharding_config = _replicated_sharding("q_weight", "k_weight")

    def __init__(self, config: Config):
        super().__init__()
        self.buckets = config.buckets
        self.heads = config.heads
        self.dim = config.dim
        self.max_ngram = config.max_ngram
        self.width = config.width
        self.hc_mult = config.hc_mult
        self.eps = config.eps
        self.table = config.table.build()  # pyrefly: ignore [missing-attribute]
        self.project = config.project.build()  # pyrefly: ignore [missing-attribute]
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.width))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.width))

    def reset_parameters(self) -> None:
        # Reference init: ``torch.ones`` for the gate weights; the table and
        # projection keep their nn.Embedding/nn.Linear defaults.
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def forward(self, h: torch.Tensor, ids: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
        columns, masks = [], []
        for n in range(2, self.max_ngram + 1):
            code = torch.zeros_like(ids)
            eligible = text_valid.clone()
            for offset in range(n):
                shifted = F.pad(ids, (offset, 0))[:, : ids.shape[1]]
                mask = F.pad(text_valid, (offset, 0), value=False)[:, : ids.shape[1]]
                # Mod at every step avoids signed int64 overflow for long IDs.
                code = (code * 1000003 + shifted + 1).remainder(self.buckets)
                eligible = eligible & mask
            for head in range(self.heads):
                column = len(columns)
                hashed = (code * (2 * head + 1) + head * 104729).remainder(self.buckets)
                columns.append(hashed + column * self.buckets)
                masks.append(eligible)
        emb = self.table(torch.stack(columns, -1)) * torch.stack(masks, -1)[..., None]
        key, value = self.project(emb.flatten(-2)).split((self.hc_mult * self.width, self.width), -1)
        key = key.float().view(*ids.shape, self.hc_mult, self.width)
        hf = h.float()
        norm = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * key * self.q_weight.float() * self.k_weight.float()).sum(-1) * norm / math.sqrt(self.width)
        gate = torch.sigmoid(torch.sign(dot) * dot.abs().clamp_min(1e-6).sqrt())
        gate = gate * text_valid[..., None]
        return (hf + gate[..., None] * value.float().unsqueeze(-2)).to(h.dtype)


# ============================================================================
# mHC hyper-connections (reference verbatim)
# ============================================================================


class HyperMix(Module):
    """Single-pass mHC: produces pre for NEXT sublayer, post/comb for current."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int
        width: int
        eps: float
        sinkhorn_iters: int
        fn: Linear.Config

        def __post_init__(self) -> None:
            if self.fn.in_features != self.hc_mult * self.width:
                raise ValueError("mHC expand must consume the multi-stream width")
            if self.fn.out_features != (self.hc_mult + 2) * self.hc_mult:
                raise ValueError("mHC expand must emit (n+2)*n mixing values")
            if self.sharding_config is None:
                # fn is a child Linear (sharding.py); base/scale are direct.
                self.sharding_config = _replicated_sharding("base", "scale")

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.eps
        self.sinkhorn_iters = config.sinkhorn_iters
        self.fn = config.fn.build()  # pyrefly: ignore [missing-attribute]
        self.base = nn.Parameter(torch.zeros((config.hc_mult + 2) * config.hc_mult))
        self.scale = nn.Parameter(torch.full((3,), 0.01))

    def reset_parameters(self) -> None:
        # Reference init: zero biases, 0.01 scale gates.
        nn.init.zeros_(self.base)
        nn.init.constant_(self.scale, 0.01)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, eps = self.hc_mult, self.eps
        flat = h.float().flatten(-2)
        z = F.linear(flat, self.fn.weight.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        pre, post, matrix = z.split((n, n, n * n), -1)
        bpre, bpost, bmatrix = self.base.float().split((n, n, n * n), -1)
        pre = torch.sigmoid(pre * self.scale[0].float() + bpre) + eps
        post = 2 * torch.sigmoid(post * self.scale[1].float() + bpost)
        matrix = (matrix * self.scale[2].float() + bmatrix).view(*h.shape[:2], n, n).softmax(-1) + eps
        for _ in range(self.sinkhorn_iters):
            matrix = matrix / matrix.sum(-2, keepdim=True).clamp_min(eps)
            matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(eps)
        return pre, post, matrix

    @staticmethod
    def collapse(h: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (h.float() * pre[..., None]).sum(-2).to(h.dtype)

    @staticmethod
    def expand(y: torch.Tensor, h: torch.Tensor, post: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # comb[i,j] routes input stream i into output stream j.
        mixed = torch.einsum("blij,blid->bljd", matrix, h.float())
        return (mixed + post[..., None] * y.float().unsqueeze(-2)).to(h.dtype)


# ============================================================================
# CSA2 sparse attention (reference verbatim)
# ============================================================================


@dataclass
class AttentionRuntime:
    """Per-forward CSA2 state: never global and never shared across batches.

    A plain class on purpose: FSDP's pre-forward input rebuild reconstructs
    dataclass/namedtuple arguments (``dataclasses.replace``), which would
    sever the in-place ``ratio``/KV threading between the layer-paired
    prefix and action streams. Non-container objects pass through by
    reference, so every layer mutates the same runtime instance.
    """

    __slots__ = ("candidates", "index_key", "indices", "latent", "latent_valid", "ratio", "rotated")

    def __init__(self) -> None:
        self.latent: torch.Tensor | None = None
        self.latent_valid: torch.Tensor | None = None
        self.rotated: torch.Tensor | None = None
        self.index_key: torch.Tensor | None = None
        self.indices: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.ratio: int = 0

    def copy(self) -> "AttentionRuntime":
        """Shallow copy for block-level copy-on-write.

        Full AC recomputes blocks from their saved inputs; a block must never
        mutate the runtime object its predecessor returned, or the recomputed
        layer would observe the final forward state instead of its own
        mid-forward state.
        """
        out = AttentionRuntime()
        out.latent = self.latent
        out.latent_valid = self.latent_valid
        out.rotated = self.rotated
        out.index_key = self.index_key
        out.indices = self.indices
        out.candidates = self.candidates
        out.ratio = self.ratio
        return out


class Compressor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        width: int
        head_dim: int
        eps: float
        kv: Linear.Config
        gate: Linear.Config | None
        norm: RMSNorm.Config

        def __post_init__(self) -> None:
            if self.kv.out_features != self.head_dim:
                raise ValueError("compressor KV projection must emit the head dim")
            if (self.gate is not None) != (self.ratio > 1):
                raise ValueError("compression gating exists exactly for ratio > 1")

    def __init__(self, config: Config):
        super().__init__()
        self.ratio = config.ratio
        self.kv = config.kv.build()  # pyrefly: ignore [missing-attribute]
        self.gate = config.gate.build() if config.gate is not None else None
        self.norm = config.norm.build()  # pyrefly: ignore [missing-attribute]

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r, length = self.ratio, x.shape[1]
        count = length // r
        y = self.kv(x[:, : count * r]).view(x.shape[0], count, r, -1)
        vm = valid[:, : count * r].view(x.shape[0], count, r)
        if r > 1:
            gate = self.gate
            assert gate is not None, "ratio > 1 requires the gate projection"
            score = gate(x[:, : count * r]).view_as(y).float()
            prob = score.masked_fill(~vm[..., None], -1e9).softmax(2) * vm[..., None]
            prob = prob / prob.sum(2, keepdim=True).clamp_min(1e-9)
            y = (y.float() * prob).sum(2).to(x.dtype)
        else:
            y = y.squeeze(2)
        return self.norm(y), vm.any(-1)


class Indexer(Module):
    """CSA2 top-k index retrieval with distillation aux (reference verbatim)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        owns_k: bool
        index_heads: int
        index_dim: int
        index_topk: int
        rope_dim: int
        head_dim: int
        query_chunk: int
        candidate_source: int
        candidate_blocks: int
        candidate_block_size: int
        q_rank: int
        width: int
        eps: float
        q: Linear.Config
        weights: Linear.Config
        key: Linear.Config | None
        norm: RMSNorm.Config | None

        def __post_init__(self) -> None:
            if (self.key is not None) != self.owns_k or (self.norm is not None) != self.owns_k:
                raise ValueError("indexer key/norm exist exactly when the layer owns its index key")

    def __init__(self, config: Config):
        super().__init__()
        self.owns_k = config.owns_k
        self.index_heads = config.index_heads
        self.index_dim = config.index_dim
        self.index_topk = config.index_topk
        self.rope_dim = config.rope_dim
        self.head_dim = config.head_dim
        self.query_chunk = config.query_chunk
        self.candidate_source = config.candidate_source
        self.candidate_blocks = config.candidate_blocks
        self.candidate_block_size = config.candidate_block_size
        self.eps = config.eps
        self.q = config.q.build()  # pyrefly: ignore [missing-attribute]
        self.weights = config.weights.build()  # pyrefly: ignore [missing-attribute]
        self.key = config.key.build() if config.key is not None else None
        self.norm = config.norm.build() if config.norm is not None else None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        main_q: torch.Tensor,
        valid: torch.Tensor,
        runtime: AttentionRuntime,
        layer: int,
    ) -> torch.Tensor:
        b, length = x.shape[0], x.shape[1]
        latent = runtime.latent
        assert latent is not None, "indexer runs on compressed Full Mode KV"
        r, m = runtime.ratio, latent.shape[1]
        kpos = torch.arange(m, device=x.device) * r
        if self.owns_k:
            key, norm = self.key, self.norm
            assert key is not None and norm is not None, "index-key owner owns key/norm"
            ik = norm(key(latent))
            runtime.index_key = rotary(ik[:, :, None], kpos, self.rope_dim, 160000.0)[:, :, 0]
        iq = self.q(qr).view(b, length, self.index_heads, self.index_dim)
        iq = rotary(iq, torch.arange(length, device=x.device), self.rope_dim, 160000.0)
        weights = self.weights(x).float() / math.sqrt(self.index_dim * self.index_heads)
        index_key = runtime.index_key
        assert index_key is not None, "index keys are owned or carried from a previous layer"
        latent_valid = runtime.latent_valid
        assert latent_valid is not None, "compressed validity is set by the KV source layer"
        result, candidates, auxs = [], [], []
        for start in range(0, length, self.query_chunk):
            end = min(start + self.query_chunk, length)
            pos = torch.arange(start, end, device=x.device)
            scores = torch.einsum("bqhd,bmd->bqhm", iq[:, start:end].float(), index_key.float()).relu()
            scores = (scores * weights[:, start:end, :, None]).sum(2)
            allowed = ((torch.arange(m, device=x.device) + 1) * r <= pos[:, None] + 1)[None]
            allowed = allowed & latent_valid[:, None]
            if layer > self.candidate_source and runtime.candidates is not None:
                allowed = allowed & runtime.candidates[:, start:end]
            safe_scores = scores.masked_fill(~allowed, -1e9)
            if layer == self.candidate_source:
                blocks = F.pad(safe_scores, (0, -m % self.candidate_block_size), value=-1e9)
                block_score = blocks.unflatten(-1, (-1, self.candidate_block_size)).amax(-1)
                last = (pos // r) // self.candidate_block_size
                block_score = block_score.masked_fill(
                    torch.arange(block_score.shape[-1], device=x.device)[None, None] == last[None, :, None], 1e9
                )
                block_ids = block_score.topk(min(self.candidate_blocks, block_score.shape[-1]), -1).indices
                keep = torch.zeros_like(block_score, dtype=torch.bool).scatter_(-1, block_ids, True)
                candidates.append(keep.repeat_interleave(self.candidate_block_size, -1)[..., :m] & allowed)
            top = safe_scores.topk(min(self.index_topk, m), -1)
            good = allowed.gather(-1, top.indices)
            result.append(torch.where(good, top.indices, -1))
            if self.training:
                # Hard Top-K has no derivative. Distill dense main-attention
                # probabilities into index scores, so indexer weights DO train.
                rotated = runtime.rotated
                assert rotated is not None, "distillation target reads the rotated Full Mode KV"
                with torch.no_grad():
                    target_score = torch.einsum("bqhd,bmd->bqhm", main_q[:, start:end].float(), rotated.float())
                    target_score = target_score.mean(2) / math.sqrt(self.head_dim)
                    target = target_score.masked_fill(~allowed, -1e9).softmax(-1) * allowed
                    target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
                lp = safe_scores.log_softmax(-1)
                kl = (target * (target.clamp_min(1e-9).log() - lp)).sum(-1)
                active = allowed.any(-1) & valid[:, start:end]
                auxs.append((kl * active).sum() / active.sum().clamp_min(1))
        if candidates:
            runtime.candidates = torch.cat(candidates, 1)
        runtime.indices = torch.cat(result, 1)
        return torch.stack(auxs).mean() if auxs else x.new_zeros(())


class SparseAttention(Module):
    """Float reference CSA2: local raw KV + shared compressed global KV.

    Full/Reindex/Reuse ownership matches the source schedule. Scores/gathers
    are chunked, but index retrieval is still O(S^2), not a production kernel.
    ``capture_kv`` additionally returns the layer's rotated local KV — the
    prefix KV the paired action expert consumes (reference ``VLA.forward``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer: int
        ratio: int
        width: int
        q_rank: int
        heads: int
        head_dim: int
        rope_dim: int
        o_groups: int
        o_rank: int
        window: int
        eps: float
        query_chunk: int
        qa: Linear.Config
        qnorm: RMSNorm.Config
        qb: Linear.Config
        kv: Linear.Config
        kvnorm: RMSNorm.Config
        ob: Linear.Config
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        rope: RopeGeometry

        def __post_init__(self) -> None:
            if self.heads % self.o_groups:
                raise ValueError("grouped output projection needs heads to divide into groups")
            if self.sharding_config is None:
                # The q/kv/ob projections are child Linears (sharding.py); the
                # grouped output factor and the sink are direct parameters.
                self.sharding_config = _replicated_sharding("oa", "sink")

        @property
        def n_heads(self) -> int:
            # Framework contract: ``Decoder.Config.update_from_config`` reads
            # ``first_attention.n_heads`` for its TP validation.
            return self.heads

    def __init__(self, config: Config):
        super().__init__()
        self.layer = config.layer
        self.ratio = config.ratio
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.o_groups = config.o_groups
        self.window = config.window
        self.query_chunk = config.query_chunk
        group_dim = config.heads * config.head_dim // config.o_groups
        self.qa = config.qa.build()  # pyrefly: ignore [missing-attribute]
        self.qnorm = config.qnorm.build()  # pyrefly: ignore [missing-attribute]
        self.qb = config.qb.build()  # pyrefly: ignore [missing-attribute]
        self.kv = config.kv.build()  # pyrefly: ignore [missing-attribute]
        self.kvnorm = config.kvnorm.build()  # pyrefly: ignore [missing-attribute]
        self.oa = nn.Parameter(torch.empty(config.o_groups, config.o_rank, group_dim))
        self.ob = config.ob.build()  # pyrefly: ignore [missing-attribute]
        self.sink = nn.Parameter(torch.zeros(config.heads))
        self.compressor = config.compressor.build() if config.compressor is not None else None
        self.indexer = config.indexer.build() if config.indexer is not None else None

    def reset_parameters(self) -> None:
        # Reference init: ``randn * group_dim**-0.5`` for the grouped output
        # low-rank factor; the attention sink starts at zero.
        group_dim = self.oa.shape[-1]
        nn.init.normal_(self.oa, std=group_dim**-0.5)
        nn.init.zeros_(self.sink)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        runtime: AttentionRuntime,
        capture_kv: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, length = x.shape[0], x.shape[1]
        positions = torch.arange(length, device=x.device)
        theta = 160000.0 if self.ratio else 10000.0
        qr = self.qnorm(self.qa(x))
        q = rotary(self.qb(qr).view(b, length, self.heads, self.head_dim), positions, self.rope_dim, theta)
        local = rotary(self.kvnorm(self.kv(x))[:, :, None], positions, self.rope_dim, theta)[:, :, 0]
        aux = x.new_zeros(())
        if self.ratio:
            if self.compressor is not None:
                runtime.latent, runtime.latent_valid = self.compressor(x, valid)
                runtime.ratio = self.ratio
                runtime.rotated = rotary(
                    runtime.latent[:, :, None],
                    torch.arange(runtime.latent.shape[1], device=x.device) * self.ratio,
                    self.rope_dim,
                    theta,
                )[:, :, 0]
            if runtime.ratio != self.ratio:
                raise RuntimeError("Missing Full Mode KV source for this ratio")
            rotated = runtime.rotated
            assert rotated is not None, "Full Mode attends the rotated compressed KV"
            if self.indexer is not None:
                aux = self.indexer(x, qr, q, valid, runtime, self.layer)
            all_kv = torch.cat((local, rotated), 1)
        else:
            all_kv = local
        out = []
        for start in range(0, length, self.query_chunk):
            end = min(start + self.query_chunk, length)
            pos = positions[start:end]
            local_ids = pos[:, None] - torch.arange(self.window - 1, -1, -1, device=x.device)[None]
            local_ok = local_ids >= 0
            ids = local_ids.clamp_min(0)[None].expand(b, -1, -1)
            ok = local_ok[None] & valid[:, ids[0]]
            if self.ratio:
                indices = runtime.indices
                assert indices is not None, "Full Mode gathers the indexer-selected keys"
                global_ids = indices[:, start:end]
                ids = torch.cat((ids, global_ids.clamp_min(0) + length), -1)
                ok = torch.cat((ok, global_ids >= 0), -1)
            # Advanced indexing gathers only W+K values per query, not S*S*D.
            values = all_kv[torch.arange(b, device=x.device)[:, None, None], ids]
            score = torch.einsum("bqhd,bqkd->bqhk", q[:, start:end].float(), values.float()) / math.sqrt(self.head_dim)
            score = score.masked_fill(~ok[:, :, None], -torch.inf)
            sink = self.sink.float().view(1, 1, -1, 1).expand(b, end - start, -1, -1)
            probs = torch.cat((score, sink), -1).softmax(-1)[..., :-1]
            y = torch.einsum("bqhk,bqkd->bqhd", probs, values.float()).to(x.dtype)
            out.append(y)
        y = rotary(torch.cat(out, 1), positions, self.rope_dim, theta, inverse=True)
        grouped = y.reshape(b, length, self.o_groups, -1)
        y = torch.einsum("blgd,grd->blgr", grouped, self.oa.to(x.dtype))
        result = self.ob(y.flatten(2)) * valid[..., None]
        return (result, aux, local) if capture_kv else (result, aux)


class Block(Module):
    """One VLM layer: optional Engram, CSA2 attention, routed MoE, mHC pair."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer: int
        attention: SparseAttention.Config
        moe: MoE.Config
        attn_mix: HyperMix.Config
        ffn_mix: HyperMix.Config
        n1: RMSNorm.Config
        n2: RMSNorm.Config
        engram: Engram.Config | None

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()  # pyrefly: ignore [missing-attribute]
        self.moe = config.moe.build()  # pyrefly: ignore [missing-attribute]
        self.attn_mix = config.attn_mix.build()  # pyrefly: ignore [missing-attribute]
        self.ffn_mix = config.ffn_mix.build()  # pyrefly: ignore [missing-attribute]
        self.n1 = config.n1.build()  # pyrefly: ignore [missing-attribute]
        self.n2 = config.n2.build()  # pyrefly: ignore [missing-attribute]
        self.engram = config.engram.build() if config.engram is not None else None
        # Shared FSDP helper contract: every backbone block is an MoE block.
        self.moe_enabled = True

    def forward(
        self,
        h: torch.Tensor,
        pre: torch.Tensor,
        valid: torch.Tensor,
        image_mask: torch.Tensor,
        ids: torch.Tensor,
        runtime: AttentionRuntime,
        capture_kv: bool = False,
    ):
        # Copy-on-write: this block's runtime updates must not leak into the
        # object the previous block returned (full-AC recompute reads the
        # saved input objects; see AttentionRuntime.copy). The updated
        # runtime is returned so the caller threads the state functionally.
        runtime = runtime.copy()
        if self.engram is not None:
            h = self.engram(h, ids, valid & ~image_mask)
        attn_pre, post, comb = self.attn_mix(h)
        result = self.attention(self.n1(HyperMix.collapse(h, pre)), valid, runtime, capture_kv)
        y, aux = result[:2]
        h = HyperMix.expand(y, h, post, comb)
        ffn_pre, post, comb = self.ffn_mix(h)
        y = self.moe(self.n2(HyperMix.collapse(h, attn_pre)), image_mask, valid)
        output = (HyperMix.expand(y, h, post, comb), ffn_pre, aux)
        kv = result[2] if capture_kv else None
        if capture_kv:
            return (*output, kv, runtime)
        return (*output, runtime)


# ============================================================================
# Action stream: layer-paired MoT experts (reference verbatim)
# ============================================================================


def time_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    periods = 0.004 * 1000 ** torch.linspace(0, 1, width // 2, device=t.device)
    angle = t.float()[:, None] * (2 * math.pi) / periods
    return torch.cat((angle.sin(), angle.cos()), -1)


class ActionExpert(Module):
    """One modality-specific Transformer expert with ONE joint attention.

    Shared latent K=V and inverse output RoPE follow the DeepSeek prefix
    geometry; action Q/output projections and dense FFN have their own weights.
    A learned sink is included in the same softmax as prefix and suffix keys.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer: int
        theta: float
        action_width: int
        action_q_rank: int
        action_o_rank: int
        action_mlp: int
        heads: int
        head_dim: int
        rope_dim: int
        o_groups: int
        eps: float
        n1: RMSNorm.Config
        n2: RMSNorm.Config
        qa: Linear.Config
        qnorm: RMSNorm.Config
        qb: Linear.Config
        kv: Linear.Config
        kvnorm: RMSNorm.Config
        ob: Linear.Config
        ffn: SwiGLU.Config
        modulation: Linear.Config

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                # The projections are child Linears (sharding.py); the grouped
                # output factor and the sink are direct parameters.
                self.sharding_config = _replicated_sharding("oa", "sink")

    def __init__(self, config: Config):
        super().__init__()
        self.theta = config.theta
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.o_groups = config.o_groups
        self.n1 = config.n1.build()  # pyrefly: ignore [missing-attribute]
        self.n2 = config.n2.build()  # pyrefly: ignore [missing-attribute]
        self.qa = config.qa.build()  # pyrefly: ignore [missing-attribute]
        self.qnorm = config.qnorm.build()  # pyrefly: ignore [missing-attribute]
        self.qb = config.qb.build()  # pyrefly: ignore [missing-attribute]
        self.kv = config.kv.build()  # pyrefly: ignore [missing-attribute]
        self.kvnorm = config.kvnorm.build()  # pyrefly: ignore [missing-attribute]
        group_dim = config.heads * config.head_dim // config.o_groups
        self.oa = nn.Parameter(torch.empty(config.o_groups, config.action_o_rank, group_dim))
        self.ob = config.ob.build()  # pyrefly: ignore [missing-attribute]
        self.sink = nn.Parameter(torch.zeros(config.heads))
        self.ffn = config.ffn.build()  # pyrefly: ignore [missing-attribute]
        self.modulation = config.modulation.build()  # pyrefly: ignore [missing-attribute]

    def reset_parameters(self) -> None:
        # Reference init: ``randn * group_dim**-0.5`` for the grouped output
        # low-rank factor; the attention sink starts at zero.
        group_dim = self.oa.shape[-1]
        nn.init.normal_(self.oa, std=group_dim**-0.5)
        nn.init.zeros_(self.sink)

    def project(
        self, x: torch.Tensor, cond: torch.Tensor, prefix_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        b, n = x.shape[0], x.shape[1]
        shift, scale, gate, fshift, fscale, fgate = self.modulation(F.silu(cond)).chunk(6, -1)
        y = self.n1(x) * (1 + scale[:, None]) + shift[:, None]
        pos = prefix_length + torch.arange(n, device=x.device)
        qr = self.qnorm(self.qa(y))
        q = rotary(self.qb(qr).view(b, n, self.heads, self.head_dim), pos, self.rope_dim, self.theta)
        kv = rotary(self.kvnorm(self.kv(y))[:, :, None], pos, self.rope_dim, self.theta)[:, :, 0]
        return q, kv, (gate, fshift, fscale, fgate), pos

    def finish(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        b, n = x.shape[0], x.shape[1]
        gate, shift, scale, fgate = modulation
        y = rotary(y, pos, self.rope_dim, self.theta, inverse=True).reshape(b, n, self.o_groups, -1)
        y = torch.einsum("blgd,grd->blgr", y, self.oa.to(x.dtype))
        x = x + gate.tanh()[:, None] * self.ob(y.flatten(2))
        return x + fgate.tanh()[:, None] * self.ffn(self.n2(x) * (1 + scale[:, None]) + shift[:, None])

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, prefix_kv: torch.Tensor, prefix_valid: torch.Tensor
    ) -> torch.Tensor:
        q, akv, modulation, pos = self.project(x, cond, prefix_kv.shape[1])
        kv = torch.cat((prefix_kv, akv), 1)
        # Prefix -> prefix only; state -> prefix/state; actions -> all.
        # This is the action-query block of a modality-routed joint attention.
        n = x.shape[1]
        suffix = torch.ones(n, n, device=x.device, dtype=torch.bool)
        suffix[0, 1:] = False
        allowed = torch.cat((prefix_valid[:, None].expand(-1, n, -1), suffix[None].expand(x.shape[0], -1, -1)), -1)
        scores = torch.einsum("bqhd,bkd->bqhk", q.float(), kv.float()) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed[:, :, None], -torch.inf)
        sink = self.sink.float()[None, None, :, None].expand(x.shape[0], n, -1, -1)
        probs = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        y = torch.einsum("bqhk,bkd->bqhd", probs, kv.float()).to(x.dtype)
        return self.finish(x, y, modulation, pos)


class ActionMoT(Module):
    """The action stream: per-layer experts fed the VLM layers' native KV.

    Only the embedding/decoding pieces and the expert list are needed by the
    joint training path (the model pairs ``self.blocks`` with the backbone
    layers directly); the reference's cached sampling scaffold is not ported.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        action_dim: int
        horizon: int
        blocks: list[ActionExpert.Config]
        state_in: Linear.Config
        action_in: Linear.Config
        time_in: Linear.Config
        time_out: Linear.Config
        norm: RMSNorm.Config
        out: Linear.Config

        def __post_init__(self) -> None:
            if self.sharding_config is None:
                # The state/action/time/out projections are child Linears
                # (sharding.py); the position codes are a direct parameter.
                self.sharding_config = _replicated_sharding("position")

    def __init__(self, config: Config):
        super().__init__()
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.blocks = ModuleList([block.build() for block in config.blocks])
        self.state_in = config.state_in.build()  # pyrefly: ignore [missing-attribute]
        self.action_in = config.action_in.build()  # pyrefly: ignore [missing-attribute]
        self.time_in = config.time_in.build()  # pyrefly: ignore [missing-attribute]
        self.time_out = config.time_out.build()  # pyrefly: ignore [missing-attribute]
        self.position = nn.Parameter(torch.empty(1, config.horizon, config.action_in.out_features))
        self.norm = config.norm.build()  # pyrefly: ignore [missing-attribute]
        self.out = config.out.build()  # pyrefly: ignore [missing-attribute]

    def reset_parameters(self) -> None:
        # Reference init: ``randn(1, horizon, width) * 0.02`` position codes.
        nn.init.normal_(self.position, std=0.02)

    def embed(self, state: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape != (state.shape[0], self.horizon, self.action_dim) or t.shape != (state.shape[0],):
            raise ValueError("Expected actions [B,50,action_dim], timestep [B]")
        x = torch.cat((self.state_in(state)[:, None], self.action_in(actions) + self.position.to(actions.dtype)), 1)
        # Reference ``nn.Sequential(Linear, SiLU, Linear)`` as leaf modules.
        cond = self.time_out(F.silu(self.time_in(time_embedding(t, self.time_in.out_features).to(state.dtype))))
        return x, cond

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(x[:, -self.horizon :]))


# ============================================================================
# Full model
# ============================================================================


def synthetic_action_batch(
    batch: int, action_dim: int, horizon: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic state/action/noise/timestep batch for capture inputs.

    The reference seeds torch RNG and samples Beta timesteps; capture is
    structure-only and the Meta device has no generator, so a fixed arithmetic
    recipe replaces the RNG. The flow-matching loss derives the velocity target
    ``noise - actions`` from the SAME recipe, keeping the captured loss graph
    identical to the reference's flow objective.
    """
    lin = torch.arange(batch * horizon * action_dim, device=device, dtype=torch.float32)
    actions = ((lin * 0.618) % 1.0).reshape(batch, horizon, action_dim) - 0.5
    noise = ((lin * 0.382) % 1.0).reshape(batch, horizon, action_dim) - 0.5
    state = ((torch.arange(batch * action_dim, device=device, dtype=torch.float32) * 0.7) % 1.0 - 0.5).reshape(
        batch, action_dim
    )
    t = (torch.arange(batch, device=device, dtype=torch.float32) + 1) / (batch + 1)
    return state, actions, noise, t


class Sys2MotModel(Decoder):
    """Layer-paired MoT VLA; prefix cannot attend to the state/action stream.

    The training path is the reference's uncached joint forward: each backbone
    layer executes (capturing its native attention-update KV) before the paired
    action expert consumes that KV, so one microbatch carries both streams. The
    head is executed by the loss when ``_skip_lm_head`` is set; the forward
    returns the action velocity, the final prefix hidden states, the indexer
    distillation aux, and the encoder-boundary hidden (diagnostic, reference
    verbatim).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # The decoder-wide norm is this family's fp32 RMSNorm (upstream field
        # type is the common RMSNorm.Config).
        norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        vision: DeepSeekViT.Config
        action: ActionMoT.Config
        # Reference geometry the forward needs at model level.
        encoder_layers: int
        n_index_sources: int
        hc_mult: int
        image_tokens: int
        image_size: int
        max_views: int
        max_text: int

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "sys2mot does not implement tensor parallelism: the CSA2 head axis, the "
                    "grouped output low-rank factors and the mHC branch mixing would all need "
                    "sharded layouts; capture covers DP/FSDP only"
                )
            if getattr(parallelism, "expert_tensor_parallel_degree", 1) > 1:
                raise NotImplementedError(
                    "sys2mot does not implement expert tensor parallelism: the grouped-GEMM "
                    "experts shard across the EP mesh only"
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            from .sharding import set_sys2mot_sharding_config

            set_sys2mot_sharding_config(
                self,
                enable_ep=getattr(parallelism, "expert_parallel_degree", 1) > 1,
            )

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            # Debug-grade estimate: touched parameter elements per prefix token
            # (attention projections, indexer, routed + shared experts, mHC
            # mixing, norms), plus the vision/action streams and the
            # embedding/LM-head GEMM; FLOPs use the 6*active*seq convention.
            nparams = sum(parameter.numel() for parameter in model.parameters())
            active = 0
            for layer_cfg in self.layers:
                attn = layer_cfg.attention
                d, qr = attn.width, attn.q_rank
                active += d * qr + qr * attn.heads * attn.head_dim  # qa/qb
                active += d * attn.head_dim  # kv projection (shared latent K=V)
                active += attn.o_groups * attn.o_rank * attn.heads * attn.head_dim // attn.o_groups
                active += attn.ob.in_features * attn.ob.out_features
                active += attn.qnorm.dim + attn.kvnorm.dim + layer_cfg.n1.dim + layer_cfg.n2.dim
                if attn.indexer is not None:
                    active += qr * attn.indexer.index_heads * attn.indexer.index_dim
                    active += d * attn.indexer.index_heads
                    if attn.indexer.key is not None:
                        active += attn.head_dim * attn.indexer.index_dim + attn.indexer.index_dim
                for mix in (layer_cfg.attn_mix, layer_cfg.ffn_mix):
                    active += 2 * (mix.fn.in_features * mix.fn.out_features)
                    active += (mix.hc_mult + 2) * mix.hc_mult + 3
                moe = layer_cfg.moe
                active += moe.router.width * moe.router.num_experts
                inner = moe.routed_experts.inner_experts
                active += moe.router.topk * 3 * inner.width * inner.expert_dim
                active += 3 * moe.shared.gate.out_features * moe.shared.gate.in_features
                if layer_cfg.engram is not None:
                    active += layer_cfg.engram.project.in_features * layer_cfg.engram.project.out_features
                    active += 2 * layer_cfg.engram.hc_mult * layer_cfg.engram.width
            for expert_cfg in self.action.blocks:
                d, qr = expert_cfg.action_width, expert_cfg.action_q_rank
                active += d * qr + qr * expert_cfg.heads * expert_cfg.head_dim
                active += d * expert_cfg.head_dim + expert_cfg.kvnorm.dim
                active += (
                    expert_cfg.o_groups
                    * expert_cfg.action_o_rank
                    * expert_cfg.heads
                    * expert_cfg.head_dim
                    // expert_cfg.o_groups
                )
                active += expert_cfg.ob.in_features * expert_cfg.ob.out_features
                active += expert_cfg.n1.dim + expert_cfg.n2.dim
                active += 6 * d * d  # AdaLN modulation
                active += 3 * d * expert_cfg.action_mlp
                active += 2 * d * d  # time MLP share
            vit = self.vision
            active += vit.patch.in_channels * vit.patch.out_channels * vit.patch.kernel_size**2
            for block in vit.blocks:
                active += 3 * block.vision_width * block.vision_width + 3 * block.vision_width * block.vision_mlp
            active += (
                vit.align1.in_features * vit.align1.out_features + vit.align2.in_features * vit.align2.out_features
            )
            active += self.dim * self.vocab_size  # embedding lookup + head GEMM
            return nparams, 6 * active * seq_len

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision = config.vision.build()
        self.action = config.action.build()
        self.encoder_layers = config.encoder_layers
        self.n_index_sources = config.n_index_sources
        self.hc_mult = config.hc_mult
        self.image_tokens = config.image_tokens
        self.image_size = config.image_size
        self.max_views = config.max_views
        self.max_text = config.max_text
        self.action_dim = config.action.action_dim
        self.horizon = config.action.horizon

    # ------------------------------------------------------------------
    # Prefix embedding and final head (reference Backbone verbatim)
    # ------------------------------------------------------------------
    def embed_prefix(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ):
        b, views = images.shape[0], images.shape[1]
        if images.ndim != 5 or tuple(images.shape[2:]) != (3, self.image_size, self.image_size):
            raise ValueError("images must be [B,V,3,image_size,image_size], normalized floats")
        if (
            not 3 <= views <= self.max_views
            or text.ndim != 2
            or text.shape[0] != b
            or not 100 <= text.shape[1] <= self.max_text
        ):
            raise ValueError("Need 3..16 views, text [B,100..200]")
        text_mask = torch.ones_like(text, dtype=torch.bool) if text_mask is None else text_mask
        view_mask = torch.ones(b, views, device=images.device, dtype=torch.bool) if view_mask is None else view_mask
        if (
            text_mask.shape != text.shape
            or view_mask.shape != (b, views)
            or text_mask.dtype != torch.bool
            or view_mask.dtype != torch.bool
        ):
            raise ValueError("Invalid boolean masks")
        visual = self.vision(images)
        visual_valid = view_mask.repeat_interleave(self.image_tokens, 1)
        valid = torch.cat((visual_valid, text_mask), 1)
        image_mask = torch.cat((torch.ones_like(visual_valid), torch.zeros_like(text_mask)), 1)
        ids = torch.cat((torch.zeros_like(visual_valid, dtype=torch.long), text), 1)
        x = torch.cat((visual, self.tok_embeddings(text)), 1)
        h = x.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        # Identity initial collapse selects the first residual stream.
        pre = h.new_zeros(*h.shape[:2], self.hc_mult)
        pre[..., 0] = 1
        return h, pre, valid, image_mask, ids, visual.shape[1]

    def finish(self, h: torch.Tensor, pre: torch.Tensor, auxs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.norm(HyperMix.collapse(h, pre))
        indexer_loss = torch.stack(auxs).sum() / self.n_index_sources
        return {"hidden": hidden, "indexer_aux": indexer_loss}

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
        *,
        attention_metadata=None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        noisy_actions: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del positions, attention_masks, attention_metadata  # synthetic rows are full length
        if images is None or state is None or noisy_actions is None or t is None:
            raise ValueError("sys2mot forward requires images, state, noisy_actions and t (multimodal capture)")
        if state.shape != (images.shape[0], self.action_dim):
            raise ValueError("state must be [B,action_dim]")
        h, pre, valid, image_mask, ids, visual_tokens = self.embed_prefix(images, tokens, text_mask, view_mask)
        a, cond = self.action.embed(state, noisy_actions, t)
        runtime, auxs, encoder_hidden = AttentionRuntime(), [], None
        for i, (layer, expert) in enumerate(zip(self.layers.values(), self.action.blocks, strict=True)):
            # The first decoder Full Mode projects final encoder representations.
            # No decoder layer constructs global KV from its own later hidden state.
            if i == self.encoder_layers:
                encoder_hidden = HyperMix.collapse(h, pre)
            h, pre, aux, kv, runtime = layer(h, pre, valid, image_mask, ids, runtime, capture_kv=True)
            a = expert(a, cond, kv, valid)
            auxs.append(aux)
        features = self.finish(h, pre, auxs)
        velocity = self.action.decode(a)
        # The encoder boundary always exists (0 < encoder_layers < layers).
        assert encoder_hidden is not None, "encoder boundary hidden must be captured"
        if getattr(self, "_distill_capture", False):
            # The loss-side teacher forward consumes the raw multimodal batch;
            # carry it through the prediction dict (capture-local contract).
            pred = {
                "velocity": velocity,
                "hidden": features["hidden"],
                "indexer_aux": features["indexer_aux"],
                "encoder_hidden": encoder_hidden,
                "distill_images": images,
                "distill_tokens": tokens,
                "distill_text_mask": text_mask,
            }
            if not self._skip_lm_head:
                pred["logits"] = self.lm_head(features["hidden"][:, visual_tokens:])
            return pred
        if self._skip_lm_head:
            return {
                "velocity": velocity,
                "hidden": features["hidden"],
                "indexer_aux": features["indexer_aux"],
                "encoder_hidden": encoder_hidden,
                "distill_text_mask": text_mask,
            }
        return {
            "velocity": velocity,
            "logits": self.lm_head(features["hidden"][:, visual_tokens:]),
            "indexer_aux": features["indexer_aux"],
            "encoder_hidden": encoder_hidden,
            "distill_text_mask": text_mask,
        }

    # ------------------------------------------------------------------
    # Runner input-prep seam (multimodal capture inputs)
    # ------------------------------------------------------------------
    def backbone_forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reference ``VLA.backbone``: ViT + VLM layers + LM head, no action loop.

        The distillation teacher runs this per step in eval mode; the indexer
        aux branch is skipped by ``self.training == False`` and the returned
        dict carries the teacher logits plus the text-span hidden states the
        ``Distiller`` formulas consume.
        """
        h, pre, valid, image_mask, ids, visual_tokens = self.embed_prefix(images, tokens, text_mask, view_mask)
        runtime, auxs = AttentionRuntime(), []
        for layer in self.layers.values():
            h, pre, aux, runtime = layer(h, pre, valid, image_mask, ids, runtime, capture_kv=False)
            auxs.append(aux)
        features = self.finish(h, pre, auxs)
        span = features["hidden"][:, visual_tokens:]
        return {"logits": self.lm_head(span), "text_hidden": span}

    def build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
        """Runner input-prep seam.

        Synthesizes the deterministic VLA inputs in front of the text span
        (3 views at the debug image geometry, minimum text length) plus the
        state/noisy-action/timestep batch from the shared deterministic
        recipe, and returns full-sequence positions. The labels' final column
        is nulled so the CE target set matches the reference's shifted
        ``logits[:, :-1] vs text[1:]`` objective. CP is rejected: the CSA2
        compressed grid, Engram hash and the per-layer prefix/action joint
        attention all span the full sequence.
        """
        if cp_mesh is not None:
            if load_balancer_type is not None:
                raise NotImplementedError("sys2mot capture does not use a CP load balancer")
            raise NotImplementedError("sys2mot does not support context parallel")
        b, s_text = tokens.shape
        device = tokens.device
        views = 3  # reference minimum; the debug flavor always uses it
        images = torch.zeros(b, views, 3, self.image_size, self.image_size, device=device)
        state, actions, noise, t = synthetic_action_batch(b, self.action_dim, self.horizon, device)
        noisy_actions = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        text_mask = torch.ones_like(tokens, dtype=torch.bool)
        view_mask = torch.ones(b, views, device=device, dtype=torch.bool)
        labels = labels.clone()
        labels[:, -1] = -100
        positions = torch.arange(views * self.image_tokens + s_text, device=device).unsqueeze(0).expand(b, -1)
        kwargs = dict(extra_inputs)
        kwargs.update(
            {
                "images": images,
                "state": state,
                "noisy_actions": noisy_actions,
                "t": t,
                "text_mask": text_mask,
                "view_mask": view_mask,
                "positions": positions,
            }
        )
        return tokens, labels, kwargs
