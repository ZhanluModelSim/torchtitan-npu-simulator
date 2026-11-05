# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates.
"""Logical, single-rank block-diffusion training and inference reference.

Only PyTorch and the standard library are required. BlockDiffusionModel executes one phase:
``prefill`` and ``commit`` append per-layer KV; ``denoise`` reads the prefix
without changing it. Attention policy and output-head positions are explicit.
No additional commit pass, sampling algorithm, or distributed layout is assumed.
``forward_training`` shares the same backbone, keeps gradients enabled, and
uses an exact full-sequence block mask with same-position masked-token loss.
It does not reuse detached inference KV or approximate attention via FLOP scaling.

The numerical path is a correctness reference, not an NPU latency prediction.
In particular PyTorch may lower SDPA and concatenate caches differently from an
NPU's fused/paged implementation. The workload adapter must map these logical
operations to the chosen backend. Distributed execution belongs to that adapter;
this module never simulates TP/EP by silently shrinking an incomplete network.

MoE executes one SwiGLU per occupied expert (no weight tensor per routed pair).
A RoutingPlan replays expert assignments with static Python sizes, permitting
meta execution without reading tensor values. It overrides the chosen experts,
not the router's score computation; it is a synthetic workload, not a quality
measurement. The default numerical path respects the actual top-k selection.

ALoDLMModel is a separate dense, uncached recurrent paradigm. Its core weights
are shared across depths and its training uses explicit two-stream batches.
The standard block-diffusion sub-contract remains at version 1; the combined
reference bundle is version 2. Native adaptation of ALoDLM is not implied.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

MODEL_CONTRACT_VERSION = 2
STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION = 1
ALODLM_CONTRACT_VERSION = 1

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


@dataclass(frozen=True)
class ModelConfig:
    """Logical model dimensions; dtype controls weights and activations.

    Router weights/scores use fp32; RMSNorm parameters use ``dtype`` and its
    reduction uses fp32. KV may use a different floating storage dtype, with
    explicit conversion at the attention read/write. Quantization is not
    implemented and integer dtypes are rejected. Shared experts are optional.
    """

    vocab_size: int = 256
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 64
    num_shared_experts: int = 0
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.float32
    kv_dtype: torch.dtype | None = None
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "num_experts_per_tok",
            "moe_intermediate_size",
        ):
            _positive_int(name, getattr(self, name))
        _positive_int("num_experts", self.num_experts, allow_zero=True)
        _positive_int("num_shared_experts", self.num_shared_experts, allow_zero=True)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.num_experts and self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if not self.num_experts and self.num_shared_experts:
            raise ValueError("num_shared_experts applies only to a routed MoE")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be a bool")
        if self.dtype not in _FLOAT_DTYPES:
            raise ValueError("dtype must be float32, float16 or bfloat16; quantization is unsupported")
        if self.kv_dtype is None:
            object.__setattr__(self, "kv_dtype", self.dtype)
        if self.kv_dtype not in _FLOAT_DTYPES:
            raise ValueError("kv_dtype must be float32, float16 or bfloat16")
        for name in ("rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Load this schema strictly: misspelled/unsupported fields are errors."""
        data = json.loads(Path(path).read_text())
        for name in ("dtype", "kv_dtype"):
            if name in data and isinstance(data[name], str):
                value = data[name].removeprefix("torch.")
                if value not in ("float32", "float16", "bfloat16"):
                    raise ValueError(f"unsupported {name}: {data[name]}")
                data[name] = getattr(torch, value)
        return cls(**data)


@dataclass(frozen=True)
class LayerKV:
    """A single layer's stored K/V, each [batch, kv_heads, prefix, head_dim]."""

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError("LayerKV key/value must have the same rank-4 shape")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("LayerKV key/value device and dtype must agree")


@dataclass(frozen=True)
class KVCache:
    """Immutable cache container; a writing phase returns a new container.

    Tensor contents are not made read-only by Python. Callers must not mutate
    cached tensors. Distinct layers own distinct state, even if shapes match.
    """

    layers: tuple[LayerKV, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if self.layers:
            first = self.layers[0].key
            for layer in self.layers:
                if layer.key.shape != first.shape or layer.key.device != first.device or layer.key.dtype != first.dtype:
                    raise ValueError("all cache layers must have matching shape, device and dtype")

    @property
    def length(self) -> int:
        return self.layers[0].key.shape[2] if self.layers else 0

    def memory_bytes(self) -> int:
        """Logical tensor storage bytes, deduplicating identical tensor objects."""
        tensors = {id(t): t for layer in self.layers for t in (layer.key, layer.value)}
        return sum(t.numel() * t.element_size() for t in tensors.values())


@dataclass(frozen=True)
class RoutingPlan:
    """Top-k expert IDs per flattened (batch, token) row, in token-major order.

    Duplicate IDs within one token are invalid. Empty experts are valid. A plan
    is reused at every layer when passed directly; pass a tuple of plans to
    specify a different assignment at each layer. Routing weights still come
    from normalized router scores for the selected experts.
    """

    expert_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_indices", tuple(tuple(row) for row in self.expert_indices))
        for row in self.expert_indices:
            if not row or any(type(expert) is not int or expert < 0 for expert in row):
                raise ValueError("routing IDs must be nonnegative integers in nonempty rows")
            if len(set(row)) != len(row):
                raise ValueError("an expert may be selected only once per token")

    @classmethod
    def balanced(cls, num_tokens: int, num_experts: int, top_k: int) -> RoutingPlan:
        _positive_int("num_tokens", num_tokens, allow_zero=True)
        _positive_int("num_experts", num_experts)
        _positive_int("top_k", top_k)
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        return cls(
            tuple(tuple((token * top_k + slot) % num_experts for slot in range(top_k)) for token in range(num_tokens))
        )

    def validate(self, num_tokens: int, num_experts: int, top_k: int) -> None:
        if len(self.expert_indices) != num_tokens:
            raise ValueError("routing plan must provide one row per batch/token")
        if any(len(row) != top_k or any(expert >= num_experts for expert in row) for row in self.expert_indices):
            raise ValueError("routing plan top-k width or expert ID is invalid")

    def counts(self, num_experts: int) -> tuple[int, ...]:
        _positive_int("num_experts", num_experts)
        counts = [0] * num_experts
        for row in self.expert_indices:
            for expert in row:
                if expert >= num_experts:
                    raise ValueError("routing expert ID exceeds num_experts")
                counts[expert] += 1
        return tuple(counts)


@dataclass(frozen=True)
class PhaseOutput:
    logits: torch.Tensor | None
    cache: KVCache | None


@dataclass(frozen=True)
class TrainingSpec:
    """Exact attention visibility and loss convention for one training window.

    ``block_causal`` allows each query to see every preceding block and all of
    its own block. ``prefix_causal_canvas_bidirectional`` makes the clean prefix
    token-causal and the final block bidirectional with access to that prefix.
    The caller supplies corruption and same-position labels; unselected labels
    must contain ``ignore_index``. No timestep or loss-weighting policy is hidden.
    """

    block_size: int
    attention_policy: str = "block_causal"
    ignore_index: int = -100

    def __post_init__(self) -> None:
        _positive_int("block_size", self.block_size)
        if self.attention_policy not in ("block_causal", "prefix_causal_canvas_bidirectional"):
            raise ValueError("unsupported training attention_policy")
        if type(self.ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")


def training_attention_mask(
    sequence_length: int,
    spec: TrainingSpec,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return [sequence, sequence] boolean visibility (True means visible).

    Both policies require complete blocks. Construction depends only on static
    dimensions and works on meta without reading tensor values.
    """
    _positive_int("sequence_length", sequence_length)
    if sequence_length < spec.block_size or sequence_length % spec.block_size:
        raise ValueError("training sequence length must be a positive multiple of block_size")
    positions = torch.arange(sequence_length, device=device)
    query, key = positions[:, None], positions[None, :]
    if spec.attention_policy == "block_causal":
        return query // spec.block_size >= key // spec.block_size
    prefix_length = sequence_length - spec.block_size
    return (key <= query) | (query >= prefix_length)


@dataclass(frozen=True)
class TrainingOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    loss_sum: torch.Tensor
    valid_tokens: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    left, right = xf.chunk(2, dim=-1)
    return (xf * cos + torch.cat((-right, left), dim=-1) * sin).to(x.dtype)


class Attention(nn.Module):
    """GQA as a semantic SDPA operation, with prefix-aware causal masking.

    Backend lowering determines whether the attention is fused and whether a
    physical concatenation is needed. The CPU reference concatenates tensors;
    it must not be mistaken for a paged-KV NPU memory-traffic specification.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False, dtype=config.dtype)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False, dtype=config.dtype)
        # Parameter-free defaults preserve the standard block-diffusion v1 state.
        # ALoDLM installs per-head RMSNorm here without changing the base profile.
        self.q_norm: nn.Module = nn.Identity()
        self.k_norm: nn.Module = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKV | None = None,
        *,
        attention_policy: str = "bidirectional",
        position_offset: int = 0,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        cfg = self.config
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is None:
            rope = _rope_tables(cfg, length, position_offset, x.device)
        q, k = _rope(q, *rope), _rope(k, *rope)
        current = LayerKV(k, v)
        prefix = 0 if cache is None else cache.key.shape[2]
        if cache is not None:
            k = torch.cat((cache.key.to(x.dtype), k), dim=2)
            v = torch.cat((cache.value.to(x.dtype), v), dim=2)
        if attention_policy not in ("bidirectional", "causal"):
            raise ValueError("attention_policy must be bidirectional or causal")
        causal = attention_policy == "causal"
        mask = attention_mask
        if mask is not None:
            if causal:
                raise ValueError("an explicit attention mask cannot be combined with causal policy")
            if mask.dtype != torch.bool or mask.device != x.device or tuple(mask.shape) != (length, prefix + length):
                raise ValueError("attention_mask must be bool [query, prefix+query] on the input device")
        # SDPA's standard causal alignment is only correct without a prefix.
        if causal and prefix:
            q_positions = prefix + torch.arange(length, device=x.device)
            key_positions = torch.arange(prefix + length, device=x.device)
            mask = key_positions[None, :] <= q_positions[:, None]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal and not prefix,
            enable_gqa=cfg.num_attention_heads != cfg.num_key_value_heads,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended), current


def _rope_tables(config: ModelConfig, length: int, offset: int, device: torch.device):
    """Build only the needed positions once per phase, shared across layers."""
    frequencies = 1.0 / (
        config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim)
    )
    positions = torch.arange(offset, offset + length, dtype=torch.float32, device=device)
    angles = positions[:, None] * frequencies[None, :]
    doubled = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    return doubled.cos(), doubled.sin()


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=torch.float32)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.moe_intermediate_size, config.dtype) for _ in range(config.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.intermediate_size, config.dtype)
                for _ in range(config.num_shared_experts)
            ]
        )

    def forward(self, x: torch.Tensor, routing_plan: RoutingPlan | None = None) -> torch.Tensor:
        cfg = self.config
        tokens = x.reshape(-1, cfg.hidden_size)
        num_tokens = tokens.shape[0]
        if routing_plan is None and x.device.type == "meta":
            raise ValueError("meta MoE execution requires a RoutingPlan")
        if routing_plan is not None:
            routing_plan.validate(num_tokens, cfg.num_experts, cfg.num_experts_per_tok)
        # The routing precision is part of the contract, including under AMP.
        context = nullcontext() if tokens.device.type == "meta" else torch.autocast(tokens.device.type, enabled=False)
        with context:
            scores = self.router(tokens.float())
        _, selected = scores.topk(cfg.num_experts_per_tok, dim=-1)
        if routing_plan is not None:
            selected = torch.tensor(routing_plan.expert_indices, dtype=torch.long, device=x.device)
        # Softmax over selected logits equals renormalizing selected full-softmax
        # probabilities, but remains finite when replay selects unlikely experts.
        weights = F.softmax(scores.gather(1, selected), dim=-1)
        output = torch.zeros_like(tokens)
        replay_groups: list[list[tuple[int, int]]] | None = None
        if routing_plan is not None:
            replay_groups = [[] for _ in range(cfg.num_experts)]
            for token, row in enumerate(routing_plan.expert_indices):
                for slot, expert_id in enumerate(row):
                    replay_groups[expert_id].append((token, slot))
        for expert_id, expert in enumerate(self.experts):
            if replay_groups is not None:
                pairs = replay_groups[expert_id]
                if not pairs:
                    continue
                rows = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=x.device)
                slots = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=x.device)
            else:
                rows, slots = torch.where(selected == expert_id)
                if rows.numel() == 0:
                    continue
            expert_input = tokens.index_select(0, rows)
            expert_output = expert(expert_input)
            weighted = expert_output * weights[rows, slots].to(x.dtype).unsqueeze(-1)
            output = output.index_add(0, rows, weighted)
        for expert in self.shared_experts:
            output = output + expert(tokens)
        return output.reshape(x.shape)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        self.mlp = (
            MoE(config) if config.num_experts else SwiGLU(config.hidden_size, config.intermediate_size, config.dtype)
        )

    def forward(self, x, cache, *, attention_policy, position_offset, rope, routing_plan, attention_mask=None):
        attention, current = self.self_attn(
            self.input_layernorm(x),
            cache,
            attention_policy=attention_policy,
            position_offset=position_offset,
            rope=rope,
            attention_mask=attention_mask,
        )
        x = x + attention
        normalized = self.post_attention_layernorm(x)
        feedforward = self.mlp(normalized, routing_plan) if isinstance(self.mlp, MoE) else self.mlp(normalized)
        return x + feedforward, current


class BlockDiffusionModel(nn.Module):
    """Shared backbone with explicit inference and gradient-enabled training.

    ``logits_positions=None`` evaluates the output head at every input position;
    ``()`` skips it entirely; a tuple of nonnegative positions selects the same
    positions for every request. An int64 tensor of shape ``[batch, head_tokens]``
    selects different positions per request. Selection happens BEFORE the output
    matrix multiplication; an empty selection skips both final norm and head.
    Tensor indices stay on the input device and are bounds-checked by gather
    during numerical execution (meta execution only checks shape/dtype/device).
    ``phase='denoise'`` never updates the
    cache. ``prefill``/``commit`` append newly encoded state and return it, without
    mutating the supplied cache. No hidden autoregressive/diffusion loop runs.
    ``forward_training`` computes a full window without an inference cache and
    returns same-position cross-entropy plus its explicit normalization count.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def parameter_memory_bytes(self) -> int:
        """Resident parameter tensor bytes, including fp32 router and norms.

        Module.parameters deduplicates shared Parameter objects, including tied
        embedding/output weights. This is not a peak workspace/traffic estimate.
        """
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def _validate_cache(self, input_ids: torch.Tensor, cache: KVCache | None) -> None:
        if cache is None:
            return
        cfg = self.config
        if len(cache.layers) != cfg.num_hidden_layers:
            raise ValueError("cache must contain one LayerKV per transformer layer")
        shape = (input_ids.shape[0], cfg.num_key_value_heads, cache.length, cfg.head_dim)
        for layer in cache.layers:
            if tuple(layer.key.shape) != shape:
                raise ValueError("cache batch/head/length dimensions do not match the model input")
            if layer.key.device != input_ids.device or layer.key.dtype != cfg.kv_dtype:
                raise ValueError("cache device/dtype does not match the input and configured KV dtype")

    def forward_training(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        spec: TrainingSpec,
        routing_plan: RoutingPlan | tuple[RoutingPlan, ...] | None = None,
    ) -> TrainingOutput:
        """Compute a differentiable full-window masked-token training loss.

        Labels are int64 [batch, sequence], aligned with input positions rather
        than shifted as in AR training. Only labels unequal to ``ignore_index``
        contribute. The loss is their summed fp32 cross-entropy divided by the
        valid count; an empty target set has differentiable zero loss. This is a
        single-rank normalization; distributed global loss reduction belongs to
        the training driver. Logits cover the full window, including ignored
        positions, so the executed output-head workload is unambiguous.
        """
        if not isinstance(spec, TrainingSpec):
            raise ValueError("spec must be a TrainingSpec")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a rank-2 integer tensor")
        batch, length = input_ids.shape
        if batch < 1 or length < 1:
            raise ValueError("input_ids batch and sequence dimensions must be positive")
        if labels.dtype != torch.int64 or labels.shape != input_ids.shape or labels.device != input_ids.device:
            raise ValueError("labels must be int64 with the input shape and device")
        if routing_plan is not None and not self.config.num_experts:
            raise ValueError("routing_plan applies only to MoE models")
        if isinstance(routing_plan, RoutingPlan) or routing_plan is None:
            plans = (routing_plan,) * self.config.num_hidden_layers
        else:
            plans = tuple(routing_plan)
            if len(plans) != self.config.num_hidden_layers or any(not isinstance(plan, RoutingPlan) for plan in plans):
                raise ValueError("routing_plan must contain one RoutingPlan per transformer layer")
        for plan in plans:
            if plan is not None:
                plan.validate(batch * length, self.config.num_experts, self.config.num_experts_per_tok)
        mask = training_attention_mask(length, spec, device=input_ids.device)
        rope = _rope_tables(self.config, length, 0, input_ids.device)
        hidden = self.embed_tokens(input_ids)
        for layer, plan in zip(self.layers, plans, strict=False):
            hidden, _ = layer(
                hidden,
                None,
                attention_policy="bidirectional",
                position_offset=0,
                rope=rope,
                routing_plan=plan,
                attention_mask=mask,
            )
        logits = self.lm_head(self.norm(hidden))
        loss_sum = F.cross_entropy(
            logits.float().reshape(-1, self.config.vocab_size),
            labels.reshape(-1),
            ignore_index=spec.ignore_index,
            reduction="sum",
        )
        valid_tokens = (labels != spec.ignore_index).sum()
        loss = loss_sum / valid_tokens.clamp_min(1)
        return TrainingOutput(logits, loss, loss_sum, valid_tokens)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        *,
        phase: str = "denoise",
        attention_policy: str = "bidirectional",
        logits_positions: Sequence[int] | torch.Tensor | None = None,
        routing_plan: RoutingPlan | tuple[RoutingPlan, ...] | None = None,
    ) -> PhaseOutput:
        if phase not in ("prefill", "denoise", "commit"):
            raise ValueError("phase must be prefill, denoise or commit")
        if attention_policy not in ("causal", "bidirectional"):
            raise ValueError("attention_policy must be causal or bidirectional")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a rank-2 integer tensor")
        batch, length = input_ids.shape
        if batch < 1 or length < 1:
            raise ValueError("input_ids batch and sequence dimensions must be positive")
        self._validate_cache(input_ids, cache)
        if isinstance(logits_positions, torch.Tensor):
            positions = logits_positions
            if positions.ndim != 2 or positions.shape[0] != batch:
                raise ValueError("tensor logits_positions must have shape [batch, head_tokens]")
            if positions.dtype != torch.int64 or positions.device != input_ids.device:
                raise ValueError("tensor logits_positions must be int64 on the input device")
            head_tokens = positions.shape[1]
        else:
            positions = None if logits_positions is None else tuple(logits_positions)
            if positions is not None and any(type(pos) is not int or pos < 0 or pos >= length for pos in positions):
                raise ValueError("logits_positions must contain valid nonnegative input positions")
            head_tokens = length if positions is None else len(positions)
        if routing_plan is not None and not self.config.num_experts:
            raise ValueError("routing_plan applies only to MoE models")
        if isinstance(routing_plan, RoutingPlan) or routing_plan is None:
            plans = (routing_plan,) * self.config.num_hidden_layers
        else:
            plans = tuple(routing_plan)
            if len(plans) != self.config.num_hidden_layers or any(not isinstance(plan, RoutingPlan) for plan in plans):
                raise ValueError("routing_plan must contain one RoutingPlan per transformer layer")
        for plan in plans:
            if plan is not None:
                plan.validate(batch * length, self.config.num_experts, self.config.num_experts_per_tok)
        prefix = 0 if cache is None else cache.length
        rope = _rope_tables(self.config, length, prefix, input_ids.device)
        hidden = self.embed_tokens(input_ids)
        updated = []
        for index, layer in enumerate(self.layers):
            previous = None if cache is None else cache.layers[index]
            hidden, current = layer(
                hidden,
                previous,
                attention_policy=attention_policy,
                position_offset=prefix,
                rope=rope,
                routing_plan=plans[index],
            )
            if phase != "denoise":
                key = current.key.to(self.config.kv_dtype)
                value = current.value.to(self.config.kv_dtype)
                if previous is not None:
                    key = torch.cat((previous.key, key), dim=2)
                    value = torch.cat((previous.value, value), dim=2)
                updated.append(LayerKV(key, value))
        logits = None
        if head_tokens:
            if isinstance(positions, torch.Tensor):
                selection = positions.unsqueeze(-1).expand(batch, head_tokens, self.config.hidden_size)
                hidden = hidden.gather(1, selection)
            elif positions is not None:
                selection = torch.tensor(positions, dtype=torch.long, device=hidden.device)
                hidden = hidden.index_select(1, selection)
            logits = self.lm_head(self.norm(hidden))
        result_cache = cache if phase == "denoise" else KVCache(tuple(updated))
        return PhaseOutput(logits, result_cache)


@dataclass(frozen=True)
class ALoDLMSpec:
    """Dense recurrent paradigm, independently specified from arXiv:2610.04198v1.

    Layer interval [loop_start, loop_end) is shared across recurrent passes.
    ``feedback_policy`` selects inference feedback only: cumulative implements
    the paper-style accumulated commitments; newly_committed matches only the
    released inference feedback rule. Training always uses cumulative gold feedback.
    Neither profile implements the approximate depth-aware serving caches.
    """

    loop_start: int
    loop_end: int
    max_depth: int
    mask_token_id: int
    feedback_policy: str = "cumulative"
    normalize_carry: bool = True
    qk_norm: bool = True
    gate_init_mass: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _positive_int("loop_start", self.loop_start, allow_zero=True)
        _positive_int("loop_end", self.loop_end)
        _positive_int("max_depth", self.max_depth)
        _positive_int("mask_token_id", self.mask_token_id, allow_zero=True)
        if self.loop_start >= self.loop_end:
            raise ValueError("loop_start must precede loop_end")
        if self.feedback_policy not in ("cumulative", "newly_committed"):
            raise ValueError("feedback_policy must be cumulative or newly_committed")
        if type(self.normalize_carry) is not bool or type(self.qk_norm) is not bool:
            raise ValueError("normalize_carry and qk_norm must be bool")
        if self.gate_init_mass is not None:
            mass = tuple(self.gate_init_mass)
            if (
                len(mass) != self.max_depth
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in mass
                )
                or not math.isclose(sum(mass), 1.0, rel_tol=1e-7, abs_tol=1e-7)
            ):
                raise ValueError("gate_init_mass must contain max_depth positive probabilities summing to one")
            object.__setattr__(self, "gate_init_mass", mass)

    def validate_model(self, model: ModelConfig) -> None:
        if model.num_experts:
            raise ValueError("the ALoDLM reference currently supports dense backbones only")
        if self.loop_end > model.num_hidden_layers:
            raise ValueError("loop_end exceeds the number of physical layers")
        if self.mask_token_id >= model.vocab_size:
            raise ValueError("mask_token_id must be inside the model vocabulary")


@dataclass(frozen=True)
class ALoDLMBatch:
    """Independent sequences represented as clean + reordered-noisy streams.

    Token fields have shape [batch, 2 * source_length]. ``targets`` holds original
    token IDs, not shifted labels. ``ar_labels`` holds the caller's clean-stream
    AR targets and ignore_index on the noisy stream. ``attention_mask`` is shared
    physical-row visibility; ``positions`` retains original logical positions.
    Packed documents and random corruption are deliberately outside this helper.
    """

    input_ids: torch.Tensor
    targets: torch.Tensor
    positions: torch.Tensor
    masked: torch.Tensor
    mask_probability: torch.Tensor
    ar_labels: torch.Tensor
    attention_mask: torch.Tensor
    source_length: int
    block_size: int
    ignore_index: int = -100


def alodlm_attention_mask(source_length: int, block_size: int, *, device=None) -> torch.Tensor:
    """Exact two-stream visibility, using physical order inside each block.

    A row sees preceding clean blocks and the causal portion of its own block
    in its own stream. No noisy row can see its same-block clean target.
    """
    _positive_int("source_length", source_length)
    _positive_int("block_size", block_size)
    rows = torch.arange(2 * source_length, device=device)
    stream = rows >= source_length
    blocks = rows.remainder(source_length) // block_size
    same_local_block = (stream[:, None] == stream[None, :]) & (blocks[:, None] == blocks[None, :])
    ordered_rows = rows[:, None] >= rows[None, :]
    prior_clean = (blocks[None, :] < blocks[:, None]) & ~stream[None, :]
    return (same_local_block & ordered_rows) | prior_clean


def build_alodlm_batch(
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    masked: torch.Tensor,
    mask_probability: torch.Tensor | float,
    *,
    block_size: int,
    mask_token_id: int,
    ignore_index: int = -100,
) -> ALoDLMBatch:
    """Deterministically construct two streams from caller-supplied corruption.

    The stable within-block permutation places observed rows before masked rows.
    Probability tensors may be scalar or [B,S]; selected entries must lie in
    (0,1] and have a non-ignored label. Tensor assertions check real inputs;
    these remain caller value invariants for meta inputs. Input token IDs and labels
    are supplied explicitly: this function neither samples corruption nor shifts
    AR labels. Each batch row is an independent sequence; partial last blocks
    are allowed. All tensor transformations have static output shapes.
    """
    _positive_int("block_size", block_size)
    _positive_int("mask_token_id", mask_token_id, allow_zero=True)
    if type(ignore_index) is not int:
        raise ValueError("ignore_index must be an integer")
    if clean_ids.ndim != 2 or clean_ids.dtype != torch.int64 or min(clean_ids.shape) < 1:
        raise ValueError("clean_ids must be a nonempty int64 [batch, sequence] tensor")
    if labels.shape != clean_ids.shape or labels.dtype != torch.int64 or labels.device != clean_ids.device:
        raise ValueError("labels must be int64 with the clean input shape and device")
    if masked.shape != clean_ids.shape or masked.dtype != torch.bool or masked.device != clean_ids.device:
        raise ValueError("masked must be bool with the clean input shape and device")
    if isinstance(mask_probability, torch.Tensor):
        if not mask_probability.is_floating_point() or mask_probability.device != clean_ids.device:
            raise ValueError("mask_probability must be floating point on the input device")
        if mask_probability.ndim != 0 and mask_probability.shape != clean_ids.shape:
            raise ValueError("mask_probability must be scalar or match clean_ids")
        probability = mask_probability.to(torch.float32).expand(clean_ids.shape)
    else:
        if (
            isinstance(mask_probability, bool)
            or not isinstance(mask_probability, (int, float))
            or not (math.isfinite(mask_probability) and 0 < mask_probability <= 1)
        ):
            raise ValueError("scalar mask_probability must lie in (0,1]")
        probability = torch.full(clean_ids.shape, mask_probability, dtype=torch.float32, device=clean_ids.device)
    # These tensor guards are traceable: real inputs are checked without Python
    # scalar extraction, while meta execution carries only their shape contract.
    torch._assert_async((~masked | (labels != ignore_index)).all(), "masked rows require supervised labels")
    valid_probability = torch.isfinite(probability) & (probability > 0) & (probability <= 1)
    torch._assert_async((~masked | valid_probability).all(), "active mask_probability must be finite and in (0,1]")
    batch, length = clean_ids.shape
    chunks = []
    for start in range(0, length, block_size):
        stop = min(start + block_size, length)
        local_order = masked[:, start:stop].to(torch.int32).argsort(dim=1, stable=True)
        chunks.append(local_order + start)
    order = torch.cat(chunks, dim=1)
    original_positions = torch.arange(length, device=clean_ids.device).expand(batch, length)
    selected = masked.gather(1, order)
    noisy_ids = torch.where(masked, mask_token_id, clean_ids).gather(1, order)
    probability = torch.where(selected, probability.gather(1, order), 0.0)
    return ALoDLMBatch(
        input_ids=torch.cat((clean_ids, noisy_ids), dim=1),
        targets=torch.cat((clean_ids, clean_ids.gather(1, order)), dim=1),
        positions=torch.cat((original_positions, original_positions.gather(1, order)), dim=1),
        masked=torch.cat((torch.zeros_like(masked), selected), dim=1),
        mask_probability=torch.cat((torch.zeros_like(probability), probability), dim=1),
        ar_labels=torch.cat((labels, torch.full_like(labels, ignore_index)), dim=1),
        attention_mask=alodlm_attention_mask(length, block_size, device=clean_ids.device),
        source_length=length,
        block_size=block_size,
        ignore_index=ignore_index,
    )


@dataclass(frozen=True)
class ALoDLMOutput:
    """Full-depth training readouts; exit depths are one-based (observed rows: K)."""

    logits: torch.Tensor
    gate_logits: torch.Tensor
    exit_depths: torch.Tensor
    final_carry: torch.Tensor


@dataclass(frozen=True)
class ALoDLMOuterOutput:
    """One uncached inference replay; commit depths are not gate halt decisions."""

    logits: torch.Tensor
    gate_logits: torch.Tensor
    commit_depths: torch.Tensor
    committed: torch.Tensor
    tokens: torch.Tensor
    final_carry: torch.Tensor
    executed_steps: int


class ALoDLMExitGate(nn.Module):
    """Learnable conditional hazards initialized from positive exit-depth mass.

    The terminal projection and depth bias are still evaluated. Consumers force
    its *effective* hazard to one; they must not discard the executed gate op.
    """

    def __init__(self, hidden_size: int, spec: ALoDLMSpec, dtype: torch.dtype):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1, bias=True, dtype=dtype)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        mass = spec.gate_init_mass or (1.0 / spec.max_depth,) * spec.max_depth
        # P(exit at depth d | survived d-1) = mass[d] / sum(mass[d:]).
        biases = [math.log(mass[depth] / sum(mass[depth + 1 :])) for depth in range(spec.max_depth - 1)]
        self.depth_bias = nn.Parameter(torch.tensor([*biases, 0.0], dtype=dtype))

    def forward(self, features: torch.Tensor, depth: int) -> torch.Tensor:
        return self.projection(features.detach()).squeeze(-1) + self.depth_bias[depth]


def _alodlm_rope(config: ModelConfig, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, device=positions.device, dtype=torch.float32) / config.head_dim
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    full_angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return full_angles.cos(), full_angles.sin()


def _alodlm_depth_plan(
    depths: torch.Tensor | tuple[tuple[int, ...], ...],
    input_ids: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Validate plan metadata; callers add traceable mask-specific value guards."""
    if isinstance(depths, torch.Tensor):
        if depths.dtype != torch.int64 or depths.shape != input_ids.shape or depths.device != input_ids.device:
            raise ValueError("depth plan must be int64 with the input shape and device")
        return depths
    rows = tuple(tuple(row) for row in depths)
    if len(rows) != input_ids.shape[0] or any(len(row) != input_ids.shape[1] for row in rows):
        raise ValueError("static depth plan must match [batch, sequence]")
    if any(type(depth) is not int or depth < 0 or depth > max_depth for row in rows for depth in row):
        raise ValueError("static depth plan values must be between zero and the executed depth")
    return torch.tensor(rows, dtype=torch.int64, device=input_ids.device)


class ALoDLMModel(nn.Module):
    """Independent dense ALoDLM paradigm on the standard reference backbone.

    This implements recurrence equations and explicit dense replay, not the
    authors' code or checkpoint format. It intentionally has no KV cache, no
    approximate depth fallback, no streaming driver and no implicit token-pruning.
    Heads run over every physical row at every executed depth. A new outer call
    always rebuilds the embedding/prelude state; recurrent carry is local to it.
    """

    def __init__(self, config: ModelConfig, spec: ALoDLMSpec):
        super().__init__()
        spec.validate_model(config)
        self.config = config
        self.spec = spec
        self.backbone = BlockDiffusionModel(config)
        if spec.qk_norm:
            for layer in self.backbone.layers:
                assert isinstance(layer, TransformerBlock)
                layer.self_attn.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, config.dtype)
                layer.self_attn.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, config.dtype)
        self.exit_gate = ALoDLMExitGate(config.hidden_size, spec, config.dtype)

    def parameter_memory_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def layer_evaluations(self, steps: int | None = None) -> int:
        """Count physical-layer calls only, not FLOPs/token or latency."""
        steps = self.spec.max_depth if steps is None else steps
        _positive_int("steps", steps)
        if steps > self.spec.max_depth:
            raise ValueError("steps cannot exceed max_depth")
        return self.spec.loop_start + steps * (self.config.num_hidden_layers - self.spec.loop_start)

    def logical_buffer_count(self) -> int:
        """Depth-aware KV *pair* count only; this reference allocates no caches."""
        return self.config.num_hidden_layers + (self.spec.max_depth - 1) * (self.spec.loop_end - self.spec.loop_start)

    def _validate_inputs(self, input_ids, position_ids, attention_mask, masked, cache) -> None:
        if cache is not None:
            raise ValueError("ALoDLM uncached reference does not support ordinary or depth-aware KV caches")
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
        if (
            position_ids.dtype != torch.int64
            or position_ids.shape != input_ids.shape
            or position_ids.device != input_ids.device
        ):
            raise ValueError("position_ids must be int64 with the input shape and device")
        torch._assert_async((position_ids >= 0).all(), "position_ids must be nonnegative")
        if masked.dtype != torch.bool or masked.shape != input_ids.shape or masked.device != input_ids.device:
            raise ValueError("masked must be bool with the input shape and device")
        length = input_ids.shape[1]
        if (
            attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != (length, length)
            or attention_mask.device != input_ids.device
        ):
            raise ValueError("attention_mask must be explicit bool [sequence, sequence] on the input device")

    def _layers(self, hidden, start, stop, rope, mask):
        for index in range(start, stop):
            hidden, _ = self.backbone.layers[index](
                hidden,
                None,
                attention_policy="bidirectional",
                position_offset=0,
                rope=rope,
                routing_plan=None,
                attention_mask=mask,
            )
        return hidden

    def _readout(self, carry, rope, mask, depth):
        readout = self._layers(carry, self.spec.loop_end, self.config.num_hidden_layers, rope, mask)
        features = self.backbone.norm(readout)
        return self.backbone.lm_head(features), self.exit_gate(features, depth)

    def forward(
        self,
        batch: ALoDLMBatch,
        *,
        exit_depths: torch.Tensor | tuple[tuple[int, ...], ...] | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOutput:
        return self.rollout_training(batch, exit_depths=exit_depths, cache=cache)

    def rollout_training(
        self,
        batch: ALoDLMBatch,
        *,
        exit_depths: torch.Tensor | tuple[tuple[int, ...], ...] | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOutput:
        """Full-K differentiable rollout with cumulative masked gold feedback.

        Replay exit depths at masked rows must be in [1,K]; observed entries are
        ignored and reported as K. Without replay, independent conditional gate
        draws determine exits, with terminal exit forced. All K readout/gate
        tensors are retained, including readouts after a token's commitment.
        Gate features are detached; carry and teacher embeddings are not.
        """
        self._validate_inputs(batch.input_ids, batch.positions, batch.attention_mask, batch.masked, cache)
        if (
            batch.targets.shape != batch.input_ids.shape
            or batch.targets.dtype != torch.int64
            or batch.targets.device != batch.input_ids.device
        ):
            raise ValueError("batch targets must be int64 with the input shape and device")
        plan = None if exit_depths is None else _alodlm_depth_plan(exit_depths, batch.input_ids, self.spec.max_depth)
        if plan is not None:
            in_range = (plan >= 1) & (plan <= self.spec.max_depth)
            torch._assert_async((~batch.masked | in_range).all(), "masked exit depths must be in [1,K]")
            observed_sentinel = (plan == 0) | (plan == self.spec.max_depth)
            torch._assert_async((batch.masked | observed_sentinel).all(), "observed exit depths must be zero or K")
        rope = _alodlm_rope(self.config, batch.positions)
        embedded = self.backbone.embed_tokens(batch.input_ids)
        carry = self._layers(embedded, 0, self.spec.loop_start, rope, batch.attention_mask)
        teacher = self.backbone.embed_tokens(batch.targets)
        exits = torch.full_like(batch.input_ids, self.spec.max_depth)
        logits, gates = [], []
        for depth in range(self.spec.max_depth):
            if depth:
                feedback = batch.masked & (exits <= depth)
                carry = torch.where(feedback.unsqueeze(-1), teacher, carry)
            carry = self._layers(carry, self.spec.loop_start, self.spec.loop_end, rope, batch.attention_mask)
            prediction, gate = self._readout(carry, rope, batch.attention_mask, depth)
            logits.append(prediction)
            gates.append(gate)
            if depth + 1 < self.spec.max_depth:
                with torch.no_grad():
                    chosen = (
                        torch.rand_like(gate.float()) < gate.float().sigmoid() if plan is None else plan == depth + 1
                    )
                    chosen = chosen & batch.masked & (exits == self.spec.max_depth)
                exits = torch.where(chosen, depth + 1, exits)
                if self.spec.normalize_carry:
                    carry = self.backbone.norm(carry)
        return ALoDLMOutput(torch.stack(logits), torch.stack(gates), exits, carry)

    @torch.no_grad()
    def run_outer(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked: torch.Tensor,
        *,
        commit_depths: torch.Tensor | tuple[tuple[int, ...], ...],
        steps: int | None = None,
        feedback_tokens: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOuterOutput:
        """Replay an uncached outer window with explicit commit and stop schedules.

        Zero denotes no commit, positive depths are one-based. ``steps`` controls
        executed depth independently of gate hazards and confidence commitments;
        it is not an implementation of adaptive halting. Supplied feedback tokens
        replace argmax predictions at their planned commit depth. Committed rows
        remain in all subsequent dense core/coda/head computations.
        """
        self._validate_inputs(input_ids, position_ids, attention_mask, masked, cache)
        steps = self.spec.max_depth if steps is None else steps
        self.layer_evaluations(steps)
        plan = _alodlm_depth_plan(commit_depths, input_ids, steps)
        torch._assert_async(((plan >= 0) & (plan <= steps)).all(), "commit depths must be in [0,steps]")
        torch._assert_async((masked | (plan == 0)).all(), "observed positions cannot have a commit depth")
        if feedback_tokens is not None and (
            feedback_tokens.dtype != torch.int64
            or feedback_tokens.shape != input_ids.shape
            or feedback_tokens.device != input_ids.device
        ):
            raise ValueError("feedback_tokens must be int64 with the input shape and device")
        rope = _alodlm_rope(self.config, position_ids)
        carry = self._layers(self.backbone.embed_tokens(input_ids), 0, self.spec.loop_start, rope, attention_mask)
        tokens = input_ids
        committed = torch.zeros_like(masked)
        newly_committed = torch.zeros_like(masked)
        actual_depths = torch.zeros_like(input_ids)
        logits, gates = [], []
        for depth in range(steps):
            if depth:
                feedback = committed if self.spec.feedback_policy == "cumulative" else newly_committed
                replacement = self.backbone.embed_tokens(tokens)
                carry = torch.where(feedback.unsqueeze(-1), replacement, carry)
            carry = self._layers(carry, self.spec.loop_start, self.spec.loop_end, rope, attention_mask)
            prediction, gate = self._readout(carry, rope, attention_mask, depth)
            logits.append(prediction)
            gates.append(gate)
            selected_tokens = prediction.argmax(-1) if feedback_tokens is None else feedback_tokens
            newly_committed = masked & ~committed & (plan == depth + 1)
            tokens = torch.where(newly_committed, selected_tokens, tokens)
            actual_depths = torch.where(newly_committed, depth + 1, actual_depths)
            committed = committed | newly_committed
            if depth + 1 < steps and self.spec.normalize_carry:
                carry = self.backbone.norm(carry)
        return ALoDLMOuterOutput(
            torch.stack(logits),
            torch.stack(gates),
            actual_depths,
            committed,
            tokens,
            carry,
            steps,
        )


@dataclass(frozen=True)
class ALoDLMObjective:
    """Released-practical surrogate components, not the unbiased basic NELBO.

    Probabilities and detached denoising weights have shape [depth,batch,tokens]
    and are zero outside the initially supervised masked positions.
    """

    loss: torch.Tensor
    denoising: torch.Tensor
    actor: torch.Tensor
    autoregressive: torch.Tensor
    masked_tokens: torch.Tensor
    exit_probabilities: torch.Tensor
    denoising_weights: torch.Tensor


def alodlm_training_objective(
    output: ALoDLMOutput,
    batch: ALoDLMBatch,
    *,
    depth_prior_c: float = 0.4,
    kl_individual: float = 0.1,
    kl_aggregate: float = 1.0,
    ignore_index: int | None = None,
) -> ALoDLMObjective:
    """Compute the dense, unpacked released-practical ALoDLM objective.

    The recurrence supplies all depth readouts. This function implements
    detached pre-exit supervision, sequence-level score-function credit with a
    first-pass baseline, and clean-stream AR supervision averaged over depths.
    The release divides their sum by two; this convention is explicit here.
    The original Eq. 7 unbiasedness claim does not apply to this relaxed loss.
    A caller replaying fixed exits gets a conditional trajectory loss, not a
    claim that those exits were sampled from the current gate distribution.
    """
    for name, value in (
        ("depth_prior_c", depth_prior_c),
        ("kl_individual", kl_individual),
        ("kl_aggregate", kl_aggregate),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(batch.ignore_index) is not int:
        raise ValueError("batch.ignore_index must be an integer")
    if ignore_index is not None and (type(ignore_index) is not int or ignore_index != batch.ignore_index):
        raise ValueError("objective ignore_index must agree with the batch")
    ignore_index = batch.ignore_index
    scores = output.logits
    gate = output.gate_logits.float()
    if scores.ndim != 4 or gate.shape != scores.shape[:-1]:
        raise ValueError("logits must be [depth,batch,tokens,vocab] with matching gate logits")
    depth_count, batch_count, tokens, vocab = scores.shape
    if min(depth_count, batch_count, tokens, vocab) < 1:
        raise ValueError("objective dimensions must be positive")
    expected = (batch_count, tokens)
    if (
        batch.targets.shape != expected
        or batch.targets.dtype != torch.int64
        or batch.masked.shape != expected
        or batch.masked.dtype != torch.bool
        or batch.mask_probability.shape != expected
        or not batch.mask_probability.is_floating_point()
        or output.exit_depths.shape != expected
        or output.exit_depths.dtype != torch.int64
        or batch.ar_labels.shape != expected
        or batch.ar_labels.dtype != torch.int64
    ):
        raise ValueError("batch targets, masks, probabilities, AR labels and exit depths must match [batch,tokens]")
    if tokens != 2 * batch.source_length:
        raise ValueError("practical loss requires an unpacked clean/noisy two-stream batch")
    for tensor in (gate, batch.targets, batch.masked, batch.mask_probability, output.exit_depths, batch.ar_labels):
        if tensor.device != scores.device:
            raise ValueError("all objective tensors must share a device")

    valid = batch.masked
    torch._assert_async(
        (
            ~valid
            | (torch.isfinite(batch.mask_probability) & (batch.mask_probability > 0) & (batch.mask_probability <= 1))
        ).all(),
        "supervised mask probabilities must be finite and lie in (0,1]",
    )
    torch._assert_async(
        (~valid | ((output.exit_depths >= 1) & (output.exit_depths <= depth_count))).all(),
        "supervised exit depths must lie in [1, max_depth]",
    )
    count = valid.sum()
    denominator = count.clamp_min(1).float()
    per_depth_ce = F.cross_entropy(
        scores.float().reshape(-1, vocab),
        batch.targets.unsqueeze(0).expand(depth_count, -1, -1).reshape(-1),
        reduction="none",
    ).reshape(depth_count, batch_count, tokens)
    depth_ids = torch.arange(1, depth_count + 1, device=scores.device).view(-1, 1, 1)
    log_survival = torch.cat((torch.zeros_like(gate[:1]), F.logsigmoid(-gate).cumsum(0)[:-1]), dim=0)
    # Force the terminal depth to absorb all remaining survival mass.
    log_exit_profile = torch.where(depth_ids == depth_count, log_survival, log_survival + F.logsigmoid(gate))
    exits = output.exit_depths
    # Observed positions have no exit event; safe gather indices are discarded
    # by the mask. A rollout must validate supervised exits in [1, max_depth].
    indices = torch.where(valid, exits - 1, 0).unsqueeze(0)
    chosen_log_probability = log_exit_profile.gather(0, indices).squeeze(0)
    exit_cost = per_depth_ce.gather(0, indices).squeeze(0)

    log_supervision = torch.where(depth_ids == exits.unsqueeze(0), log_survival, log_survival + F.logsigmoid(gate))
    before_feedback = (depth_ids <= exits.unsqueeze(0)) & valid.unsqueeze(0)
    supervision = torch.where(before_feedback, log_supervision.exp(), 0.0).detach()
    inverse_probability = torch.where(valid, (batch.mask_probability.float() + 1e-8).reciprocal(), 0.0)
    normalized_inverse = inverse_probability / inverse_probability.sum().clamp_min(1e-12)
    denoising = ((per_depth_ce * supervision).sum(0) * normalized_inverse).sum()

    probabilities = log_exit_profile.exp() * valid.unsqueeze(0)
    with torch.no_grad():
        aggregate = probabilities.detach().sum((1, 2)) / denominator
        aggregate_log = aggregate.clamp_min(1e-12).log()
        prior_log = F.log_softmax(
            -depth_prior_c * torch.arange(1, depth_count + 1, device=scores.device, dtype=torch.float32), dim=0
        )
        selected_aggregate = aggregate_log[indices.squeeze(0)]
        selected_prior = prior_log[indices.squeeze(0)]
        regularization = kl_individual * (chosen_log_probability - selected_aggregate) + kl_aggregate * (
            selected_aggregate - selected_prior
        )
        advantage = (exit_cost - per_depth_ce[0] + regularization) * valid
    # Separate batch rows are independently attended sequences. Packed segment
    # credit is deliberately not inferred from a flat, ambiguous token stream.
    sequence_score = (chosen_log_probability * valid).sum(dim=1)
    actor = (advantage.sum(dim=1) * sequence_score).sum() / denominator

    clean_length = batch.source_length
    if clean_length > 1:
        next_labels = batch.ar_labels[:, 1:clean_length]
        ar_count = (next_labels != ignore_index).sum().clamp_min(1)
        ar_terms = F.cross_entropy(
            scores[:, :, : clean_length - 1].float().reshape(-1, vocab),
            next_labels.unsqueeze(0).expand(depth_count, -1, -1).reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        )
        autoregressive = ar_terms.sum() / (depth_count * ar_count)
    else:
        autoregressive = scores.sum() * 0.0
    loss = (denoising + actor + autoregressive) / 2
    # Keep zero-valued terminal gate/empty-target gradients explicit.
    loss = loss + gate.sum() * 0.0
    return ALoDLMObjective(loss, denoising, actor, autoregressive, count, probabilities, supervision)


@dataclass(frozen=True)
class ALoDLMDecision:
    tokens: torch.Tensor
    newly_committed: torch.Tensor
    committed: torch.Tensor
    unresolved: torch.Tensor
    cumulative_halt: torch.Tensor
    stop: torch.Tensor
    entropy: torch.Tensor


@torch.no_grad()
def alodlm_decision(
    logits: torch.Tensor,
    gate_logits: torch.Tensor,
    unresolved: torch.Tensor,
    committed: torch.Tensor,
    cumulative_halt: torch.Tensor,
    *,
    entropy_threshold: float,
    halt_threshold: float,
    terminal: bool = False,
    strict_entropy: bool = False,
    position_penalty: torch.Tensor | None = None,
) -> ALoDLMDecision:
    """Tensor-only one-pass commitment/halting decision with greedy predictions.

    Confidence commits tokens; the gate stops recurrence. They are separate
    decisions. Defaults use the paper's inclusive entropy threshold. The release
    profile can request a strict threshold and supply its explicit position cost.
    The caller drives outer/inner loops; meta callers can replay their schedule.
    """
    for name, value in (("entropy_threshold", entropy_threshold), ("halt_threshold", halt_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if entropy_threshold < 0 or not 0 <= halt_threshold <= 1:
        raise ValueError("entropy_threshold must be nonnegative and halt_threshold in [0,1]")
    if type(terminal) is not bool or type(strict_entropy) is not bool:
        raise ValueError("terminal and strict_entropy must be bool")
    if logits.ndim != 3 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating [batch,tokens,vocab] tensor")
    shape = logits.shape[:2]
    for tensor in (gate_logits, unresolved, committed, cumulative_halt):
        if tensor.shape != shape or tensor.device != logits.device:
            raise ValueError("decision tensors must match [batch,tokens] and device")
    if unresolved.dtype != torch.bool or committed.dtype != torch.bool:
        raise ValueError("unresolved and committed must be boolean")
    if not gate_logits.is_floating_point() or not cumulative_halt.is_floating_point():
        raise ValueError("gate logits and cumulative halt must be floating")
    if position_penalty is not None and (
        position_penalty.shape != shape
        or position_penalty.device != logits.device
        or not position_penalty.is_floating_point()
    ):
        raise ValueError("position_penalty must be floating [batch,tokens] on the input device")
    torch._assert_async((~(unresolved & committed)).all(), "unresolved and committed masks must be disjoint")
    torch._assert_async(
        (torch.isfinite(cumulative_halt) & (cumulative_halt >= 0) & (cumulative_halt <= 1)).all(),
        "cumulative halt probabilities must be finite and lie in [0,1]",
    )
    if position_penalty is not None:
        torch._assert_async(
            (torch.isfinite(position_penalty) & (position_penalty >= 0)).all(),
            "position penalties must be finite and nonnegative",
        )
    working_logits = logits.float()
    torch._assert_async(
        (~torch.isnan(working_logits) & (working_logits != float("inf"))).all(),
        "logits may mask vocabulary entries with -inf, but cannot contain NaN or +inf",
    )
    torch._assert_async(
        torch.isfinite(working_logits).any(-1).all(),
        "each logits row must contain at least one finite vocabulary value",
    )
    log_probabilities = F.log_softmax(working_logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -torch.xlogy(probabilities, probabilities).sum(-1)
    ranking = entropy if position_penalty is None else entropy + position_penalty
    hazard = torch.ones_like(gate_logits, dtype=torch.float32) if terminal else gate_logits.float().sigmoid()
    cumulative = torch.where(unresolved, cumulative_halt + (1 - cumulative_halt) * hazard, cumulative_halt)
    confident = ranking < entropy_threshold if strict_entropy else ranking <= entropy_threshold
    new = unresolved & confident
    all_committed = committed | new
    residual = unresolved & ~new
    remaining = residual.sum(-1)
    mean_halt = (cumulative * residual).sum(-1) / remaining.clamp_min(1)
    stop = (remaining == 0) | (mean_halt >= halt_threshold) | terminal
    ensure_progress = stop & ~all_committed.any(-1) & unresolved.any(-1)
    lowest = ranking.masked_fill(~unresolved, float("inf")).argmin(-1)
    fallback = F.one_hot(lowest, num_classes=logits.shape[1]).bool() & ensure_progress.unsqueeze(-1)
    new = new | fallback
    all_committed = committed | new
    residual = unresolved & ~new
    return ALoDLMDecision(logits.argmax(-1), new, all_committed, residual, cumulative, stop, entropy)


if __name__ == "__main__":
    config = ModelConfig()
    model = BlockDiffusionModel(config).eval()
    prompt = torch.arange(8).unsqueeze(0)
    prefix = model(prompt, phase="prefill", attention_policy="causal", logits_positions=()).cache
    canvas = torch.zeros((1, 4), dtype=torch.long)
    denoised = model(canvas, prefix, phase="denoise", logits_positions=(0, 2))
    committed = model(canvas, prefix, phase="commit", logits_positions=())
    print(
        {
            "logits_shape": tuple(denoised.logits.shape),
            "prefix_length": prefix.length,
            "committed_length": committed.cache.length,
            "weight_bytes": model.parameter_memory_bytes(),
            "kv_bytes": committed.cache.memory_bytes(),
        }
    )
