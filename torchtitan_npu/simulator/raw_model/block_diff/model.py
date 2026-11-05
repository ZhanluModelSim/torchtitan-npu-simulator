# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates.
"""Logical, single-rank block-diffusion training and inference reference.

Only PyTorch and the standard library are required. A call executes one phase:
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
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

MODEL_CONTRACT_VERSION = 1

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
