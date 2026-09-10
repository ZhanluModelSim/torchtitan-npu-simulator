# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ar_llm model: hybrid CSA/HCA/KDA attention + LatentMoE + mHC + Engram.

Contract: see MODEL_CONTRACT.md. Deviations from the raw reference
(``torchtitan_npu/simulator/raw_model/ar_llm``) are recorded there: the Engram
branch input fix, the sampling-loss wiring fix, removal of the unused KDA
``da_proj`` and indexer Hadamard rotation, untied output head, and MoR being an
inference-only scheme (training captures per-layer KV as usual).
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module, ModuleDict

from .attention import ArLlmAttention
from .feed_forward import ArLlmMoE, estimate_expert_params

logger = logging.getLogger(__name__)


class SinkhornIteration(nn.Module):
    def __init__(self, n_iters: int):
        super().__init__()
        self.n_iters = n_iters

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = torch.exp(w)
        for _ in range(self.n_iters):
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            w = w / w.sum(dim=-2, keepdim=True).clamp(min=1e-12)
        return w


class HyperConnectionBlock(nn.Module):
    """mHC block: expand -> Sinkhorn mix -> contract, with residual and norms."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        d, hc = model_args.dim, model_args.hc_mult
        self.hc_mult = hc
        self.expand = Linear.Config(in_features=d, out_features=hc * d, bias=False).build()
        self.sinkhorn = SinkhornIteration(model_args.sinkhorn_iters)
        self.mix_weights = nn.Parameter(torch.randn(hc, hc) * 0.02)
        self.contract = Linear.Config(in_features=hc * d, out_features=d, bias=False).build()
        self.pre_norm = RMSNorm.Config(normalized_shape=d, eps=model_args.norm_eps).build()
        self.post_norm = RMSNorm.Config(normalized_shape=d, eps=model_args.norm_eps).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        residual = x
        xn = self.pre_norm(x)
        expanded = self.expand(xn).view(b, s, self.hc_mult, d)
        mix = self.sinkhorn(self.mix_weights)
        mixed = torch.einsum("bshd,ho->bsod", expanded, mix)
        merged = mixed.reshape(b, s, self.hc_mult * d)
        return residual + self.post_norm(self.contract(merged))


class PolynomialRollingHash(nn.Module):
    """Multi-head polynomial rolling hash over n-gram token windows."""

    BASE = 257

    def __init__(self, order: int, capacity: int, num_heads: int, layer_idx: int, seed_offset: int = 0):
        super().__init__()
        self.order = order
        self.capacity = capacity
        self.num_heads = num_heads
        gen = torch.Generator()
        gen.manual_seed(layer_idx * 100 + order * 10 + seed_offset)
        self.register_buffer(
            "multipliers",
            torch.randint(1, 2**31 - 1, (num_heads,), generator=gen).to(torch.int64),
        )
        self.register_buffer(
            "base_powers",
            torch.tensor([self.BASE**i for i in range(order)], dtype=torch.int64),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        padded = F.pad(input_ids, (self.order - 1, 0), value=0)
        windows = torch.stack(
            [padded[:, offset : offset + s] for offset in range(self.order)], dim=0
        )
        hh = (windows * self.base_powers.view(-1, 1, 1)).sum(dim=0)
        return (hh.unsqueeze(-1) * self.multipliers) % self.capacity


class MultiHeadHashTable(nn.Module):
    """Per-head hash embedding table: ``[num_heads * capacity, per_head_dim]``.

    Deployment follows the Engram paper's training scheme (MODEL_CONTRACT.md
    §6): the table is sharded by hash bucket (contiguous ``Shard(0)``) across
    the EP/TP mesh, the lookup dispatches keys to the owning rank via
    All-to-All (forward gather) and autograd returns the row gradients via the
    reverse All-to-All (backward dispatch). With no sharding mesh the lookup
    degenerates to a local embedding gather.
    """

    def __init__(self, capacity: int, num_heads: int, memory_dim: int):
        super().__init__()
        self.capacity = capacity
        self.num_heads = num_heads
        self.per_head_dim = memory_dim // num_heads
        self.embedding = nn.Embedding(num_heads * capacity, self.per_head_dim)
        self.register_buffer("head_offsets", (torch.arange(num_heads) * capacity).to(torch.int64))
        self._table_group = None
        self._table_world_size = 1
        self._table_rank = 0
        self._table_local_size = num_heads * capacity
        self._table_local_offset = 0
        self._force_balance = False

    def forward(self, hash_indices: torch.Tensor) -> torch.Tensor:
        b, s, nh = hash_indices.shape
        idx = hash_indices + self.head_offsets
        if self._table_group is None or self._table_world_size <= 1:
            out = self.embedding(idx)
            return out.reshape(b, s, nh * self.per_head_dim)
        return self._forward_sharded(idx, b, s, nh)

    def _forward_sharded(self, idx: torch.Tensor, b: int, s: int, nh: int) -> torch.Tensor:
        world = self._table_world_size
        keys = idx.reshape(-1)
        total_keys = keys.shape[0]
        if self._force_balance:
            # Deterministic round-robin routing (simulator / forced load
            # balance): owner assignments and All-to-All splits become pure
            # functions of shapes, so no tensor values are ever read on meta.
            if total_keys % world != 0:
                raise ValueError(
                    f"Engram lookup keys ({total_keys}) must be divisible by the "
                    f"table-sharding world size ({world}) under forced load balance"
                )
            ar = torch.arange(total_keys, device=keys.device, dtype=torch.int64)
            owner = ar % world
            keys = (owner * self._table_local_size) + ((ar // world) % self._table_local_size)
            splits = [total_keys // world] * world
        else:
            if keys.device.type == "meta":
                raise RuntimeError(
                    "Engram sharded lookup on meta tensors requires forced load "
                    "balance (debug.moe_force_load_balance)"
                )
            owner = keys // self._table_local_size
            counts = torch.histc(owner.float(), bins=world, min=0, max=world - 1).to(torch.int64)
            splits = counts.tolist()

        send_perm = torch.argsort(owner, stable=True)
        sorted_keys = keys[send_perm].contiguous()
        recv_keys = funcol.all_to_all_single_autograd(
            sorted_keys, splits, splits, self._table_group
        )
        local_rows = F.embedding(
            (recv_keys - self._table_local_offset), self._weight_local()
        )
        recv_rows = funcol.all_to_all_single_autograd(
            local_rows.contiguous(), splits, splits, self._table_group
        )
        out = torch.zeros_like(recv_rows)
        out.index_add_(0, send_perm, recv_rows.to(out.dtype))
        return out.reshape(b, s, nh * self.per_head_dim)

    def _weight_local(self) -> torch.Tensor:
        weight = self.embedding.weight
        return weight.to_local() if isinstance(weight, DTensor) else weight


class ContextAwareGate(nn.Module):
    def __init__(self, hidden_size: int, memory_dim: int, eps: float):
        super().__init__()
        self.query_proj = Linear.Config(in_features=hidden_size, out_features=memory_dim, bias=False).build()
        self.query_norm = RMSNorm.Config(normalized_shape=memory_dim, eps=eps).build()
        self.key_norm = RMSNorm.Config(normalized_shape=memory_dim, eps=eps).build()
        self.scale = 1.0 / math.sqrt(memory_dim)

    def forward(self, hidden_states: torch.Tensor, memory_vectors: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(self.query_proj(hidden_states))
        k = self.key_norm(memory_vectors)
        sim = (q * k).sum(dim=-1, keepdim=True) * self.scale
        return torch.sigmoid(sim)


class ShortTermConv(nn.Module):
    def __init__(self, hidden_size: int, max_ngram_order: int):
        super().__init__()
        self.kernel_size = 4
        self.dilation = max_ngram_order
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=hidden_size,
            padding=0,
            bias=False,
        )
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.dilation * (self.kernel_size - 1)
        xt = F.pad(x.transpose(1, 2), (pad, 0))
        return self.conv(xt)[..., : x.shape[1]].transpose(1, 2)


class EngramModule(nn.Module):
    """Conditional memory: n-gram hash lookup + context gate + short conv."""

    def __init__(self, model_args: "ArLlmModel.Config", layer_idx: int):
        super().__init__()
        d = model_args.dim
        memory_dim = model_args.engram_memory_dim
        orders = model_args.engram_ngram_orders
        num_heads = model_args.engram_num_hash_heads
        capacity = model_args.engram_table_capacity

        self.hash_fns = nn.ModuleList(
            PolynomialRollingHash(order, capacity, num_heads, layer_idx, n_idx)
            for n_idx, order in enumerate(orders)
        )
        self.hash_tables = nn.ModuleList(
            MultiHeadHashTable(capacity, num_heads, memory_dim // len(orders)) for _ in orders
        )
        self.memory_proj = Linear.Config(in_features=memory_dim, out_features=d, bias=False).build()
        self.gate = ContextAwareGate(d, memory_dim, model_args.norm_eps)
        self.short_term = ShortTermConv(d, max_ngram_order=max(orders))
        self.input_norm = RMSNorm.Config(normalized_shape=d, eps=model_args.norm_eps).build()
        self.memory_norm = RMSNorm.Config(normalized_shape=d, eps=model_args.norm_eps).build()
        self.gate_bias = nn.Parameter(torch.zeros(1))
        # Sequence window of the local rank inside the GLOBAL token ids under
        # TP/CP sequence sharding (set by parallelize). Hash windows are
        # evaluated on the global ids and sliced to the local positions, so
        # n-grams keep cross-rank context without gathering hidden states.
        self._ids_window: tuple[int, int] | None = None

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        h = self.input_norm(hidden_states)

        memory_parts = []
        for hash_fn, table in zip(self.hash_fns, self.hash_tables):
            indices = hash_fn(input_ids)
            if self._ids_window is not None:
                start, end = self._ids_window
                indices = indices[:, start:end]
            memory_parts.append(table(indices))
        memory_vectors = torch.cat(memory_parts, dim=-1)

        gate = self.gate(h, memory_vectors) + self.gate_bias
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()

        memory_out = self.memory_norm(self.memory_proj(memory_vectors))
        conv_out = self.short_term(h)
        return residual + gate * memory_out + conv_out


class ArLlmBlock(Module):
    """Block: [Engram] -> mHC -> attention -> mHC -> LatentMoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        model_args: "ArLlmModel.Config"
        layer_id: int = 0

    def __init__(self, config: Config):
        super().__init__()
        model_args = config.model_args
        self.layer_id = config.layer_id
        self.mor_type = model_args.mor_type_for_layer(config.layer_id)

        self.has_engram = config.layer_id in model_args.engram_layers
        if self.has_engram:
            self.engram = EngramModule(model_args, config.layer_id)

        self.hc_pre_attn = HyperConnectionBlock(model_args)
        self.attention_norm = RMSNorm.Config(normalized_shape=model_args.dim, eps=model_args.norm_eps).build()
        self.attention = ArLlmAttention(model_args, config.layer_id)
        self.attn_post_norm = RMSNorm.Config(normalized_shape=model_args.dim, eps=model_args.norm_eps).build()

        self.hc_pre_ffn = HyperConnectionBlock(model_args)
        self.moe_pre_norm = RMSNorm.Config(normalized_shape=model_args.dim, eps=model_args.norm_eps).build()
        self.moe = ArLlmMoE(model_args, self.mor_type)
        self.moe_post_norm = RMSNorm.Config(normalized_shape=model_args.dim, eps=model_args.norm_eps).build()
        self.moe_enabled = True

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_masks, positions
        if self.has_engram:
            x = x + self.engram(x, input_ids)

        h = self.hc_pre_attn(x)
        h = h + self.attention(self.attention_norm(h), rope_cos, rope_sin)
        h = self.attn_post_norm(h)

        h = self.hc_pre_ffn(h)
        h = h + self.moe(self.moe_pre_norm(h))
        h = self.moe_post_norm(h)
        return h


class ArLlmModel(Module):
    """DeepSeekV4-Sparse hybrid model (ar_llm)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int = 524288
        dim: int = 16384
        n_layers: int = 60
        n_heads: int = 128
        head_dim: int = 256
        qk_nope_head_dim: int = 192
        qk_rope_head_dim: int = 64
        q_lora_rank: int = 4096
        kv_lora_rank: int = 1024
        o_groups: int = 32
        o_lora_rank: int = 4096

        unit_size: int = 6
        csa_per_unit: int = 1
        hca_per_unit: int = 1

        max_seq_len: int = 4096
        rope_theta: float = 16384.0
        yarn_factor: float = 32.0
        yarn_original_max: int = 65536

        kda_d_state: int = 256
        kda_d_k: int = 128
        kda_d_v: int = 128

        csa_compress_ratio: int = 16
        csa_window_size: int = 1024

        hca_compress_ratio: int = 256
        indexer_n_heads: int = 64
        indexer_head_dim: int = 128
        indexer_topk: int = 4096

        num_routed_experts: int = 2048
        num_shared_experts: int = 2
        moe_intermediate_size: int = 4096
        moe_latent_dim: int = 7168
        num_experts_per_token: int = 16
        router_score_function: str = "sqrtsoftplus"
        route_scale: float = 2.5
        mor_expert_ratio: float = 0.05
        mor_expert_capacity: float = 1.0
        debug_force_load_balance: bool = False

        hc_mult: int = 4
        sinkhorn_iters: int = 20
        use_attn_sink: bool = True

        engram_layers: list[int] = field(default_factory=list)
        engram_ngram_orders: list[int] = field(default_factory=lambda: [2, 3])
        engram_num_hash_heads: int = 8
        engram_table_capacity: int = 8_388_608
        engram_memory_dim: int = 4096

        swiglu_clamp: float = 10.0
        attn_softmax_clamp: float = 50.0
        norm_eps: float = 1e-6

        @property
        def layers(self):
            return range(self.n_layers)

        def layer_type(self, layer_idx: int) -> str:
            pos = layer_idx % self.unit_size
            if pos < self.csa_per_unit:
                return "csa"
            if pos < self.csa_per_unit + self.hca_per_unit:
                return "hca"
            return "kda"

        def expert_choice_layers(self) -> list[int]:
            num_expert_layers = max(1, int(self.n_layers * self.mor_expert_ratio))
            candidates = [
                idx for idx in range(self.n_layers) if self.layer_type(idx) in ("csa", "hca")
            ]
            if num_expert_layers >= len(candidates):
                return candidates
            step = len(candidates) // num_expert_layers
            return sorted({candidates[idx * step] for idx in range(num_expert_layers)})

        def mor_type_for_layer(self, layer_idx: int) -> str:
            return "expert" if layer_idx in self.expert_choice_layers() else "token"

        def validate(self) -> None:
            if self.n_layers % self.unit_size != 0:
                raise ValueError(
                    f"n_layers={self.n_layers} must be divisible by unit_size={self.unit_size}"
                )
            if self.csa_per_unit + self.hca_per_unit >= self.unit_size:
                raise ValueError("unit must contain at least one KDA layer")
            if self.n_heads % self.o_groups != 0:
                raise ValueError(f"n_heads={self.n_heads} must be divisible by o_groups={self.o_groups}")
            if self.head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
                raise ValueError("head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
            if self.kda_d_k != self.kda_d_v:
                raise ValueError(
                    f"kda_d_k={self.kda_d_k} must equal kda_d_v={self.kda_d_v} "
                    "(alpha gate/output alignment)"
                )
            if self.kda_d_state < self.kda_d_k:
                raise ValueError("kda_d_state must be >= kda_d_k")
            if self.dim % self.o_groups != 0:
                raise ValueError(f"dim={self.dim} must be divisible by o_groups={self.o_groups}")
            if self.engram_memory_dim % len(self.engram_ngram_orders) != 0:
                raise ValueError("engram_memory_dim must be divisible by the number of n-gram orders")
            if self.engram_memory_dim % self.engram_num_hash_heads != 0:
                raise ValueError("engram_memory_dim must be divisible by engram_num_hash_heads")
            if any(layer_idx < 0 or layer_idx >= self.n_layers for layer_idx in self.engram_layers):
                raise ValueError("engram_layers entries must be valid layer indices")
            if self.router_score_function not in {"sqrtsoftplus", "softmax", "sigmoid"}:
                raise ValueError(f"Unsupported router_score_function: {self.router_score_function}")

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            seq_len = trainer_config.training.seq_len
            if seq_len > self.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
                )
            self.max_seq_len = seq_len
            self.debug_force_load_balance = trainer_config.debug.moe_force_load_balance

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, float]:
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            flops_per_token = 6.0 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.model_args = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = ModuleDict(
            {
                str(idx): ArLlmBlock(ArLlmBlock.Config(model_args=config, layer_id=idx))
                for idx in range(config.n_layers)
            }
        )
        self.norm = RMSNorm.Config(normalized_shape=config.dim, eps=config.norm_eps).build()
        self.output = Linear.Config(in_features=config.dim, out_features=config.vocab_size, bias=False).build()
        self.register_buffer("rope_cos", torch.zeros(1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(1), persistent=False)

    def verify_module_protocol(self) -> None:
        pass

    def _build_rope_buffers(self, seq_len: int, device: torch.device) -> None:
        rope_dim = self.model_args.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.model_args.rope_theta
            ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim)
        )
        angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        if seq_len > self.model_args.yarn_original_max:
            ramp = torch.linspace(0, 1, seq_len, device=device)
            yarn_scale = (
                1.0 / self.model_args.yarn_factor + (1.0 - 1.0 / self.model_args.yarn_factor) * ramp
            ).clamp(min=0.1)
            emb = emb / yarn_scale.unsqueeze(-1)
        self.rope_cos = emb.cos()
        self.rope_sin = emb.sin()

    def init_weights(self, *, buffer_device=None) -> None:
        device = buffer_device if buffer_device is not None else self.rope_cos.device
        self._build_rope_buffers(self.model_args.max_seq_len, device)

        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, (PolynomialRollingHash, MultiHeadHashTable)):
                for name, buf in module.named_buffers(recurse=False):
                    module.register_buffer(name, buf.to(device))

        for layer in self.layers.values():
            model_args = self.model_args
            nn.init.normal_(layer.hc_pre_attn.mix_weights, mean=0.0, std=init_std)
            nn.init.normal_(layer.hc_pre_ffn.mix_weights, mean=0.0, std=init_std)
            if layer.attention.o_proj is not None:
                nn.init.normal_(
                    layer.attention.o_proj.o_down,
                    mean=0.0,
                    std=init_std / math.sqrt(layer.attention.o_proj.o_down.shape[-1]),
                )
                nn.init.normal_(layer.attention.o_proj.o_up, mean=0.0, std=init_std)
                if layer.attention.attn_sink is not None:
                    nn.init.zeros_(layer.attention.attn_sink)
            if layer.moe.shared_experts is not None:
                for weight in (
                    layer.moe.shared_experts.gate_down,
                    layer.moe.shared_experts.up_down,
                    layer.moe.shared_experts.latent_to_inter,
                    layer.moe.shared_experts.inter_to_latent,
                    layer.moe.shared_experts.latent_to_out,
                ):
                    nn.init.normal_(weight, mean=0.0, std=init_std)
            nn.init.normal_(
                layer.moe.router.gate.weight,
                mean=0.0,
                std=init_std / math.sqrt(model_args.num_routed_experts),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        # Full-length rope buffers: CP gathers the sequence inside attention
        # and re-slices there; attention itself slices to its (possibly
        # gathered) sequence length.
        for layer in self.layers.values():
            x = layer(x, tokens, self.rope_cos, self.rope_sin, attention_masks, positions)
        x = self.norm(x)
        return self.output(x)


def estimate_ar_llm_params(config: "ArLlmModel.Config") -> dict[str, int]:
    """Independent parameter-count formula (MODEL_CONTRACT.md section 7)."""
    d = config.dim
    nh = config.n_heads
    hd = config.head_dim
    nope = config.qk_nope_head_dim
    rope = config.qk_rope_head_dim
    g = config.o_groups
    idx_dim = config.indexer_n_heads * config.indexer_head_dim

    embedding = config.vocab_size * d
    output_head = config.vocab_size * d

    csa_hca_attn = (
        d * config.q_lora_rank
        + config.q_lora_rank * nh * nope
        + config.q_lora_rank * nh * rope
        + d * (config.kv_lora_rank + rope)
        + config.kv_lora_rank * nh * nope
        + config.kv_lora_rank * nh * hd
        + g * (nh // g) * hd * config.o_lora_rank
        + g * config.o_lora_rank * (d // g)
        + config.q_lora_rank
        + nope
        + (config.kv_lora_rank + rope)
        + config.kv_lora_rank
        + nh * config.use_attn_sink
    )
    hca_indexer = 2 * d * idx_dim + 2 * idx_dim

    kda_per_layer = (
        d * nh * config.kda_d_k
        + d * nh * config.kda_d_k
        + d * nh * config.kda_d_v
        + nh * config.kda_d_v * d
        + d * nh * config.kda_d_k
        + d * 2 * nh * config.kda_d_state
    )

    expert_per = estimate_expert_params(d, config.moe_latent_dim, config.moe_intermediate_size)
    shared_per_layer = config.num_shared_experts * expert_per
    router_per_layer = d * config.num_routed_experts
    mhc_per_layer = 2 * (
        config.hc_mult * d * d + config.hc_mult * config.hc_mult + config.hc_mult * d * d
    )
    block_norms = 4 * d + 4 * d

    engram_per_layer = 0
    if config.engram_layers:
        per_head_dim = config.engram_memory_dim // len(config.engram_ngram_orders) // config.engram_num_hash_heads
        tables = (
            len(config.engram_ngram_orders)
            * config.engram_num_hash_heads
            * config.engram_table_capacity
            * per_head_dim
        )
        engram_per_layer = (
            tables
            + 2 * d * config.engram_memory_dim
            + 2 * config.engram_memory_dim
            + 2 * d
            + d * 4
            + 1
        )

    csa_hca_count = sum(1 for idx in range(config.n_layers) if config.layer_type(idx) in ("csa", "hca"))
    hca_count = sum(1 for idx in range(config.n_layers) if config.layer_type(idx) == "hca")
    kda_count = sum(1 for idx in range(config.n_layers) if config.layer_type(idx) == "kda")

    attention_total = (
        csa_hca_count * csa_hca_attn + hca_count * hca_indexer + kda_count * kda_per_layer
    )
    moe_total = config.n_layers * (
        config.num_routed_experts * expert_per + shared_per_layer + router_per_layer
    )
    mhc_total = config.n_layers * mhc_per_layer
    norms_total = config.n_layers * block_norms + d
    engram_total = len(config.engram_layers) * engram_per_layer

    total = (
        embedding
        + output_head
        + attention_total
        + moe_total
        + mhc_total
        + norms_total
        + engram_total
    )
    return {
        "embedding": embedding,
        "output_head": output_head,
        "attention": attention_total,
        "moe": moe_total,
        "mhc": mhc_total,
        "norms": norms_total,
        "engram": engram_total,
        "total": total,
    }
