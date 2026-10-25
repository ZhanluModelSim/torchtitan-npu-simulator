# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 attention modules: KDA (Kimi Delta Attention) and Gated MLA.

Fork reason: Upstream torchtitan has no K3 support. KDA is a novel linear
attention mechanism (delta rule) not present in any upstream model.
Reference: moonshotai/Kimi-K3, revision f831ab66814297da540d832a5235f8e904f29d06.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.protocols.module import Module


class ShortConvolution(Module):
    """Causal short convolution with optional activation (kernel_size=4 for KDA)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        conv: Conv1d.Config
        activation: str = "silu"

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.conv.in_channels
        self.kernel_size = config.conv.kernel_size
        self.activation = config.activation
        self.conv = config.conv.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        x_t = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        weight = self.conv.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        x_t = F.conv1d(
            x_t,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=x_t.shape[1],
        )[..., : x.shape[1]]
        x = x_t.transpose(1, 2)
        if self.activation == "silu":
            x = F.silu(x)
        return x


class RMSNormGated(Module):
    """RMSNorm with sigmoid gating: norm(x) * sigmoid(gate)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm = config.norm.build()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * torch.sigmoid(gate.float()).to(x.dtype)


class _FullContextAttention(Module):
    """Gather causal context after TP/SP gather; select CP rows before output projection.

    A regular slice has zero-filled backward outside the local interval.
    This avoids duplicating every CP rank's loss gradient before the input
    all-gather's reduce-scatter backward.
    """

    _cp_group = None
    _cp_rank: int = 0
    _cp_size: int = 1

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        mesh = parallel_dims.get_optional_mesh("cp")
        if mesh is not None:
            self._cp_group = mesh.get_group()
            self._cp_rank = mesh.get_local_rank()
            self._cp_size = mesh.size()

    def _gather_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._cp_group is None:
            return hidden_states
        gathered = funcol.all_gather_tensor_autograd(hidden_states.contiguous(), gather_dim=1, group=self._cp_group)
        if isinstance(gathered, funcol.AsyncCollectiveTensor):
            gathered = torch.ops._c10d_functional.wait_tensor(gathered)
        return gathered

    def _local_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._cp_group is None:
            return hidden_states
        if hidden_states.shape[1] % self._cp_size:
            raise ValueError("Kimi full context must be divisible by CP degree")
        return hidden_states.chunk(self._cp_size, dim=1)[self._cp_rank].contiguous()


class KimiDeltaAttention(_FullContextAttention):
    """Kimi Delta Attention (KDA) — linear attention via delta rule.

    69 of 93 layers in Kimi K3 use this O(n) attention variant.
    Core computation is supplied through Config.kernel at the fused boundary.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv1d: ShortConvolution.Config
        k_conv1d: ShortConvolution.Config
        v_conv1d: ShortConvolution.Config
        f_a_proj: Linear.Config
        f_b_proj: Linear.Config
        b_proj: Linear.Config
        o_norm: RMSNormGated.Config
        o_proj: Linear.Config
        g_proj: Linear.Config | None = None
        g_a_proj: Linear.Config | None = None
        g_b_proj: Linear.Config | None = None

        dim: int = 7168
        num_heads: int = 96
        head_dim: int = 128
        conv_kernel_size: int = 4
        gate_lower_bound: float | None = -5.0
        use_full_rank_gate: bool = True
        norm_eps: float = 1e-5
        kernel: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        projection_size = self.num_heads * self.head_dim

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv1d = config.q_conv1d.build()
        self.k_conv1d = config.k_conv1d.build()
        self.v_conv1d = config.v_conv1d.build()
        self.f_a_proj = config.f_a_proj.build()
        self.f_b_proj = config.f_b_proj.build()
        self.b_proj = config.b_proj.build()
        self.o_norm = config.o_norm.build()
        self.o_proj = config.o_proj.build()
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.use_full_rank_gate = config.use_full_rank_gate
        if self.use_full_rank_gate:
            if config.g_proj is None:
                raise ValueError("Full rank KDA gate requires g_proj")
            self.g_proj = config.g_proj.build()
        else:
            if config.g_a_proj is None or config.g_b_proj is None:
                raise ValueError("Low rank KDA gate requires g_a_proj and g_b_proj")
            self.g_a_proj = config.g_a_proj.build()
            self.g_b_proj = config.g_b_proj.build()

        self.gate_lower_bound = config.gate_lower_bound
        self.kernel = config.kernel

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        if self.num_heads % parallel_dims.tp:
            raise ValueError("KDA heads must be divisible by TP degree")
        self.num_heads //= parallel_dims.tp

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.A_log, 1.0, 16.0)
        with torch.no_grad():
            self.A_log.log_()
        nn.init.zeros_(self.dt_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: Any | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self._gather_context(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        # Gate and beta
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Core KDA computation
        o = self._chunk_kda(q, k, v, g, beta)

        # Output gate + norm
        out_g = self.g_proj(hidden_states) if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(hidden_states))
        out_g = out_g.view(batch_size, seq_len, self.num_heads, self.head_dim)

        o = self.o_norm(o, out_g)
        o = o.reshape(batch_size, seq_len, -1)
        return self.o_proj(self._local_context(o))

    def _chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke an explicitly supplied KDA provider at the retained fusion boundary."""
        if self.kernel is None:
            raise RuntimeError("Kimi KDA needs an explicit kernel provider; graph capture uses the simulator runtime")

        A_log = self.A_log.to_local() if isinstance(self.A_log, DTensor) else self.A_log
        dt_bias = self.dt_bias.to_local() if isinstance(self.dt_bias, DTensor) else self.dt_bias
        o, _ = self.kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=None,
        )
        return o


class KimiGatedMLA(_FullContextAttention):
    """Gated Multi-head Latent Attention for Kimi K3.

    24 of 93 layers use this. Same as DSv3 MLA but with an output gate:
    attn_output *= sigmoid(g_proj(hidden_states))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        q_a_proj: Linear.Config
        q_a_layernorm: RMSNorm.Config
        q_b_proj: Linear.Config
        kv_a_proj_with_mqa: Linear.Config
        kv_a_layernorm: RMSNorm.Config
        kv_b_proj: Linear.Config
        g_proj: Linear.Config
        o_proj: Linear.Config

        dim: int = 7168
        n_heads: int = 96
        q_lora_rank: int = 1536
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)

        self.q_a_proj = config.q_a_proj.build()
        self.q_a_layernorm = config.q_a_layernorm.build()
        self.q_b_proj = config.q_b_proj.build()
        self.kv_a_proj_with_mqa = config.kv_a_proj_with_mqa.build()
        self.kv_a_layernorm = config.kv_a_layernorm.build()
        self.kv_b_proj = config.kv_b_proj.build()
        self.g_proj = config.g_proj.build()
        self.o_proj = config.o_proj.build()

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        if self.num_heads % parallel_dims.tp:
            raise ValueError("MLA heads must be divisible by TP degree")
        self.num_heads //= parallel_dims.tp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_masks: Any | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self._gather_context(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape

        # Query
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(k_compressed))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Kimi K3 sets mla_use_nope=True, so the legacy q_rope/k_rope slices
        # participate in attention without rotary position encoding.
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope_expanded], dim=-1)

        # Pad v to match q_head_dim for flash attention compatibility
        v_padded = F.pad(v, [0, self.q_head_dim - self.v_head_dim]) if self.v_head_dim != self.q_head_dim else v

        # Scaled dot-product attention
        query = query.transpose(1, 2)  # (B, H, S, D)
        key = key.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        attn_output = self._attention_core(query, key, v_padded)
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)

        # Trim padding
        if self.v_head_dim != self.q_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output gate (K3-specific)
        gate = torch.sigmoid(self.g_proj(hidden_states))
        attn_output = attn_output * gate

        return self.o_proj(self._local_context(attn_output))

    def _attention_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MLA attention core after all projection/layout work."""
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
