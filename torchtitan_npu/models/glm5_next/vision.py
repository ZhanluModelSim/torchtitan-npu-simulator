# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash vision tower (reference Glm5NextVisionModel).

24 blocks of RMSNorm + full (non-causal) attention with per-image RoPE +
clamped-swiglu MLP, followed by a 2x2 spatial-merge Conv2d and the patch
merger. Production runs ``torch_npu.npu_fusion_attention`` with per-image
cu_seqlens (varlen). The eager fallback exploits the uniform square grid
(every image shares the config-known gh/gw), so the batched SDPA path needs
no value reads and stays meta-safe: cu_seqlens are built with ``arange`` and
the vision RoPE uses coordinate ranges instead of gathered grid values.

The tower internals stay plain ``nn`` children (Conv3d has no protocol
counterpart); :meth:`GlmVisionTower.reset_parameters` owns their
initialization, which ``init_states`` calls through the Module walk.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

from .feed_forward import clamp_swiglu_eager


def _vision_position_ids(num_images: int, grid_size: int) -> torch.Tensor:
    """Qwen-VL-style 2D vision RoPE coordinates for a uniform square grid.

    Returns ``[num_images * grid_size**2, 2]`` int64 (h_idx, w_idx) built
    purely from ranges — no grid_thw value reads (meta-safe)."""
    h = torch.arange(grid_size)
    w = torch.arange(grid_size)
    per_image = torch.stack(
        (
            h.unsqueeze(1).expand(grid_size, grid_size).reshape(-1),
            w.unsqueeze(0).expand(grid_size, grid_size).reshape(-1),
        ),
        dim=-1,
    )
    return per_image.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, 2)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference folds gate/up weights into one linear (fused gate_up).
        gate_up = F.linear(
            x,
            torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0),
            torch.cat([self.gate_proj.bias, self.up_proj.bias]),
        )
        gate_up = torch.clamp(gate_up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(clamp_swiglu_eager(gate_up, self.swiglu_limit))


class VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_norm = RMSNorm.Config(normalized_shape=self.head_dim, eps=norm_eps).build()
        self.k_norm = RMSNorm.Config(normalized_shape=self.head_dim, eps=norm_eps).build()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query = self.q_norm(query)
        key = self.k_norm(key)

        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        query = ((query.float() * cos) + (self._rotate_half(query.float()) * sin)).to(query.dtype)
        key = ((key.float() * cos) + (self._rotate_half(key.float()) * sin)).to(key.dtype)

        # Uniform-square fallback: every image has the same length, so a
        # plain batched non-causal SDPA over [N, H, L, D] is equivalent to
        # the production packed varlen fusion attention.
        total, heads, dim = query.shape[0], query.shape[1], query.shape[2]
        per_image = cu_seqlens.shape[0] - 1
        length = total // per_image
        attn_output = (
            F.scaled_dot_product_attention(
                query.view(per_image, length, heads, dim).transpose(1, 2),
                key.view(per_image, length, heads, dim).transpose(1, 2),
                value.view(per_image, length, heads, dim).transpose(1, 2),
                scale=self.scaling,
            )
            .transpose(1, 2)
            .reshape(total, heads * dim)
        )
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, norm_eps: float, swiglu_limit: float):
        super().__init__()
        self.norm1 = RMSNorm.Config(normalized_shape=hidden_size, eps=norm_eps).build()
        self.norm2 = RMSNorm.Config(normalized_shape=hidden_size, eps=norm_eps).build()
        self.attn = VisionAttention(hidden_size, num_heads, norm_eps)
        self.mlp = VisionMLP(hidden_size, intermediate_size, swiglu_limit)

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, swiglu_limit: float):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = F.gelu(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_state).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class GlmVisionTower(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 1024
        depth: int = 24
        num_heads: int = 16
        intermediate_size: int = 4096
        out_hidden_size: int = 4096
        """Deployed GLM-5.3-Flash HF config value (must equal the text dim:
        the merger output is masked-scattered into the embedding stream).
        The MindSpeed Glm5NextVisionConfig default (1536) is overridden by
        the checkpoint config.json."""
        projection_intermediate_size: int = 10240
        patch_size: int = 14
        temporal_patch_size: int = 1
        """The torchtitan mm dataset emits single-frame image tensors with
        D = in_channels * patch_size^2 (no temporal stacking), so the patch
        conv must consume t=1 groups even though the official ckpt ships a
        t=2 kernel (unused for the image-only simulator path)."""
        spatial_merge_size: int = 2
        in_channels: int = 3
        norm_eps: float = 1e-5
        swiglu_limit: float = 10.0
        image_size: int = 448
        """Forced square image size (patches per side = image_size // patch_size)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        kernel = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.patch_embed = nn.Conv3d(config.in_channels, config.hidden_size, kernel_size=kernel, stride=kernel)
        head_dim = config.hidden_size // config.num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float) / (head_dim // 2)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.intermediate_size,
                    config.norm_eps,
                    config.swiglu_limit,
                )
                for _ in range(config.depth)
            ]
        )
        self.post_layernorm = RMSNorm.Config(normalized_shape=config.hidden_size, eps=config.norm_eps).build()
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = VisionPatchMerger(
            config.out_hidden_size,
            config.projection_intermediate_size,
            config.swiglu_limit,
        )

    def reset_parameters(self) -> None:
        """Own initialization for the plain-nn internals (the Module walk
        only initializes protocol modules)."""
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.inv_freq.device
        head_dim = self.config.hidden_size // self.config.num_heads
        self.inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float, device=device) / (head_dim // 2))
        )

    def _rotary(self, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.to(position_ids.device)
        return freqs.flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.config
        num_images = pixel_values.shape[0]
        grid_size = cfg.image_size // cfg.patch_size
        length = grid_size * grid_size

        hidden_states = self.patch_embed(
            pixel_values.view(-1, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        ).view(-1, cfg.hidden_size)

        position_ids = _vision_position_ids(num_images, grid_size).to(hidden_states.device)
        rotary_emb = self._rotary(position_ids)
        emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        cu_seqlens = torch.arange(0, num_images + 1, device=hidden_states.device) * length

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, cos, sin)

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(-1, cfg.spatial_merge_size, cfg.spatial_merge_size, cfg.hidden_size)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, cfg.out_hidden_size)
        return self.merger(hidden_states)
