# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""glm5_next (GLM-5.3-Flash) model: mHC hybrid attention MoE + vision tower.

Contract: see MODEL_CONTRACT.md. Key decisions implemented here:

- 41 unique text blocks: 16 pre + 1 shared looped block (executed
  ``loop_train_steps`` times, no halting) + 24 post. The residual stream is
  ``[B, S, hc_mult, D]`` throughout; mHC reuses deepseek_v4's paramless
  ``HcPre``/``HcPost`` modules (same math) with a parameterless mean head.
- NoPE everywhere (``qk_rope_head_dim=0``); positions are accepted and ignored.
- The vision tower early-fuses patch-merged embeddings into the image token
  slots before the text backbone. v1 requires a uniform grid per batch
  (``image_size // patch_size`` patches per side, config-known) and never
  reads grid values on meta tensors.
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module, ModuleDict

from torchtitan_npu.models.deepseek_v4.model import HcPost, HcPre
from torchtitan_npu.models.multimodal import DenseMaskSDPA, build_valid_patch_mask

from .attention import GlmDeltaAttention, GlmDsaAttention
from .feed_forward import GlmMLP, GlmSparseMoeBlock

logger = logging.getLogger(__name__)


def _fuse_visual_embeddings(
    inputs_embeds: torch.Tensor,
    tokens: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Write visual embeddings into image-token slots (meta-clean).

    Equivalent to ``scatter_visual_embeddings`` with an all-valid visual mask:
    the k-th image slot (row-major) receives the k-th visual embedding via a
    cumsum-order gather instead of ``masked_select`` (no meta kernel).
    """
    b, s, d = inputs_embeds.shape
    image_slots = tokens == image_token_id  # [B, S]
    n, lm = visual_embeds.shape[0], visual_embeds.shape[1]
    visual_flat = visual_embeds.reshape(n * lm, d)
    order = image_slots.reshape(-1).to(torch.long).cumsum(dim=0) - 1
    order = order.clamp(min=0)
    gathered = visual_flat[order].view(b, s, d)
    return torch.where(image_slots.unsqueeze(-1), gathered.to(inputs_embeds.dtype), inputs_embeds)


class GlmHcHeadMean(nn.Module):
    """Final GLM HC-stream collapse: unweighted mean (no parameters)."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class GlmVisionAttention(nn.Module):
    """Per-image full attention with per-head QK norm and axial 2D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = Linear.Config(
            in_features=hidden_size, out_features=hidden_size * 3, bias=attention_bias
        ).build()
        self.proj = Linear.Config(
            in_features=hidden_size, out_features=hidden_size, bias=attention_bias
        ).build()
        self.q_norm = RMSNorm.Config(normalized_shape=self.head_dim, eps=rms_norm_eps).build()
        self.k_norm = RMSNorm.Config(normalized_shape=self.head_dim, eps=rms_norm_eps).build()
        self.inner_attention = DenseMaskSDPA.Config().build()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim // 2, 2).float() / (self.head_dim // 2))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axial_rope(self, grid_thw: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Axial 2D rope from per-patch (t, h, w) coordinates, meta-clean.

        ``grid_thw``: [N, L, 3]. Returns cos/sin of shape [N, L, head_dim].
        """
        pos_h = grid_thw[:, :, 1].float()
        pos_w = grid_thw[:, :, 2].float()
        freqs_h = pos_h[:, :, None] * self.inv_freq.to(pos_h.device)  # [N, L, hd/4]
        freqs_w = pos_w[:, :, None] * self.inv_freq.to(pos_w.device)
        freq_hw = torch.cat([freqs_h, freqs_w], dim=-1)  # [N, L, hd/2]
        freqs = torch.cat([freq_hw, freq_hw], dim=-1)  # [N, L, hd]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [N, L, hidden]
        cos: torch.Tensor,  # [N, L, head_dim]
        sin: torch.Tensor,
        attention_masks: torch.Tensor | None,  # [N, L, L] bool
    ) -> torch.Tensor:
        n, l, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(n, l, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)  # [N, L, H, D]

        query = self.q_norm(query)
        key = self.k_norm(key)

        cos = cos.unsqueeze(2)  # [N, L, 1, D]
        sin = sin.unsqueeze(2)
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin

        output = self.inner_attention(query, key, value, attention_masks=attention_masks)
        output = output.reshape(n, l, -1)
        return self.proj(output)


class GlmVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float, bias: bool = True):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.gate_proj = Linear.Config(
            in_features=hidden_size, out_features=intermediate_size, bias=bias
        ).build()
        self.up_proj = Linear.Config(
            in_features=hidden_size, out_features=intermediate_size, bias=bias
        ).build()
        self.down_proj = Linear.Config(
            in_features=intermediate_size, out_features=hidden_size, bias=bias
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class GlmVisionBlock(nn.Module):
    def __init__(self, vision_args: "GlmVisionTower.Config"):
        super().__init__()
        self.norm1 = RMSNorm.Config(normalized_shape=vision_args.hidden_size, eps=vision_args.rms_norm_eps).build()
        self.norm2 = RMSNorm.Config(normalized_shape=vision_args.hidden_size, eps=vision_args.rms_norm_eps).build()
        self.attn = GlmVisionAttention(
            hidden_size=vision_args.hidden_size,
            num_heads=vision_args.num_heads,
            rope_theta=vision_args.rope_theta,
            rms_norm_eps=vision_args.rms_norm_eps,
            attention_bias=vision_args.attention_bias,
        )
        self.mlp = GlmVisionMLP(
            hidden_size=vision_args.hidden_size,
            intermediate_size=vision_args.intermediate_size,
            swiglu_limit=vision_args.swiglu_limit,
            bias=vision_args.attention_bias,
        )

    def forward(self, x, cos, sin, attention_masks):
        x = x + self.attn(self.norm1(x), cos, sin, attention_masks)
        x = x + self.mlp(self.norm2(x))
        return x


class GlmVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, projection_intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.proj = Linear.Config(in_features=dim, out_features=dim, bias=False).build()
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = Linear.Config(
            in_features=dim, out_features=projection_intermediate_size, bias=False
        ).build()
        self.up_proj = Linear.Config(
            in_features=dim, out_features=projection_intermediate_size, bias=False
        ).build()
        self.down_proj = Linear.Config(
            in_features=projection_intermediate_size, out_features=dim, bias=False
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.gelu(self.post_projection_norm(x))
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class GlmVisionTower(nn.Module):
    """GLM vision tower: patchify -> blocks -> downsample conv -> merger."""

    @dataclass(kw_only=True, slots=True)
    class Config:
        depth: int = 32
        hidden_size: int = 2048
        num_heads: int = 32
        intermediate_size: int = 8192
        out_hidden_size: int = 24576
        patch_size: int = 14
        spatial_merge_size: int = 2
        temporal_patch_size: int = 1
        in_channels: int = 3
        image_size: int = 672
        projection_intermediate_size: int = 49152
        swiglu_limit: float = 10.0
        rms_norm_eps: float = 1e-5
        rope_theta: float = 10000.0
        attention_bias: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"vision image_size={config.image_size} must be divisible by "
                f"patch_size={config.patch_size}"
            )
        self.patches_per_side = config.image_size // config.patch_size
        self.num_patches = self.patches_per_side**2
        if self.num_patches % config.spatial_merge_size**2 != 0:
            raise ValueError("patches per image must be divisible by spatial_merge_size^2")

        self.patch_embed = nn.Conv2d(
            config.in_channels * config.temporal_patch_size,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            GlmVisionBlock(config) for _ in range(config.depth)
        )
        self.post_layernorm = RMSNorm.Config(
            normalized_shape=config.hidden_size, eps=config.rms_norm_eps
        ).build()
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = GlmVisionPatchMerger(
            dim=config.out_hidden_size,
            projection_intermediate_size=config.projection_intermediate_size,
            swiglu_limit=config.swiglu_limit,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,  # [N, L, C*T*P*P]
        grid_thw: torch.Tensor,  # [N, L, 3] per-patch coordinates
    ) -> torch.Tensor:
        n, l, _ = pixel_values.shape
        if l != self.num_patches:
            raise ValueError(
                "glm5_next vision v1 requires a uniform grid: pixel patches per "
                f"image L={l} must equal image_size//patch_size squared "
                f"{self.num_patches}; see MODEL_CONTRACT.md section 7"
            )

        c = self.config.in_channels * self.config.temporal_patch_size
        p = self.config.patch_size
        patches = pixel_values.reshape(n * l, c, p, p)
        hidden = self.patch_embed(patches).view(n, l, self.config.hidden_size)

        attention_masks = None
        if grid_thw is not None:
            valid = build_valid_patch_mask(grid_thw[:, :, 1:])
            valid_pair = valid[:, :, None] & valid[:, None, :]
            diagonal = torch.eye(l, dtype=torch.bool, device=valid.device)
            attention_masks = valid_pair | (~valid[:, :, None] & diagonal)
            cos, sin = self.blocks[0].attn._axial_rope(grid_thw, hidden.dtype)

        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_masks)
        hidden = self.post_layernorm(hidden)
        m = self.config.spatial_merge_size
        hidden = hidden.view(n, self.patches_per_side, self.patches_per_side, self.config.hidden_size)
        hidden = hidden.permute(0, 3, 1, 2)
        hidden = self.downsample(hidden)
        hidden = hidden.reshape(n, self.num_patches // (m * m), self.config.out_hidden_size)
        return self.merger(hidden)


class Glm5NextTransformerBlock(Module):
    """Text block: mHC -> attention -> mHC -> dense/MoE, 4-stream residual."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        model_args: "Glm5NextTextModel.Config"
        layer_id: int = 0
        attention_type: str = "kda"  # "kda" | "dsa"
        moe_enabled: bool = True

    def __init__(self, config: Config):
        super().__init__()
        model_args = config.model_args
        self.layer_id = config.layer_id
        self.attention_type = config.attention_type
        self.moe_enabled = config.moe_enabled
        self.hc_mult = model_args.hc_mult

        # mHC parameters (fp32, DSv4-compatible layout: fn on the block,
        # paramless HcPre/HcPost modules shared between both sites).
        mix_hc = (2 + model_args.hc_mult) * model_args.hc_mult
        hc_dim = model_args.hc_mult * model_args.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)
        self.hc_pre = HcPre.Config(
            hc_mult=model_args.hc_mult,
            hc_sinkhorn_iters=model_args.hc_sinkhorn_iters,
            hc_eps=model_args.hc_eps,
            norm_eps=model_args.norm_eps,
        ).build()
        self.hc_post = HcPost.Config().build()

        self.input_layernorm = RMSNorm.Config(
            normalized_shape=model_args.hidden_size,
            eps=model_args.norm_eps,
        ).build()
        self.post_attention_layernorm = RMSNorm.Config(
            normalized_shape=model_args.hidden_size,
            eps=model_args.norm_eps,
        ).build()

        if config.attention_type == "kda":
            self.attention = GlmDeltaAttention(
                hidden_size=model_args.hidden_size,
                num_heads=model_args.kda_num_heads,
                head_dim=model_args.kda_head_dim,
                conv_kernel_size=model_args.kda_conv_kernel_size,
                gate_lower_bound=model_args.kda_gate_lower_bound,
                norm_eps=model_args.norm_eps,
            )
        elif config.attention_type == "dsa":
            self.attention = GlmDsaAttention(
                hidden_size=model_args.hidden_size,
                num_heads=model_args.num_attention_heads,
                q_lora_rank=model_args.q_lora_rank,
                kv_lora_rank=model_args.kv_lora_rank,
                qk_nope_head_dim=model_args.qk_nope_head_dim,
                v_head_dim=model_args.v_head_dim,
                indexer_heads=model_args.indexer_n_heads,
                indexer_head_dim=model_args.indexer_head_dim,
                index_topk=model_args.index_topk,
                index_kpool=model_args.index_kpool,
                index_kpool_always_select_tail=model_args.index_kpool_always_select_tail,
                rms_norm_eps=model_args.norm_eps,
            )
        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")

        if config.moe_enabled:
            self.mlp = None
            self.moe = GlmSparseMoeBlock(
                hidden_size=model_args.hidden_size,
                num_experts=model_args.n_routed_experts,
                num_experts_per_tok=model_args.num_experts_per_tok,
                num_shared_experts=model_args.n_shared_experts,
                moe_intermediate_size=model_args.moe_intermediate_size,
                routed_scaling_factor=model_args.routed_scaling_factor,
                swiglu_limit=model_args.swiglu_limit,
                debug_force_load_balance=model_args.debug_force_load_balance,
            )
        else:
            self.moe = None
            self.mlp = GlmMLP(
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                swiglu_limit=model_args.swiglu_limit,
            )

    def forward(
        self,
        hidden_streams: torch.Tensor,  # [B, S, hc_mult, D]
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_streams
        hidden_states, post, comb = self.hc_pre(
            hidden_streams, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_masks, positions)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.moe is not None:
            hidden_states = self.moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states


class Glm5NextModel(Module):
    """GLM-5.3-Flash: vision tower + NoPE hybrid MoE text backbone.

    The text backbone is flat (``tok_embeddings`` / ``layers`` / ``norm`` /
    ``output``) so the upstream AC/PP conventions (``model.layers``) apply;
    the vision tower hangs off ``self.visual`` and early-fuses into the
    image-token slots.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # nominal structure
        vocab_size: int = 154880
        hidden_size: int = 24576
        num_hidden_layers: int = 96  # nominal, for bookkeeping
        pre_layers: int = 16
        looped_layers: int = 56
        post_layers: int = 24
        loop_train_steps: int = 4
        share_loop_weights: bool = True

        # attention
        num_attention_heads: int = 192
        kda_num_heads: int = 192
        kda_head_dim: int = 128
        kda_conv_kernel_size: int = 4
        kda_gate_lower_bound: float = -5.0
        q_lora_rank: int = 6144
        kv_lora_rank: int = 2048
        qk_nope_head_dim: int = 256
        v_head_dim: int = 256
        qk_rope_head_dim: int = 0  # NoPE; kept for state-dict/config parity
        indexer_n_heads: int = 64
        indexer_head_dim: int = 128
        index_topk: int = 8192
        index_kpool: int = 8
        index_kpool_always_select_tail: bool = True

        # mlp / moe
        first_k_dense_replace: int = 4
        intermediate_size: int = 73728
        n_routed_experts: int = 2048
        num_experts_per_tok: int = 16
        n_shared_experts: int = 1
        moe_intermediate_size: int = 3072
        routed_scaling_factor: float = 2.5

        # norms / hc
        norm_eps: float = 1e-5
        hc_mult: int = 4
        hc_sinkhorn_iters: int = 20
        hc_eps: float = 1e-6
        swiglu_limit: float = 10.0

        debug_force_load_balance: bool = False
        max_seq_len: int = 4096

        # multimodal fusion
        vision_config: GlmVisionTower.Config = field(default_factory=GlmVisionTower.Config)
        image_token_id: int = 154854
        video_start_token_id: int = 154832

        def layer_type(self, layer_id: int) -> str:
            return "dsa" if layer_id % 4 == 3 else "kda"

        def is_dense_layer(self, layer_id: int) -> bool:
            return layer_id < self.first_k_dense_replace

        def validate(self) -> None:
            if not self.share_loop_weights:
                raise ValueError(
                    "glm5_next v1 only models the weight-shared loop region "
                    "(share_loop_weights=True); see MODEL_CONTRACT.md section 3"
                )
            if not 1 <= self.loop_train_steps <= 4:
                raise ValueError(
                    f"loop_train_steps={self.loop_train_steps} must be within "
                    "train_min/max_steps [1, 4]; adaptive halting is not modeled"
                )
            if self.pre_layers + self.looped_layers + self.post_layers != self.num_hidden_layers:
                raise ValueError("pre + looped + post layers must equal num_hidden_layers")
            if self.qk_rope_head_dim != 0:
                raise ValueError("glm5_next is a NoPE model (qk_rope_head_dim must be 0)")
            if self.hidden_size % self.hc_mult != 0:
                raise ValueError("hidden_size must be divisible by hc_mult")
            if self.qk_nope_head_dim != self.v_head_dim:
                raise ValueError("qk_nope_head_dim must equal v_head_dim (fused kv latent)")
            if self.n_routed_experts % self.num_experts_per_tok != 0:
                raise ValueError("n_routed_experts must be divisible by num_experts_per_tok")
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

    @property
    def loop_block_id(self) -> int:
        return self.config.pre_layers

    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.config = config
        self.visual = GlmVisionTower(config.vision_config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # All blocks live in one ``layers`` ModuleDict so the upstream AC/PP
        # conventions (``model.layers``) apply. The weight-shared loop block
        # occupies slot ``pre_layers`` and is executed ``loop_train_steps``
        # times per forward (MODEL_CONTRACT.md section 3).
        layers: dict[str, Glm5NextTransformerBlock] = {}
        for layer_id in range(config.pre_layers):
            layers[str(layer_id)] = Glm5NextTransformerBlock(
                Glm5NextTransformerBlock.Config(
                    model_args=config,
                    layer_id=layer_id,
                    attention_type=config.layer_type(layer_id),
                    moe_enabled=not config.is_dense_layer(layer_id),
                )
            )
        layers[str(config.pre_layers)] = Glm5NextTransformerBlock(
            Glm5NextTransformerBlock.Config(
                model_args=config,
                layer_id=config.pre_layers,
                attention_type="kda",
                moe_enabled=True,
            )
        )
        for layer_id in range(config.pre_layers + config.looped_layers, config.num_hidden_layers):
            layers[str(layer_id)] = Glm5NextTransformerBlock(
                Glm5NextTransformerBlock.Config(
                    model_args=config,
                    layer_id=layer_id,
                    attention_type=config.layer_type(layer_id),
                    moe_enabled=not config.is_dense_layer(layer_id),
                )
            )
        self.layers = ModuleDict(layers)

        self.norm = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.hc_head = GlmHcHeadMean()
        self.output = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        ).build()

    def verify_module_protocol(self) -> None:
        pass

    def init_weights(self, *, buffer_device=None) -> None:
        del buffer_device
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, Glm5NextTransformerBlock):
                for hc_param in (
                    module.hc_attn_fn,
                    module.hc_attn_base,
                    module.hc_attn_scale,
                    module.hc_ffn_fn,
                    module.hc_ffn_base,
                    module.hc_ffn_scale,
                ):
                    nn.init.trunc_normal_(hc_param, mean=0.0, std=init_std)
                if isinstance(module.attention, GlmDeltaAttention):
                    gate = module.attention.forget_gate
                    nn.init.zeros_(gate.A_log)
                    with torch.no_grad():
                        dt = torch.empty_like(gate.dt_bias).uniform_(
                            math.log(1e-3), math.log(1e-1)
                        ).exp().clamp_min(1e-4)
                        gate.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                    nn.init.ones_(module.attention.o_norm.weight)
                if isinstance(module.attention, GlmDsaAttention):
                    indexer = module.attention.indexer
                    nn.init.zeros_(indexer.kpool_ape)
                    nn.init.ones_(indexer.kpool_gate)
                if module.moe is not None:
                    nn.init.normal_(module.moe.gate.e_score_correction_bias, mean=0.0, std=init_std)

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        inputs_embeds = self.tok_embeddings(tokens)

        if pixel_values is not None and grid_thw is not None:
            visual_embeds = self.visual(pixel_values, grid_thw)  # [N, L/m^2, D]
            image_token_mask = tokens == self.config.image_token_id
            if tokens.device.type != "meta":
                if (tokens == self.config.video_start_token_id).any():
                    raise ValueError(
                        "glm5_next v1 does not support the video token path; "
                        "see MODEL_CONTRACT.md section 7"
                    )
                num_image_tokens = int(image_token_mask.sum())
                num_visual = int(visual_embeds.shape[0] * visual_embeds.shape[1])
                if num_image_tokens != num_visual:
                    raise ValueError(
                        f"Image tokens ({num_image_tokens}) do not match visual "
                        f"embeddings ({num_visual})"
                    )
            inputs_embeds = _fuse_visual_embeddings(
                inputs_embeds,
                tokens,
                visual_embeds,
                self.config.image_token_id,
            )

        hidden_streams = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
        hidden_streams = hidden_streams.contiguous()

        for layer_id in range(self.config.pre_layers):
            hidden_streams = self.layers[str(layer_id)](hidden_streams, attention_masks, positions)
        loop_block = self.layers[str(self.config.pre_layers)]
        for _ in range(self.config.loop_train_steps):
            hidden_streams = loop_block(hidden_streams, attention_masks, positions)
        for layer_id in range(
            self.config.pre_layers + self.config.looped_layers, self.config.num_hidden_layers
        ):
            hidden_streams = self.layers[str(layer_id)](hidden_streams, attention_masks, positions)

        hidden_streams = self.norm(self.hc_head(hidden_streams))
        return self.output(hidden_streams)


def estimate_glm5_next_params(config: "Glm5NextModel.Config") -> dict[str, int]:
    """Independent parameter-count formula (MODEL_CONTRACT.md section 8)."""
    text = config  # text fields live directly on the top-level config
    d = text.hidden_size
    nh = text.num_attention_heads
    kq = text.kda_num_heads * text.kda_head_dim

    embedding = text.vocab_size * d
    output_head = text.vocab_size * d

    kda_per_block = (
        3 * d * kq
        + 3 * kq * text.kda_conv_kernel_size
        + d * text.kda_head_dim
        + text.kda_head_dim * kq
        + kq
        + text.kda_num_heads
        + d * text.kda_num_heads
        + d * text.kda_head_dim
        + text.kda_head_dim * kq
        + text.kda_head_dim
        + kq * d
    )
    dsa_per_block = (
        d * text.q_lora_rank
        + text.q_lora_rank
        + text.q_lora_rank * nh * text.qk_nope_head_dim
        + d * text.kv_lora_rank
        + text.kv_lora_rank
        + text.kv_lora_rank * nh * (text.qk_nope_head_dim + text.v_head_dim)
        + (nh * text.v_head_dim) * d
        + text.q_lora_rank * text.indexer_n_heads * text.indexer_head_dim
        + d * text.indexer_head_dim
        + 2 * text.indexer_head_dim  # indexer k_norm LayerNorm weight + bias
        + d * text.indexer_n_heads
        + text.index_kpool * text.indexer_head_dim
        + text.indexer_head_dim * d
    )
    moe_per_block = (
        text.n_routed_experts * (2 * text.moe_intermediate_size * d + d * text.moe_intermediate_size)
        + d * text.n_routed_experts
        + text.n_routed_experts
        + (2 * text.moe_intermediate_size * d + d * text.moe_intermediate_size) * text.n_shared_experts
    )
    dense_per_block = 2 * text.intermediate_size * d + d * text.intermediate_size
    hc_per_block = 2 * (
        ((2 * text.hc_mult + text.hc_mult**2) * (text.hc_mult * d))
        + (2 * text.hc_mult + text.hc_mult**2)
        + 3
    )
    norms_per_block = 2 * d

    pre_dense = sum(1 for i in range(text.pre_layers) if text.is_dense_layer(i))
    pre_sparse = text.pre_layers - pre_dense
    pre_dsa = sum(1 for i in range(text.pre_layers) if text.layer_type(i) == "dsa")
    pre_kda = text.pre_layers - pre_dsa
    post_dsa = sum(
        1 for i in range(text.pre_layers + text.looped_layers, text.num_hidden_layers)
        if text.layer_type(i) == "dsa"
    )
    post_kda = text.post_layers - post_dsa

    attention_total = (
        pre_kda * kda_per_block
        + pre_dsa * dsa_per_block
        + kda_per_block  # loop block
        + post_kda * kda_per_block
        + post_dsa * dsa_per_block
    )
    moe_total = (
        (pre_sparse + text.post_layers + 1) * moe_per_block
        + pre_dense * dense_per_block
    )
    mhc_total = (text.pre_layers + text.post_layers + 1) * hc_per_block
    norms_total = (text.pre_layers + text.post_layers + 1) * norms_per_block + d

    vision = config.vision_config
    v_h = vision.hidden_size
    head_dim = v_h // vision.num_heads
    # attention_bias=True: qkv/proj/mlp carry biases (raw vision config)
    vision_block_params = (
        3 * v_h * v_h + 3 * v_h  # qkv + bias
        + v_h * v_h + v_h  # proj + bias
        + 2 * head_dim  # q/k RMSNorm weights
        + 2 * v_h * vision.intermediate_size + 2 * vision.intermediate_size  # gate/up + biases
        + vision.intermediate_size * v_h + v_h  # down + bias
        + 2 * v_h  # norm1/norm2
    )
    vision_total = (
        (vision.in_channels * vision.temporal_patch_size * vision.patch_size**2 + 1) * v_h
        + vision.depth * vision_block_params
        + v_h  # post_layernorm
        + (v_h * vision.out_hidden_size * vision.spatial_merge_size**2 + vision.out_hidden_size)
        + (
            vision.out_hidden_size * vision.out_hidden_size
            + 2 * vision.out_hidden_size  # merger LayerNorm weight + bias
            + 2 * vision.out_hidden_size * vision.projection_intermediate_size
            + vision.projection_intermediate_size * vision.out_hidden_size
        )
    )

    total = (
        embedding
        + output_head
        + attention_total
        + moe_total
        + mhc_total
        + norms_total
        + vision_total
    )
    return {
        "embedding": embedding,
        "output_head": output_head,
        "attention": attention_total,
        "moe": moe_total,
        "mhc": mhc_total,
        "norms": norms_total,
        "vision": vision_total,
        "total": total,
    }
