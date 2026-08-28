# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview model: video+audio diffusion transformer for flow matching.

Fork reason: Upstream torchtitan has no MAGI-2 support. MAGI-2-preview is a
114B joint video/audio diffusion MoE transformer: per-token modality-sorted
packing, multi-modality grouped projections, MHC (multi-stream hyper-connect)
wrapped sublayers, sigmoid-gated varlen attention with learned sinks and
partial RoPE, and a multi-head core MoE with sigmoid routing.
Reference: /tmp/magi2-preview/inference/model/magi2_preview.py (Apache-2.0)

Training-port deviations from the official inference code:
- Plain torch everywhere (no triton / flash-attn); attention is a plain
  per-segment softmax implementation, autograd friendly.
- ``init_weights`` gives a live randomly-initialized network for smoke
  training; the official skip-load scheme zeroes the MoE expert tensors
  (W_gate/W_up/W_down) which would produce a dead network without a
  checkpoint. Official checkpoints are loaded via the state dict adapter.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum

import torch
from torch import nn

from torchtitan.protocols.module import Module, ModuleDict

from .attention import Magi2Attention
from .embeddings import ElementWiseFourierEmbed
from .feed_forward import CoreMultiHeadMoE, Magi2MLP, MultiHeadMoELayer
from .grouped_linear import GroupedLinear
from .mhc import apply_hpre, hyper_connect, sigmoid_affine, sinkhorn_knopp
from .norms import MultiModalityRMSNorm

logger = logging.getLogger(__name__)


class Modality(IntEnum):
    """Per-token modality ids carried by ``modality_mapping``."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    # Marker modality for tokens whose first channels carry the timestep
    # embedding; remapped to TEXT at model entry (official parity).
    TIME = 3


class PreAdapter(nn.Module):
    """Projects channel-padded raw tokens into the MHC stream width.

    Per-modality linear embedders write into a zero-initialized fp32 buffer
    via ``index_copy_`` (tokens stay in original packed order), plus the
    element-wise Fourier RoPE embedding computed from the 9-dim coordinates.
    """

    def __init__(
        self,
        adapter_dim: int,
        video_in_channels: int,
        audio_in_channels: int,
        text_in_channels: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.adapter_dim = adapter_dim
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels
        self.video_embedder = nn.Linear(video_in_channels, adapter_dim, bias=True)
        self.text_embedder = nn.Linear(text_in_channels, adapter_dim, bias=True)
        self.audio_embedder = nn.Linear(audio_in_channels, adapter_dim, bias=True)
        self.rope = ElementWiseFourierEmbed(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_idx: torch.Tensor,
        audio_idx: torch.Tensor,
        text_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        output = torch.zeros(
            x.shape[0], self.adapter_dim, device=x.device, dtype=torch.float32
        )
        if video_idx.numel() > 0:
            output.index_copy_(
                0,
                video_idx,
                self.video_embedder(
                    x.index_select(0, video_idx)[:, : self.video_in_channels]
                ),
            )
        if audio_idx.numel() > 0:
            output.index_copy_(
                0,
                audio_idx,
                self.audio_embedder(
                    x.index_select(0, audio_idx)[:, : self.audio_in_channels]
                ),
            )
        if text_idx.numel() > 0:
            output.index_copy_(
                0,
                text_idx,
                self.text_embedder(
                    x.index_select(0, text_idx)[:, : self.text_in_channels]
                ),
            )
        return output, rope


class PostAdapter(nn.Module):
    """Projects the stream back to per-modality prediction channels.

    Video rows fill columns ``[:48]``, audio rows fill ``[:64]``; text rows
    stay zero so the MSE loss is naturally masked there.
    """

    def __init__(
        self,
        adapter_dim: int,
        video_in_channels: int,
        audio_in_channels: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.final_out_dim = max(video_in_channels, audio_in_channels)
        self.final_norm_video = MultiModalityRMSNorm(adapter_dim, eps=eps)
        self.final_norm_audio = MultiModalityRMSNorm(adapter_dim, eps=eps)
        self.final_linear_video = nn.Linear(
            adapter_dim, video_in_channels, bias=False
        )
        self.final_linear_audio = nn.Linear(
            adapter_dim, audio_in_channels, bias=False
        )

    def forward(
        self, x: torch.Tensor, video_idx: torch.Tensor, audio_idx: torch.Tensor
    ) -> torch.Tensor:
        x_out = torch.zeros(
            x.shape[0], self.final_out_dim, device=x.device, dtype=torch.float32
        )
        if video_idx.numel() > 0:
            x_video = self.final_norm_video(x.index_select(0, video_idx))
            x_video = self.final_linear_video(x_video)
            x_out[:, : self.video_in_channels].index_copy_(0, video_idx, x_video)
        if audio_idx.numel() > 0:
            x_audio = self.final_norm_audio(x.index_select(0, audio_idx))
            x_audio = self.final_linear_audio(x_audio)
            x_out[:, : self.audio_in_channels].index_copy_(0, audio_idx, x_audio)
        return x_out


class TransformerLayer(nn.Module):
    """One MAGI-2-preview layer: MHC-wrapped attention + dense/MoE MLP.

    Owns the attention module, the MLP module and the 14 MHC parameters
    (6 alphas, 6 biases, 2 fused phi projections; the MHC norm is a child
    module). The stream state ``s`` has shape ``(T, num_stream, hidden)``.
    """

    def __init__(self, config: "Magi2PreviewModel.Config", layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_stream = config.num_stream
        self.hidden_size = config.hidden_size
        self.mhc_matmul_scale = 1.0 / math.sqrt(
            float(config.num_stream * config.hidden_size)
        )
        num_modality = 3 if layer_id in config.mm_layers else 1
        self.attention = Magi2Attention(
            Magi2Attention.Config(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_modality=num_modality,
                norm_eps=config.norm_eps,
                sink_token_num=config.sink_token_num,
            )
        )
        self.mlp: MultiHeadMoELayer | Magi2MLP
        if layer_id in config.moe_layers:
            self.mlp = MultiHeadMoELayer(
                MultiHeadMoELayer.Config(
                    hidden_size=config.hidden_size,
                    num_modality=3,
                    moe_num_heads=config.moe_num_heads,
                    num_experts=config.num_experts,
                    moe_top_k=config.moe_top_k,
                    expert_intermediate_size=config.expert_intermediate_size,
                    shared_expert_intermediate_size=(
                        config.shared_expert_intermediate_size
                    ),
                    route_scale=config.route_scale,
                    norm_eps=config.norm_eps,
                )
            )
        else:
            self.mlp = Magi2MLP(
                Magi2MLP.Config(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.dense_intermediate_size,
                    num_modality=num_modality,
                    norm_eps=config.norm_eps,
                )
            )

        n = config.num_stream
        c = config.hidden_size
        phi_out = n + n + n * n
        self.mhc_alpha_pre_attn = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_alpha_post_attn = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_alpha_res_attn = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_alpha_pre_mlp = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_alpha_post_mlp = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_alpha_res_mlp = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mhc_bias_pre_attn = nn.Parameter(torch.empty(n, dtype=torch.float32))
        self.mhc_bias_post_attn = nn.Parameter(torch.empty(n, dtype=torch.float32))
        self.mhc_bias_pre_mlp = nn.Parameter(torch.empty(n, dtype=torch.float32))
        self.mhc_bias_post_mlp = nn.Parameter(torch.empty(n, dtype=torch.float32))
        self.mhc_bias_res_attn = nn.Parameter(torch.empty(n, n, dtype=torch.float32))
        self.mhc_bias_res_mlp = nn.Parameter(torch.empty(n, n, dtype=torch.float32))
        self.mhc_phi_fused_attn = nn.Parameter(
            torch.empty(n * c, phi_out, dtype=torch.float32)
        )
        self.mhc_phi_fused_mlp = nn.Parameter(
            torch.empty(n * c, phi_out, dtype=torch.float32)
        )
        self.mhc_norm = MultiModalityRMSNorm(
            n * c, num_modality=num_modality, eps=config.norm_eps,
            out_dtype=torch.float32,
        )

    def _mhc_project(
        self, s: torch.Tensor, phi: torch.Tensor, m_splits: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Norm the flattened stream and split phi output into pre/post/res."""
        n = self.num_stream
        s_flat = s.reshape(s.shape[0], n * self.hidden_size)
        h_fused = torch.matmul(self.mhc_norm(s_flat, m_splits).float(), phi)
        h_pre, h_post, h_res = torch.split(h_fused, [n, n, n * n], dim=-1)
        return h_pre, h_post, h_res.reshape(-1, n, n)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        sort_idx: torch.Tensor,
        inv_sort_idx: torch.Tensor,
        m_splits: list[int],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = hidden_states.shape[0]
        s = hidden_states.reshape(t, self.num_stream, self.hidden_size)

        # Pre-attn mix -> attention -> post/res attn stream update.
        h_pre, h_post_attn, h_res_attn = self._mhc_project(
            s, self.mhc_phi_fused_attn, m_splits
        )
        attn_in = apply_hpre(
            sigmoid_affine(
                h_pre,
                self.mhc_alpha_pre_attn,
                self.mhc_bias_pre_attn,
                self.mhc_matmul_scale,
            ),
            s,
        )
        attn_out = self.attention(
            attn_in, rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens
        )
        s = hyper_connect(
            s,
            attn_out,
            sigmoid_affine(
                h_post_attn,
                self.mhc_alpha_post_attn,
                self.mhc_bias_post_attn,
                self.mhc_matmul_scale,
                sigmoid_scale=2.0,
            ),
            sinkhorn_knopp(
                self.mhc_alpha_res_attn * self.mhc_matmul_scale * h_res_attn
                + self.mhc_bias_res_attn
            ),
        )

        # Pre-mlp mix -> MLP -> post/res mlp stream update.
        h_pre, h_post_mlp, h_res_mlp = self._mhc_project(
            s, self.mhc_phi_fused_mlp, m_splits
        )
        mlp_in = apply_hpre(
            sigmoid_affine(
                h_pre,
                self.mhc_alpha_pre_mlp,
                self.mhc_bias_pre_mlp,
                self.mhc_matmul_scale,
            ),
            s,
        )
        mlp_out = self.mlp(mlp_in, m_splits)
        s = hyper_connect(
            s,
            mlp_out,
            sigmoid_affine(
                h_post_mlp,
                self.mhc_alpha_post_mlp,
                self.mhc_bias_post_mlp,
                self.mhc_matmul_scale,
                sigmoid_scale=2.0,
            ),
            sinkhorn_knopp(
                self.mhc_alpha_res_mlp * self.mhc_matmul_scale * h_res_mlp
                + self.mhc_bias_res_mlp
            ),
        )
        return s.reshape(t, self.num_stream * self.hidden_size)


class TransformerBlock(nn.Module):
    """Stack of MAGI-2-preview layers keyed ``layers.{i}`` for checkpoint keys."""

    def __init__(self, config: "Magi2PreviewModel.Config") -> None:
        super().__init__()
        self.layers = ModuleDict(
            {
                str(i): TransformerLayer(config, i)
                for i in range(config.num_layers)
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        sort_idx: torch.Tensor,
        inv_sort_idx: torch.Tensor,
        m_splits: list[int],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x, rope, sort_idx, inv_sort_idx, m_splits, cu_seqlens)
        return x


class Magi2PreviewModel(Module):
    """MAGI-2-preview: joint video/audio diffusion transformer.

    Packed tokens arrive in original order as ``x (T, max_in_channels)`` with
    channel padding, plus per-token coordinates, modality ids, the sinusoidal
    timestep embedding and varlen ``cu_seqlens``. Tokens are remapped
    TIME->TEXT, modality-sorted for the grouped layers, and the predictions
    are returned in original order as ``(T, 64)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_layers: int = 40
        hidden_size: int = 3072
        head_dim: int = 128
        num_stream: int = 4
        video_in_channels: int = 48
        audio_in_channels: int = 64
        text_in_channels: int = 5120
        time_channel_dim: int = 64
        dense_intermediate_size: int = 8192
        mm_layers: list[int] = field(default_factory=lambda: [0, 1, 38, 39])
        moe_layers: list[int] = field(default_factory=lambda: list(range(2, 38)))
        moe_num_heads: int = 12
        num_experts: int = 256
        moe_top_k: int = 6
        expert_intermediate_size: int = 1280
        shared_expert_intermediate_size: int = 1280
        route_scale: float = 4.9
        sink_token_num: int = 1
        norm_eps: float = 1e-6
        alpha_init: float = 0.01

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            """No-op: MAGI-2-preview has no trainer-runtime config to sync."""
            del trainer_config, kwargs

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
            """Return (total_params, flops_per_token) for MFU calculation."""
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            # Approximate: 6 * nparams (2 for fwd + 4 for bwd).
            flops_per_token = 6.0 * nparams
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        mm_set = set(config.mm_layers)
        moe_set = set(config.moe_layers)
        if mm_set & moe_set:
            raise ValueError(
                f"mm_layers and moe_layers must be disjoint, got overlap "
                f"{sorted(mm_set & moe_set)}"
            )
        out_of_range = sorted(
            i for i in mm_set | moe_set if not 0 <= i < config.num_layers
        )
        if out_of_range:
            raise ValueError(
                f"layer ids {out_of_range} out of range for "
                f"num_layers={config.num_layers}"
            )
        if config.hidden_size % config.head_dim != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"head_dim ({config.head_dim})"
            )
        self.time_channel_dim = config.time_channel_dim
        adapter_dim = config.hidden_size * config.num_stream
        self.pre_adapter = PreAdapter(
            adapter_dim,
            config.video_in_channels,
            config.audio_in_channels,
            config.text_in_channels,
            config.head_dim,
        )
        self.block = TransformerBlock(config)
        self.post_adapter = PostAdapter(
            adapter_dim,
            config.video_in_channels,
            config.audio_in_channels,
            config.norm_eps,
        )

    def verify_module_protocol(self) -> None:
        """Verify model conforms to torchtitan's module protocol."""
        pass

    def init_weights(self, *, buffer_device=None) -> None:
        """Initialize every parameter/buffer; called by Trainer after to_empty.

        Deviation from the official skip-load: the official scheme zeroes the
        MoE W_gate/W_up/W_down tensors (they are overwritten by the
        checkpoint). Here they get ``normal_(std=0.02)`` instead so a freshly
        initialized network is live for smoke training; official runs load
        the checkpoint through the state dict adapter.
        """
        del buffer_device
        init_std = 0.02
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, MultiModalityRMSNorm):
                    # Zero weights give an identity (gain = weight + 1) norm.
                    nn.init.zeros_(module.weight)
                elif isinstance(module, GroupedLinear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, ElementWiseFourierEmbed):
                    module.reset_parameters()
                elif isinstance(module, CoreMultiHeadMoE):
                    nn.init.normal_(module.gate, mean=0.0, std=init_std)
                    nn.init.normal_(module.W_gate, mean=0.0, std=init_std)
                    nn.init.normal_(module.W_up, mean=0.0, std=init_std)
                    nn.init.normal_(module.W_down, mean=0.0, std=init_std)
                    module.router.expert_bias.zero_()
                    module.router.expert_bias_ema.zero_()
                elif isinstance(module, Magi2Attention):
                    nn.init.zeros_(module.sinks)
                elif isinstance(module, TransformerLayer):
                    for name, parameter in module.named_parameters(
                        recurse=False
                    ):
                        if name.startswith("mhc_alpha_"):
                            parameter.fill_(self.config.alpha_init)
                        elif name.startswith(("mhc_bias_", "mhc_phi_fused_")):
                            parameter.zero_()

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor | None = None,
        modality_mapping: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords_mapping is None or modality_mapping is None:
            raise ValueError(
                "magi2_preview forward requires coords_mapping and "
                "modality_mapping"
            )
        # TIME marks tokens carrying the timestep embedding; the official
        # model has no TIME tokens internally, remap to TEXT for parity.
        modality_mapping = modality_mapping.clone()
        modality_mapping[modality_mapping == Modality.TIME] = Modality.TEXT

        sort_idx = torch.argsort(modality_mapping)
        inv_sort_idx = torch.argsort(sort_idx)
        m_splits = [
            int(v)
            for v in torch.bincount(modality_mapping, minlength=3).tolist()
        ]

        video_idx = (modality_mapping == Modality.VIDEO).nonzero().flatten()
        audio_idx = (modality_mapping == Modality.AUDIO).nonzero().flatten()
        text_idx = (modality_mapping == Modality.TEXT).nonzero().flatten()

        x_emb, rope = self.pre_adapter(
            x, coords_mapping, video_idx, audio_idx, text_idx
        )
        if time_embedding is not None:
            x_emb[:, : self.time_channel_dim] = time_embedding.to(x_emb.dtype)

        h = x_emb.index_select(0, sort_idx)
        h = self.block(h, rope, sort_idx, inv_sort_idx, m_splits, cu_seqlens)
        h = h.index_select(0, inv_sort_idx)
        return self.post_adapter(h, video_idx, audio_idx)
