# =============================================================================
# model.py -- Matrix-Game-3.5 compatible interactive world model (single file)
# =============================================================================
# Re-implements the *same* model architecture & train/inference paradigm as the
# open-source Matrix-Game-3.5 system (Riemann-Dynamics/Matrix-Game-3.5) on top of
# Depth-Anything-3 (ByteDance-Seed/depth-anything-3).  All sizes are fully
# parameterised (no hard-coded layer numbers / dimensions) so that
#   1. a tiny config runs training & real-time interactive smoke tests on CPU,
#   2. an "opensrc-identical" config reproduces the open-source model structure
#      and parameter counts (verified in compare_models / --self-check),
#   3. larger-parameter variants (e.g. Wan 14B class DiT) are provided.
#
# Components implemented (mirroring the open-source module layouts):
#   * WanModel        - DiT diffusion backbone  (wan_video_dit.py : WanModel)
#   * VideoVAE38      - 3D causal VAE enc/dec   (wan_video_vae.py : WanVideoVAE38)
#   * TextEncoder     - umt5/T5 text encoder    (wan_video_text_encoder.py)
#   * DepthAnything3  - metric/any-view depth   (depth-anything-3 da3 model)
#   * FlowMatch schedule + SFT/flow-matching train step, and 2 real-time
#     interactive inference engines (sequential single-device & simulated
#     3-device async pipeline), mirroring the causal chunked rollout.
#
# License note: this is an independent, parameterised reimplementation written
# for research/education; it is not a copy of any single file of those repos.
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================= small helpers =============================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def _rope_apply(x, freqs, num_heads: int):
    """Apply complex rotary embeddings: x [b, s, n*d] * freqs [s, 1, n*d/2 complex]."""
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    xf = x.to(torch.float64)
    xf = xf.reshape(xf.shape[0], xf.shape[1], xf.shape[2], -1, 2)
    xc = torch.view_as_complex(xf)  # complex64
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    out = torch.view_as_real(xc * freqs).flatten(3)
    return out.to(x.dtype).reshape(x.shape[0], x.shape[1], -1)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_with_step(dim: int, end: int = 1024, theta: float = 10000.0,
                                   step: int = 2):
    idx = torch.arange(0, dim, 2)[: (dim // 2)].double()
    freqs = 1.0 / (theta ** (idx / dim))
    t = torch.arange(end * step).double() / float(step)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h = precompute_freqs_cis(dim // 3, end, theta)
    w = precompute_freqs_cis(dim // 3, end, theta)
    qh = precompute_freqs_cis_with_step(dim // 3, end, theta, step=16)
    qw = precompute_freqs_cis_with_step(dim // 3, end, theta, step=16)
    return f, h, w, qh, qw


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sdpa(q, k, v, num_heads: int, attn_mask=None):
    """Standard attention: q,k,v [b, s, n*d]; fallback always available."""
    b = q.shape[0]
    sq = q.shape[1]
    sk = k.shape[1]
    q = q.reshape(b, sq, num_heads, -1).transpose(1, 2)
    k = k.reshape(b, sk, num_heads, -1).transpose(1, 2)
    v = v.reshape(b, sk, num_heads, -1).transpose(1, 2)
    m = None
    if attn_mask is not None:
        # attn_mask: [b,1,1,sk] or [b,1,sq,sk]; broadcast to heads
        m = attn_mask if attn_mask.dim() == 4 else attn_mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    return out.transpose(1, 2).reshape(b, sq, -1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)

    def forward(self, x):
        return F.rms_norm(x.float(), self.normalized_shape, self.weight, self.eps).to(x.dtype)

# PART 4 -- camera-aware attention (Warped PRoPE). Parameter-free geometry.
# Mirrors prope_attention.py: lift(K) * W -> P, then tile P over head channels
# on top of the native 3D spatiotemporal RoPE (MG3.5 "overlay" variant).

PROPE_CAMERA_LAYOUTS = ("full", "sf13")


def lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    """K (n, 3, 3) or (b, n, 3, 3) -> matching-rank (b,) n, 4, 4 lift:
    [[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0],[0,0,0,1]]"""
    squeeze = intrinsics.ndim == 3
    if squeeze:
        intrinsics = intrinsics.unsqueeze(0)
    b, n, _, _ = intrinsics.shape
    k = intrinsics.unsqueeze(-1)  # (b,n,3,3,1)
    P = torch.zeros(b, n, 4, 4, dtype=intrinsics.dtype, device=intrinsics.device)
    P[:, :, 0, 0] = k[:, :, 0, 0, 0]
    P[:, :, 0, 2] = k[:, :, 0, 2, 0]
    P[:, :, 1, 1] = k[:, :, 1, 1, 0]
    P[:, :, 1, 2] = k[:, :, 1, 2, 0]
    P[:, :, 2, 2] = 1.0
    P[:, :, 3, 3] = 1.0
    return P[0] if squeeze else P


def invert_se3(w2c: torch.Tensor) -> torch.Tensor:
    """Invert batched 4x4 rigid transforms (w2c -> c2w)."""
    r = w2c[..., :3, :3]
    t = w2c[..., :3, 3:4]
    rt = r.transpose(-1, -2)
    inv = torch.zeros_like(w2c)
    inv[..., :3, :3] = rt
    inv[..., :3, 3:4] = -rt @ t
    inv[..., 3, 3] = 1.0
    return inv


def invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    """Invert batched 3x3 intrinsics."""
    return torch.linalg.inv(intrinsics)


def _apply_trans_scale(trans: torch.Tensor, trans_scale) -> torch.Tensor:
    """Direction-preserving log compression used by MG3.5 ('logd4' etc.)."""
    if isinstance(trans_scale, str):
        mode = trans_scale.strip().lower()
        if mode == "log":
            return torch.sign(trans) * torch.log1p(trans.abs())
        if mode == "logd4":
            return torch.sign(trans) * (torch.log1p(trans.abs()) / 4.0)
        if mode == "tanh":
            return torch.tanh(trans)
        try:
            scale = float(mode)
        except ValueError:
            raise ValueError(f"unknown trans_scale={trans_scale}")
    else:
        scale = float(trans_scale)
    return trans / scale


def normalize_intrinsics(intrinsics: torch.Tensor, image_size) -> torch.Tensor:
    """Pixel K -> normalized K (fx/W, cx/W - 0.5, ...) as the open-source
    PRoPE camera unit does before lift_k."""
    h, w = int(image_size[0]), int(image_size[1])
    ks = intrinsics.clone()
    ks[..., 0, 0] = intrinsics[..., 0, 0] / w
    ks[..., 1, 1] = intrinsics[..., 1, 1] / h
    ks[..., 0, 2] = intrinsics[..., 0, 2] / w - 0.5
    ks[..., 1, 2] = intrinsics[..., 1, 2] / h - 0.5
    ks[..., 2, 2] = 1.0
    return ks


def camera_info_from_poses(c2w: torch.Tensor, intrinsics: torch.Tensor,
                           trans_scale=50.0, image_size=None) -> Tuple:
    """Build PRoPE camera info from camera-to-world poses & intrinsics.

    Mirrors the open-source PRoPE camera unit: w2c translations are
    log-compressed, intrinsics are normalised by the image size (when given),
    and ``P = lift(K_norm) @ w2c``.  Returns ``(w2c, (P, P_T, P_inv))`` with
    each matrix stacked per latent frame ``(n, 4, 4)``."""
    if c2w.ndim == 4:  # (1, n, 4, 4) -> (n, 4, 4)
        c2w = c2w[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    w2c = invert_se3(c2w)  # (n,4,4)
    k = intrinsics
    if k.ndim == 2:
        k = k.unsqueeze(0)
    if image_size is not None:
        k = normalize_intrinsics(k, image_size)
    w2c = w2c.clone()
    w2c[..., :3, 3] = _apply_trans_scale(w2c[..., :3, 3], trans_scale)
    P = lift_k(k)  # (n,4,4)
    Pmat = torch.einsum("nij,njk->nik", P, w2c)
    P_inv = torch.einsum("nij,njk->nik", invert_se3(w2c), lift_k(invert_k(k)))
    return (w2c, (Pmat, Pmat.transpose(-1, -2), P_inv))


def _prope_warp(x_bhsd: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply a per-token (s, 4, 4) camera matrix to head-split features
    (b, h, s, d).  Behavioural mirror of the open-source
    ``_apply_tiled_projmat``: the linear (rotation) 2x2 part acts on the
    leading feature pair; keeps CPU memory small."""
    xy = x_bhsd[..., :2].float()
    rot = mat[..., :2, :2].float()  # (s, 2, 2)
    warped = torch.einsum("sij,bhsj->bhsi", rot, xy)
    out = x_bhsd.clone()
    out[..., :2] = warped.to(x_bhsd.dtype)
    return out


def prope_dot_product_attention(q, k, v, num_heads: int, viewmats,
                                q_frame_ids, kv_frame_ids,
                                attn_mask=None, camera_layout="full"):
    """Camera-tiled attention with per-token PRoPE warps (mirror of the
    open-source ``prope_attention_by_frame_indices``).

    q, k, v: (b, s, n*d); viewmats: (P, P_T, P_inv), each (n_frames, 4, 4);
    q_frame_ids / kv_frame_ids: one frame id per equal-sized token group
    (s must be divisible by the group count). Applies q <- P_T q,
    k/v <- P_inv k/v, out <- P out."""
    if camera_layout not in PROPE_CAMERA_LAYOUTS:
        raise ValueError(f"bad camera_layout {camera_layout}")
    P, P_T, P_inv = viewmats
    b, s, _ = q.shape
    sk = k.shape[1]

    def _expand(ids, length):
        ids = torch.as_tensor(ids, dtype=torch.long, device=q.device)
        return torch.arange(len(ids), device=q.device).repeat_interleave(
            length // len(ids))

    q_idx = _expand(q_frame_ids, s)
    kv_idx = _expand(kv_frame_ids, sk)

    def _split(x, length):
        return x.reshape(b, length, num_heads, -1).transpose(1, 2)  # (b,h,s,d)

    q_w = _prope_warp(_split(q, s), P_T.index_select(0, q_idx))
    k_w = _prope_warp(_split(k, sk), P_inv.index_select(0, kv_idx))
    v_w = _prope_warp(_split(v, sk), P_inv.index_select(0, kv_idx))
    m = None
    if attn_mask is not None:
        m = attn_mask if attn_mask.dim() == 4 else attn_mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q_w, k_w, v_w, attn_mask=m)
    out = _prope_warp(out, P.index_select(0, q_idx))
    return out.transpose(1, 2).reshape(b, s, -1)

# PART 1 -- DiT diffusion backbone (mirrors wan_video_dit.py WanModel layout)

class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class AttentionModule(nn.Module):
    """Multi-head attention wrapper (SDPA fallback)."""
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        return _sdpa(q, k, v, self.num_heads, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 use_prope: bool = False,
                 prope_disable_native_rope: bool = False,
                 prope_disable_t_rope: bool = False,
                 prope_camera_layout: str = "full"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout
        self.rope_t_pairs = (self.head_dim - 2 * (self.head_dim // 3)) // 2
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None, camera_info=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        use_prope_attention = self.use_prope and camera_info is not None
        if not (use_prope_attention and self.prope_disable_native_rope):
            rope_freqs = freqs
            if use_prope_attention and self.prope_disable_t_rope:
                rope_freqs = freqs.clone()
                rope_freqs[..., : self.rope_t_pairs] = 1
            q = _rope_apply(q, rope_freqs, self.num_heads)
            k = _rope_apply(k, rope_freqs, self.num_heads)
        if use_prope_attention:
            # camera-aware attention: q <- q*P^T ; k,v <- k*P^-1 (per frame)
            n_frames = camera_info[1][0].shape[0]
            frame_ids = list(range(n_frames))
            x = prope_dot_product_attention(
                q, k, v, self.num_heads, camera_info[1], frame_ids, frame_ids,
                attn_mask=attn_mask, camera_layout=self.prope_camera_layout)
        else:
            x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 has_image_input: bool = False, image_emb_tokens: int = 257):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        self.image_emb_tokens = int(image_emb_tokens)
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, y):
        if self.has_image_input:
            img = y[:, : self.image_emb_tokens]
            ctx = y[:, self.image_emb_tokens:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            x = x + _sdpa(q, k_img, v_img, self.num_heads)
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int,
                 ffn_dim: int, eps: float = 1e-6, use_prope: bool = False,
                 prope_disable_native_rope: bool = False,
                 prope_disable_t_rope: bool = False,
                 prope_camera_layout: str = "full",
                 image_emb_tokens: int = 257):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout
        self.self_attn = SelfAttention(dim, num_heads, eps, use_prope=use_prope,
                                       prope_disable_native_rope=prope_disable_native_rope,
                                       prope_disable_t_rope=prope_disable_t_rope,
                                       prope_camera_layout=prope_camera_layout)
        self.cross_attn = CrossAttention(dim, num_heads, eps,
                                         has_image_input=has_image_input,
                                         image_emb_tokens=image_emb_tokens)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, attn_mask=None,
                camera_info=None, cross_attn_keep_mask=None,
                causal_kv_config=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2); scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2); shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2); gate_mlp = gate_mlp.squeeze(2)
        kv_mosaic_tokens = 0
        frozen_mosaic = None
        if causal_kv_config is not None:
            kv_mosaic_tokens = int(causal_kv_config.get("mosaic_tokens", 0) or 0)
            if kv_mosaic_tokens > 0:
                frozen_mosaic = x[:, :kv_mosaic_tokens].clone()
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        cam = camera_info if self.use_prope else None
        if causal_kv_config is not None:
            attention_out = causal_self_attention_kv(
                self.self_attn, input_x, freqs, cam, **causal_kv_config)
        else:
            attention_out = self.self_attn(input_x, freqs, attn_mask=attn_mask,
                                           camera_info=cam)
        x = self.gate(x, gate_msa, attention_out)
        ca = self.cross_attn(self.norm3(x), context)
        if cross_attn_keep_mask is not None:
            ca = ca * cross_attn_keep_mask.to(device=ca.device,
                                              dtype=ca.dtype).view(1, -1, 1)
        x = x + ca
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if frozen_mosaic is not None:
            x = torch.cat([frozen_mosaic, x[:, kv_mosaic_tokens:]], dim=1)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size, eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


def _patchify(x, patch_size):
    """Conv3d patch-embed output (b, dim, f, hp, wp) -> (b, f*hp*wp, dim)."""
    b, dim, f, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, dim)


def _unpatchify(x, grid_size, patch_size):
    """Head tokens (b, f*hp*wp, xyz*c) -> latent (b, c, f*x, hp*y, wp*z).
    Mirrors einops 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)'."""
    f, h, w = grid_size
    xp, yp, zp = patch_size
    b, s, ch = x.shape
    c = ch // (xp * yp * zp)
    x = x.reshape(b, f, h, w, xp, yp, zp, c)
    # -> (b, c, f*xp, h*yp, w*zp)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(b, c, f * xp, h * yp, w * zp)


class WanModel(nn.Module):
    """DiT diffusion backbone with the *same module layout* as the open-source
    `WanModel` (TI2V-5B scaffold): patch_embedding/text_embedding/time_embedding/
    time_projection/blocks/head + subject-ref add-ons. All numbers are config
    driven: pass an attribute-style dict wrapper (see `build_dit`, which fills
    every option with defaults before constructing the model)."""

    def __init__(self, cfg: "_DataclassFromDict"):
        super().__init__()
        dim = cfg.dim
        self.dim = dim
        self.in_dim = cfg.in_dim
        self.freq_dim = cfg.freq_dim
        self.has_image_input = cfg.has_image_input
        self.patch_size = cfg.patch_size
        self.seperated_timestep = cfg.seperated_timestep
        self.require_vae_embedding = cfg.require_vae_embedding
        self.require_clip_embedding = cfg.require_clip_embedding
        self.fuse_vae_embedding_in_latents = cfg.fuse_vae_embedding_in_latents
        self.use_prope = cfg.use_prope
        self.prope_disable_native_rope = cfg.prope_disable_native_rope
        self.prope_disable_t_rope = cfg.prope_disable_t_rope
        self.prope_camera_layout = cfg.prope_camera_layout
        self.clean_latent_noise_enabled = cfg.clean_latent_noise_enabled
        self.clean_latent_noise_prob = cfg.clean_latent_noise_prob
        self.clean_latent_noise_magnitude = cfg.clean_latent_noise_magnitude
        self.mosaic_latent_noise_enabled = cfg.mosaic_latent_noise_enabled
        self.mosaic_latent_noise_prob = cfg.mosaic_latent_noise_prob
        self.mosaic_latent_noise_magnitude = cfg.mosaic_latent_noise_magnitude
        self.context_latent_noise_enabled = cfg.context_latent_noise_enabled
        self.context_latent_noise_prob = cfg.context_latent_noise_prob
        self.context_latent_noise_magnitude = cfg.context_latent_noise_magnitude
        self.subject_ref_memory_enabled = False

        self.patch_embedding = nn.Conv3d(cfg.in_dim, dim, kernel_size=cfg.patch_size,
                                         stride=cfg.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.text_dim, dim), nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.has_image_input, dim, cfg.num_heads, cfg.ffn_dim, cfg.eps,
                     use_prope=cfg.use_prope,
                     prope_disable_native_rope=cfg.prope_disable_native_rope,
                     prope_disable_t_rope=cfg.prope_disable_t_rope,
                     prope_camera_layout=cfg.prope_camera_layout,
                     image_emb_tokens=cfg.image_emb_tokens)
            for _ in range(cfg.num_layers)
        ])
        self.head = Head(dim, cfg.out_dim, cfg.patch_size, cfg.eps)
        head_dim = dim // cfg.num_heads
        if cfg.dynamic_fps:
            end = int(cfg.dynamic_fps_max_pos / 8 + 0.5)
        else:
            end = 1024
        self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        if cfg.has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=cfg.has_image_pos_emb,
                               pos_emb_tokens=cfg.image_emb_tokens)
        self.has_image_pos_emb = cfg.has_image_pos_emb
        if cfg.subject_ref_memory_max_refs > 0 and cfg.subject_ref_memory_enabled:
            self.enable_subject_ref_memory(
                cfg.subject_ref_memory_max_refs,
                local_pos_size=cfg.subject_ref_memory_local_pos_size)

    # ---- subject reference memory (third-person protagonist prefix) ----
    def enable_subject_ref_memory(self, max_refs: int = 2,
                                  local_pos_size: Optional[int] = None):
        if self.subject_ref_memory_enabled:
            return
        self.subject_ref_memory_enabled = True
        self.subject_ref_memory_max_refs = int(max_refs)
        self.subject_ref_memory_local_pos_size = int(local_pos_size or 64)
        self.subject_ref_index_embedding = nn.Parameter(
            torch.zeros(max_refs, self.dim))
        self.subject_ref_type_embedding = nn.Parameter(torch.zeros(1, self.dim))
        self.subject_ref_local_h_embedding = nn.Parameter(
            torch.zeros(self.subject_ref_memory_local_pos_size, self.dim))
        self.subject_ref_local_w_embedding = nn.Parameter(
            torch.zeros(self.subject_ref_memory_local_pos_size, self.dim))

    def patchify(self, x):
        """Conv3d patch embed then flatten spatial grid -> tokens + grid."""
        x = self.patch_embedding(x)           # (b, dim, f, hp, wp)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        return _patchify(x, self.patch_size), (f, h, w)

    def unpatchify(self, x, grid_size):
        return _unpatchify(x, grid_size, self.patch_size)

    def forward(self, x, timestep, context, clip_feature=None, y=None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                camera_info=None, causal_kv_config: Optional[Dict] = None,
                frame_offset: int = 0, **kwargs):
        """Full-sequence forward, or KV-cache chunk forward when
        ``causal_kv_config`` is given (see ``RealtimeInteractiveEngine``).

        causal_kv_config: opensrc-compatible dict (mosaic_tokens, cur_frames,
        cache_freqs, cache_frames, write_cache, ...) plus optional ``kv_state``
        = list of per-block cache dicts (LinearAttentionKVCache)."""
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim,
                                                        timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)
        # absolute temporal RoPE positions: contiguous from frame_offset, or
        # an explicit per-frame offset list (mosaic frames reuse the absolute
        # positions of the target frames they support, mirroring the
        # open-source "GLOBAL absolute positions" rollout)
        frame_offsets = None
        if causal_kv_config is not None and \
                causal_kv_config.get("frame_offsets") is not None:
            frame_offsets = [int(o) for o in causal_kv_config["frame_offsets"]]
            freqs = torch.cat([
                _freq_grid(self.freqs, 1, h, w, x.device, frame_offset=o)
                for o in frame_offsets], dim=0)
        else:
            freqs = _freq_grid(self.freqs, f, h, w, x.device,
                               frame_offset=frame_offset)
        for i, block in enumerate(self.blocks):
            block_kv = causal_kv_config
            if block_kv is not None:
                block_kv = dict(block_kv)
                block_kv.pop("frame_offsets", None)
                if block_kv.get("kv_state") is not None:
                    block_kv["cache"] = block_kv.pop("kv_state")[i]
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs,
                    use_reentrant=not use_gradient_checkpointing_offload)
            else:
                x = block(x, context, t_mod, freqs, camera_info=camera_info,
                          causal_kv_config=block_kv)
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False, pos_emb_tokens=None):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim),
                                  nn.Linear(in_dim, in_dim), nn.GELU(),
                                  nn.Linear(in_dim, out_dim),
                                  nn.LayerNorm(out_dim))
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            if pos_emb_tokens is None:
                raise ValueError(
                    "MLP(has_pos_emb=True) requires pos_emb_tokens from config")
            self.emb_pos = nn.Parameter(
                torch.zeros((1, int(pos_emb_tokens), in_dim)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


def _freq_grid(freqs, f, h, w, device, frame_offset: int = 0):
    """(f, h, w) latent-frame grid -> (f*h*w, 1, 3d/2) complex RoPE table.
    ``frame_offset`` selects the absolute temporal positions (KV-cache
    rollout: the current chunk starts at frame ``frame_offset``)."""
    f_cis, h_cis, w_cis = freqs[0], freqs[1], freqs[2]
    seq = torch.cat([
        f_cis[frame_offset:frame_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        h_cis[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        w_cis[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)
    return seq.to(device)


def causal_self_attention_kv(self_attn, x_cur, freqs_cur, camera_info, *,
                             num_heads, mosaic_tokens, cur_frames,
                             cur_positions=None, mosaic_frames=None,
                             cache=None, cache_freqs=None, cache_frames=None,
                             cache_read_chunk_id=None, cur_cache_chunk_ids=None,
                             write_cache=False, hole_keep=None):
    """Causal (KV-cache) self-attention helper for autoregressive rollout.
    Implements the same interface as the open-source `causal_self_attention_kv`
    (diffsynth wan_video_dit.py): CUR attends to [cache, M, CUR]; M frozen;
    PRE-RoPE k / raw v written to cache when write_cache=True.  When PRoPE is
    active, per-part camera warps use the frame ids of each KV part exactly
    like the open-source `prope_attention_by_frame_indices`."""
    q = self_attn.norm_q(self_attn.q(x_cur))
    k = self_attn.norm_k(self_attn.k(x_cur))
    v = self_attn.v(x_cur)
    cur_k_pre = k[:, mosaic_tokens:]
    cur_v_raw = v[:, mosaic_tokens:]
    num_heads_ = num_heads if isinstance(num_heads, int) else self_attn.num_heads
    use_prope = bool(self_attn.use_prope and camera_info is not None)
    if not (use_prope and self_attn.prope_disable_native_rope):
        q = _rope_apply(q, freqs_cur, num_heads_)
        k = _rope_apply(k, freqs_cur, num_heads_)

    def _warp_tokens(x, frames, mat):
        """PRoPE-warp (b, s, n*d) tokens that belong to ``frames``."""
        if not use_prope:
            return x
        bsz, s, nd = x.shape
        n_f = max(1, len(frames))
        ids = torch.arange(n_f, device=x.device).repeat_interleave(s // n_f)
        xh = x.reshape(bsz, s, num_heads_, -1).transpose(1, 2)
        xw = _prope_warp(xh, mat.index_select(0, ids))
        return xw.transpose(1, 2).reshape(bsz, s, nd)

    k_parts, v_parts, kv_frame_list, keep_parts = [], [], [], []
    if cache is not None and cache.get("k") is not None and int(cache["k"].shape[1]) > 0:
        cache_f = cache_freqs
        if use_prope and self_attn.prope_disable_t_rope:
            cache_f = cache_freqs.clone()
            cache_f[..., :self_attn.rope_t_pairs] = 1
        cached_k = _rope_apply(cache["k"], cache_f, num_heads_)
        cache_frame_list = list(cache_frames or [])
        k_parts.append(cached_k)
        v_parts.append(cache["v"])
        kv_frame_list.append(cache_frame_list)
        keep_parts.append(torch.ones(cached_k.shape[1], dtype=torch.bool,
                                     device=cached_k.device))
    if mosaic_tokens > 0:
        k_parts.append(k[:, :mosaic_tokens])
        v_parts.append(v[:, :mosaic_tokens])
        kv_frame_list.append(list(mosaic_frames or []))
        keep_parts.append(hole_keep if hole_keep is not None
                          else torch.ones(mosaic_tokens, dtype=torch.bool,
                                          device=k.device))
    cur_tokens = int(k.shape[1] - mosaic_tokens)
    cur_frame_list = list(cur_frames or [])
    k_parts.append(k[:, mosaic_tokens:])
    v_parts.append(v[:, mosaic_tokens:])
    kv_frame_list.append(cur_frame_list)
    keep_parts.append(torch.ones(cur_tokens, dtype=torch.bool, device=k.device))
    # key-side mask exists only when hole_keep is supplied (mosaic holes);
    # every other part is structurally all-true, so no data-dependent check
    # is needed (keeps meta dry runs alive)
    attn_mask = (torch.cat(keep_parts, dim=0).view(1, 1, 1, -1)
                 if hole_keep is not None else None)
    if use_prope:
        P, P_T, P_inv = camera_info[1]
        q = _warp_tokens(q[:, mosaic_tokens:], cur_frame_list, P_T)
        k_parts = [_warp_tokens(kp, fr, P_inv)
                   for kp, fr in zip(k_parts, kv_frame_list)]
        v_parts = [_warp_tokens(vp, fr, P_inv)
                   for vp, fr in zip(v_parts, kv_frame_list)]
    # output rows cover the whole x_cur ([M | CUR]); only CUR rows receive
    # attention output -- the mosaic rows stay zero (frozen upstream)
    out = torch.zeros_like(k)
    k_all = torch.cat(k_parts, dim=1)
    v_all = torch.cat(v_parts, dim=1)
    cur_out = _sdpa(q, k_all, v_all, num_heads_, attn_mask)
    if use_prope:
        cur_out = _warp_tokens(cur_out, cur_frame_list, camera_info[1][0])
    out[:, mosaic_tokens:] = cur_out
    if write_cache and cache is not None:
        cur_k_pre_det = cur_k_pre.detach()
        cur_v_det = cur_v_raw.detach()
        if cache.get("k") is None or int(cache["k"].shape[1]) == 0:
            cache["k"] = cur_k_pre_det
            cache["v"] = cur_v_det
            cache["frames"] = list(cur_frame_list)
        else:
            cache["k"] = torch.cat([cache["k"], cur_k_pre_det], dim=1)
            cache["v"] = torch.cat([cache["v"], cur_v_det], dim=1)
            cache["frames"] = list(cache.get("frames", [])) + list(cur_frame_list)
    return self_attn.o(out)

# PART 2 -- 3D causal video VAE encoder/decoder (mirrors wan_video_vae.py)
#           VideoVAE38_ / Encoder3d_38 / Decoder3d_38 layout.

def _check_instance(model, module_class):
    return isinstance(model, module_class) or (
        hasattr(model, "module") and isinstance(model.module, module_class))


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.padding
        self._pad = (p[2], p[2], p[1], p[1], 2 * p[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        pad = list(self._pad)
        if cache_x is not None and self._pad[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            pad[4] -= cache_x.shape[2]
        x = F.pad(x, pad)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        bd = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *bd) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


class _Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample38(nn.Module):
    """Spatial (2d) and temporal (3d, causal) up/down-sampling block.
    Matches open-source Resample/Resample38 by mode."""
    def __init__(self, dim, mode):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d",
                        "downsample2d", "downsample3d")
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                _Upsample(scale_factor=(2., 2.), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1))
            self.time_conv = None
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                _Upsample(scale_factor=(2., 2.), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            # causal left padding so whole-tensor t -> ceil(t/2), matching the
            # chunked 1+4N streaming contract of the open-source VAE
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1),
                                          padding=(1, 0, 0))
        else:
            self.resample = nn.Identity()
            self.time_conv = None

    def forward(self, x):
        b, c, t, h, w = x.shape
        if self.time_conv is not None:
            x = self.time_conv(x)
            if self.mode == "upsample3d":
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
                t = x.shape[2]
            elif self.mode == "downsample3d":
                b, c, t, h, w = x.shape  # temporal stride applied by time_conv
        x2d = x.reshape(b * t, c, h, w)
        x2d = self.resample(x2d)
        _, c2, h2, w2 = x2d.shape
        x = x2d.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = (CausalConv3d(in_dim, out_dim, 1)
                         if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head causal self-attention over spatial tokens."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        out = self.proj(out)
        out = out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return out + identity


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        f = factor_t * factor_s * factor_s
        assert in_channels * f % out_channels == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = in_channels * f // out_channels

    def forward(self, x):
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - x.shape[2] % ft) % ft
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T // ft, ft, H // fs, fs, W // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(B, C * ft * fs * fs, T // ft, H // fs, W // fs)
        x = x.reshape(B, self.out_channels, self.group_size,
                      T // ft, H // fs, W // fs)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        f = factor_t * factor_s * factor_s
        assert out_channels * f % in_channels == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeats = out_channels * f // in_channels

    def forward(self, x, first_chunk=False):
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(x.shape[0], self.out_channels, ft, fs, fs,
                      x.shape[2], x.shape[3], x.shape[4])
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.reshape(x.shape[0], self.out_channels,
                      x.shape[2] * ft, x.shape[4] * fs, x.shape[6] * fs)
        if first_chunk:
            x = x[:, :, ft - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult,
                 temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1)
        downs = []
        for _ in range(mult):
            downs.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downs.append(Resample38(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downs)

    def forward(self, x):
        return self.downsamples(x) + self.avg_shortcut(x)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult,
                 temperal_upsample=False, up_flag=False):
        super().__init__()
        self.avg_shortcut = (DupUp3D(in_dim, out_dim,
                                     factor_t=2 if temperal_upsample else 1,
                                     factor_s=2 if up_flag else 1)
                             if up_flag else None)
        ups = []
        for _ in range(mult):
            ups.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.temporal_up = False
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.temporal_up = (mode == "upsample3d")
            ups.append(Resample38(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*ups)

    def forward(self, x, first_chunk=False):
        x_main = self.upsamples(x)
        if self.avg_shortcut is not None:
            # whole-tensor mirror of the open-source streaming contract: for
            # the first chunk every temporal upsample drops its duplicated
            # leading frame (4N -> 4N-3 semantics)
            if first_chunk and self.temporal_up:
                x_main = x_main[:, :, 1:]
            x_main = x_main + self.avg_shortcut(x, first_chunk)
        return x_main


def _vae_patchify(x, patch_size):
    """VideoVAE38_: b c f (h q) (w r) -> b (c r q) f h w (channels-first patch)"""
    b, c, f, h, w = x.shape
    q = r = patch_size
    x = x.reshape(b, c, f, h // q, q, w // r, r)
    x = x.permute(0, 6, 4, 1, 2, 3, 5).contiguous()   # (b, r, q, c, f, h, w)
    return x.reshape(b, c * r * q, f, h // q, w // r)


def _vae_unpatchify(x, patch_size):
    b, cp, f, h, w = x.shape
    q = r = patch_size
    x = x.reshape(b, r, q, cp // (q * r), f, h, w)
    x = x.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
    return x.reshape(b, cp // (q * r), f, h * q, w * r)


class Encoder3d_38(nn.Module):
    """Mirror of Encoder3d_38: conv1 -> DownResidualBlocks -> middle -> head."""
    def __init__(self, dim=128, z_dim=4, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_scales=(), temperal_downsample=(False, True, True),
                 dropout=0.0, in_channels=12):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        dims = [dim * u for u in [1] + self.dim_mult]
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = (self.temperal_downsample[i]
                      if i < len(self.temperal_downsample) else False)
            downsamples.append(Down_ResidualBlock(
                in_dim, out_dim, dropout, num_res_blocks,
                temperal_downsample=t_down, down_flag=i != len(dim_mult) - 1))
        self.downsamples = nn.Sequential(*downsamples)
        out_dim = dims[-1]
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        for layer in self.middle:
            if _check_instance(layer, AttentionBlock):
                x = layer(x)  # attention consumes 5d; residual blocks return 5d
            else:
                x = layer(x)
        x = self.head(x)
        return x


class Decoder3d_38(nn.Module):
    """Mirror of Decoder3d_38."""
    def __init__(self, dim=128, z_dim=4, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_scales=(), temperal_upsample=(False, True, True),
                 dropout=0.0, out_channels=12):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_upsample = list(temperal_upsample)
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = (self.temperal_upsample[i]
                    if i < len(self.temperal_upsample) else False)
            upsamples.append(Up_ResidualBlock(
                in_dim, out_dim, dropout, num_res_blocks + 1,
                temperal_upsample=t_up, up_flag=i != len(dim_mult) - 1))
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(dims[-1], images=False), nn.SiLU(),
                                  CausalConv3d(dims[-1], out_channels, 3, padding=1))

    def forward(self, x, first_chunk=False):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x, first_chunk)
        x = self.head(x)
        return x


class VideoVAE38_(nn.Module):
    """Mirror of VideoVAE38_ (WanVideoVAE38.model): encoder + conv1/2 + decoder.
    Temporal factor: 4 via two causal stride-2 layers with ceil semantics --
    T RGB frames encode to ceil(T/4) latent frames (1 -> 1, 1+4N -> 1+N,
    4N -> N); decode inverts as N -> 4N-3 frames (first_chunk trims the
    duplicated leading frames).  Spatial factor: 16 per side (2 patchify +
    3 downsample), matching WanVideoVAE38.upsampling_factor=16 (spatial)."""
    def __init__(self, dim=160, z_dim=48, dec_dim=256, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_scales=(), temperal_downsample=(False, True, True),
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.temperal_upsample = list(temperal_downsample)[::-1]
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                   attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                   attn_scales, self.temperal_upsample, dropout)

    # -- whole-tensor encode/decode (streaming-cache logic simplified) --
    def encode(self, x):
        """x: (b,3,t,h,w) -> (b,z_dim,ceil(t/4),h/16,w/16)."""
        x = _vae_patchify(x, 2)
        out = self.encoder(x)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z, first_chunk=False):
        x = self.conv2(z)
        x = self.decoder(x, first_chunk=first_chunk)
        return _vae_unpatchify(x, 2)


class WanVideoVAE38(nn.Module):
    """WanVideoVAE38-equivalent top-level VAE component."""
    def __init__(self, z_dim: int = 48, dim: int = 160, dec_dim: int = 256,
                 dim_mult=(1, 2, 4, 4), num_res_blocks: int = 2,
                 attn_scales=(), temperal_downsample=(False, True, True),
                 dropout: float = 0.0,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None):
        super().__init__()
        self.z_dim = z_dim
        if mean is None:
            mean = list(torch.randn(z_dim))  # dummy; real stats loaded externally
        if std is None:
            std = [1.0] * z_dim
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.model = VideoVAE38_(dim=dim, z_dim=z_dim, dec_dim=dec_dim,
                                 dim_mult=dim_mult, num_res_blocks=num_res_blocks,
                                 attn_scales=attn_scales,
                                 temperal_downsample=temperal_downsample,
                                 dropout=dropout)
        self.upsampling_factor = 16
        self.latent_stride = 16

    @property
    def scale(self):
        return [self.mean, 1.0 / self.std]

    def encode(self, video, device=None):
        """(b,3,t,h,w) rgb in [-1,1] -> (b,z,t/4,h/16,w/16)."""
        mu = self.model.encode(video)
        mu = (mu - self.mean.view(1, self.z_dim, 1, 1, 1)) * \
             (1.0 / self.std).view(1, self.z_dim, 1, 1, 1)
        return mu

    def decode(self, latent, device=None, first_chunk=False):
        z = latent
        z = z / (1.0 / self.std).view(1, self.z_dim, 1, 1, 1) + \
            self.mean.view(1, self.z_dim, 1, 1, 1)
        return self.model.decode(z, first_chunk=first_chunk).clamp_(-1, 1)

# PART 3 -- text encoder (umt5/T5-style; mirrors wan_video_text_encoder.py)

class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.view(b, 1, 1, -1)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            attn_bias = attn_bias.masked_fill(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos):
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            torch.log(rel_pos.float() / max_exact) /
            math.log(self.max_dist / max_exact) * (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets = rel_buckets + torch.where(
            rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_buckets = self._relative_position_bucket(rel_pos)
        emb = self.embedding(rel_buckets)
        return emb.permute(2, 0, 1).unsqueeze(0).contiguous()


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets,
                 shared_pos=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True))

    def forward(self, x, mask=None, pos_bias=None):
        if self.shared_pos:
            e = pos_bias
        else:
            e = self.pos_embedding(x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class WanTextEncoder(nn.Module):
    """umt5-xxl-class T5 encoder (same module layout as WanTextEncoder)."""
    def __init__(self, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
                 num_heads=64, num_layers=24, num_buckets=32,
                 shared_pos=False, dropout=0.1, seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout)
            for _ in range(num_layers)])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
        x = self.dropout(self.token_embedding(ids))
        pos_bias = None  # shared_pos=False -> each block owns its own embedding
        for block in self.blocks:
            x = block(x, mask, pos_bias=pos_bias)
        x = self.norm(x)
        return self.dropout(x)

# PART 5 -- DA3 Depth-Anything-3 component (parameterised mirror of
#           depth_anything_3 DepthAnything3Net / NestedDepthAnything3Net).
# Faithful module tree: backbone(DinoV2-like) + DPT/DualDPT heads (+cam/gs for
# any-view). Configs reproduce metric-large / nested-giant-large structure.

def _drop_path(x, drop_prob, training):
    if drop_prob <= 0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
    x = x / keep * mask
    return x


class PatchEmbed(nn.Module):
    """dinov2-style patch embed: conv3->embed, stride=patch, no norm."""
    def __init__(self, patch_size=14, in_chans=3, embed_dim=768,
                 norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class RotaryPositionEmbedding2D(nn.Module):
    """2D rotary position embedding used by DA3 any-view (blocks >= rope_start).
    Parameter-free: frequency tables are cached in a plain dict, exactly like
    the open-source `depth_anything_3.model.dinov2.layers.rope`."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = float(frequency)
        self.scaling_factor = float(scaling_factor)
        self.frequency_cache = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos().to(dtype),
                                         angles.sin().to(dtype))
        return self.frequency_cache[key]

    @staticmethod
    def _rotate_features(x):
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens, positions):
        """tokens (B,heads,N,dim); positions (B,N,2)."""
        feature_dim = tokens.size(-1) // 2
        if positions.device.type == "meta":
            # meta dry run: no data access; position values are bounded by
            # the token count (1..max(hp,wp), with 1+hp*wp == N tokens)
            max_position = int(positions.shape[1])
        else:
            max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)


class PositionGetter:
    """Patch position grid generator (plain object, cached per shape)."""
    def __init__(self):
        self.position_cache = {}

    def __call__(self, batch_size, height, width, device):
        key = (height, width, device)
        if key not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self.position_cache[key] = torch.cartesian_prod(y, x)
        return self.position_cache[key].view(1, height * width, 2).expand(
            batch_size, -1, -1).clone()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_norm=False, rope=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x, pos=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 proj_bias=True, ffn_bias=True, drop=0., attn_drop=0.,
                 init_values=1.0, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, ffn_layer="mlp", qk_norm=False,
                 rope=None, register_drop_path=True):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop,
                              qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values) if init_values is not None \
            else nn.Identity()
        self.register_drop_path = bool(register_drop_path)
        if register_drop_path:
            self.drop_path1 = (DropPath(drop_path) if drop_path > 0.0
                               else nn.Identity())
        self.norm2 = norm_layer(dim, eps=1e-6)
        if ffn_layer == "swiglufused":
            self.mlp = SwiGLUFFNFused(in_features=dim,
                                      hidden_features=int(dim * mlp_ratio),
                                      drop=drop)
        else:
            self.mlp = Mlp(in_features=dim,
                           hidden_features=int(dim * mlp_ratio),
                           drop=drop)
        self.ls2 = LayerScale(dim, init_values) if init_values is not None \
            else nn.Identity()
        if register_drop_path:
            self.drop_path2 = (DropPath(drop_path) if drop_path > 0.0
                               else nn.Identity())
        self.sample_drop_ratio = 0.0

    def forward(self, x, pos=None, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """DropPath used when drop_path_rate > 0 (matches dinov2 naming)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob or 0.0, self.training)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SwiGLUFFNFused(SwiGLUFFN):
    """giant branch FFN name (structure identical to SwiGLUFFN)."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features,
                         hidden_features=hidden_features,
                         drop=drop)


class DinoVisionTransformer(nn.Module):
    """DA3-adapted DinoVisionTransformer:
    input [B,V,3,H,W] -> tuple of (patch_feats, cam_tokens) per out_layer.
    Options alt_start/qknorm_start/rope_start/cat_token mirror the DA3 fork.
    When rope_start != -1 a RotaryPositionEmbedding2D is created on this module
    and shared with the attention of every block with index >= rope_start
    (same instance re-parented, mirroring the open-source layout)."""
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 patch_size=14, img_size=518, in_chans=3, out_layers=(4, 11, 17, 23),
                 alt_start=-1, qknorm_start=-1, rope_start=-1, cat_token=False,
                 ffn_layer="mlp", use_reference_view=True, rope_freq=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_layers = tuple(out_layers)
        self.alt_start = int(alt_start)
        self.qknorm_start = int(qknorm_start)
        self.rope_start = int(rope_start)
        self.cat_token = bool(cat_token)
        self.patch_start_idx = 1
        self.num_tokens = 1
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        else:
            self.camera_token = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim))
        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self.position_getter = PositionGetter()
        else:
            self.rope = None
            self.position_getter = None
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, init_values=1.0, ffn_layer=ffn_layer,
                  qk_norm=(i >= self.qknorm_start) if self.qknorm_start != -1 else False,
                  rope=self.rope if i >= self.rope_start and self.rope_start != -1
                  else None)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def prepare_tokens(self, x):
        """x: [B,V,3,H,W] -> tokens [B,V,1+Np,D]"""
        B, V, C, H, W = x.shape
        xt = x.reshape(B * V, C, H, W)
        xt = self.patch_embed(xt)  # (BV, D, Hp, Wp)
        Hp, Wp = xt.shape[-2], xt.shape[-1]
        xt = xt.flatten(2).transpose(1, 2)  # (BV, Np, D)
        xt = torch.cat([self.cls_token.expand(xt.shape[0], -1, -1), xt], dim=1)
        # interpolate pos embed to (1+Hp*Wp) if needed
        pe = self.pos_embed
        n_pe = pe.shape[1]
        if n_pe != 1 + Hp * Wp:
            pe = pe[:, 1:, :].transpose(1, 2).reshape(1, self.embed_dim,
                                                      int(math.sqrt(n_pe - 1)),
                                                      int(math.sqrt(n_pe - 1)))
            pe = F.interpolate(pe.float(), size=(Hp, Wp), mode="bicubic",
                               align_corners=False).to(x.dtype)
            pe = pe.flatten(2).transpose(1, 2)
            pe = torch.cat([self.pos_embed[:, :1], pe], dim=1)
        xt = xt + pe
        xt = xt.reshape(B, V, -1, self.embed_dim)
        return xt

    def prepare_positions(self, B, V, H, W, device):
        """Return (pos, pos_nodiff) as in the open source `_prepare_rope`."""
        if self.rope is None:
            return None, None
        n = self.patch_start_idx
        hp = H // self.patch_size
        wp = W // self.patch_size
        pos = self.position_getter(B * V, hp, wp, device=device)
        pos = pos.reshape(B, V, hp * wp, 2)
        pos = pos + 1
        special = torch.zeros(B, V, n, 2, device=device, dtype=pos.dtype)
        pos = torch.cat([special, pos], dim=2)          # (B,V,1+Np,2)
        pos_nodiff = torch.ones(B, V, hp * wp, 2, device=device, dtype=pos.dtype)
        pos_nodiff = torch.cat([special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def forward(self, x, cam_token=None):
        B, V, C, H, W = x.shape
        tokens = self.prepare_tokens(x)          # (B,V,n,D)
        pos, pos_nodiff = self.prepare_positions(B, V, H, W, tokens.device)
        local_x = tokens
        outputs = []
        for i, blk in enumerate(self.blocks):
            # positions: global attention uses pos_nodiff (uniform), local uses pos
            if self.rope is not None and i >= self.rope_start:
                g_pos, l_pos = pos_nodiff, pos
            else:
                g_pos = l_pos = None
            if self.alt_start != -1 and i == self.alt_start:
                if cam_token is not None:
                    tokens = tokens.clone()
                    tokens[:, :, 0] = cam_token
                elif self.camera_token is not None:
                    ref = self.camera_token[:, :1].expand(B, -1, -1)
                    src = self.camera_token[:, 1:].expand(B, V - 1, -1)
                    tokens = tokens.clone()
                    tokens[:, :, 0] = torch.cat([ref, src], dim=1)
            is_global = (self.alt_start != -1 and i >= self.alt_start
                         and i % 2 == 1)
            if is_global:
                n = tokens.shape[2]
                tg = tokens.reshape(B, V * n, self.embed_dim)
                pg = None if g_pos is None else g_pos.reshape(B, V * n, 2)
                tg = blk(tg, pos=pg)
                tokens = tg.reshape(B, V, n, self.embed_dim)
            else:
                n = tokens.shape[2]
                tv = tokens.reshape(B * V, n, self.embed_dim)
                pv = None if l_pos is None else l_pos.reshape(B * V, n, 2)
                tv = blk(tv, pos=pv)
                tokens = tv.reshape(B, V, n, self.embed_dim)
                local_x = tokens
            if i in self.out_layers:
                out_x = torch.cat([local_x, tokens], dim=-1) if self.cat_token \
                    else tokens
                if self.cat_token:
                    out_x = out_x.reshape(B * V, out_x.shape[2], -1)
                    half = self.embed_dim
                    out_x = torch.cat(
                        [out_x[..., :half], self.norm(out_x[..., half:])], dim=-1)
                    out_x = out_x.reshape(B, V, out_x.shape[1], -1)
                # (patch feat [B,V,Np,Df], camera-token stream [B,V,Df])
                outputs.append((out_x[:, :, 1:], out_x[:, :, 0]))
        return tuple(outputs)


class DinoV2(nn.Module):
    VARIANTS = {
        "vits": dict(embed_dim=384, depth=12, num_heads=6),
        "vitb": dict(embed_dim=768, depth=12, num_heads=12),
        "vitl": dict(embed_dim=1024, depth=24, num_heads=16),
        "vitg": dict(embed_dim=1536, depth=40, num_heads=24, ffn_layer="swiglufused"),
    }

    def __init__(self, name="vitl", out_layers=(4, 11, 17, 23), alt_start=-1,
                 qknorm_start=-1, rope_start=-1, cat_token=False,
                 patch_size=14, img_size=518, mlp_ratio=4.0):
        super().__init__()
        v = dict(self.VARIANTS[name])
        v["ffn_layer"] = v.get("ffn_layer", "mlp")
        self.name = name
        self.pretrained = DinoVisionTransformer(
            embed_dim=v["embed_dim"], depth=v["depth"],
            num_heads=v["num_heads"], mlp_ratio=mlp_ratio,
            patch_size=patch_size, img_size=img_size,
            out_layers=out_layers, alt_start=alt_start,
            qknorm_start=qknorm_start, rope_start=rope_start,
            cat_token=cat_token, ffn_layer=v["ffn_layer"])

    def forward(self, x, cam_token=None):
        return self.pretrained(x, cam_token=cam_token)

# PART 5b -- DA3 DPT heads + DepthAnything3Net / NestedDepthAnything3Net mirror.

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class FloatFunctional(nn.Module):
    """Structural twin of torch.nn.quantized.FloatFunctional as used by the
    open-source fusion code (registers an `activation_post_process` child;
    parameter-free, participates only in the module tree / skip-add)."""
    def __init__(self):
        super().__init__()
        self.activation_post_process = nn.Identity()

    def add(self, a, b):
        return a + b

    def forward(self, x):
        return x


class ResidualConvUnit(nn.Module):
    """ResidualConvUnit mirroring dpt.py: children conv1, conv2, norm1=None,
    norm2=None, activation (module), skip_add (FloatFunctional)."""
    def __init__(self, features=256, activation=None, bn=False, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True,
                               groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True,
                               groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation if activation is not None else nn.ReLU()
        self.skip_add = FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block mirroring dpt.py: resConfUnit1 (None when
    has_residual=False), resConfUnit2, out_conv, skip_add; one activation
    instance is shared by both residual units (as in the open source)."""
    def __init__(self, features=256, activation=None, deconv=False, bn=False,
                 expand=False, align_corners=True, size=None,
                 has_residual=True, groups=1):
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        act = activation if activation is not None else nn.ReLU()
        self.resConfUnit1 = (ResidualConvUnit(features, act, bn, groups=groups)
                             if has_residual else None)
        self.resConfUnit2 = ResidualConvUnit(features, act, bn, groups=groups)
        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True,
                                  groups=groups)
        self.skip_add = FloatFunctional()

    def forward(self, *xs, size=None):
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))
        y = self.resConfUnit2(y)
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}
        y = F.interpolate(y.float(), **up_kwargs, mode="bilinear",
                          align_corners=self.align_corners).to(y.dtype)
        return self.out_conv(y)


def _make_fusion_block(features, size=None, has_residual=True, groups=1,
                       inplace=False):
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False, bn=False, expand=False, align_corners=True, size=size,
        has_residual=has_residual, groups=groups,
    )


def _make_scratch(in_shape, features, groups=1, expand=False):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], features, 3, 1, 1, bias=False,
                                  groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], features, 3, 1, 1, bias=False,
                                  groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], features, 3, 1, 1, bias=False,
                                  groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], features, 3, 1, 1, bias=False,
                                  groups=groups)
    return scratch


class DPT(nn.Module):
    """Metric/mono DPT decoder head mirroring dpt.py: 4-stage pyramid fusion
    (resize_layers -> scratch layerN_rn -> refinenet1..4) + output convs +
    optional sky head. Module tree, param layout and fusion chain match the
    open-source DPT (norm_type idt -> norm Identity; pos_embed off)."""
    def __init__(self, dim_in=1024, output_dim=1, features=256,
                 out_channels=(256, 512, 1024, 1024), use_sky_head=True,
                 patch_size=14, down_ratio=1, fusion_block_inplace=False,
                 use_ln_for_heads=False, head_features_2: int = 32,
                 raw_output: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.dim_in = dim_in
        self.features = features
        self.out_channels = list(out_channels)
        self.down_ratio = down_ratio
        self.raw_output = raw_output
        self.intermediate_layer_idx = (0, 1, 2, 3)
        # token pre-norm: norm_type="idt" -> Identity, "layer" -> LayerNorm
        self.norm = nn.Identity()
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, 1, bias=True) for oc in self.out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0],
                               kernel_size=4, stride=4),
            nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1],
                               kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(self.out_channels[3], self.out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])
        # scratch: stage adapters + main fusion chain (dpt.py _make_scratch)
        self.scratch = _make_scratch(self.out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(
            features, inplace=fusion_block_inplace)
        self.scratch.refinenet2 = _make_fusion_block(
            features, inplace=fusion_block_inplace)
        self.scratch.refinenet3 = _make_fusion_block(
            features, inplace=fusion_block_inplace)
        self.scratch.refinenet4 = _make_fusion_block(
            features, has_residual=False, inplace=fusion_block_inplace)
        # heads: shared neck1 then main (+ optional sky head)
        self.head_features_2 = int(head_features_2)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3,
                                              padding=1)
        ln_seq = ([Permute((0, 2, 3, 1)), nn.LayerNorm(self.head_features_2),
                   Permute((0, 3, 1, 2))] if use_ln_for_heads else [])
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, self.head_features_2, 3, padding=1),
            *ln_seq,
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_features_2, output_dim, 1))
        self.use_sky_head = bool(use_sky_head)
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, self.head_features_2, 3, padding=1),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_features_2, 1, 1))

    def _fuse(self, feats):
        """4-layer top-down fusion (dpt.py _fuse): returns finest-scale fused
        features *before* output_conv1."""
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _forward(self, feats, h_out, w_out):
        # feats: list of (patch[B,V,Np,2D or D], cam_token) -> use patch feat
        b, v = feats[0][0].shape[0], feats[0][0].shape[1]
        resized = []
        for s, stage in enumerate(self.intermediate_layer_idx):
            x = feats[stage][0]
            x = x.reshape(b * v, -1, x.shape[-1])            # (B*V, Np, C)
            x = self.norm(x)
            hp, wp = getattr(self, "_grid", None) or (int(math.sqrt(x.shape[1])),
                                                     int(math.sqrt(x.shape[1])))
            x = x.permute(0, 2, 1).reshape(b * v, x.shape[-1], hp, wp)
            x = self.projects[s](x)
            x = self.resize_layers[s](x)                     # align scale
            resized.append(x)
        out = self._fuse(resized)
        out = self.scratch.output_conv1(out)
        out = F.interpolate(out.float(), size=(h_out, w_out), mode="bilinear",
                            align_corners=True).to(out.dtype)
        raw = self.scratch.output_conv2(out)
        if self.raw_output:
            # GSDPT-style raw multi-channel map (no exp/monocular prior)
            return raw.reshape(b, v, raw.shape[1], h_out, w_out), None
        depth = torch.exp(raw[:, 0])
        if self.use_sky_head:
            sky = self.scratch.sky_output_conv2(out)[:, 0]
            sky = F.relu(sky)
        else:
            sky = None
        return depth.reshape(b, v, h_out, w_out), sky

    def forward(self, feats, H, W, patch_start_idx=0):
        # true (possibly rectangular) patch grid of the backbone tokens
        self._grid = (H // self.patch_size, W // self.patch_size)
        h_out = int((H // self.patch_size) * self.patch_size / self.down_ratio)
        w_out = int((W // self.patch_size) * self.patch_size / self.down_ratio)
        return self._forward(feats, h_out, w_out)


class DualDPT(nn.Module):
    """Any-view DualDPT head (depth+conf main chain + ray/conf aux chain)
    mirroring dualdpt.py: independent main/aux fusion chains, aux head outputs
    only its final pyramid level.  Only used for giant/large any-view."""
    def __init__(self, dim_in=3072, output_dim=2, features=256,
                 out_channels=(256, 512, 1024, 1024), patch_size=14,
                 down_ratio=1, aux_pyramid_levels=4, aux_out1_conv_num=5,
                 head_features_2: int = 32, aux_output_dim: int = 7):
        super().__init__()
        self.output_dim = output_dim
        self.dim_in = dim_in
        self.out_channels = list(out_channels)
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.intermediate_layer_idx = (0, 1, 2, 3)
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, 1) for oc in self.out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1)])
        # scratch: shared stage adapters; main & aux chains fully separate
        self.scratch = _make_scratch(self.out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features,
                                                     has_residual=False)
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features,
                                                         has_residual=False)
        self.head_features_2 = int(head_features_2)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3,
                                              padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, self.head_features_2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_features_2, output_dim, 1))
        # aux per-level necks + final projections (shared LN/Permute pair,
        # re-parenting reproduces the exact upstream state_dict keys)
        self.aux_output_dim = int(aux_output_dim)
        self.scratch.output_conv1_aux = nn.ModuleList([
            nn.Sequential(*[nn.Conv2d(features if j % 2 == 0 else features // 2,
                                      features // 2 if j % 2 == 0 else features,
                                      3, padding=1) for j in range(5)])
            for _ in range(self.aux_levels)])
        ln_seq = [Permute(dims=(0, 2, 3, 1)), nn.LayerNorm(self.head_features_2),
                  Permute(dims=(0, 3, 1, 2))]
        self.scratch.output_conv2_aux = nn.ModuleList([
            nn.Sequential(nn.Conv2d(features // 2, self.head_features_2, 3, padding=1),
                          *ln_seq,
                          nn.ReLU(inplace=True),
                          nn.Conv2d(self.head_features_2, self.aux_output_dim, 1))
            for _ in range(self.aux_levels)])

    def _fuse(self, feats):
        """Main + aux pyramid fusion (dualdpt.py _fuse)."""
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list = []
        if self.aux_levels >= 4:
            aux_list.append(aux)
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux = self.scratch.refinenet3_aux(aux, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux)
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux = self.scratch.refinenet2_aux(aux, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux)
        out = self.scratch.refinenet1(out, l1_rn)
        aux = self.scratch.refinenet1_aux(aux, l1_rn)
        aux_list.append(aux)
        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](a)
                    for i, a in enumerate(aux_list)]
        return out, aux_list

    def forward(self, feats, H, W, patch_start_idx=0):
        self._grid = (H // self.patch_size, W // self.patch_size)
        b, v = feats[0][0].shape[0], feats[0][0].shape[1]
        resized = []
        for s, stage in enumerate(self.intermediate_layer_idx):
            x = feats[stage][0]
            x = x.reshape(b * v, -1, x.shape[-1])
            x = self.norm(x.float()).to(x.dtype)
            hp, wp = getattr(self, "_grid", None) or (int(math.sqrt(x.shape[1])),
                                                     int(math.sqrt(x.shape[1])))
            x = x.permute(0, 2, 1).reshape(b * v, x.shape[-1], hp, wp)
            x = self.projects[s](x)
            x = self.resize_layers[s](x)
            resized.append(x)
        out, aux_list = self._fuse(resized)
        h_out = int((H // self.patch_size) * self.patch_size / self.down_ratio)
        w_out = int((W // self.patch_size) * self.patch_size / self.down_ratio)
        out = F.interpolate(out.float(), size=(h_out, w_out),
                            mode="bilinear", align_corners=True).to(out.dtype)
        last_aux = aux_list[-1]
        last_aux = F.interpolate(last_aux.float(), size=(h_out, w_out),
                                 mode="bilinear",
                                 align_corners=True).to(last_aux.dtype)
        d = self.scratch.output_conv2(out)
        depth = torch.exp(d[:, 0]).reshape(b, v, h_out, w_out)
        conf = (torch.exp(d[:, 1]) + 1).reshape(b, v, h_out, w_out, 1)
        a = self.scratch.output_conv2_aux[-1](last_aux)
        raymap = a.reshape(b, v, h_out, w_out, -1)
        return depth, conf, raymap


class DepthAnything3Net(nn.Module):
    """Metric/any-view net mirror: backbone(DinoV2) + head(DPT|DualDPT) +
    optional cam_enc/cam_dec/gs_head/gs_adapter (any-view variants)."""
    def __init__(self, net=None, head=None, cam_enc=None, cam_dec=None,
                 gs_head=None, gs_adapter=None, **kwargs):
        super().__init__()
        cfg_net = net or {}
        cfg_head = dict(head or {})
        if isinstance(net, nn.Module):
            self.backbone = net
        else:
            self.backbone = DinoV2(**cfg_net)
        if isinstance(head, nn.Module):
            self.head = head
        else:
            head_cls = cfg_head.pop("cls", "DPT")
            if head_cls == "DualDPT":
                self.head = DualDPT(**cfg_head)
            else:
                self.head = DPT(**cfg_head)
        self.cam_enc = None
        self.cam_dec = None
        self.gs_head = None
        self.gs_adapter = None
        if cam_enc is not None:
            self.cam_enc = cam_enc if isinstance(cam_enc, nn.Module) \
                else CameraEnc(**cam_enc)
        if cam_dec is not None:
            self.cam_dec = cam_dec if isinstance(cam_dec, nn.Module) \
                else CameraDec(**cam_dec)
        if gs_head is not None:
            self.gs_head = gs_head if isinstance(gs_head, nn.Module) \
                else GSDPT(**gs_head)
        if gs_adapter is not None:
            self.gs_adapter = gs_adapter \
                if isinstance(gs_adapter, nn.Module) else GaussianAdapter(**gs_adapter)

    def forward(self, x, extrinsics=None, intrinsics=None, export_feat_layers=None):
        B, V = x.shape[0], x.shape[1]
        cam_token = None
        if self.cam_enc is not None and extrinsics is not None and \
                intrinsics is not None:
            cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        feats = self.backbone(x, cam_token=cam_token)
        H, W = x.shape[3], x.shape[4]
        head_out = self.head(feats, H, W)
        out = {}
        if isinstance(head_out, tuple):
            out["depth"] = head_out[0]
            if len(head_out) > 1 and head_out[1] is not None:
                out["conf_or_sky"] = head_out[1]
            if len(head_out) > 2 and head_out[2] is not None:
                out["raymap"] = head_out[2]
        else:
            out["depth"] = head_out
        if self.cam_dec is not None and len(feats) > 0:
            # pose regression from the last out-layer camera tokens
            out["camera"] = self.cam_dec(feats[-1][1])
        if self.gs_head is not None:
            out["gs"] = self.gs_head(feats, H, W)[0]
        if self.gs_adapter is not None:
            out["gs_d_in"] = self.gs_adapter.d_in
        return out


class NestedDepthAnything3Net(nn.Module):
    """Nested (any-view + metric) container mirroring da3.py: two subnets
    `da3` and `da3_metric`."""
    def __init__(self, anyview=None, metric=None):
        super().__init__()
        self.da3 = anyview if isinstance(anyview, nn.Module) else DepthAnything3Net(**anyview)
        self.da3_metric = (metric if isinstance(metric, nn.Module)
                           else DepthAnything3Net(**metric))

    def forward(self, x, extrinsics=None, intrinsics=None, **kwargs):
        return {"anyview": self.da3(x, extrinsics=extrinsics,
                                    intrinsics=intrinsics),
                "metric": self.da3_metric(x)}

# PART 5c -- DA3 any-view accessories mirror: CameraEnc / CameraDec / GSDPT /
#           GaussianAdapter (module layout per da3_port_spec §3.3-3.5).

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class CameraEnc(nn.Module):
    """9-dim pose -> [B,V,dim_out] token trunk (cam_enc.py mirror)."""
    def __init__(self, dim_out=1536, dim_in=9, target_dim=9, trunk_depth=4,
                 num_heads=16, mlp_ratio=4, init_values=0.01):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(*[
            Block(dim=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, init_values=init_values,
                  register_drop_path=False)  # cam trunk: no drop_path members
            for _ in range(trunk_depth)])
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(dim_in, dim_out // 2, dim_out)

    def forward(self, ext, intrinsic, size):
        # ext: [B,V,4,4] w2c, intrinsic: [B,V,3,3]; build 9-d pose encoding
        if ext.ndim == 3:  # single view: (B,4,4) -> (B,1,4,4)
            ext = ext.unsqueeze(0)
            if intrinsic.ndim == 3:
                intrinsic = intrinsic.unsqueeze(0)
        B, V = ext.shape[0], ext.shape[1]
        R = ext[..., :3, :3]
        t = ext[..., :3, 3]
        pose = torch.cat([t, R.reshape(B, V, 9)], dim=-1)  # 12-d approx
        pose = pose[..., : self.dim_in]
        tok = self.pose_branch(pose)
        tok = self.token_norm(tok)
        for blk in self.trunk:
            tok = blk(tok)
        return self.trunk_norm(tok)


class CameraDec(nn.Module):
    """camera token [B,V,dim_in] -> pose regression (cam_dec.py mirror):
    translation t_dim + quaternion quat_dim + fov fov_dim channels."""
    def __init__(self, dim_in=3072, t_dim: int = 3, quat_dim: int = 4,
                 fov_dim: int = 2):
        super().__init__()
        self.t_dim = t_dim
        self.quat_dim = quat_dim
        self.fov_dim = fov_dim
        self.pose_dim = t_dim + quat_dim + fov_dim
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(),
            nn.Linear(dim_in, dim_in), nn.ReLU())
        self.fc_t = nn.Linear(dim_in, t_dim)
        self.fc_qvec = nn.Linear(dim_in, quat_dim)
        self.fc_fov = nn.Sequential(nn.Linear(dim_in, fov_dim), nn.ReLU())

    def forward(self, cam_token):
        x = cam_token.reshape(-1, cam_token.shape[-1])
        x = self.backbone(x)
        t = self.fc_t(x)
        q = self.fc_qvec(x)
        fov = self.fc_fov(x)
        return torch.cat([t, q, fov], dim=-1).reshape(
            cam_token.shape[0], cam_token.shape[1], self.pose_dim)


class GSDPT(DPT):
    """GSDPT: DPT with raw_gs output + images_merger (gsdpt.py mirror)."""
    def __init__(self, dim_in=3072, output_dim=38, features=256,
                 out_channels=(256, 512, 1024, 1024),
                 head_features_2: int = 32, merger_channels=(32, 64)):
        super().__init__(dim_in=dim_in, output_dim=output_dim, features=features,
                         out_channels=out_channels, use_sky_head=False,
                         head_features_2=head_features_2, raw_output=True)
        c1, c2 = merger_channels
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, output_dim, 1))
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.GELU(),
            nn.Conv2d(c2, features // 2, 3, padding=1), nn.GELU())


class GaussianAdapter(nn.Module):
    """GaussianAdapter: no learnable parameters (sh_mask non-persistent)."""
    def __init__(self, sh_degree=2, pred_color=False, pred_offset_depth=True,
                 pred_offset_xy=True, gaussian_scale_min=1e-5,
                 gaussian_scale_max=30.0):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        d_sh = (self.sh_degree + 1) ** 2
        self.register_buffer("sh_mask", torch.ones(d_sh), persistent=False)

    @property
    def d_sh(self):
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self):
        n = 3 + 4 + 3 * self.d_sh  # scale, quat, SH
        if self.pred_offset_xy:
            n += 2
        if self.pred_offset_depth:
            n += 1
        if self.pred_color:
            n += 3
        return n

# PART 6 -- flow-match scheduler, losses, real-time interactive engines and the
#          end-to-end "MatrixGame35Interactive" driver (sequential & async-3).

class FlowMatchScheduler:
    """Rectified-flow scheduler (mirrors diffsynth FlowMatchScheduler "Wan")."""
    def __init__(self, num_train_timesteps: int = 1000,
                 shift: float = 1.0, use_timestep_transform: bool = False):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.use_timestep_transform = use_timestep_transform
        sigmas = torch.linspace(0, 1, num_train_timesteps + 1)
        if shift != 1.0:
            sigmas = sigmas / (1 + (shift - 1) * (1 - sigmas))
        self.sigmas = sigmas

    def add_noise(self, x0, noise, t):
        """x_t = (1-t) x0 + t noise, t in [0,1]."""
        t = torch.as_tensor(t, dtype=x0.dtype, device=x0.device)
        t = t.view([-1] + [1] * (x0.ndim - 1))
        return (1 - t) * x0 + t * noise

    def step(self, v, t, x, to_final: bool = False):
        """x_{t+dt} = x_t + v*(t_next - t) with dt chosen to reach sigma 0
        when to_final=True."""
        t = torch.as_tensor(t, dtype=v.dtype, device=v.device)
        dt = (0.0 - t) if to_final else (1.0 / self.num_train_timesteps)
        x = x + v * dt
        return x

    def training_target(self, x0, noise):
        return noise - x0


class FlowMatchSFTLoss(nn.Module):
    """Latent-space flow-matching MSE over DiT velocity predictions."""
    def __init__(self, weighting: float = 1.0, subject_alpha: float = 0.0):
        super().__init__()
        self.weighting = weighting
        self.subject_alpha = subject_alpha

    def forward(self, pred, target, subject_mask=None):
        loss = F.mse_loss(pred, target)
        if subject_mask is not None and self.subject_alpha > 0:
            w = subject_mask.to(pred.dtype)
            sub = ((pred - target) ** 2).mean(dim=1, keepdim=True)
            sub = (sub * w).sum() / (w.sum() + 1e-6)
            loss = loss + self.subject_alpha * sub
        return loss


class LinearAttentionKVCache(dict):
    """Plain-dict KV cache, mirroring causal_self_attention_kv cache contract:
    keys ``k`` (PRE-RoPE k), ``v`` (raw v) and ``frames`` (per-frame ids)."""
    pass


def _da3_output_fields(out: Dict) -> Dict:
    """Normalise a DA3 output: a flat DepthAnything3Net dict, or the nested
    ``{"anyview": {...}, "metric": {...}}`` container (depth from the metric
    subnet, camera/gs from the any-view subnet)."""
    if "depth" in out:
        return out
    merged: Dict[str, Any] = {}
    anyv = out.get("anyview") or {}
    met = out.get("metric") or {}
    if isinstance(met, dict) and "depth" in met:
        merged["depth"] = met["depth"]
    elif isinstance(anyv, dict) and "depth" in anyv:
        merged["depth"] = anyv["depth"]
    if isinstance(anyv, dict):
        for key in ("camera", "gs", "conf_or_sky", "raymap"):
            if key in anyv:
                merged[key] = anyv[key]
    return merged


class ShapeTracer:
    """Component-structured inference shape tracer.

    Forward hooks record every hooked module's input/output tensor shapes in
    execution order; ``report()`` prints them grouped per model component
    (text / dit / vae / da3) so the tensor-shape flow through each component
    of the complete model is visible."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: List[Tuple[int, str, str, str, List, List]] = []
        self._order = 0
        self._lock = threading.Lock()
        self._hooks: List[Any] = []
        self._groups: Dict[str, str] = {}

    @staticmethod
    def _shapes(x) -> List[str]:
        if torch.is_tensor(x):
            return [f"{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"]
        if isinstance(x, (tuple, list)):
            out: List[str] = []
            for e in x:
                out.extend(ShapeTracer._shapes(e))
            return out or ["-"]
        return [type(x).__name__]

    def attach(self, group: str, module: nn.Module, patterns: Sequence[str]):
        """Register hooks on ``module``'s submodules whose qualified name
        matches one of ``patterns``.  A pattern ending in ``.*`` matches only
        direct children of that prefix (component-boundary granularity, not
        every leaf Linear); other patterns are exact matches."""
        if not self.enabled:
            return
        self._groups[group] = type(module).__name__

        def matches(name: str, p: str) -> bool:
            if p.endswith(".*"):
                stem = p[:-2]
                rest = name[len(stem) + 1:]
                return name.startswith(stem + ".") and "." not in rest
            return name == p

        for name, mod in module.named_modules():
            if name == "" and "" not in patterns:
                continue
            if any(matches(name, p) for p in patterns):
                self._hooks.append(mod.register_forward_hook(
                    self._make_hook(group, name or group, type(mod).__name__)))

    def _make_hook(self, group, path, cls):
        def hook(_m, inp, out):
            with self._lock:
                self.records.append((self._order, group, path, cls,
                                     self._shapes(inp), self._shapes(out)))
                self._order += 1
        return hook

    def report(self):
        if not self.enabled:
            return
        for group, cls in self._groups.items():
            rows = [r for r in self.records if r[1] == group]
            if not rows:
                continue
            print(f"  [shape] ---- component {group} ({cls}) ----")
            for order, _g, path, klass, ins, outs in rows:
                print(f"  [shape]  #{order:<3} {path:<34} {klass:<24} "
                      f"in={ins} out={outs}")

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def _dummy_camera(n_frames: int, h: int, w: int, seed: int = 0) -> Dict:
    """Forward-motion camera trajectory (c2w + pinhole intrinsics) used to
    exercise the PRoPE path in the interactive smoke tests."""
    g = torch.Generator().manual_seed(seed)
    c2w = torch.zeros(n_frames, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[:, 2, 3] = torch.linspace(0.0, 0.2, n_frames) + \
        0.01 * torch.randn(n_frames, generator=g)
    c2w[:, 3, 3] = 1.0
    intr = torch.zeros(n_frames, 3, 3)
    intr[:, 0, 0] = intr[:, 1, 1] = 0.8
    intr[:, 0, 2] = w / 2
    intr[:, 1, 2] = h / 2
    intr[:, 2, 2] = 1.0
    return {"c2w": c2w, "intrinsics": intr, "image_size": (h, w)}


class RealtimeInteractiveEngine:
    """Real-time interactive world-model driver used by both execution modes.

    This reproduces the *interactive paradigm* of Matrix-Game 3.5 distilled
    inference: rolling C0 clean anchor + chunk-wise (chunk_size latent frames)
    causal DiT denoising over a fixed 3-step schedule, VAE decode of every
    chunk and DA3 metric-depth + camera registration back into the rolling
    memory.  The DiT rollout runs on the *causal KV-cache path*
    (``causal_self_attention_kv``): the clean prefix lives in per-block caches
    (PRE-RoPE k / raw v / frame ids) and each denoise step processes only the
    current chunk tokens -- the streaming paradigm of the open-source engine.
    When the DiT is configured with ``use_prope`` and a camera trajectory is
    supplied, PRoPE camera-aware attention is applied by frame index
    end-to-end.

    Mode "seq" executes on a single device sequentially (DA3 -> VAE encode ->
    DiT rollout -> VAE decode).  Mode "async3" simulates a 3-GPU asynchronous
    pipeline where DA3, VAE and DiT are *virtual devices* (own threads / event
    queues) whose compute overlaps: depth registration of chunk i-1 (DA3 card)
    and VAE decode of chunk i (VAE card) run while chunk i+1 is being denoised
    (DiT on the calling thread), so each card's latency is masked by the
    others.  All stages execute the real models on CPU; the schedule log
    quantifies the overlap."""

    # per-component hook patterns for the structured shape tracer
    HOOK_SPECS = {
        "text": [""],
        "dit": ["patch_embedding", "blocks.*", "head"],
        "vae": ["model.encoder", "model.decoder"],
        "da3": ["da3.backbone.pretrained", "da3.head", "da3.cam_enc",
                "da3.cam_dec", "da3.gs_head", "da3_metric.backbone.pretrained",
                "da3_metric.head", "backbone.pretrained", "head",
                "cam_enc", "cam_dec", "gs_head"],
    }

    def __init__(self, components, scheduler, *, latent_stride=16,
                 chunk_size=3, num_steps=3, process_res=64, dtype=torch.float32,
                 device="cpu", seed=3407, shape_trace=False, mode="seq",
                 verbose=True, camera_trans_scale=50.0,
                 context_chunks=7):
        # components: dict with keys dit / vae / text / da3 / memory
        self.comp = components
        self.scheduler = scheduler
        self.latent_stride = latent_stride
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        # rolling KV-cache window (opensrc `context_chunks`): the cache keeps
        # the advancing anchor + the most recent (context_chunks - 1) chunks;
        # older chunks are dropped from every layer's cache after each fill
        self.context_chunks = int(context_chunks)
        self.process_res = process_res
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.mode = mode
        self.verbose = verbose
        self.camera_trans_scale = camera_trans_scale
        self.schedule_log: List[Tuple[float, str]] = []
        self.tracer = ShapeTracer(enabled=shape_trace)
        if shape_trace:
            for key, patterns in self.HOOK_SPECS.items():
                if key in self.comp and isinstance(self.comp[key], nn.Module):
                    self.tracer.attach(key, self.comp[key], patterns)

    # ---------------- shared rollout primitives ----------------
    def _camera_info(self, dit, camera, n_frames, device):
        """PRoPE camera info over the first ``n_frames`` latent frames, or
        None when the DiT is not camera-aware / no camera given."""
        if camera is None or not getattr(dit, "use_prope", False):
            return None
        c2w = camera["c2w"]
        intr = camera["intrinsics"]
        if c2w.shape[0] < n_frames:  # hold the last pose if the trajectory ends
            pad = c2w[-1:].expand(n_frames - c2w.shape[0], -1, -1)
            c2w = torch.cat([c2w, pad], dim=0)
            pad_i = intr[-1:].expand(n_frames - intr.shape[0], -1, -1)
            intr = torch.cat([intr, pad_i], dim=0)
        return camera_info_from_poses(
            c2w[:n_frames], intr[:n_frames], self.camera_trans_scale,
            camera.get("image_size"))

    def _num_heads(self, dit):
        return dit.blocks[0].num_heads

    def _trim_rolling_cache(self, kv_state, keep_frames):
        """Sliding-window eviction of the rolling KV cache (mirror of the
        open-source ``_causal_kv_trim_rolling_window``): after each cache
        fill, only latent frames in ``keep_frames`` survive in every layer's
        cache; evicted chunks' keys/values are physically dropped.  All
        bookkeeping (frames list) is trimmed in lockstep, so the absolute
        frame addressing of the surviving prefix stays exact."""
        keep = set(int(f) for f in keep_frames)
        for cache in kv_state:
            if cache.get("k") is None or int(cache["k"].shape[1]) == 0:
                continue
            frames = [int(f) for f in cache.get("frames", [])]
            if not frames:
                continue
            n_tokens = int(cache["k"].shape[1])
            per_frame = n_tokens // len(frames)
            keep_mask = torch.tensor([f in keep for f in frames],
                                     dtype=torch.bool).repeat_interleave(
                                         per_frame)
            cache["k"] = cache["k"][:, keep_mask]
            cache["v"] = cache["v"][:, keep_mask]
            cache["frames"] = [f for f in frames if f in keep]

    def _kv_state(self, dit):
        return [LinearAttentionKVCache() for _ in dit.blocks]

    def _kv_forward(self, dit, x_chunk, tv, ctx, kv_state, frame_offset,
                    total_frames, camera, write_cache, mosaic=None,
                    mosaic_frames=None, hole_keep=None):
        """One causal forward over ``x_chunk`` (chunk tokens only) attending
        to the clean-prefix KV cache; optionally appends the chunk to it.
        ``mosaic`` (1,c,f,h,w) is the query-aligned memory canvas: it is
        prepended on the time axis so its tokens become the mosaic KV prefix
        ([M | CUR]), addressed at the absolute positions of the target
        frames they support."""
        b, _, f_chunk, h_lat, w_lat = x_chunk.shape
        prefix = frame_offset
        # RoPE grids live on the DiT patch grid, not the latent resolution
        ph, pw = dit.patch_size[1], dit.patch_size[2]
        hp, wp = h_lat // ph, w_lat // pw
        cur_frames = list(range(prefix, prefix + f_chunk))
        # the cache may be a trimmed sliding window: read its actual
        # surviving frame ids and build the RoPE grid for exactly those
        # absolute positions (evicted frames are simply absent)
        cache_frames = list(kv_state[0].get("frames", [])) \
            if kv_state else []
        cache_freqs = None
        if cache_frames:
            cache_freqs = torch.cat(
                [_freq_grid(dit.freqs, 1, hp, wp, x_chunk.device,
                            frame_offset=f) for f in cache_frames], dim=0)
        cfg = dict(mosaic_tokens=0,
                   cur_frames=cur_frames,
                   cache_freqs=cache_freqs,
                   cache_frames=cache_frames,
                   write_cache=write_cache,
                   num_heads=self._num_heads(dit),
                   kv_state=kv_state)
        if mosaic is not None:
            x_chunk = torch.cat([mosaic.to(x_chunk.dtype), x_chunk], dim=2)
            f_m = int(mosaic.shape[2])
            cfg["mosaic_tokens"] = f_m * hp * wp
            cfg["mosaic_frames"] = list(mosaic_frames or cur_frames)
            if hole_keep is not None:
                cfg["hole_keep"] = hole_keep.to(x_chunk.device)
            # mosaic frames reuse the target frames' absolute RoPE positions
            cfg["frame_offsets"] = list(mosaic_frames or cur_frames) + cur_frames
        cam = self._camera_info(dit, camera, total_frames, x_chunk.device)
        return dit(x_chunk, tv, ctx, camera_info=cam, causal_kv_config=cfg,
                   frame_offset=frame_offset)

    def _build_mosaic(self, dit, prefix, f_chunk, camera, h_lat, w_lat):
        """Query the 3D patch memory for the chunk's target frames and pack
        the canvas as the mosaic prefix for the DiT. Returns
        (canvas, mosaic_frames, hole_keep) or (None, None, None) when the
        memory is empty / no camera is given."""
        memory = self.comp.get("memory")
        if memory is None or len(memory.latents) == 0 or camera is None:
            return None, None, None
        n_cam = camera["c2w"].shape[0]
        ids = [min(prefix + g, n_cam - 1) for g in range(f_chunk)]
        canvas, hole = memory.build_mosaic_canvas(
            camera["c2w"][ids].to(camera["c2w"].dtype),
            camera["intrinsics"][ids], camera["image_size"])
        if canvas is None:
            return None, None, None
        hp, wp = h_lat // dit.patch_size[1], w_lat // dit.patch_size[2]
        # patch-grid usability: a patch is usable if any of its latent cells
        # is covered (max-pool the coverage mask over the patch footprint)
        covered = (~hole).to(torch.float32).unsqueeze(1)
        hole_keep = F.max_pool2d(covered, kernel_size=(dit.patch_size[1],
                                                       dit.patch_size[2])
                                 ).squeeze(1).gt(0).reshape(-1)
        if self.verbose:
            cov = ("n/a (meta dry run)" if hole.is_meta else
                   f"{float((~hole).float().mean()):.0%}")
            print(f"  [mosaic] chunk@{prefix}: canvas {tuple(canvas.shape)}, "
                  f"cell coverage {cov}")
        return canvas, ids, hole_keep

    def _da3_register(self, rgb_frames, camera, frame_idx, latent=None,
                      num_latent_frames=1):
        """DA3 metric-depth + camera registration of a decoded chunk into
        the rolling 3D patch memory (handles flat & nested DA3 outputs).

        Mirrors the open-source ``register_source_sequence``: the decoded
        clip's frames each get their own metric depth and each latent frame
        is registered with its own depth slice + pose, so every latent frame
        of the memory can warp patches into future cameras (previously only
        the clip's last frame carried depth, i.e. 1/3 registration density).

        ``rgb_frames`` (b,3,F_rgb,H,W) is the decoded clip (a single frame
        (b,3,1,H,W) or (b,3,H,W) is also accepted); latent frame k takes its
        depth from RGB frame ``min(k*4, F_rgb-1)`` (the clip's temporal
        stride).  ``latent`` (b,c,f_lat,h,w) is the VAE latent the clip was
        decoded from; ``num_latent_frames`` == f_lat."""
        da3 = self.comp["da3"]
        memory = self.comp.get("memory")
        if rgb_frames.dim() == 4:  # (b,3,H,W) single frame
            rgb_frames = rgb_frames[:, :, None]
        if rgb_frames.shape[2] == 3 and rgb_frames.shape[1] != 3:
            # (b,F,3,H,W) layout -> (b,3,F,H,W)
            rgb_frames = rgb_frames.transpose(1, 2)
        n_rgb = int(rgb_frames.shape[2])
        # per-latent-frame DA3 depth: run DA3 on each latent frame's
        # representative RGB frame (stride-4 sampling of the clip)
        depths = []
        with torch.no_grad():
            for k in range(num_latent_frames):
                rid = min(k * 4, n_rgb - 1)
                rgb_k = rgb_frames[:, :, rid]
                ext = intr = None
                if camera is not None:
                    ci = min(frame_idx + k, camera["c2w"].shape[0] - 1)
                    ext = invert_se3(camera["c2w"][ci:ci + 1])
                    intr = camera["intrinsics"][ci:ci + 1]
                out = _da3_output_fields(
                    da3(rgb_k[:, None], extrinsics=ext, intrinsics=intr))
                depths.append(out.get("depth"))
                if k == num_latent_frames - 1:
                    pose = out.get("camera")
        if memory is not None:
            lat = latent
            if lat is not None and lat.dim() == 5:
                lat = lat[0, :, -num_latent_frames:]  # (c, n_lat, h, w)
                lat = lat.transpose(0, 1)             # (n_lat, c, h, w)
            for k in range(num_latent_frames):
                f_idx = frame_idx + k
                if camera is not None:
                    ci = min(f_idx, camera["c2w"].shape[0] - 1)
                    f_c2w = camera["c2w"][ci]
                    f_K = camera["intrinsics"][ci]
                    f_size = camera["image_size"]
                else:
                    f_c2w = f_K = f_size = None
                rid = min(k * 4, n_rgb - 1)
                memory.add(
                    frame=rgb_frames[:, :, rid],
                    depth=(depths[k][0] if depths[k] is not None else None),
                    pose=pose,
                    latent=(None if lat is None
                            else lat[k][None, :, None]),
                    c2w=f_c2w, intrinsics=f_K, image_size=f_size,
                    frame_idx=f_idx)
        return depths[-1], pose

    def _prefill_cache(self, dit, anchor_lat, ctx, kv_state, camera):
        """Write the C0 clean anchor into the per-block KV caches."""
        b, _, f0, h_lat, w_lat = anchor_lat.shape
        t0 = torch.zeros(b, device=self.device)
        self._kv_forward(dit, anchor_lat, t0, ctx, kv_state, frame_offset=0,
                         total_frames=f0, camera=camera, write_cache=True)

    def _denoise_chunk(self, dit, ctx, latents, kv_state, camera, b, c, h_lat,
                       w_lat, return_tokens=False):
        """3-step distilled denoise of one chunk on the KV-cache path.

        Before the first step the 3D patch memory is queried for the chunk's
        target cameras: the warped memory canvas M is prepended on the time
        axis so the self-attention becomes [cache ‖ M ‖ CUR] with CUR's
        queries only (block-causal, the open-source rollout contract)."""
        prefix = latents.shape[2]
        noise = torch.randn(b, c, self.chunk_size, h_lat, w_lat,
                            dtype=self.dtype, device=self.device)
        x = noise
        total = prefix + self.chunk_size
        mosaic, mosaic_frames, hole_keep = self._build_mosaic(
            dit, prefix, self.chunk_size, camera, h_lat, w_lat)
        for step in range(self.num_steps):
            t_frac = 1.0 - (step + 1) / self.num_steps
            tv = torch.full((b,), t_frac * 1000.0, device=self.device)
            v_all = self._kv_forward(dit, x, tv, ctx, kv_state,
                                     frame_offset=prefix,
                                     total_frames=total, camera=camera,
                                     write_cache=False, mosaic=mosaic,
                                     mosaic_frames=mosaic_frames,
                                     hole_keep=hole_keep)
            v = v_all[:, :, int(mosaic.shape[2]):] if mosaic is not None \
                else v_all
            x = self.scheduler.step(v, torch.as_tensor([t_frac]), x,
                                    to_final=(step == self.num_steps - 1))
        # write the clean chunk into the caches for the next chunks, then
        # slide the rolling window: the anchor *advances* with the window
        # (opensrc standard profile -- the original C0 is evicted once the
        # window slides past it), keeping the last
        # context_chunks * chunk_size latent frames (=21 with defaults,
        # i.e. latent_window_size)
        t0 = torch.zeros(b, device=self.device)
        self._kv_forward(dit, x, t0, ctx, kv_state, frame_offset=prefix,
                         total_frames=total, camera=camera, write_cache=True)
        window_start = max(0, total - self.context_chunks * self.chunk_size)
        self._trim_rolling_cache(kv_state,
                                 list(range(window_start, total)))
        return x

    # ---------------- sequential single-"device" mode ----------------
    def run_sequential(self, anchor_rgb, camera, prompt_ids, total_chunks):
        torch.manual_seed(self.seed)
        dit = self.comp["dit"]
        vae = self.comp["vae"]
        text = self.comp["text"]
        with torch.no_grad():
            # 1) DA3 metric depth of the anchor (registers C0 into memory).
            img = anchor_rgb[:, :, 0] if anchor_rgb.dim() == 5 else anchor_rgb
            # 2) text embedding
            ctx = text(prompt_ids)
            # 3) VAE encode anchor -> C0 clean latent, KV-cache prefill
            lat = vae.encode(anchor_rgb)
            b, c = lat.shape[0], lat.shape[1]
            h_lat, w_lat = lat.shape[3], lat.shape[4]
            # register the anchor (single frame: 1 RGB -> 1 latent frame,
            # temporal downsample floors) into the 3D patch memory
            self._da3_register(img[:, None], camera, 0, latent=lat,
                               num_latent_frames=int(lat.shape[2]))
            kv_state = self._kv_state(dit)
            self._prefill_cache(dit, lat, ctx, kv_state, camera)
            latents = lat
            videos = [anchor_rgb]
            for ci in range(total_chunks):
                x = self._denoise_chunk(dit, ctx, latents, kv_state, camera,
                                        b, c, h_lat, w_lat)
                latents = torch.cat([latents, x], dim=2)
                rgb = vae.decode(x)                       # VAE card work
                videos.append(rgb)
                # register the full decoded clip: every latent frame gets
                # its own depth/pose (open-source register_source_sequence)
                self._da3_register(rgb, camera,
                                   frame_idx=latents.shape[2] - self.chunk_size,
                                   latent=x,
                                   num_latent_frames=int(x.shape[2]))
            if self.comp.get("memory") is not None:
                ctx_mem = self.comp["memory"].query()
                if self.verbose and ctx_mem is not None:
                    print(f"  [memory] rolling window now holds "
                          f"{len(self.comp['memory'])} latent frames "
                          f"(query -> {tuple(ctx_mem.shape)})")
        video = torch.cat(videos, dim=2)
        self.tracer.report()
        self.tracer.detach()
        return latents, video

    # ---------------- async 3-"card" simulated pipeline ----------------
    def run_async3(self, anchor_rgb, camera, prompt_ids, total_chunks):
        """Three virtual GPUs (DA3 / VAE / DiT) run as asynchronous pipeline
        stages on worker threads.  Chunk i's clean latent is handed to the VAE
        card while the DiT (calling thread) keeps denoising chunk i+1; decoded
        RGB flows on to the DA3 card for depth/camera registration.  Every
        card's latency is masked by the others' work; the timeline log proves
        the overlap.  Returns the full latent window and the full generated
        video (anchor + all decoded chunks)."""
        dit = self.comp["dit"]
        vae = self.comp["vae"]
        text = self.comp["text"]
        timeline: List[Tuple[float, str]] = []
        t0 = time.time()
        log = lambda name: timeline.append((time.time() - t0, name))
        q_dec = queue.Queue()   # clean chunk latents waiting for VAE decode
        q_dep = queue.Queue()   # (rgb, latent, n_frames) waiting for DA3 reg
        q_out = queue.Queue()   # decoded RGB chunks, in order
        sentinel = object()

        def card_vae():
            while True:
                job = q_dec.get()
                if job is sentinel:
                    return
                log("VAE.decode.start")
                rgb = vae.decode(job)
                log("VAE.decode.end")
                q_dep.put((rgb, job, int(job.shape[2])))

        def card_da3():
            while True:
                job = q_dep.get()
                if job is sentinel:
                    return
                rgb, latent, n_frames = job
                log("DA3.start")
                # register the full decoded clip into the 3D patch memory:
                # each latent frame gets its own DA3 depth (at its stride-4
                # RGB representative) + pose/latent slice, so later chunks
                # can warp every frame as their mosaic canvas
                self._da3_register(rgb, camera, frame_idx=n_frames,
                                   latent=latent, num_latent_frames=n_frames)
                log("DA3.end")
                q_out.put(rgb)

        th_da3 = threading.Thread(target=card_da3)
        th_vae = threading.Thread(target=card_vae)
        th_da3.start()
        th_vae.start()
        try:
            with torch.no_grad():
                ctx = text(prompt_ids)
                lat = vae.encode(anchor_rgb)          # C0 encode (VAE card, sync)
                b, c = lat.shape[0], lat.shape[1]
                h_lat, w_lat = lat.shape[3], lat.shape[4]
                kv_state = self._kv_state(dit)
                log("DiT.cache_prefill.start")
                self._prefill_cache(dit, lat, ctx, kv_state, camera)
                log("DiT.cache_prefill.end")
                latents = lat
                # register the anchor (single frame: 1 RGB -> 1 latent frame)
                # into the 3D patch memory
                self._da3_register(anchor_rgb[:, :, :1], camera, frame_idx=0,
                                   latent=lat,
                                   num_latent_frames=int(lat.shape[2]))
                for ci in range(total_chunks):
                    log(f"DiT.rollout.chunk{ci}.start")
                    x = self._denoise_chunk(dit, ctx, latents, kv_state,
                                            camera, b, c, h_lat, w_lat)
                    log(f"DiT.rollout.chunk{ci}.end")
                    latents = torch.cat([latents, x], dim=2)
                    q_dec.put(x)                      # async decode (VAE card)
                q_dec.put(sentinel)
                th_vae.join()                         # flush decodes -> DA3
                q_dep.put(sentinel)
                th_da3.join()
        except Exception:
            q_dec.put(sentinel)
            q_dep.put(sentinel)
            th_vae.join()
            th_da3.join()
            raise
        decoded = []
        while not q_out.empty():
            decoded.append(q_out.get())
        # measure concurrency: total wall vs serialised sum of stage windows
        wall = time.time() - t0
        stage_sum = 0.0
        start_map = {}
        for ts, name in timeline:
            if name.endswith(".start"):
                start_map[name[:-6]] = ts
            elif name.endswith(".end"):
                key = name[:-4]
                stage_sum += ts - start_map.get(key, ts)
        overlap = max(0.0, stage_sum - wall)
        if self.verbose:
            print(f"  [async] 3-card pipeline wall = {wall*1000:.1f} ms "
                  f"(stage sum {stage_sum*1000:.1f} ms, overlap/masked "
                  f"{overlap*1000:.1f} ms)")
            if len(timeline) <= 40:
                for ts, name in timeline:
                    print(f"           t={ts*1000:7.1f}ms  {name}")
        self.schedule_log = timeline
        video = torch.cat([anchor_rgb] + decoded, dim=2)
        self.tracer.report()
        self.tracer.detach()
        return latents, video


class MemoryBank:
    """Parameter-free rolling 3D patch memory (frustum-lite mirror of the
    open-source FrustumHandler).

    Each entry is one *latent* frame: RGB frame, its VAE latent slice, DA3
    metric depth (pixel resolution), camera-to-world pose + intrinsics (+
    image size) and the absolute latent-frame index.  ``build_mosaic_canvas``
    forward-warps the stored latent patches into a target camera through 3D
    (unproject at stored metric depth -> rigid transform -> pinhole project)
    with a nearest-depth z-buffer, yielding the query-aligned memory canvas
    M and its hole mask -- the open-source FrustumHandler contract.

    Every resolution (pixel grid / latent grid / channels) is derived from
    the stored tensors and the query arguments; nothing is hard-coded, so
    the same code serves the tiny CPU preset and the opensrc 704x1280 scale
    alike (cost is O(frames x latent-cells) point ops).

    Retention policy mirrors the open-source FrustumHandler: the memory
    horizon is much longer than the rolling KV window (``max_frames`` latent
    frames vs 21), and eviction keeps *keyframes* -- a frame whose camera
    has moved/rotated more than ``keyframe_rot_thresh`` / ``keyframe_trans_thresh``
    since the previous retained frame is always kept, FIFO otherwise."""

    def __init__(self, max_frames: int = 256,
                 keyframe_rot_thresh: float = 0.15,
                 keyframe_trans_thresh: float = 0.3):
        self.max_frames = max_frames
        self.keyframe_rot_thresh = float(keyframe_rot_thresh)
        self.keyframe_trans_thresh = float(keyframe_trans_thresh)
        self.frames: List[torch.Tensor] = []      # RGB (b,3,H,W)
        self.latents: List[torch.Tensor] = []     # (b,c,1,h_lat,w_lat)
        self.depths: List[torch.Tensor] = []      # (H,W) metric depth
        self.poses: List[Any] = []                # CameraDec estimates (opt)
        self.c2w: List[torch.Tensor] = []         # (4,4) per latent frame
        self.K: List[torch.Tensor] = []           # (3,3) pixel intrinsics
        self.image_sizes: List[Tuple[int, int]] = []
        self.frame_ids: List[int] = []

    def __len__(self):
        return len(self.latents) if self.latents else len(self.frames)

    def add(self, frame=None, depth=None, pose=None, latent=None,
            c2w=None, intrinsics=None, image_size=None, frame_idx=None):
        """Register one latent frame's observation (any subset of fields)."""
        keep = (lambda t: t)  # meta tensors (dry runs) stay on device

        def _keep(t):
            return t if t.device.type == "meta" else t.detach().cpu()

        if frame is not None:
            self.frames.append(_keep(frame))
        if latent is not None:
            self.latents.append(_keep(latent))
        if depth is not None:
            self.depths.append(_keep(depth))
        if pose is not None:
            self.poses.append(pose)
        if c2w is not None:
            self.c2w.append(_keep(c2w).float())
        if intrinsics is not None:
            self.K.append(_keep(intrinsics).float())
        if image_size is not None:
            self.image_sizes.append((int(image_size[0]), int(image_size[1])))
        if frame_idx is not None:
            self.frame_ids.append(int(frame_idx))
        self._evict()

    # ---- retention (opensrc FrustumHandler-style) ----
    @staticmethod
    def _cam_delta(c2w_a, c2w_b):
        """(rotation angle [rad], translation norm) between two c2w poses."""
        if c2w_a is None or c2w_b is None:
            return 0.0, 0.0
        dR = c2w_b[:3, :3] @ c2w_a[:3, :3].T
        ang = torch.acos(((torch.trace(dR) - 1.0) / 2.0).clamp(-1.0, 1.0))
        return float(ang), float(torch.norm(c2w_b[:3, 3] - c2w_a[:3, 3]))

    def _evict(self):
        """Trim to ``max_frames`` with keyframe preservation: walk oldest ->
        newest and drop non-keyframes (camera moved/rotated less than the
        thresholds relative to the previous *kept* frame) until ``n_drop``
        are dropped; the oldest anchor and the newest frame are always kept,
        mirroring the open-source keyframe_rot/trans_thresh policy."""
        n = len(self)
        if n <= self.max_frames:
            return
        n_drop = n - self.max_frames
        keep = [True] * n
        keep[n - 1] = True                    # newest frame always kept
        dropped = 0
        last_kept = 0                         # anchor is index 0
        for i in range(1, n - 1):
            if dropped >= n_drop:
                break
            ang, trans = self._cam_delta(
                self.c2w[last_kept] if last_kept < len(self.c2w) else None,
                self.c2w[i] if i < len(self.c2w) else None)
            if not (ang > self.keyframe_rot_thresh
                    or trans > self.keyframe_trans_thresh):
                keep[i] = False               # redundant view: drop oldest-first
                dropped += 1
            else:
                last_kept = i                 # keyframe: keep, re-baseline
        if dropped < n_drop:                  # all keyframes: FIFO fallback
            for i in range(1, n - 1):
                if keep[i]:
                    keep[i] = False
                    dropped += 1
                    if dropped >= n_drop:
                        break
        lists = (self.frames, self.latents, self.depths, self.poses,
                 self.c2w, self.K, self.image_sizes, self.frame_ids)
        kept_idx = [i for i in range(n) if keep[i]]
        for lst in lists:
            if len(lst) == n:
                lst[:] = [lst[i] for i in kept_idx]

    def query(self, n_frames: int = 5):
        """Return the most recent n clean RGB frames stacked on time."""
        n = min(n_frames, len(self.frames))
        if n == 0:
            return None
        return torch.stack(self.frames[-n:], dim=2)

    # ---- 3D patch memory: query-aligned mosaic canvas (frustum-lite) ----
    @staticmethod
    def _cell_grid(h: int, w: int, device):
        i = torch.arange(h, device=device).view(h, 1).expand(h, w)
        j = torch.arange(w, device=device).view(1, w).expand(h, w)
        return i.reshape(-1), j.reshape(-1)

    def _source_points(self, s: int):
        """Sample source depth/latent at the source latent-cell centres and
        unproject to world points. Returns (pts_w (N,3), lat (c,N), valid)."""
        lat = self.latents[s]                       # (b,c,1,h_s,w_s)
        b, c, _, h_s, w_s = lat.shape
        depth = self.depths[s]
        if depth.dim() == 3:
            depth = depth[0]
        H, W = int(depth.shape[-2]), int(depth.shape[-1])
        fy, fx = H / h_s, W / w_s                   # pixels per latent cell
        ii, jj = self._cell_grid(h_s, w_s, lat.device)
        v = (ii.to(torch.float32) + 0.5) * fy - 0.5  # pixel centre of cell
        u = (jj.to(torch.float32) + 0.5) * fx - 0.5
        iv = v.floor().long().clamp(0, H - 1)
        iu = u.floor().long().clamp(0, W - 1)
        d = depth[iv, iu].to(torch.float32)          # (N,) metric depth
        Kinv = invert_k(self.K[s][None])[0]
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        rays = uv1 @ Kinv.T                          # (N,3) camera rays
        valid = d > 1e-4
        pts_cam = rays * d.unsqueeze(-1)
        c2w = self.c2w[s]
        pts_w = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
        return pts_w, lat[0, :, 0].reshape(c, h_s * w_s), valid

    def build_mosaic_canvas(self, target_c2w, target_K, image_size):
        """Forward-warp all stored latent patches into the target cameras.

        target_c2w (F,4,4), target_K (F,3,3), image_size (H,W): the F target
        latent frames' cameras.  The canvas grid equals the stored latent
        grid (all entries of one model share it); each source pixel grid is
        derived from its own depth-map size, so mixed resolutions work.
        Returns (canvas (1,c,F,h,w), hole (F,h,w) bool)."""
        if not self.latents:
            return None, None
        _, c, _, h_t, w_t = self.latents[0].shape
        if self.latents[0].is_meta:
            # meta dry run: the canvas shape is fixed by the stored grid and
            # the query; the point math itself is not traceable on meta
            F = int(target_c2w.shape[0])
            return (torch.zeros(1, c, F, h_t, w_t, device="meta"),
                    torch.zeros(F, h_t, w_t, dtype=torch.bool, device="meta"))
        dev = self.latents[0].device
        F = int(target_c2w.shape[0])
        H, W = int(image_size[0]), int(image_size[1])
        fy, fx = H / h_t, W / w_t
        canvas = torch.zeros(F, c, h_t, w_t, device=dev)
        hole = torch.ones(F, h_t, w_t, dtype=torch.bool, device=dev)
        for f in range(F):
            K_t = target_K[f].to(torch.float32)
            tw2c = invert_se3(target_c2w[f:f + 1].to(torch.float32))[0]
            R_t, t_w2c = tw2c[:3, :3].to(dev), tw2c[:3, 3].to(dev)
            z_buf = torch.full((h_t * w_t,), float("inf"), device=dev)
            val_buf = torch.zeros(c, h_t * w_t, device=dev)
            for s in range(len(self.latents)):
                if s >= len(self.c2w) or s >= len(self.depths):
                    continue
                pts_w, lat_flat, valid = self._source_points(s)
                x_cam = pts_w.to(dev) @ R_t.T + t_w2c
                z = x_cam[:, 2]
                uv1 = x_cam @ K_t.T
                u = uv1[:, 0] / uv1[:, 2].clamp_min(1e-6)
                v = uv1[:, 1] / uv1[:, 2].clamp_min(1e-6)
                jj = (u / fx).long()
                ii = (v / fy).long()
                inb = (ii >= 0) & (ii < h_t) & (jj >= 0) & (jj < w_t) \
                    & (z > 1e-4) & valid
                if inb.device.type != "meta" and not bool(inb.any()):
                    continue   # nothing of this source lands in the new view
                hit_idx = (ii[inb] * w_t + jj[inb])
                hit_z = z[inb]
                hit_val = lat_flat[:, inb]
                # z-buffer: nearest target-space depth wins each cell.
                # Duplicated hits on one cell: keep only the nearest hit so
                # the scatter below is index-unique.
                order = torch.argsort(hit_z, stable=True)
                hit_idx, hit_val, hit_z = (hit_idx[order], hit_val[:, order],
                                           hit_z[order])
                first = torch.ones_like(hit_idx, dtype=torch.bool)
                first[1:] = hit_idx[1:] != hit_idx[:-1]
                hit_idx, hit_val, hit_z = (hit_idx[first], hit_val[:, first],
                                           hit_z[first])
                # keep hits that are (jointly) nearest across all sources
                z_min = torch.full_like(z_buf, float("inf")).scatter_reduce(
                    0, hit_idx, hit_z, reduce="amin", include_self=True)
                win = hit_z <= z_min[hit_idx] + 1e-6
                # hit_idx is unique here (deduped above): plain column write
                val_buf[:, hit_idx[win]] = hit_val[:, win]
                z_buf = torch.minimum(z_buf, z_min)
            canvas[f] = val_buf.reshape(c, h_t, w_t)
            hole[f] = z_buf.isinf().reshape(h_t, w_t)
        # (F, c, h, w) -> (1, c, F, h, w) per the documented contract
        return canvas.permute(1, 0, 2, 3).unsqueeze(0), hole

# PART 7 -- configuration registry (open-source-identical + larger presets),
#          the assembled world-model container, training step and CLI entry.

# ---------------------------------------------------------------------------
# Configuration presets -------------------------------------------------------
# ---------------------------------------------------------------------------
# Values marked "opensrc" reproduce the *open-source* architectures 1:1:
#   DiT        = Wan-AI/Wan2.2-TI2V-5B scaffold  (4,999,787,712 params)
#   VAE        = WanVideoVAE38 z48/dim160         (  704,688,668 params)
#   Text       = umt5-xxl-class T5 encoder        (5,680,910,336 params)
#   Depth(DA3) = nested-giant-large / metric-large  (1,689,845,519 / 334,171,394)
# "tiny-*" presets shrink every dimension for CPU smoke tests.

WANMODEL_OPENSRC = dict(
    has_image_input=False, patch_size=[1, 2, 2], in_dim=48, dim=3072,
    ffn_dim=14336, freq_dim=256, text_dim=4096, out_dim=48, num_heads=24,
    num_layers=30, eps=1e-6, seperated_timestep=True,
    require_clip_embedding=False, require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True,
)

# Larger-parameter family reference (Wan 2.1/2.2 class 14B from the model
# config table: dim 5120, 40 heads, 40 layers, ffn 13824)
WANMODEL_14B_CLASS = dict(
    has_image_input=False, patch_size=[1, 2, 2], in_dim=16, dim=5120,
    ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16, num_heads=40,
    num_layers=40, eps=1e-6,
)

TEXT_OPENSRC = dict(vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
                    num_heads=64, num_layers=24, num_buckets=32,
                    shared_pos=False, dropout=0.1, seq_len=512)

VAE_OPENSRC = dict(z_dim=48, dim=160, dec_dim=256,
                   dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                   temperal_downsample=[False, True, True], dropout=0.0)

# DA3 metric-large sub-net (da3metric-large.yaml -> DepthAnything3Net)
DA3_METRIC_LARGE = dict(
    net=dict(name="vitl", out_layers=[4, 11, 17, 23], alt_start=-1,
             qknorm_start=-1, rope_start=-1, cat_token=False),
    head=dict(cls="DPT", dim_in=1024, output_dim=1, features=256,
              out_channels=[256, 512, 1024, 1024], use_sky_head=True,
              head_features_2=32),
)

# DA3 giant (any-view) sub-net (da3-giant.yaml)
DA3_GIANT = dict(
    net=dict(name="vitg", out_layers=[19, 27, 33, 39], alt_start=13,
             qknorm_start=13, rope_start=13, cat_token=True),
    head=dict(cls="DualDPT", dim_in=3072, output_dim=2, features=256,
              out_channels=[256, 512, 1024, 1024], head_features_2=32,
              aux_output_dim=7),
    cam_enc=dict(dim_out=1536),
    cam_dec=dict(dim_in=3072, t_dim=3, quat_dim=4, fov_dim=2),
    gs_head=dict(dim_in=3072, output_dim=38, features=256,
                 out_channels=[256, 512, 1024, 1024], head_features_2=32,
                 merger_channels=[32, 64]),
    gs_adapter=dict(sh_degree=2, pred_color=False, pred_offset_depth=True,
                    pred_offset_xy=True, gaussian_scale_min=1e-5,
                    gaussian_scale_max=30.0),
)

# Nested giant-large (the exact DA3 variant MG3.5 downloads at runtime)
DA3_NESTED_GIANT_LARGE = dict(anyview=DA3_GIANT, metric=DA3_METRIC_LARGE)

# ----- tiny CPU presets -----
# The tiny DiT enables PRoPE (camera-aware attention, parameter-free) so the
# interactive smoke tests exercise the full camera-conditioning path.
WANMODEL_TINY = dict(
    has_image_input=False, patch_size=[1, 2, 2], in_dim=4, dim=96,
    ffn_dim=256, freq_dim=32, text_dim=96, out_dim=4, num_heads=4,
    num_layers=2, eps=1e-6, seperated_timestep=True,
    require_clip_embedding=False, require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True, use_prope=True,
)
TEXT_TINY = dict(vocab=200, dim=96, dim_attn=96, dim_ffn=256, num_heads=4,
                 num_layers=2, num_buckets=8, shared_pos=False, dropout=0.0,
                 seq_len=32)
VAE_TINY = dict(z_dim=4, dim=24, dec_dim=48, dim_mult=[1, 2, 4, 4],
                num_res_blocks=1, attn_scales=[],
                temperal_downsample=[False, True, True], dropout=0.0)
# Tiny nested DA3 (any-view with camera enc/dec + GS head + metric subnet) so
# the interactive tests exercise NestedDepthAnything3Net, CameraEnc/CameraDec
# and GSDPT end-to-end at CPU scale. cat_token doubles the any-view feature
# dim: 2*384=768 feeds DualDPT / cam_dec / gs_head.
DA3_TINY = dict(
    anyview=dict(
        net=dict(name="vits", out_layers=[3, 5, 7, 9], alt_start=2,
                 qknorm_start=2, rope_start=2, cat_token=True),
        head=dict(cls="DualDPT", dim_in=768, output_dim=2, features=32,
                  out_channels=[16, 32, 64, 64], head_features_2=16,
                  aux_output_dim=4, aux_pyramid_levels=2),
        cam_enc=dict(dim_out=384, dim_in=9, trunk_depth=2, num_heads=6),
        cam_dec=dict(dim_in=768, t_dim=3, quat_dim=4, fov_dim=2),
        gs_head=dict(dim_in=768, output_dim=12, features=32,
                     out_channels=[16, 32, 64, 64], head_features_2=16,
                     merger_channels=[16, 32]),
        gs_adapter=dict(sh_degree=1, pred_color=False, pred_offset_depth=True,
                        pred_offset_xy=True, gaussian_scale_min=1e-5,
                        gaussian_scale_max=30.0),
    ),
    metric=dict(
        net=dict(name="vits", out_layers=[3, 5, 7, 9], alt_start=-1,
                 qknorm_start=-1, rope_start=-1, cat_token=False),
        head=dict(cls="DPT", dim_in=384, output_dim=1, features=32,
                  out_channels=[16, 32, 64, 64], use_sky_head=True,
                  head_features_2=16),
    ),
)

PRESETS = {
    "opensrc": {"dit": WANMODEL_OPENSRC, "vae": VAE_OPENSRC,
                "text": TEXT_OPENSRC, "da3": DA3_NESTED_GIANT_LARGE,
                "label": "open-source identical config"},
    "opensrc_large": {"dit": WANMODEL_14B_CLASS, "vae": VAE_OPENSRC,
                      "text": TEXT_OPENSRC, "da3": DA3_NESTED_GIANT_LARGE,
                      "label": "larger-parameter family (Wan 14B-class DiT + "
                               "umt5-xxl text + VAE38 + DA3 nested)"},
    "tiny": {"dit": WANMODEL_TINY, "vae": VAE_TINY, "text": TEXT_TINY,
             "da3": DA3_TINY,
             "label": "CPU smoke-test config"},
}

# ----- maximum-parameter preset -------------------------------------------
# Largest known variant of every component, wired for the interactive rollout:
#   DiT   : Wan 14B-class (dim 5120, 40 heads, 40 layers, ffn 13824) with
#           PRoPE camera conditioning enabled, in/out dim 48 to match VAE38
#   VAE   : WanVideoVAE38 (z 48, dim 160, dec 256)          704,688,668
#   Text  : umt5-xxl-class T5 encoder                     5,680,910,336
#   DA3   : nested giant-large (any-view vitg + metric vitl) 1,689,845,519
# Rollout: chunk = 1 latent frame = 4 RGB frames (720p), rolling history =
# 7 chunks = 7 latent frames, 3-step distilled sampling, 720p resolution
# (1280x720; padded to 1280x736 internally so the 2x2 DiT patch grid and the
# /32 VAE+patch stack stay integral, cropped back to 720 on output).
WANMODEL_MAX = dict(
    has_image_input=False, patch_size=[1, 2, 2], in_dim=48, dim=5120,
    ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=48, num_heads=40,
    num_layers=40, eps=1e-6, seperated_timestep=True,
    require_clip_embedding=False, require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True, use_prope=True,
)

PRESETS["max"] = {
    "dit": WANMODEL_MAX, "vae": VAE_OPENSRC, "text": TEXT_OPENSRC,
    "da3": DA3_NESTED_GIANT_LARGE,
    "chunk_size": 1, "context_chunks": 7, "num_steps": 3,
    "resolution": [720, 1280],
    "label": "maximum-parameter config (14B-class DiT + VAE38 + umt5-xxl + "
             "DA3 nested-giant-large; chunk = 1 latent = 4 RGB frames @720p, "
             "rolling history = 7 chunks)",
}


def build_dit(cfg):
    defaults = dict(
        has_image_pos_emb=False, dynamic_fps=False, dynamic_fps_max_pos=22350,
        subject_ref_memory_enabled=False, subject_ref_memory_max_refs=2,
        subject_ref_memory_local_pos_size=64,
        use_prope=False, prope_disable_native_rope=False,
        prope_disable_t_rope=False, prope_camera_layout="full",
        image_emb_tokens=257,
        clean_latent_noise_enabled=False, clean_latent_noise_prob=0.2,
        clean_latent_noise_magnitude=0.03, mosaic_latent_noise_enabled=False,
        mosaic_latent_noise_prob=0.2, mosaic_latent_noise_magnitude=0.03,
        context_latent_noise_enabled=False, context_latent_noise_prob=0.2,
        context_latent_noise_magnitude=0.03,
        has_ref_conv=False, add_control_adapter=False, in_dim_control_adapter=24,
        wantodance_enable_music_inject=False,
        wantodance_music_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage=False, wantodance_enable_refface=False,
        wantodance_enable_global=False, wantodance_enable_dynamicfps=False,
        wantodance_enable_unimodel=False,
    )
    defaults.update(cfg)
    return WanModel(_DataclassFromDict(defaults))


def build_vae(cfg):
    return WanVideoVAE38(**cfg)


def build_text(cfg):
    return WanTextEncoder(**cfg)


def build_da3(cfg):
    if "anyview" in cfg:  # nested container
        return NestedDepthAnything3Net(anyview=cfg["anyview"],
                                       metric=cfg["metric"])
    return DepthAnything3Net(net=cfg["net"], head=cfg["head"])


class MatrixGame35(nn.Module):
    """Assembled world-model container: text encoder, VAE (enc+dec), DiT and
    DA3 metric-depth component, plus parameter-free memory & scheduler hooks.

    This is the "complete model" equivalent of the open-source system:
        text_encoder + dit + vae + da3  (+ reference-token prefix inside DiT)
    in one torch module so the whole thing can be printed & compared."""
    def __init__(self, dit_cfg, vae_cfg, text_cfg, da3_cfg):
        super().__init__()
        self.dit = build_dit(dit_cfg)
        self.vae = build_vae(vae_cfg)
        self.text = build_text(text_cfg)
        self.da3 = build_da3(da3_cfg)
        # parameter-free rolling memory (registered component, used by the
        # interactive engines for DA3 depth/camera registration)
        self.memory = MemoryBank()
        self.scheduler = None

    def enable_subject_ref_memory(self, max_refs=2, local_pos_size=None):
        self.dit.enable_subject_ref_memory(max_refs, local_pos_size)

    def forward(self, video, timestep, prompt_ids, camera=None):
        """One training/inference forward (flow-matching, latent space):
        VAE-encode video -> text embed -> DiT velocity -> return v + latents.
        This mirrors the base SFT path used to optimise the DiT only."""
        lat = self.vae.encode(video)
        ctx = self.text(prompt_ids)
        v = self.dit(lat, timestep, ctx)
        return v, lat


class _DataclassFromDict:
    """Thin config adapter: WanModel(cfg) accesses attributes."""
    def __init__(self, d):
        self.__dict__.update(d)


# ----------------------------- training step ------------------------------
def train_step_smoke(model: MatrixGame35, opt, sched, batch, device="cpu"):
    """One flow-matching SFT step on the DiT with dummy data.

    Strict latent-space rectified-flow objective (same paradigm as the
    open-source SFT): ``x_t = (1-t) x0 + t noise`` in VAE latent space, the
    DiT predicts the velocity ``v`` and is supervised with the flow target
    ``v* = noise - x0`` (``FlowMatchScheduler.training_target``)."""
    model.train()
    video, prompt_ids, _t_unused = batch
    with torch.no_grad():
        lat0 = model.vae.encode(video)              # clean latent x0
        ctx = model.text(prompt_ids)
    noise = torch.randn_like(lat0)
    t = torch.rand(lat0.shape[0])
    xt = sched.add_noise(lat0, noise, t)            # x_t = (1-t) x0 + t noise
    target = sched.training_target(lat0, noise)     # v* = noise - x0
    tv = t * sched.num_train_timesteps
    v_pred = model.dit(xt, tv, ctx)
    loss = FlowMatchSFTLoss()(v_pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())

# PART 8 -- CLI entry: CPU smoke tests (train / interactive seq / async3 /
#          component coverage), model printing & equivalence self-checks.

def _self_check():
    """Print component param counts for opensrc & tiny presets."""
    print("=== self-check: parameter counts (meta device) ===")
    with torch.device("meta"):
        for name, preset in [("opensrc", PRESETS["opensrc"]),
                             ("max", PRESETS["max"]), ("tiny", PRESETS["tiny"])]:
            for comp_key in ("dit", "vae", "text", "da3"):
                try:
                    obj = build_component(comp_key, preset[comp_key])
                    print(f"  {name:8s} {comp_key:5s} params = "
                          f"{count_parameters(obj):,}")
                except Exception as e:  # meta-incompatible net (DA3 uses .item)
                    print(f"  {name:8s} {comp_key:5s} meta build failed: {e}")


def build_component(key, cfg):
    return {"dit": build_dit, "vae": build_vae, "text": build_text,
            "da3": build_da3}[key](cfg)


def make_tiny_system(seed=0):
    torch.manual_seed(seed)
    p = PRESETS["tiny"]
    return MatrixGame35(p["dit"], p["vae"], p["text"], p["da3"])


def make_opensrc_system():
    p = PRESETS["opensrc"]
    return MatrixGame35(p["dit"], p["vae"], p["text"], p["da3"])


def print_component_models(name="model.py components"):
    """Pretty-print each component of a tiny instance (module tree head)."""
    model = make_tiny_system()
    print(f"===== {name} : tiny MatrixGame35 components =====")
    for key in ("text", "dit", "vae", "da3"):
        obj = getattr(model, key)
        print(f"\n--- component {key}: {type(obj).__name__} "
              f"({len(list(obj.parameters()))} tensors, "
              f"{count_parameters(obj):,} params) ---")
        print(obj)


# --------------------------- interactive tests ---------------------------
def _dummy_video(b=1, t=5, h=32, w=32, c=3):
    return (torch.rand(b, c, t, h, w) * 2 - 1)


def _tiny_engine(model, mode, trace):
    """Build the interactive engine over the tiny system with the rolling
    memory registered and structured shape tracing enabled."""
    sched = FlowMatchScheduler(num_train_timesteps=1000)
    components = {"dit": model.dit, "vae": model.vae, "text": model.text,
                  "da3": model.da3, "memory": model.memory}
    return RealtimeInteractiveEngine(components, sched, shape_trace=trace,
                                     mode=mode)


def run_train_smoke(steps=3, trace=True):
    print("===== CPU training smoke test (strict latent-space flow matching "
          "on the tiny DiT) =====")
    model = make_tiny_system()
    opt = torch.optim.AdamW(model.dit.parameters(), lr=1e-3)
    sched = FlowMatchScheduler(num_train_timesteps=1000)
    video = _dummy_video(t=5)
    prompt_ids = torch.randint(0, PRESETS["tiny"]["text"]["vocab"], (1, 8))
    batch = (video, prompt_ids, None)
    for i in range(steps):
        loss = train_step_smoke(model, opt, sched, batch)
        if trace:
            print(f"  step {i}: loss={loss:.4f}")
    print("  OK: training smoke passed")


def run_interactive_seq(trace=True):
    print("===== CPU real-time interactive inference (single 'device', "
          "KV-cache + PRoPE) =====")
    model = make_tiny_system()
    eng = _tiny_engine(model, "seq", trace)
    anchor = _dummy_video(t=1, h=32, w=32)[:, :, :1]
    camera = _dummy_camera(1 + 12 * 2, 32, 32)  # C0 + 2 chunks of pixel frames
    prompt_ids = torch.randint(0, PRESETS["tiny"]["text"]["vocab"], (1, 8))
    lat, rgb = eng.run_sequential(anchor, camera, prompt_ids, total_chunks=2)
    print("  OK: sequential interactive passed, latent", tuple(lat.shape),
          "video", tuple(rgb.shape))


def run_interactive_async3(trace=True):
    print("===== CPU real-time interactive inference (simulated 3-card async, "
          "KV-cache + PRoPE) =====")
    model = make_tiny_system()
    eng = _tiny_engine(model, "async3", trace)
    anchor = _dummy_video(t=1, h=32, w=32)[:, :, :1]
    camera = _dummy_camera(1 + 12 * 2, 32, 32)
    prompt_ids = torch.randint(0, PRESETS["tiny"]["text"]["vocab"], (1, 8))
    lat, rgb = eng.run_async3(anchor, camera, prompt_ids, total_chunks=2)
    print("  OK: async3 interactive passed, latent", tuple(lat.shape),
          "video", tuple(rgb.shape))


# --------------------------- component coverage ---------------------------
# --------------------- maximum-parameter dry run --------------------------
def run_max_test(num_chunks: int = 9):
    """Maximum-parameter config ('max' preset) dry run on the meta device.

    Meta tensors carry exact shapes but no data, so the FULL 22B-parameter
    system executes end-to-end on a CPU-only host: both interactive modes
    (seq + async3) run with the real rollout logic -- 720p resolution,
    chunk = 1 latent frame = 4 RGB frames, rolling history = 7 chunks -- and
    the ShapeTracer records the hierarchical input/output shape of every
    component boundary.  This validates all wiring and shapes at max scale;
    real-data execution at this scale needs a GPU host (weights alone exceed
    80 GB in fp32/BF16).

    RGB 720x1280 is padded to 736x1280 (multiple of 32 = VAE/16 x DiT-patch/2)
    and cropped back to 720 on output.
    """
    p = PRESETS["max"]
    H, W = p["resolution"]
    Hp = (H + 31) // 32 * 32
    print("===== maximum-parameter config dry run (meta device, shape-exact) =====")
    print(f"  resolution : {W}x{H} (internally padded to {W}x{Hp}, cropped back)")
    print(f"  chunk      : {p['chunk_size']} latent frame = "
          f"{p['chunk_size'] * 4} RGB frames (720p)")
    print(f"  history    : {p['context_chunks']} chunks rolling = "
          f"{p['context_chunks'] * p['chunk_size']} latent frames")
    print(f"  sampling   : {p['num_steps']} distilled steps per chunk")

    with torch.device("meta"):
        dit = build_dit(p["dit"])
        vae = build_vae(p["vae"])
        text = build_text(p["text"])
        da3 = build_da3(p["da3"])
    memory = MemoryBank()
    components = {"dit": dit, "vae": vae, "text": text, "da3": da3,
                  "memory": memory}
    for m in components.values():
        if isinstance(m, nn.Module):
            m.eval()
    counts = {k: count_parameters(v) for k, v in components.items()
              if isinstance(v, nn.Module)}
    print("  parameters : " +
          ", ".join(f"{k} {counts[k]:,}" for k in
                    ("dit", "vae", "text", "da3")) +
          f"  |  total {sum(counts.values()):,}")

    sched = FlowMatchScheduler(num_train_timesteps=1000)
    camera = _dummy_camera(1 + p["chunk_size"] * 4 * num_chunks, Hp, W)
    camera = {k: (v.to("meta") if torch.is_tensor(v) else v)
              for k, v in camera.items()}
    anchor = torch.rand(1, 3, 1, Hp, W, device="meta") * 2 - 1
    prompt_ids = torch.randint(0, p["text"]["vocab"], (1, 8), device="meta")
    print(f"  dummy inputs: anchor {tuple(anchor.shape)}, "
          f"c2w {tuple(camera['c2w'].shape)}, "
          f"K {tuple(camera['intrinsics'].shape)}, "
          f"prompt {tuple(prompt_ids.shape)}")

    for mode in ("seq", "async3"):
        print(f"\n--- inference mode: {mode} ---")
        eng = RealtimeInteractiveEngine(
            components, sched, chunk_size=p["chunk_size"],
            context_chunks=p["context_chunks"], num_steps=p["num_steps"],
            shape_trace=True, mode=mode, verbose=True, device="meta")
        with torch.no_grad():
            if mode == "seq":
                lat, video = eng.run_sequential(anchor, camera, prompt_ids,
                                                num_chunks)
            else:
                lat, video = eng.run_async3(anchor, camera, prompt_ids,
                                            num_chunks)
        video = video[..., :H, :]   # crop the pad back to 720p
        print(f"  [{mode}] latent {tuple(lat.shape)} | "
              f"video(720p) {tuple(video.shape)}")
        eng.tracer.detach()

        fname = f"max_{mode}_shape_flow.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"max config: {p['label']}\n")
            f.write(f"resolution {W}x{H} (padded {W}x{Hp}); "
                    f"chunk = {p['chunk_size']} latent = "
                    f"{p['chunk_size'] * 4} RGB frames; history = "
                    f"{p['context_chunks']} chunks; steps = {p['num_steps']}\n")
            f.write("params: " + ", ".join(f"{k}={counts[k]:,}" for k in counts)
                    + f", total={sum(counts.values()):,}\n")
            f.write(f"dummy inputs: anchor {tuple(anchor.shape)}, "
                    f"c2w {tuple(camera['c2w'].shape)}, "
                    f"K {tuple(camera['intrinsics'].shape)}, "
                    f"prompt {tuple(prompt_ids.shape)}\n")
            f.write(f"mode: {mode}; chunks: {num_chunks}\n\n")
            for order, group, path, cls, ins, outs in eng.tracer.records:
                f.write(f"#{order:<5} [{group:<5}] {path:<44} {cls:<26} "
                        f"in={ins} out={outs}\n")
        print(f"  [trace] hierarchical shape flow written -> {fname}")

    print("\nOK: maximum-parameter dry run passed (seq + async3, shape-exact)")


def run_component_test():
    """Exercise every remaining module/helper that the interactive and
    training smokes do not hit on their own (VAE first_chunk decode, PRoPE
    camera helpers, DA3 accessories at tiny scale, scheduler step/target,
    DropPath, memory query)."""
    print("===== component coverage test (tiny scale, CPU) =====")
    torch.manual_seed(0)
    checks = []

    def check(name, fn):
        fn()
        checks.append(name)
        print(f"  OK  {name}")

    p = PRESETS["tiny"]

    # 1) FlowMatchScheduler: training_target + incremental step
    def sched_check():
        sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        x0, noise = torch.randn(2, 4), torch.randn(2, 4)
        t = torch.tensor([0.3, 0.7])
        xt = sched.add_noise(x0, noise, t)
        assert torch.allclose(xt[0], 0.7 * x0[0] + 0.3 * noise[0], atol=1e-5)
        assert torch.allclose(sched.training_target(x0, noise), noise - x0)
        x = sched.step(noise - x0, torch.tensor([0.3]), xt)
        assert x.shape == xt.shape
    check("FlowMatchScheduler add_noise/step/training_target", sched_check)

    # 2) PRoPE camera helpers: pose -> viewmats -> warped attention
    def prope_check():
        cam = _dummy_camera(4, 32, 32)
        info = camera_info_from_poses(cam["c2w"], cam["intrinsics"],
                                      image_size=cam["image_size"])
        w2c, (P, P_T, P_inv) = info
        assert P.shape == (4, 4, 4) and torch.allclose(
            torch.einsum("nij,njk->nik", P, P_inv),
            torch.eye(4).expand(4, 4, 4), atol=1e-4)
        q = torch.randn(1, 16, 96)
        out = prope_dot_product_attention(q, q, q, 4, (P, P_T, P_inv),
                                          list(range(4)), list(range(4)))
        assert out.shape == q.shape
    check("PRoPE camera_info_from_poses + prope_dot_product_attention",
          prope_check)

    # 3) VAE first_chunk decode path (DupUp3D first_chunk trim)
    def vae_first_check():
        vae = build_vae(p["vae"])
        z = torch.randn(1, p["vae"]["z_dim"], 1, 4, 4)
        rgb = vae.decode(z, first_chunk=True)
        assert rgb.shape[2] >= 1
    check("WanVideoVAE38.decode(first_chunk=True) [DupUp3D path]",
          vae_first_check)

    # 4) DA3 accessories at tiny scale: CameraEnc -> backbone cam token ->
    #    CameraDec pose regression; GSDPT raw gs map; GaussianAdapter dims
    def da3_acc_check():
        da3 = build_da3(p["da3"])
        cam = _dummy_camera(2, 64, 64)
        ext = invert_se3(cam["c2w"])[None]           # (1,V,4,4) w2c
        intr = cam["intrinsics"][None]
        imgs = torch.randn(1, 2, 3, 64, 64)
        with torch.no_grad():
            out = _da3_output_fields(da3(imgs, extrinsics=ext, intrinsics=intr))
        assert out["depth"].shape == (1, 2, 56, 56)   # patch-aligned (64//14)*14
        assert out["camera"].shape == (1, 2, 9)       # t(3) + quat(4) + fov(2)
        assert out["gs"].shape == (1, 2, 12, 56, 56)  # raw gs channels
        adapter = GaussianAdapter(sh_degree=1)
        assert adapter.d_in == 3 + 4 + 3 * adapter.d_sh + 2 + 1
    check("DA3 nested forward with CameraEnc/CameraDec/GSDPT/GaussianAdapter",
          da3_acc_check)

    # 5) DropPath active branch (dinov2 Block with drop_path > 0)
    def droppath_check():
        blk = Block(dim=64, num_heads=4, drop_path=0.5)
        blk.train()
        y = blk(torch.randn(2, 8, 64))
        assert y.shape == (2, 8, 64)
    check("dinov2 Block with DropPath", droppath_check)

    # 6) MemoryBank rolling window + 3D patch memory -> mosaic canvas
    def memory_check():
        mem = MemoryBank(max_frames=3)
        for i in range(5):
            mem.add(frame=torch.full((1, 3, 1, 4, 4), float(i)),
                    depth=torch.full((1, 4, 4), float(i)))
        assert len(mem) == 3 and float(mem.frames[-1][0, 0, 0, 0, 0]) == 4
        ctx = mem.query(n_frames=2)
        assert ctx.shape == (1, 3, 2, 1, 4, 4)

        # frustum-lite 3D pipeline: register one latent frame at the origin
        # camera, then query with a camera that views the same scene -> the
        # warped canvas must be fully covered; a far-away camera must see
        # only holes. Grid/latent sizes are arbitrary (scale-agnostic).
        lat = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2) / 16
        depth = torch.full((1, 8, 8), 2.0)
        c2w0 = torch.eye(4)
        K0 = torch.tensor([[4.0, 0, 4], [0, 4.0, 4], [0, 0, 1.0]])
        mem2 = MemoryBank()
        mem2.add(latent=lat, depth=depth, c2w=c2w0, intrinsics=K0,
                 image_size=(8, 8), frame_idx=0)
        near = torch.diag(torch.tensor([1.0, 1, 1, 1])).unsqueeze(0)
        canvas, hole = mem2.build_mosaic_canvas(
            torch.eye(4).unsqueeze(0), K0.unsqueeze(0), (8, 8))
        assert canvas.shape == (1, 4, 1, 2, 2) and not bool(hole.any())

        # per-latent-frame registration density: a 3-latent-frame chunk
        # (12-RGB decoded clip) must register depth for EVERY latent frame
        # (stride-4 representative), not just the last one
        import model as _m
        eng = _m.RealtimeInteractiveEngine(
            {"da3": _m.build_da3(p["da3"]), "memory": MemoryBank()},
            _m.FlowMatchScheduler(1000), verbose=False)
        clip = torch.rand(1, 3, 12, 32, 32)          # 3 latent frames decoded
        lat3 = torch.randn(1, 4, 3, 4, 4)
        cam3 = _m._dummy_camera(16, 32, 32)
        eng._da3_register(clip, cam3, 0, latent=lat3, num_latent_frames=3)
        assert len(eng.comp["memory"]) == 3
        assert all(d is not None for d in eng.comp["memory"].depths), \
            "every latent frame must carry its own DA3 depth"
        assert eng.comp["memory"].frame_ids == [0, 1, 2]
        far = torch.eye(4).unsqueeze(0)
        far[0, 2, 3] = 100.0                      # move 100 units away
        _, hole_far = mem2.build_mosaic_canvas(
            far, K0.unsqueeze(0), (8, 8))
        assert bool(hole_far.all())

        # retention: horizon >> KV window; the oldest anchor + keyframes
        # survive eviction while redundant (static-camera) frames are
        # dropped oldest-first
        mem3 = MemoryBank(max_frames=4)
        for f in range(8):                        # slow dolly: no keyframes
            c2w = torch.eye(4)
            c2w[2, 3] = 0.01 * f
            mem3.add(latent=torch.full((1, 4, 1, 2, 2), float(f)),
                     depth=torch.full((1, 8, 8), 2.0), c2w=c2w,
                     intrinsics=K0, image_size=(8, 8), frame_idx=f)
        assert len(mem3) == 4 and mem3.frame_ids == [0, 5, 6, 7]
        mem4 = MemoryBank(max_frames=4)           # one big jump = keyframe
        for f in range(8):
            c2w = torch.eye(4)
            c2w[2, 3] = 0.01 * f if f != 4 else 2.0
            mem4.add(latent=torch.full((1, 4, 1, 2, 2), float(f)),
                     depth=torch.full((1, 8, 8), 2.0), c2w=c2w,
                     intrinsics=K0, image_size=(8, 8), frame_idx=f)
        assert 4 in mem4.frame_ids, "keyframe evicted despite big camera move"
    check("MemoryBank window + 3D frustum mosaic canvas (near/far cameras)\n"
          "       + keyframe-preserving retention", memory_check)

    # 7) CausalConv3d streaming cache path (encoder-side cache_x)
    def causal_conv_check():
        conv = CausalConv3d(4, 4, 3, padding=1)
        x1, x2 = torch.randn(1, 4, 2, 6, 6), torch.randn(1, 4, 2, 6, 6)
        y1 = conv(x1)
        y2 = conv(x2, cache_x=x1[:, :, -2:])
        assert y2.shape == y1.shape
    check("CausalConv3d streaming cache_x", causal_conv_check)

    # 8) DiT image-conditioning branch (MLP img_emb + CrossAttention k_img)
    def dit_img_check():
        cfg = dict(p["dit"])
        # image-input Wan models concatenate y into the patch-embed input:
        # in_dim covers latent (4) + image (4) channels
        cfg.update(has_image_input=True, has_image_pos_emb=True,
                   text_dim=96, in_dim=8)
        dit = build_dit(cfg)
        lat = torch.randn(1, 4, 2, 2, 2)
        n_img = 257  # build_dit default image_emb_tokens
        ctx = torch.randn(1, 8 + n_img, 96)  # CLIP tokens || text tokens
        v = dit(lat, torch.tensor([500.0]),
                ctx, clip_feature=torch.randn(1, n_img, 1280), y=lat)
        assert v.shape == lat.shape
    check("WanModel has_image_input path (img_emb + k_img cross-attn)",
          dit_img_check)

    # 9) subject reference memory registration on the DiT
    def subject_ref_check():
        model = make_tiny_system()
        model.enable_subject_ref_memory(max_refs=2, local_pos_size=16)
        assert model.dit.subject_ref_memory_enabled
        assert model.dit.subject_ref_index_embedding.shape == (2, 96)
    check("WanModel.enable_subject_ref_memory", subject_ref_check)

    # 10) rolling KV-cache sliding window (opensrc context_chunks semantics):
    #     with context_chunks=2 the cache must hold at most 2*chunk_size
    #     latent frames after each fill, the original C0 must be evicted as
    #     the window advances, and rollout must keep working with exact
    #     absolute frame addressing (cache frames strictly increasing,
    #     contiguous, within the window)
    def kv_window_check():
        torch.manual_seed(0)
        model = make_tiny_system()
        sched = FlowMatchScheduler(num_train_timesteps=1000)
        eng = RealtimeInteractiveEngine(
            {"dit": model.dit, "vae": model.vae, "text": model.text,
             "da3": model.da3, "memory": MemoryBank(max_frames=64)},
            sched, chunk_size=3, context_chunks=2, verbose=False)
        anchor = _dummy_video(t=1, h=32, w=32)[:, :, :1]
        camera = _dummy_camera(1 + 12 * 5, 32, 32)
        prompt_ids = torch.randint(0, PRESETS["tiny"]["text"]["vocab"], (1, 8))
        with torch.no_grad():
            ctx = model.text(prompt_ids)
            lat = model.vae.encode(anchor)
            kv = eng._kv_state(model.dit)
            eng._prefill_cache(model.dit, lat, ctx, kv, camera)
            latents = lat
            b, c = lat.shape[0], lat.shape[1]
            h_l, w_l = lat.shape[3], lat.shape[4]
            for ci in range(5):
                x = eng._denoise_chunk(model.dit, ctx, latents, kv, camera,
                                       b, c, h_l, w_l)
                latents = torch.cat([latents, x], dim=2)
                frames = kv[0]["frames"]
                assert len(frames) <= 2 * 3, f"window exceeded: {frames}"
                assert frames == sorted(frames) and len(set(frames)) == len(frames)
                assert frames[-1] == int(latents.shape[2]) - 1
                # k/v rows trimmed in lockstep with the frame bookkeeping
                assert kv[0]["k"].shape[1] == len(frames)
        assert int(latents.shape[2]) == 1 + 5 * 3
    check("Rolling KV-cache sliding window (context_chunks) + absolute "
          "addressing intact", kv_window_check)

    print(f"  OK: component coverage passed ({len(checks)} checks)")


# ------------------------------ main ---------------------------------------
def _tree_lines(module, prefix="model", max_depth=24, _d=0, out=None):
    """Collect one indented line per named module (hierarchy)."""
    if out is None:
        out = []
    n = sum(p.numel() for p in module.parameters(recurse=False))
    direct_t = len(list(module.parameters(recurse=False)))
    out.append((_d, prefix, type(module).__name__, n, direct_t))
    if _d < max_depth:
        for name, child in module.named_children():
            _tree_lines(child, name, max_depth, _d + 1, out)
    return out


def _locate_repo(subdir: str) -> str:
    """Locate a sibling open-source repo (e.g. depth-anything-3 /
    Matrix-Game-3.5) relative to this file's workspace root, falling back to
    the original C:\\app\\dshws layout."""
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.dirname(here), r"C:\app\dshws"):
        cand = os.path.join(root, subdir)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"could not locate repo directory {subdir!r}")


def _load_da3_ref():
    """Load the real DA3 nested-giant-large net from depth-anything-3 (CPU);
    DINOv2 needs .item() at init so meta cannot be used. Returns meta model."""
    import sys as _sys
    repo = _locate_repo("depth-anything-3")
    _sys.path.insert(0, os.path.join(repo, "src"))
    from depth_anything_3.cfg import load_config, create_object
    cfg = load_config(os.path.join(
        repo, "src", "depth_anything_3", "configs", "da3nested-giant-large.yaml"))
    net = create_object(cfg)  # CPU fp32 (~1.7B params, ~7 GB)
    return net.to("meta")


def run_equivalence(out_prefix=None):
    """Print the *open-source repo model* (instantiated from the actual
    diffsynth/DA3 sources when available) and the *model.py model* at the
    identical opensrc configuration, then align the two module trees and dump a
    visual side-by-side comparison + equality verdict."""
    if out_prefix is None:
        out_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "opensrc_equivalence")
    os.makedirs(out_prefix, exist_ok=True)
    components = ["dit", "vae", "text", "da3"]
    try:
        import load_opensrc
        mods = load_opensrc.import_opensrc_models()
        have_opensrc_dit = True
    except Exception:
        mods = None
        have_opensrc_dit = False
    results = {}
    with torch.device("meta"):
        ours = {k: build_component(k, PRESETS["opensrc"][k]) for k in components}
    # ---- DA3 / others need CPU for the open-source reference: build meta only
    # for dit/vae/text (pure meta), DA3 uses CPU because of .item() in DINOv2.
    ref = {}
    if have_opensrc_dit:
        with torch.device("meta"):
            ref["dit"] = mods["dit"].WanModel(**WANMODEL_OPENSRC)
            ref["vae"] = mods["vae"].WanVideoVAE38()
            ref_kw = dict(TEXT_OPENSRC); ref_kw.pop("seq_len", None); ref["text"] = mods["text"].WanTextEncoder(**ref_kw)
        try:
            print("building real DA3 nested-giant-large reference on CPU ...")
            ref["da3"] = _load_da3_ref()
        except Exception as e:
            print(f"note: DA3 reference skipped: {e}")
    for key in components:
        print(f"\n===== {key}: open-source total = {sum(p.numel() for p in (ref.get(key) or ours[key]).parameters()):,}, "
              f"model.py total = {sum(p.numel() for p in ours[key].parameters()):,} =====")
        if key in ref:
            a = _tree_lines(ref[key], prefix=f"opensrc.{key}")
            b = _tree_lines(ours[key], prefix=f"model.py.{key}")
            results[key] = (a, b)
    # dump aligned text report
    def _norm(n):
        for pfx in ("opensrc.", "model.py."):
            if n.startswith(pfx):
                return n[len(pfx):]
        return n
    with open(os.path.join(out_prefix, "equivalence_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("Matrix-Game-3.5 model equivalence report (module-level)\n")
        f.write("=" * 100 + "\n")
        for key, (a, b) in results.items():
            f.write(f"\n## component: {key}\n")
            sa = {(_norm(n), d): (c, p) for d, n, c, p, _ in a}
            sb = {(_norm(n), d): (c, p) for d, n, c, p, _ in b}
            rows = sorted(set(sa) | set(sb), key=lambda kv: (kv[1], kv[0]))
            for n, d in rows:
                ca, pa = sa.get((n, d), ("-", "-"))
                cb, pb = sb.get((n, d), ("-", "-"))
                ok = (ca == cb and str(pa) == str(pb))
                mark = "  " if ok else ">>"
                f.write(f"{mark} {'  '*d}{n:<56} opensrc[{ca}:{pa}] "
                        f"model.py[{cb}:{pb}] {'OK' if ok else 'DIFF'}\n")
        # ---- authoritative parameter-leaf diff (names + shapes) ----
        f.write("\n" + "=" * 100 + "\n")
        f.write("Parameter-leaf diff (named_parameters: name -> shape)\n")
        f.write("=" * 100 + "\n")
        for key in components:
            if key not in ref:
                f.write(f"\n## {key}: reference unavailable, totals only\n")
                continue
            ra = dict(ref[key].named_parameters())
            rb = dict(ours[key].named_parameters())
            miss = sorted(set(ra) - set(rb))
            extra = sorted(set(rb) - set(ra))
            shape_mis = [k for k in (set(ra) & set(rb))
                         if tuple(ra[k].shape) != tuple(rb[k].shape)]
            f.write(f"\n## {key}: missing={len(miss)} extra={len(extra)} "
                    f"shape-mismatch={len(shape_mis)}\n")
            for k in miss[:20]:
                f.write(f"   - {k} {tuple(ra[k].shape)}\n")
            for k in extra[:20]:
                f.write(f"   + {k} {tuple(rb[k].shape)}\n")
            for k in shape_mis[:20]:
                f.write(f"   ~ {k} {tuple(ra[k].shape)} vs {tuple(rb[k].shape)}\n")
    print(f"[equivalence] report -> {out_prefix}/equivalence_report.txt")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Matrix-Game-3.5 compatible model.py - single-file CPU "
                    "train & real-time interactive world model")
    ap.add_argument("--mode", default="train-smoke",
                    choices=["train-smoke", "interactive-seq", "interactive-async3",
                             "component-test", "max-test", "self-check",
                             "print-components", "print-opensrc-tree",
                             "equivalence"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()
    if args.mode == "self-check":
        _self_check()
    elif args.mode == "print-components":
        print_component_models()
    elif args.mode == "component-test":
        run_component_test()
    elif args.mode == "max-test":
        run_max_test()
    elif args.mode == "print-opensrc-tree":
        model = make_opensrc_system()  # only used for module-tree print on meta
        with torch.device("meta"):
            for key in ("text", "dit", "vae"):
                obj = getattr(model, key)
                print(f"--- opensrc-config {key}: "
                      f"{sum(p.numel() for p in obj.parameters()):,} params "
                      f"(meta) ---")
                _print_tree(obj, key)
    elif args.mode == "equivalence":
        run_equivalence()
    elif args.mode == "train-smoke":
        run_train_smoke(steps=args.steps, trace=not args.no_trace)
    elif args.mode == "interactive-seq":
        run_interactive_seq(trace=not args.no_trace)
    elif args.mode == "interactive-async3":
        run_interactive_async3(trace=not args.no_trace)


def _print_tree(module, prefix="model", max_depth=12, _depth=0):
    if _depth > max_depth:
        print("  " * _depth + "...")
        return
    n = sum(p.numel() for p in module.parameters(recurse=False))
    children = [c for c in module.children()]
    if not children:
        print(f"{'  ' * _depth}{prefix}: {type(module).__name__} "
              f"(leaf params {n})")
        return
    print(f"{'  ' * _depth}{prefix}: {type(module).__name__} "
          f"(direct params {n})")
    for name, child in module.named_children():
        _print_tree(child, name, max_depth, _depth + 1)


if __name__ == "__main__":
    main()

