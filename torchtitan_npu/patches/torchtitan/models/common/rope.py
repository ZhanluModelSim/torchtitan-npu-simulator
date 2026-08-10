# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport the unified single-tensor RoPE API and the unconditional YaRN
policy from TorchTitan PR #3634.

``RoPE.forward`` accepts ``key=None`` (rotate a single tensor, returning it
directly) and ``inverse`` (conjugate rotation, for backward passes); the
YaRN correction in ``ComplexRoPE._precompute_cache`` is applied whenever
``scaling == "yarn"`` and ``original_seq_len > 0`` — the HF/Megatron
convention where the correction is the encoding itself — replacing the
``end > original_seq_len`` context-extension gate (which silently disabled
YaRN for models whose max_seq_len does not exceed their original_seq_len).
The ``> 0`` guard shields ``_yarn_inv_freq``'s ``log(original_seq_len)``
from degenerate configs.  The previous ``SingleComplexRoPE`` backport is
superseded by this unified API.

Remove this module after the TorchTitan dependency includes the PR.
"""

import math

import torch
import torchtitan.models.common.rope


def _yarn_precompute_cache(self) -> torch.Tensor:
    """ComplexRoPE._precompute_cache with the unconditional YaRN policy."""
    cfg = self.config
    dim = cfg.dim
    end = cfg.max_seq_len
    theta = cfg.theta

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if cfg.scaling == "llama":
        scaling_factor = cfg.scaling_factor
        low_freq_factor = cfg.low_freq_factor
        high_freq_factor = cfg.high_freq_factor
        original_max_position_embeddings = cfg.original_max_position_embeddings
        wavelen = 2 * math.pi / freqs
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        freqs = torch.where(wavelen > low_freq_wavelen, freqs / scaling_factor, freqs)
        smooth_factor = (
            original_max_position_embeddings / wavelen - low_freq_factor
        ) / (high_freq_factor - low_freq_factor)
        smoothed_freqs = (
            1 - smooth_factor
        ) * freqs / scaling_factor + smooth_factor * freqs
        is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)
    elif cfg.scaling == "yarn" and cfg.original_seq_len > 0:
        # YaRN: the correction is the encoding itself (HF/Megatron
        # convention), applied at every position — original_seq_len is a
        # parameter of the correction range, not a length gate.
        freqs = torchtitan.models.common.rope._yarn_inv_freq(
            dim,
            theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            cfg.original_seq_len,
            cfg.truncate,
        )

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _rope_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and optional key tensors."""
    reshaped_cache = self._reshape_cache(query, positions)
    return self.apply_rotary_emb(query, key, reshaped_cache, inverse=inverse)


def _complex_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor | None,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply complex RoPE using adjacent-dim pairs; rotate query only when
    ``key`` is ``None``, and conjugate the cache for ``inverse``."""
    if inverse:
        rope_cache = rope_cache.conj()
    xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    query_out = torch.view_as_real(xq_ * rope_cache).flatten(-2).type_as(query)
    if key is None:
        return query_out
    xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    key_out = torch.view_as_real(xk_ * rope_cache).flatten(-2).type_as(key)
    return query_out, key_out


def _cossin_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor | None,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply cos/sin RoPE using the rotate-half convention; rotate query
    only when ``key`` is ``None``."""
    if inverse:
        raise NotImplementedError("CosSinRoPE does not support inverse rotation.")
    head_dim = query.shape[-1]
    cos = rope_cache[..., :head_dim]
    sin = rope_cache[..., head_dim:]
    query_f = query.float()
    xq_out = (query_f * cos) + (
        torchtitan.models.common.rope.CosSinRoPE._rotate_half(query_f) * sin
    )
    if key is None:
        return xq_out.type_as(query)
    key_f = key.float()
    xk_out = (key_f * cos) + (
        torchtitan.models.common.rope.CosSinRoPE._rotate_half(key_f) * sin
    )
    return xq_out.type_as(query), xk_out.type_as(key)


def apply() -> None:
    """Monkey-patch the upstream rope module to PR #3634's final state."""

    torchtitan.models.common.rope.RoPE.forward = (
        _rope_forward  # pyrefly: ignore [bad-assignment]
    )
    torchtitan.models.common.rope.ComplexRoPE._precompute_cache = _yarn_precompute_cache
    torchtitan.models.common.rope.ComplexRoPE.apply_rotary_emb = (
        staticmethod(  # pyrefly: ignore [bad-assignment]
            _complex_apply_rotary_emb
        )
    )
    torchtitan.models.common.rope.CosSinRoPE.apply_rotary_emb = (
        staticmethod(  # pyrefly: ignore [bad-assignment]
            _cossin_apply_rotary_emb
        )
    )


apply()
