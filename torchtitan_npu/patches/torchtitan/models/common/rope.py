# Backports the split-aware partial-RoPE API: the rotary split as a config
# field and the tail-rotation forward come from sdmyzlp/torchtitan branch
# br_dpsk_v4_1, commit 24dd1a1 ("rope: make the rotary split a config field",
# https://github.com/sdmyzlp/torchtitan/commit/24dd1a1); the query/key
# optional-key + inverse forward surface follows the API shipped upstream by
# https://github.com/pytorch/torchtitan/pull/3634 (merged as 0ff2464).
# Remove this module once the TorchTitan dependency pinned by this project
# natively carries an equivalent split config and forward semantics and the
# compatibility tests pass — keep whichever key/inverse backport pieces the
# pinned version still lacks.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport the split-aware partial RoPE API onto the installed rope module."""

import dataclasses

import torch
import torchtitan.models.common.rope

__all__ = [
    "SplitComplexRoPEConfig",
    "SplitCosSinRoPEConfig",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class SplitComplexRoPEConfig(torchtitan.models.common.rope.ComplexRoPE.Config):
    """``ComplexRoPE.Config`` plus the ``split`` prefix-channel width.

    ``build()`` still constructs the upstream ``ComplexRoPE``; instances of
    this config carry ``split`` so the patched forward can slice the rotary
    span out of full-width inputs.
    """

    split: int = 0


@dataclasses.dataclass(kw_only=True, slots=True)
class SplitCosSinRoPEConfig(torchtitan.models.common.rope.CosSinRoPE.Config):
    """``CosSinRoPE.Config`` plus the ``split`` prefix-channel width."""

    split: int = 0


_ORIG_ROPE_INIT = torchtitan.models.common.rope.RoPE.__init__


def _rope_init(self, config) -> None:
    """Upstream ``__init__`` plus ``self.split`` (br_dpsk_v4_1 keeps the
    split in the module, not just the config)."""
    _ORIG_ROPE_INIT(self, config)
    self.split = getattr(config, "split", 0)


def _split(self, x: torch.Tensor | None) -> torch.Tensor | None:
    """Return the channels of ``x`` to rotate: everything after the ``split`` prefix.

    A ``None`` tensor (an absent key) and an unset ``split`` pass through
    unchanged, so query and key can be handed over unconditionally.
    """
    if x is None or not self.split:
        return x
    return x[..., self.split :]


def _unsplit(self, rotated: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
    """Put the untouched prefix of ``x`` back in front of its ``rotated`` channels."""
    if x is None or not self.split:
        return rotated
    return torch.cat([x[..., : self.split], rotated], dim=-1)


def _rope_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and optional key tensors.

    With a non-zero ``split`` the leading ``split`` channels are left
    untouched: the trailing ``dim`` channels are rotated and the prefix is
    concatenated back, so ``apply_rotary_emb`` only ever sees the rotated
    slice.
    """
    rope_cache = self._reshape_cache(query, positions)
    rotated = self.apply_rotary_emb(self._split(query), self._split(key), rope_cache, inverse=inverse)
    if key is None:
        return self._unsplit(rotated, query)
    rotated_query, rotated_key = rotated
    return self._unsplit(rotated_query, query), self._unsplit(rotated_key, key)


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
    xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], query.shape[-1] // 2, 2))
    query_out = torch.view_as_real(xq_ * rope_cache).flatten(-2).type_as(query)
    if key is None:
        return query_out
    xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], key.shape[-1] // 2, 2))
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
    xq_out = (query_f * cos) + (torchtitan.models.common.rope.CosSinRoPE._rotate_half(query_f) * sin)
    if key is None:
        return xq_out.type_as(query)
    key_f = key.float()
    xk_out = (key_f * cos) + (torchtitan.models.common.rope.CosSinRoPE._rotate_half(key_f) * sin)
    return xq_out.type_as(query), xk_out.type_as(key)


def apply() -> None:
    """Monkey-patch the upstream rope module to the br_dpsk_v4_1 API."""

    if getattr(torchtitan.models.common.rope.RoPE, "_npu_split_rope", False):
        return

    torchtitan.models.common.rope.RoPE.__init__ = _rope_init  # pyrefly: ignore [bad-assignment]
    torchtitan.models.common.rope.RoPE._split = _split  # pyrefly: ignore [bad-assignment]
    torchtitan.models.common.rope.RoPE._unsplit = _unsplit  # pyrefly: ignore [bad-assignment]
    torchtitan.models.common.rope.RoPE.forward = _rope_forward  # pyrefly: ignore [bad-assignment]
    torchtitan.models.common.rope.ComplexRoPE.apply_rotary_emb = staticmethod(  # pyrefly: ignore [bad-assignment]
        _complex_apply_rotary_emb
    )
    torchtitan.models.common.rope.CosSinRoPE.apply_rotary_emb = staticmethod(  # pyrefly: ignore [bad-assignment]
        _cossin_apply_rotary_emb
    )
    # Configs gain the ``split`` field; ``_owner`` is inherited, so ``build()``
    # keeps constructing the upstream modules with the split-bearing config.
    torchtitan.models.common.rope.ComplexRoPE.Config = (  # pyrefly: ignore [read-only]
        SplitComplexRoPEConfig
    )
    torchtitan.models.common.rope.CosSinRoPE.Config = (  # pyrefly: ignore [read-only]
        SplitCosSinRoPEConfig
    )
    torchtitan.models.common.rope.RoPE._npu_split_rope = True


apply()
