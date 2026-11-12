# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest
import torch
from torchtitan.config import derive
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.override.common.rope import (
    AscComplexRoPE,
    AscPartialComplexRoPE,
    WorkaroundComplexRoPE,
)
from torchtitan_npu.patches.torchtitan.models.common.rope import SplitComplexRoPEConfig


@pytest.mark.parametrize("rope_cls", [WorkaroundComplexRoPE, AscComplexRoPE])
def test_interleaved_rope_caches_expanded_cos_and_sin(rope_cls):
    config = rope_cls.Config(dim=8, max_context_length=16)
    rope = rope_cls(config)
    complex_cache = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16)).cache

    assert rope.cache.shape == (2, 16, 8)
    torch.testing.assert_close(
        rope.cache[0],
        complex_cache.real.repeat_interleave(2, dim=-1),
    )
    torch.testing.assert_close(
        rope.cache[1],
        complex_cache.imag.repeat_interleave(2, dim=-1),
    )
    assert rope.cache.is_contiguous()

    rope._init_self_buffers(buffer_device=torch.device("cpu"))

    assert rope.cache.shape == (2, 16, 8)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("explicit_positions", [False, True], ids=["sequential", "packed"])
def test_workaround_rope_matches_complex_reference(position_dtype, explicit_positions):
    config = ComplexRoPE.Config(dim=8, max_context_length=16)
    reference = ComplexRoPE(config)
    workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_context_length=16))
    # The folded-token call form the migrated attention uses: ``[T, H, D]``
    # queries with flat per-token positions (two three-token rows folded).
    query = torch.randn(6, 1, 8)
    positions = torch.arange(3, dtype=position_dtype).repeat(2) if explicit_positions else None

    expected = reference(query, positions=positions)
    actual = workaround(query, positions=positions)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_workaround_split_rope_matches_real_rotation_forward_and_gradient(dtype, inverse):
    config = ComplexRoPE.Config(dim=8, max_context_length=16, split=4)
    reference = ComplexRoPE(config)
    workaround = WorkaroundComplexRoPE(derive(config, WorkaroundComplexRoPE.Config))
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(6, 2, 12, generator=generator, dtype=dtype).requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    positions = torch.tensor([0, 1, 2, 0, 1, 2])

    actual = workaround(query, positions=positions, inverse=inverse)
    cache = reference.cache[positions].unsqueeze(1)
    cos, sin = cache.real, cache.imag
    if inverse:
        sin = -sin
    even, odd = expected_query[..., 4::2].float(), expected_query[..., 5::2].float()
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    expected = torch.cat((expected_query[..., :4], rotated.to(dtype)), dim=-1)
    actual.float().sum().backward()
    expected.float().sum().backward()

    torch.testing.assert_close(query.grad, expected_query.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rope_cls", [WorkaroundComplexRoPE, AscComplexRoPE])
def test_interleaved_rope_cache_pool_reuses_compatible_cache(rope_cls):
    first = rope_cls(rope_cls.Config(dim=8, max_context_length=16))
    second = rope_cls(rope_cls.Config(dim=8, max_context_length=16))

    assert first.cache is second.cache

    first._init_self_buffers(buffer_device=torch.device("cpu"))
    second._init_self_buffers(buffer_device=torch.device("cpu"))
    assert first.cache is second.cache

    different = rope_cls(rope_cls.Config(dim=8, max_context_length=16, theta=123.0))
    assert different.cache is not first.cache


def test_interleaved_rope_cache_pool_reuses_across_implementations():
    workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_context_length=16))
    ascend = AscComplexRoPE(AscComplexRoPE.Config(dim=8, max_context_length=16))

    assert workaround.cache is ascend.cache


def test_meta_rope_cache_deferred_until_init_states():
    with torch.device("meta"):
        workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_context_length=16))
        ascend = AscComplexRoPE(AscComplexRoPE.Config(dim=8, max_context_length=16))
        module = torch.nn.ModuleList([workaround, ascend])

    assert workaround.cache.device.type == "meta"
    assert workaround.cache.numel() == 1
    module.to_empty(device="cpu")
    assert workaround.cache.device.type == "cpu"
    assert workaround.cache.numel() == 1
    assert ascend.cache.numel() == 1

    workaround.init_states(buffer_device=torch.device("cpu"))
    ascend.init_states(buffer_device=torch.device("cpu"))
    assert workaround.cache.shape == (2, 16, 8)
    assert workaround.cache is ascend.cache

    # Deferred materialization must preserve the RoPE math, not just restore
    # the shared buffer shape/alias.
    reference = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16))
    query = torch.randn(6, 1, 8)
    positions = torch.arange(3).repeat(2)
    expected = reference(query, positions=positions)
    actual = workaround(query, positions=positions)
    torch.testing.assert_close(actual, expected)


# --- split-aware partial RoPE (DSV4 ``[nope | rope]`` layouts) -------------


def test_patched_config_accepts_and_carries_split():
    config = ComplexRoPE.Config(dim=8, max_context_length=16, split=12)

    assert isinstance(config, SplitComplexRoPEConfig)
    assert config.split == 12
    # The model factories pin split per site via dataclasses.replace.
    assert dataclasses.replace(config, theta=99.0).split == 12
    # Overrides receive split through derive().
    assert derive(config, WorkaroundComplexRoPE.Config).split == 12
    assert derive(config, AscPartialComplexRoPE.Config).split == 12
    # build() still constructs the upstream module with the split config.
    rope = config.build()
    assert type(rope) is ComplexRoPE
    assert rope.config.split == 12


def test_split_aware_default_rope_matches_manual_split():
    split, rotary_dim = 12, 8
    rope = ComplexRoPE(ComplexRoPE.Config(dim=rotary_dim, max_context_length=16, split=split))
    plain = ComplexRoPE(ComplexRoPE.Config(dim=rotary_dim, max_context_length=16))
    x = torch.randn(10, 3, split + rotary_dim)
    positions = torch.arange(5).repeat(2)

    out = rope(x, positions=positions)
    expected = torch.cat((x[..., :split], plain(x[..., split:], positions=positions)), dim=-1)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.equal(out, expected)


def test_split_zero_keeps_whole_tensor_rotation():
    plain = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16))
    zero_split = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16, split=0))
    x = torch.randn(10, 3, 8)
    positions = torch.arange(5).repeat(2)

    assert torch.equal(zero_split(x, positions=positions), plain(x, positions=positions))


@pytest.mark.parametrize("inverse", [False, True])
def test_split_aware_default_rope_matches_manual_split_qk(inverse):
    """Default (no-override) path stays bitwise equal to the pre-change
    model pattern — manually slicing each tensor, rotating the tail with a
    plain whole-tensor rope, and concatenating the prefix back — for the
    query+key call form and for inverse rotation."""
    split, rotary_dim = 12, 8
    rope = ComplexRoPE(ComplexRoPE.Config(dim=rotary_dim, max_context_length=16, split=split))
    plain = ComplexRoPE(ComplexRoPE.Config(dim=rotary_dim, max_context_length=16))
    q = torch.randn(10, 3, split + rotary_dim)
    k = torch.randn(10, 3, split + rotary_dim)
    positions = torch.arange(5).repeat(2)

    out_q, out_k = rope(q, k, positions=positions, inverse=inverse)

    expected_q = torch.cat((q[..., :split], plain(q[..., split:], positions=positions, inverse=inverse)), dim=-1)
    expected_k = torch.cat((k[..., :split], plain(k[..., split:], positions=positions, inverse=inverse)), dim=-1)

    assert torch.equal(out_q, expected_q)
    assert torch.equal(out_k, expected_k)
    # One call with both tensors is also consistent with separate calls.
    assert torch.equal(out_q, rope(q, positions=positions, inverse=inverse))
    assert torch.equal(out_k, rope(k, positions=positions, inverse=inverse))


def test_split_rope_reference_api():
    """The br_dpsk_v4_1 API surface: ``self.split`` mirrors the config and
    ``_split``/``_unsplit`` are pass-throughs for unset splits and None keys."""
    rope = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16, split=12))
    x = torch.randn(2, 5, 3, 20)

    assert rope.split == 12
    assert torch.equal(rope._split(x), x[..., 12:])
    assert torch.equal(rope._unsplit(x[..., 12:], x), x)
    assert rope._split(None) is None
    rotated = x[..., 12:]
    assert rope._unsplit(rotated, None) is rotated

    plain = ComplexRoPE(ComplexRoPE.Config(dim=8, max_context_length=16))
    y = torch.randn(2, 5, 3, 8)
    assert plain.split == 0
    assert plain._split(y) is y
    assert plain._unsplit(y, y) is y
