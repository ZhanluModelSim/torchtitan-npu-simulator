# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.override.common.rope import AscComplexRoPE, WorkaroundComplexRoPE


@pytest.mark.parametrize("rope_cls", [WorkaroundComplexRoPE, AscComplexRoPE])
def test_interleaved_rope_caches_expanded_cos_and_sin(rope_cls):
    config = rope_cls.Config(dim=8, max_seq_len=16)
    rope = rope_cls(config)
    complex_cache = ComplexRoPE(ComplexRoPE.Config(dim=8, max_seq_len=16)).cache

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


def test_workaround_rope_matches_complex_reference():
    config = ComplexRoPE.Config(dim=8, max_seq_len=16)
    reference = ComplexRoPE(config)
    workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_seq_len=16))
    query = torch.randn(2, 3, 1, 8)
    positions = torch.arange(3).expand(2, -1)

    expected = reference(query, positions=positions)
    actual = workaround(query, positions=positions)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rope_cls", [WorkaroundComplexRoPE, AscComplexRoPE])
def test_interleaved_rope_cache_pool_reuses_compatible_cache(rope_cls):
    first = rope_cls(rope_cls.Config(dim=8, max_seq_len=16))
    second = rope_cls(rope_cls.Config(dim=8, max_seq_len=16))

    assert first.cache is second.cache

    first._init_self_buffers(buffer_device=torch.device("cpu"))
    second._init_self_buffers(buffer_device=torch.device("cpu"))
    assert first.cache is second.cache

    different = rope_cls(rope_cls.Config(dim=8, max_seq_len=16, theta=123.0))
    assert different.cache is not first.cache


def test_interleaved_rope_cache_pool_reuses_across_implementations():
    workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_seq_len=16))
    ascend = AscComplexRoPE(AscComplexRoPE.Config(dim=8, max_seq_len=16))

    assert workaround.cache is ascend.cache


def test_meta_rope_cache_deferred_until_init_states():
    with torch.device("meta"):
        workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_seq_len=16))
        ascend = AscComplexRoPE(AscComplexRoPE.Config(dim=8, max_seq_len=16))
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
    reference = ComplexRoPE(ComplexRoPE.Config(dim=8, max_seq_len=16))
    query = torch.randn(2, 3, 1, 8)
    positions = torch.arange(3).expand(2, -1)
    expected = reference(query, positions=positions)
    actual = workaround(query, positions=positions)
    torch.testing.assert_close(actual, expected)
