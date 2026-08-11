# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.override.common.rope import CANNComplexRoPE, WorkaroundComplexRoPE


@pytest.mark.parametrize("rope_cls", [WorkaroundComplexRoPE, CANNComplexRoPE])
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
    workaround = WorkaroundComplexRoPE(
        WorkaroundComplexRoPE.Config(dim=8, max_seq_len=16)
    )
    query = torch.randn(2, 3, 1, 8)
    positions = torch.arange(3).expand(2, -1)

    expected = reference(query, positions=positions)
    actual = workaround(query, positions=positions)

    torch.testing.assert_close(actual, expected)
