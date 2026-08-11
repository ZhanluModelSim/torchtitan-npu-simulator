# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

_previous_pre_grad_pass = torch._inductor.config.pre_grad_custom_pass
inplace_partial_rope = importlib.import_module(
    "torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope"
)
_make_parent_rope_pattern = inplace_partial_rope._make_parent_rope_pattern


def teardown_module():
    torch._inductor.config.pre_grad_custom_pass = _previous_pre_grad_pass


def _fake_inplace_partial_rotary_mul(
    x,
    cos,
    sin,
    *,
    rotary_mode,
    partial_slice,
):
    assert rotary_mode == "interleave"
    start, end = partial_slice
    rotary = x[..., start:end]
    pairs = rotary.float().reshape(*rotary.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    rotary.copy_((rotary.float() * cos + rotated * sin).type_as(rotary))


def _reference_partial_rope(x, cos, sin, inverse):
    rotary_width = cos.shape[-1]
    prefix, rotary = torch.split(
        x,
        [x.shape[-1] - rotary_width, rotary_width],
        dim=-1,
    )
    cache = torch.complex(cos[..., ::2], sin[..., ::2])
    if inverse:
        cache = cache.conj()
    rotary_complex = torch.view_as_complex(
        rotary.float().reshape(*rotary.shape[:-1], -1, 2)
    )
    rotated = torch.view_as_real(rotary_complex * cache).flatten(-2).type_as(rotary)
    return torch.cat([prefix, rotated], dim=-1)


def _partial_rope_with_different_literals(x, cos, sin):
    prefix, rotary = torch.split(x, [448, 64], dim=-1)
    rotary_float = rotary.float()
    pairs = rotary_float.reshape(1, 2048, 1, -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    rotated = rotary_float * cos + rotated * sin
    return torch.cat([prefix, rotated.type_as(rotary)], dim=-1)


@pytest.mark.parametrize(
    ("inverse", "rotary_width"),
    [
        pytest.param(False, 4, id="forward"),
        pytest.param(True, 4, id="inverse"),
        pytest.param(False, 8, id="different-width"),
    ],
)
def test_partial_rope_replacement_matches_search_fragment(
    monkeypatch,
    inverse,
    rotary_width,
):
    monkeypatch.setattr(
        inplace_partial_rope,
        "inplace_partial_rotary_mul",
        _fake_inplace_partial_rotary_mul,
    )
    pattern = _make_parent_rope_pattern(
        inverse=inverse,
    )
    x = torch.randn(2, 3, 2, 4 + rotary_width, dtype=torch.bfloat16)
    angles = torch.randn(2, 3, 1, rotary_width // 2)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    expected = _reference_partial_rope(x, cos, sin, inverse)

    actual = pattern.replacement_fn(x.clone(), cos, sin)

    torch.testing.assert_close(actual, expected)


def test_replacement_calls_inplace_partial_rotary_mul(monkeypatch):
    calls = []

    def fake_op(x, cos, sin, *, rotary_mode, partial_slice):
        calls.append((x, cos, sin, rotary_mode, partial_slice))
        x.add_(1)

    monkeypatch.setattr(
        inplace_partial_rope,
        "inplace_partial_rotary_mul",
        fake_op,
    )
    x = torch.zeros(2, 3, 2, 8)
    cos = torch.ones(2, 3, 1, 4)
    sin = torch.zeros_like(cos)
    pattern = _make_parent_rope_pattern(inverse=False)

    actual = pattern.replacement_fn(x, cos, sin)

    assert actual is not x
    torch.testing.assert_close(x, torch.zeros_like(x))
    torch.testing.assert_close(actual, torch.ones_like(actual))
    assert len(calls) == 1
    call_x, call_cos, call_sin, rotary_mode, partial_slice = calls[0]
    assert call_x is not x
    assert call_cos is cos
    assert call_sin is sin
    assert rotary_mode == "interleave"
    assert partial_slice == [4, 8]


def test_search_pattern_ignores_shape_literals():
    pattern = _make_parent_rope_pattern(inverse=False)
    graph_module = torch.fx.symbolic_trace(_partial_rope_with_different_literals)

    matches = replace_pattern_with_filters(
        graph_module,
        pattern.search_fn,
        pattern.replacement_fn,
        ignore_literals=pattern.ignore_literals,
    )

    assert len(matches) == 1


def test_replacement_does_not_repeat_rope_cache():
    pattern = _make_parent_rope_pattern(inverse=False)

    graph_module = torch.fx.symbolic_trace(pattern.replacement_fn)

    assert "repeat_interleave" not in str(graph_module.graph)


def test_isolate_gradient_preserves_shared_tangent():
    class MutatingBackward(torch.autograd.Function):
        @staticmethod
        def forward(_ctx, x):
            return x.clone()

        @staticmethod
        def backward(_ctx, grad_output):
            grad_output.mul_(2)
            return grad_output

    def fn(x):
        rotated = inplace_partial_rope._isolate_gradient(MutatingBackward.apply(x))
        return (rotated + x).sum()

    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    x = torch.randn(2, 3, 2, 8, requires_grad=True)

    compiled(x).backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 3))
