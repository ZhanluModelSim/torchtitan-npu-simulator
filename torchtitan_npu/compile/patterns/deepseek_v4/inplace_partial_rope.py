# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace DeepSeek-V4 split/RoPE/cat regions before AOTAutograd."""

from __future__ import annotations

import torch
import torch_npu
from cann_ops_transformer.ops import (  # pyrefly: ignore [missing-import]
    inplace_partial_rotary_mul,
)

from torchtitan_npu.compile.pattern_replacement import (
    PatternReplacement,
    register_pre_aot_patterns,
)

if torch_npu.npu.is_available():
    import torch_npu._inductor


class _IsolateGradient(torch.autograd.Function):
    """Prevent Inductor from sharing a tangent mutated by an in-place backward."""

    @staticmethod
    def forward(ctx, x):  # pyrefly: ignore [bad-override]
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        return grad_output.clone()


def _isolate_gradient(x):
    return _IsolateGradient.apply(x)


torch.fx.wrap("inplace_partial_rotary_mul")
torch.fx.wrap("_isolate_gradient")


def _make_parent_rope_pattern(
    *,
    inverse: bool,
) -> PatternReplacement:
    def search_fn(x, cos, sin):
        # Shape literals are placeholders generalized by ignore_literals=True.
        prefix, rotary = torch.split(x, [2, 2], dim=-1)
        rotary_float = rotary.float()
        pairs = rotary_float.reshape(1, 2, 1, -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        if inverse:
            sin = -sin
        rotated = rotary_float * cos + rotated * sin
        return torch.cat([prefix, rotated.type_as(rotary)], dim=-1)

    def replacement_fn(x, cos, sin):
        if inverse:
            sin = -sin
        end = x.shape[-1]
        output = x.clone()
        inplace_partial_rotary_mul(
            output,
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[end - cos.shape[-1], end],
        )
        return _isolate_gradient(output)

    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
        ignore_literals=True,
    )


register_pre_aot_patterns(
    {
        "dsv4_parent_rope_inverse": _make_parent_rope_pattern(inverse=True),
        "dsv4_parent_rope_forward": _make_parent_rope_pattern(inverse=False),
    },
)
