# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU test doubles for the fused partial-RoPE wrapper.

Registers CPU kernels for the two native ``(Tensor(a!)) -> ()`` CANN mutator
schemas (the conftest fake defines the schemas when no real CANN package is
present) so the *real* functional autograd.Function wrapper — clone, forward
mutator, backward mutator — can run on CPU.  The emulations mutate in place
and return ``None`` exactly like the native contracts; the device smoke
(A3, CANN w0902) showed the forward kernel is bitwise equal to this fp32
interleave formula and the backward to the per-element transpose below.

Call counters are exposed for the pass-level recompute tests: reset them
after tracing and compare execution counts across graph variants.
"""

import torch

from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE, _apply_interleaved_rope

_KERNELS_REGISTERED = False
KERNEL_CALLS = {"forward": 0, "backward": 0}


def register_cpu_kernels() -> None:
    """Register the CPU emulations once (idempotent)."""
    global _KERNELS_REGISTERED
    if _KERNELS_REGISTERED:
        return
    import torchtitan_npu.ops.ascendc.inplace_partial_rotary_mul  # noqa: F401

    def _forward_cpu(x, r1, r2, *, rotary_mode="interleave", partial_slice=None):
        partial_slice = [0, 0] if partial_slice is None else partial_slice
        start, end = partial_slice
        if start != end:
            seg = x[..., start:end].float()
            rotated = torch.stack((-seg[..., 1::2], seg[..., ::2]), dim=-1).flatten(-2)
            x[..., start:end] = (seg * r1 + rotated * r2).to(x.dtype)
        KERNEL_CALLS["forward"] += 1
        return None

    def _backward_cpu(grad_output, r1, r2, *, rotary_mode="interleave", partial_slice=None):
        partial_slice = [0, 0] if partial_slice is None else partial_slice
        start, end = partial_slice
        if start != end:
            rope = grad_output[..., start:end].float()
            even, odd = rope[..., ::2], rope[..., 1::2]
            new_even = even * r1[..., ::2] + odd * r2[..., 1::2]
            new_odd = odd * r1[..., 1::2] - even * r2[..., ::2]
            grad_output[..., start:end] = torch.stack((new_even, new_odd), dim=-1).flatten(-2).to(grad_output.dtype)
        KERNEL_CALLS["backward"] += 1
        return None

    torch.library.register_kernel("cann_ops_transformer::inplace_partial_rotary_mul", "CPU", _forward_cpu)
    torch.library.register_kernel("cann_ops_transformer::inplace_partial_rotary_mul_backward", "CPU", _backward_cpu)
    _KERNELS_REGISTERED = True


def reset_kernel_counts() -> None:
    KERNEL_CALLS["forward"] = 0
    KERNEL_CALLS["backward"] = 0


def split_workaround_reference(x, split, rotary_dim, positions, *, inverse=False):
    """The pre-split model path: slice, interleave-rotate the tail, cat back."""
    plain = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=rotary_dim, max_seq_len=16))
    prefix, rotary = x[..., :split], x[..., split:]
    cos, sin = plain._reshape_cache(rotary, positions)
    if inverse:
        sin = -sin
    return torch.cat((prefix, _apply_interleaved_rope(rotary, cos, sin)), dim=-1)
