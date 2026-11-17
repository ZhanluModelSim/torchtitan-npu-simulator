# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge a missing Meta autograd registration without changing device kernels."""

import torch


class _RotaryMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, mode, forward):
        ctx.save_for_backward(x, cos, sin)
        ctx.mode = mode
        return forward(x, cos, sin, rotary_mode=mode)

    @staticmethod
    # Fixed gradient arity follows Function outputs; the base stub uses variadic args.
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, cos, sin = ctx.saved_tensors
        dx, dcos, dsin = torch.ops.npu.npu_rotary_mul_backward.default(grad, x, cos, sin, rotary_mode=ctx.mode)
        return dx, dcos, dsin, None, None


def meta_rotary_needs_bridge(forward) -> bool:
    """Check connectivity, rather than inferring support from a dispatch key."""
    with torch.enable_grad():
        x = torch.empty(1, 2, 1, 4, device="meta", requires_grad=True)
        cos = torch.empty_like(x)
        sin = torch.empty_like(x)
        output = forward(x, cos, sin, rotary_mode="interleave")
        if not output.requires_grad:
            return True
        gradient = torch.autograd.grad(output.sum(), x, allow_unused=True)[0]
        return gradient is None


def install_rotary_bridge(scope) -> dict[str, object]:
    import spmd_types
    import torch_npu

    # Rotation is local to the head dimension; it does not communicate or
    # change the placement. Keep the native SPMD checker active for the bridge.
    spmd_types.register_local_autograd_function(_RotaryMetaAutograd)

    original = torch_npu.npu_rotary_mul
    needed = meta_rotary_needs_bridge(original)
    if not needed:
        return {"rotary_meta_autograd": "native"}
    schema = torch.ops.npu.npu_rotary_mul_backward.default._schema
    required = {"grad", "input", "r1", "r2", "rotary_mode"}
    if not required.issubset({arg.name for arg in schema.arguments}):
        raise RuntimeError(f"Unsupported rotary backward contract: {schema}")

    def rotary(x, r1, r2, rotary_mode="half", rotate=None):
        if x.device.type != "meta":
            return original(x, r1, r2, rotary_mode=rotary_mode, rotate=rotate)
        if rotate is not None:
            raise ValueError("Meta rotary bridge does not support the rotate operand")
        return _RotaryMetaAutograd.apply(x, r1, r2, rotary_mode, original)

    scope.replace(torch_npu, "npu_rotary_mul", rotary)
    return {"rotary_meta_autograd": "bridge", "backward_schema": str(schema)}
