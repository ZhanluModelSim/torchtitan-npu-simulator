# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve native SwiGLU operands when vendor autograd has wrong arity."""

from functools import wraps

import torch


class _SwigluMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_index, clamp_limit, forward):
        ctx.save_for_backward(x, weight, group_index)
        ctx.clamp_limit = clamp_limit
        return forward(x, weight, group_index, clamp_limit=clamp_limit)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, weight, group_index = ctx.saved_tensors
        y_origin = (
            torch.ops.cann_ops_nn.swiglu_group.default(x, None, group_index, clamp_limit=ctx.clamp_limit)
            if weight is not None
            else None
        )
        dx, dw = torch.ops.cann_ops_nn.swiglu_group_backward.default(
            grad,
            x,
            weight=weight,
            y_origin=y_origin,
            group_index=group_index,
            clamp_limit=0.0 if ctx.clamp_limit == -1.0 else ctx.clamp_limit,
        )
        return dx, dw, None, None, None


def install_meta_swiglu_autograd(scope):
    import spmd_types

    from torchtitan_npu.override.common.swiglu_group import ascendc

    ascendc._ensure_cann_ops_loaded()
    op = torch.ops.cann_ops_nn.swiglu_group.default
    schema = op._schema
    if [(argument.name, argument.kwarg_only) for argument in schema.arguments] != [
        ("x", False),
        ("weight", False),
        ("group_index", False),
        ("clamp_limit", True),
    ]:
        raise RuntimeError(f"SwiGLU autograd contract requires review: {schema}")
    with torch.enable_grad():
        x = torch.empty(2, 64, device="meta", requires_grad=True)
        try:
            torch.autograd.grad(op(x).sum(), x)
        except RuntimeError as error:
            if "incorrect number of gradients" not in str(error):
                raise
        else:
            return "native"
    spmd_types.register_local_autograd_function(_SwigluMetaAutograd)
    original = ascendc.swiglu_group_activation

    @wraps(original)
    def activation(h, swiglu_limit=None, routed_scores=None):
        if h.device.type != "meta":
            return original(h, swiglu_limit, routed_scores)
        weight = None if routed_scores is None else routed_scores.to(dtype=torch.float32).contiguous()
        limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
        return _SwigluMetaAutograd.apply(h.contiguous(), weight, None, limit, op)

    scope.replace(ascendc, "swiglu_group_activation", activation)
    return "bridge_vendor_gradient_arity"
