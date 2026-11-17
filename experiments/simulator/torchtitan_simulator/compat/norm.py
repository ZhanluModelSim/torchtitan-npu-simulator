# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep unit-scale normalization buffers on the explicit Meta build device."""

from functools import wraps

import torch


def install_meta_norm_buffers(scope, norm_class):
    original = norm_class._init_self_buffers

    @wraps(original)
    def initialize(module, *, buffer_device=None):
        if module.weight.device.type != "meta":
            return original(module, buffer_device=buffer_device)
        if not module.elementwise_affine:
            # The native helper selects the current NPU for this unit buffer.
            # Meta has no hardware device; retain the actual registered buffer.
            module.weight.fill_(1)

    scope.replace(norm_class, "_init_self_buffers", initialize)


class _RMSNormMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, epsilon, forward):
        output, rstd = forward(x, gamma, epsilon)
        ctx.save_for_backward(x, gamma, rstd)
        ctx.mark_non_differentiable(rstd)
        return output, rstd

    @staticmethod
    # Fixed gradient arity follows Function outputs; the base stub uses variadic args.
    # pyrefly: ignore [bad-override]
    def backward(ctx, dy, grad_rstd):
        import torch

        x, gamma, rstd = ctx.saved_tensors
        dx, dgamma = torch.ops.npu.npu_rms_norm_backward.default(dy, x, gamma, rstd)
        return dx, dgamma, None, None


def install_meta_norm_autograd(scope):
    import spmd_types
    import torch
    import torch_npu

    original = torch_npu.npu_rms_norm
    with torch.enable_grad():
        x = torch.empty(2, 8, device="meta", requires_grad=True)
        gamma = torch.empty(8, device="meta", requires_grad=True)
        output, _ = original(x, gamma)
        gradients = (
            torch.autograd.grad(output.sum(), (x, gamma), allow_unused=True) if output.requires_grad else (None, None)
        )
    if all(gradient is not None for gradient in gradients):
        return "native"
    schema = torch.ops.npu.npu_rms_norm_backward.default._schema
    if [argument.name for argument in schema.arguments] != ["dy", "input", "gamma", "rstd"]:
        raise RuntimeError(f"RMSNorm backward contract requires review: {schema}")
    spmd_types.register_local_autograd_function(_RMSNormMetaAutograd)

    @wraps(original)
    def norm(input, gamma, epsilon=1e-6):
        if input.device.type != "meta":
            return original(input, gamma, epsilon)
        return _RMSNormMetaAutograd.apply(input, gamma, epsilon, original)

    scope.replace(torch_npu, "npu_rms_norm", norm)
    return "bridge"
