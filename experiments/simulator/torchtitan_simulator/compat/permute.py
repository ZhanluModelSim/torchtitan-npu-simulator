# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge missing token-permute Meta autograd using the native gradient op."""

from functools import wraps

import torch


class _PermuteMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, indices, num_out_tokens, padded_mode, forward):
        output, sorted_indices = forward(tokens, indices, num_out_tokens, padded_mode)
        ctx.save_for_backward(tokens, indices, sorted_indices)
        ctx.padded_mode = padded_mode
        ctx.mark_non_differentiable(sorted_indices)
        return output, sorted_indices

    @staticmethod
    # Fixed gradient arity follows Function outputs; the base stub uses variadic args.
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad, grad_indices):
        tokens, indices, sorted_indices = ctx.saved_tensors
        dx = torch.ops.npu.npu_moe_token_permute_grad.default(
            tokens,
            grad,
            indices,
            sorted_indices,
            ctx.padded_mode,
        )
        return dx, None, None, None, None


def install_meta_permute_autograd(scope):
    import spmd_types
    import torch_npu

    original = torch_npu.npu_moe_token_permute
    with torch.enable_grad():
        x = torch.empty(4, 8, device="meta", requires_grad=True)
        indices = torch.empty(4, 2, device="meta", dtype=torch.int32)
        output, _ = original(x, indices)
        gradient = torch.autograd.grad(output.sum(), x, allow_unused=True)[0] if output.requires_grad else None
    if gradient is not None:
        return "native"
    schema = torch.ops.npu.npu_moe_token_permute_grad.default._schema
    if [argument.name for argument in schema.arguments] != [
        "tokens",
        "grad_permuted_tokens",
        "indices",
        "sorted_indices",
        "padded_mode",
    ]:
        raise RuntimeError(f"Token-permute backward contract requires review: {schema}")
    spmd_types.register_local_autograd_function(_PermuteMetaAutograd)

    @wraps(original)
    def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
        if tokens.device.type != "meta":
            return original(tokens, indices, num_out_tokens, padded_mode)
        return _PermuteMetaAutograd.apply(tokens, indices, num_out_tokens, padded_mode, original)

    scope.replace(torch_npu, "npu_moe_token_permute", permute)
    return "bridge"
