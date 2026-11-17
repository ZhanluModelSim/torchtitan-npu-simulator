# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta autograd bridge for the fused CANN attention (npu_fusion_attention_v3).

The vendor registration exposes a Meta shape kernel for the forward op but
does not propagate input gradients on Meta (the CppFunction formula drops
them).  The bridge follows the RMSNorm pattern: probe the native capability
first, then wrap the module-level seam of
``torchtitan_npu.override.common.varlen_attention`` with an autograd.Function
whose forward runs the real fused op (so the captured graph keeps the
``npu::npu_fusion_attention_v3`` node) and whose backward runs the real
``npu::npu_fusion_attention_grad_v3`` op.
"""

from functools import wraps

import torch


class _FusionAttentionV3MetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, atten_mask, actual_seq, head_num, scale, pre_tockens, next_tockens, original):
        result = original(
            query=query,
            key=key,
            value=value,
            head_num=head_num,
            input_layout="TND",
            atten_mask=atten_mask,
            scale=scale,
            keep_prob=1.0,
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
            actual_seq_qlen=actual_seq,
            actual_seq_kvlen=actual_seq,
            sparse_mode=7,
        )
        # (output, softmax_max, softmax_sum, softmax_in placeholder, seed, offset)
        output, softmax_max, softmax_sum = result[0], result[1], result[2]
        ctx.save_for_backward(query, key, value, atten_mask, actual_seq, softmax_max, softmax_sum, output)
        ctx.head_num = head_num
        ctx.scale = scale
        ctx.pre_tockens = pre_tockens
        ctx.next_tockens = next_tockens
        return output

    @staticmethod
    # Fixed gradient arity follows Function outputs; the base stub uses variadic args.
    # pyrefly: ignore [bad-override]
    def backward(ctx, dy):
        query, key, value, atten_mask, actual_seq, softmax_max, softmax_sum, output = ctx.saved_tensors
        gradients = torch.ops.npu.npu_fusion_attention_grad_v3(
            query=query,
            key=key,
            value=value,
            dy=dy.contiguous(),
            head_num=ctx.head_num,
            input_layout="TND",
            atten_mask=atten_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=output,
            scale_value=ctx.scale,
            keep_prob=1.0,
            pre_tockens=ctx.pre_tockens,
            next_tockens=ctx.next_tockens,
            actual_seq_qlen=actual_seq,
            actual_seq_kvlen=actual_seq,
            sparse_mode=7,
        )
        # Schema returns (dq, dk, dv, pse-grad placeholder, scalar tail).
        return gradients[0], gradients[1], gradients[2], None, None, None, None, None, None, None


def install_meta_fusion_attention_autograd(scope):
    """Bridge npu_fusion_attention_v3 autograd on Meta if the vendor formula does not."""
    import spmd_types
    import torch_npu

    from torchtitan_npu.override.common import varlen_attention

    original = torch_npu.npu_fusion_attention_v3
    with torch.enable_grad():
        probe_q = torch.empty(16, 4, 64, dtype=torch.bfloat16, device="meta", requires_grad=True)
        probe_k = torch.empty(16, 2, 64, dtype=torch.bfloat16, device="meta", requires_grad=True)
        probe_v = torch.empty(16, 2, 64, dtype=torch.bfloat16, device="meta", requires_grad=True)
        output = original(
            query=probe_q,
            key=probe_k,
            value=probe_v,
            head_num=4,
            input_layout="TND",
            scale=64**-0.5,
            keep_prob=1.0,
            pre_tockens=16,
            next_tockens=0,
            sparse_mode=7,
        )[0]
        gradients = (
            torch.autograd.grad(output.sum(), (probe_q, probe_k, probe_v), allow_unused=True)
            if output.requires_grad
            else (None, None, None)
        )
    if all(gradient is not None for gradient in gradients):
        return "native"
    # Mirror the RMSNorm bridge contract check: the gradient op must keep the
    # documented input order. The return layout (dq, dk, dv, pse placeholder,
    # scalar tail) is positional-only knowledge for torch_npu 2.14.0.dev20260918.
    grad_schema = torch.ops.npu.npu_fusion_attention_grad_v3.default._schema
    if [argument.name for argument in grad_schema.arguments][:5] != ["query", "key", "value", "dy", "head_num"]:
        raise RuntimeError(f"Fusion attention backward contract requires review: {grad_schema}")
    spmd_types.register_local_autograd_function(_FusionAttentionV3MetaAutograd)

    seam = varlen_attention._fusion_attention_v3

    @wraps(seam)
    def fusion_attention_v3(**kwargs):
        if kwargs["query"].device.type != "meta":
            return seam(**kwargs)
        return _FusionAttentionV3MetaAutograd.apply(
            kwargs["query"],
            kwargs["key"],
            kwargs["value"],
            kwargs["atten_mask"],
            kwargs["actual_seq"],
            kwargs["head_num"],
            kwargs["scale"],
            kwargs["pre_tockens"],
            kwargs["next_tockens"],
            original,
        )

    scope.replace(varlen_attention, "_fusion_attention_v3", fusion_attention_v3)
    return "bridge"
