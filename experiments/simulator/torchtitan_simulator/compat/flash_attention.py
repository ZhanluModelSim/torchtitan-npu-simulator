# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture bridge for ``npu_fusion_attention``.

The vendor kernel pair (``FlashAttentionScore`` / ``FlashAttentionScoreGrad``)
drives both the DSA dense-mask attention and the vision tower. torch_npu
registers the ops without a Meta kernel, so meta capture needs shape-only
stand-ins plus an autograd Function whose backward emits the
``npu_fusion_attention_grad`` node — matching the device profile's fused
kernel pair. Real-device forwards keep calling the vendor op.

The autograd Function takes only tensor inputs (an optional attention mask is
passed as a 0-dim placeholder and dropped at the op boundary): torch's
autograd engine rejects non-None gradients for non-tensor apply arguments,
and the spmd type checker invokes backward with a bare ctx that only exposes
standard attributes, so scalars ride in save_for_backward as 0-dim tensors.
"""

import torch

_VENDOR_OP = "npu.npu_fusion_attention.default"
_VENDOR_BACKWARD_OP = "npu.npu_fusion_attention_grad.default"

_FA_OPS_LIB = None

# The capture always uses the head-major layout the vendor kernel shows in
# the device profile ([B, N, S, D] query/key/value).
_INPUT_LAYOUT = "BNSD"


def _fa_meta(
    query,
    key,
    value,
    head_num,
    input_layout,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tockens=2147483647,
    next_tockens=2147483647,
    inner_precise=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
    softmax_layout="",
    sink=None,
    dropout_mask=None,
    seed=0,
    offset=0,
):
    if input_layout != _INPUT_LAYOUT:
        raise NotImplementedError(f"only {_INPUT_LAYOUT} is bridged for meta capture, got {input_layout}")
    b, n, s, d = query.shape
    output = torch.empty(b, n, s, d, dtype=query.dtype, device=query.device)
    softmax_stats = torch.empty(b, n, s, 8, dtype=torch.float32, device=query.device)
    attention_in = torch.empty(b, n, s, 1, dtype=torch.float32, device=query.device)
    return output, softmax_stats, softmax_stats, attention_in, 0, 0, 0


def _fa_grad_meta(query, key, value, dy, head_num, input_layout, **kwargs):
    if input_layout != _INPUT_LAYOUT:
        raise NotImplementedError(f"only {_INPUT_LAYOUT} is bridged for meta capture, got {input_layout}")
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
        torch.empty(0, device=query.device),
        torch.empty(0, device=query.device),
    )


def _register_fake_ops() -> None:
    global _FA_OPS_LIB
    if _FA_OPS_LIB is not None:
        return
    lib = torch.library.Library("npu", "FRAGMENT")
    lib.impl("npu_fusion_attention", _fa_meta, "Meta")
    lib.impl("npu_fusion_attention_grad", _fa_grad_meta, "Meta")
    _FA_OPS_LIB = lib


class _FAMetaAutograd(torch.autograd.Function):
    """Fused attention forward + ``npu_fusion_attention_grad`` node pair.

    All apply arguments are tensors: ``atten_mask`` is the [B, 1, S, S] mask
    or a 0-dim placeholder (dropped at the op boundary), ``scale`` a 0-dim
    scalar tensor riding in save_for_backward.
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, scale):
        output, softmax_max, softmax_sum, attention_in, *_ = torch.ops.npu.npu_fusion_attention.default(
            query,
            key,
            value,
            query.shape[1],
            _INPUT_LAYOUT,
            atten_mask=atten_mask if atten_mask.dim() > 0 else None,
            scale=scale.item(),
        )
        ctx.save_for_backward(query, key, value, atten_mask, scale, softmax_max, softmax_sum, attention_in)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        query, key, value, atten_mask, scale, softmax_max, softmax_sum, attention_in = ctx.saved_tensors
        dq, dk, dv, _, _ = torch.ops.npu.npu_fusion_attention_grad.default(
            query,
            key,
            value,
            grad_output,
            query.shape[1],
            _INPUT_LAYOUT,
            atten_mask=atten_mask if atten_mask.dim() > 0 else None,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attention_in,
            scale_value=scale.item(),
        )
        return dq, dk, dv, None, None


def install_meta_flash_attention(scope):
    import spmd_types
    import torch_npu

    original = torch_npu.npu_fusion_attention
    with torch.enable_grad():
        query = torch.empty(1, 2, 4, 8, device="meta", requires_grad=True)
        try:
            torch.autograd.grad(
                torch.ops.npu.npu_fusion_attention.default(query, query, query, 2, _INPUT_LAYOUT)[0].sum(), query
            )
        except RuntimeError as error:
            message = str(error)
            if (
                "arithmetic error" not in message
                and "not implemented" not in message
                and "aclInit" not in message
                and "not have been used in the graph" not in message
            ):
                raise
        else:
            return "native"
    _register_fake_ops()
    spmd_types.register_local_autograd_function(_FAMetaAutograd)

    def fusion_attention(
        query,
        key,
        value,
        head_num,
        input_layout,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=1.0,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        inner_precise=0,
        prefix=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        sparse_mode=0,
        gen_mask_parallel=True,
        sync=False,
        softmax_layout="",
        sink=None,
        dropout_mask=None,
        seed=0,
        offset=0,
    ):
        if query.device.type != "meta":
            return original(
                query,
                key,
                value,
                head_num,
                input_layout,
                pse=pse,
                padding_mask=padding_mask,
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=keep_prob,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
                inner_precise=inner_precise,
                prefix=prefix,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                sparse_mode=sparse_mode,
                gen_mask_parallel=gen_mask_parallel,
                sync=sync,
                softmax_layout=softmax_layout,
                sink=sink,
                dropout_mask=dropout_mask,
                seed=seed,
                offset=offset,
            )
        if input_layout != _INPUT_LAYOUT:
            raise NotImplementedError(f"only {_INPUT_LAYOUT} is bridged for meta capture, got {input_layout}")
        placeholder = torch.zeros((), device=query.device)
        return (
            _FAMetaAutograd.apply(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                atten_mask if atten_mask is not None else placeholder,
                torch.tensor(float(scale) if scale is not None else 1.0),
            ),
            None,
        )

    scope.replace(torch_npu, "npu_fusion_attention", fusion_attention)
    return "bridge_fake_vendor_meta"
